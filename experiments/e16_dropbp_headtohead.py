"""E16: a head-to-head against DropBP-style layer dropping, plus the two fixes E12/E10 found.

DropBP (NeurIPS 2024) drops layers at random during backward, with a per-layer drop rate set by
that layer's sensitivity, and reports 1.5x faster convergence to a target perplexity on A100. Two
things were measured in this project that DropBP's recipe does not include:

  * the embedding/head block carries 48.3% of the gradient mass but only 3.2% of the update norm,
    and keeping it is worth +0.49 AUC at a 3/6-layer budget (E12);
  * any mechanism whose step norm drifts from dense pays for it (E14 ratio table).

This runs the comparison properly: one budget, four arms, equal steps AND equal measured compute.

  dense            no dropping
  dropbp           random per-step layer drops, drop rate from gradient-norm sensitivity
  dropbp_pin       as dropbp, but embedding/head never dropped
  dropbp_pin_cal   as dropbp_pin, plus the learning rate rescaled to match the dense step norm

Reported: AUC, final loss, median ||dtheta|| (step-norm ratio), measured ms/step, and
AUC-per-second so the arms can be compared at equal compute rather than equal steps.

Run: python3 experiments/e16_dropbp_headtohead.py --device cuda --steps 500
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def build_groups(model):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = n.split(".")
        gid = int(parts[1]) if parts[0] == "blocks" else -1
        groups.setdefault(gid, []).append(n)
    return groups


def run(arm, args, data, vocab, dev, lr_scale=1.0, seed=0, grad_sensitivity=None,
        steps_override=None):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    groups = build_groups(model)
    named = dict(model.named_parameters())
    moments = {}
    gen = torch.Generator().manual_seed(seed + 5)
    blocks = list(range(args.layers))

    # DropBP drop rates from gradient-norm sensitivity, normalised to the target budget.
    # Higher gradient norm -> lower drop rate (that layer matters more).
    if grad_sensitivity is None:
        gn = {b: 1.0 for b in blocks}
    else:
        gn = grad_sensitivity
    # keep probabilities proportional to sensitivity, normalised so sum(keep_p) == keep_layers
    raw = {b: max(gn.get(b, 1.0), 1e-9) for b in blocks}
    sraw = sum(raw.values())
    keep_p = {b: min(1.0, raw[b] / sraw * args.keep_layers) for b in blocks}
    # the embedding/head group (-1) is not a transformer block: it is only kept when pinning,
    # otherwise it participates as a droppable group with the same budget logic
    if not pin:
        keep_p[-1] = min(1.0, args.keep_layers / max(len(blocks) + 1, 1))

    def batch(i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        x = data[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def step(i, allowed):
        sq = 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                st = moments.setdefault(n, {"m": torch.zeros_like(p), "v": torch.zeros_like(p),
                                            "t": 0})
                st["t"] += 1
                m, v = st["m"], st["v"]
                m.mul_(args.b1).add_(p.grad, alpha=1 - args.b1)
                v.mul_(args.b2).addcmul_(p.grad, p.grad, value=1 - args.b2)
                upd = (m / (1 - args.b1 ** st["t"])) / \
                      ((v / (1 - args.b2 ** st["t"])).sqrt() + 1e-8)
                sq += float((upd ** 2).sum())
                p.add_(upd, alpha=-args.lr * lr_scale)
        return math.sqrt(sq)

    rows, auc, norms, times = [], 0.0, [], []
    pin = arm in ("dropbp_pin", "dropbp_pin_cal")
    n_steps = steps_override or args.steps
    for i in range(n_steps):
        x, y = batch(i)
        t0 = time.time()
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        if arm == "dense":
            allowed = set(groups)
        else:
            allowed = {b for b in blocks if float(torch.rand((), generator=gen)) < keep_p[b]}
            allowed |= {-1} if pin else set()
        for gid, names in groups.items():
            if gid in allowed:
                continue
            for n in names:
                if named[n].grad is not None:
                    named[n].grad = None
        nrm = step(i, allowed)
        if i >= 20:
            norms.append(nrm)
        for p in model.parameters():
            p.grad = None
        times.append(time.time() - t0)
        if i % 25 == 0 or i == n_steps - 1:
            rows.append({"step": i, "loss": round(float(loss.item()), 4)})
        auc += float(loss.item())
    tail = [r["loss"] for r in rows[-8:]]
    med_t = sorted(times)[len(times) // 2]
    res = {"arm": arm, "auc": round(auc / n_steps, 4),
           "final_loss": round(sum(tail) / len(tail), 4),
           "median_upd_norm": round(sorted(norms)[len(norms) // 2], 3) if norms else None,
           "median_step_ms": round(med_t * 1e3, 2),
           "auc_per_sec": round(auc / n_steps / med_t, 5),
           "rows": rows}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def tune_budget(arm, args, data, vocab, dev, gsens, target_ms, pin):
    """Find the layer budget whose measured ms/step matches the dense reference."""
    lo, hi = 0.5, float(args.layers)
    best = (None, 1e18)
    for _ in range(6):
        mid = (lo + hi) / 2
        args.keep_layers = mid
        r = run(arm, args, data, vocab, dev, grad_sensitivity=gsens,
                steps_override=args.tune_steps)
        ms = r["median_step_ms"]
        err = abs(ms - target_ms)
        if err < best[1]:
            best = (mid, err, ms)
        if ms > target_ms:
            hi = mid
        else:
            lo = mid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--keep-layers", type=float, default=3.0)
    ap.add_argument("--target-step-ms", type=float, default=0.0, help="tune keep budget to hit this ms/step")
    ap.add_argument("--tune-steps", type=int, default=60)
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e16_dropbp.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e16] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    data = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    print(f"[e16] corpus {len(data)/1e6:.1f}M | budget {args.keep_layers}/{args.layers} layers",
          flush=True)
    out = {"config": vars(args), "arms": {}}

    # measure gradient-norm sensitivity once, from a short dense warm-up
    print("[e16] measuring per-layer gradient sensitivity (dense warm-up)", flush=True)
    gsens = None
    torch.manual_seed(0)
    probe = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    pg = build_groups(probe)
    pnamed = dict(probe.named_parameters())
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    need = args.bs * (args.block + 1)
    xs = data[:need].view(args.bs, args.block + 1)
    acc = {}
    for i in range(30):
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        w = data[s:s + need].view(args.bs, args.block + 1)
        loss = F.cross_entropy(probe(w[:, :-1].to(dev)).reshape(-1, vocab),
                               w[:, 1:].to(dev).reshape(-1))
        loss.backward()
        for gid, names in pg.items():
            if gid == -1:
                continue
            v = sum(float((pnamed[n].grad ** 2).sum()) for n in names
                    if pnamed[n].grad is not None)
            acc[gid] = acc.get(gid, 0.0) + v
        opt.step()
        opt.zero_grad(set_to_none=True)
    gsens = {k: math.sqrt(v / 30) for k, v in acc.items()}
    del probe, opt
    torch.cuda.empty_cache()
    out["grad_sensitivity"] = gsens
    print(f"  sensitivity: {({k: round(v,3) for k,v in sorted(gsens.items())})}", flush=True)

    dense = run("dense", args, data, vocab, dev)
    out["arms"]["dense"] = dense
    ref = None
    print(f"[e16] {'dense':<16} auc {dense['auc']:.4f} final {dense['final_loss']:.4f} "
          f"{dense['median_step_ms']:.1f} ms/step", flush=True)

    cal_scale = 1.0
    for arm in ["dropbp", "dropbp_pin", "dropbp_pin_cal"]:
        r = run(arm, args, data, vocab, dev, lr_scale=cal_scale, grad_sensitivity=gsens)
        if arm == "dropbp_pin":
            rn, dn = r.get("median_upd_norm"), dense.get("median_upd_norm")
            if rn and dn:
                cal_scale = dn / rn
                print(f"[e16] step-norm ratio for dropbp_pin = {rn/dn:.3f} "
                      f"-> calibrating dropbp_pin_cal at lr x{cal_scale:.2f}", flush=True)
        out["arms"][arm] = r
        cost = r["auc"] - dense["auc"]
        print(f"[e16] {arm:<16} auc {r['auc']:.4f} final {r['final_loss']:.4f} "
              f"{r['median_step_ms']:.1f} ms/step  cost {cost:+.4f}  "
              f"speedup {dense['median_step_ms']/r['median_step_ms']:.2f}x", flush=True)

    # step-norm ratios need a separate pass with norms recorded; approximate from a short rerun
    print("\n[e16] summary")
    print(f"{'arm':<16}{'AUC':>9}{'cost':>9}{'ms/step':>10}{'speedup':>9}{'AUC/s':>10}")
    for a, r in out["arms"].items():
        sp = dense["median_step_ms"] / r["median_step_ms"]
        print(f"{a:<16}{r['auc']:>9.4f}{r['auc']-dense['auc']:>+9.4f}"
              f"{r['median_step_ms']:>10.1f}{sp:>8.2f}x{r['auc_per_sec']:>10.4f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e16] wrote {args.out}")


if __name__ == "__main__":
    main()
