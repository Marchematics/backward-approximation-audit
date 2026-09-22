"""E17: DropBP-style layer dropping, compared at EQUAL COMPUTE (not equal steps).

E16 showed the two comparison frames disagree:
    by steps   : dropbp +0.554 cost, dropbp_pin +0.053 cost  -> "pinning the embedding is better"
    by AUC/sec : dropbp 30.56,      dropbp_pin 18.50         -> "plain dropping is better"
because dropbp runs 2.38x faster per step than dense and dropbp_pin only 1.70x. Neither frame is
the honest one: at equal steps the cheap arm does less work, and at equal wall-clock it does more
steps. The question a training system actually faces is: **at a fixed compute budget, which arm
delivers the lower loss?**

So: tune each arm's layer budget until its measured ms/step equals the dense reference, then run all
arms for the same number of steps. Equal compute per step, equal steps, comparable quality.

Arms:
    fully_random  : per-step uniform random layer drops, budget tuned to dense ms/step
    grad_aware    : drop probability from gradient-norm sensitivity (DropBP's recipe),
                    budget tuned to dense ms/step
    grad_aware_pin: grad_aware with the embedding/head block never dropped, budget tuned to dense

The tuned budgets are themselves the result: they say how much extra compute the embedding block
costs, and whether sensitivity-based allocation can buy back that cost with a better layer choice.

Run: python3 experiments/e17_equal_compute.py --device cuda --steps 400
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def groups_of(model):
    g = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = n.split(".")
        gid = int(parts[1]) if parts[0] == "blocks" else -1
        g.setdefault(gid, []).append(n)
    return g


def run(mode, args, data, vocab, dev, gsens, keep_layers, steps, pin, seed=0, lr_scale=1.0,
        warmup_keep=None):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    grp = groups_of(model)
    named = dict(model.named_parameters())
    blocks = list(range(args.layers))
    gen = torch.Generator().manual_seed(seed + 13)
    if mode == "fully_random":
        raw = {b: 1.0 for b in blocks}
    else:
        raw = {b: max(gsens.get(b, 1.0), 1e-9) for b in blocks}
    sraw = sum(raw.values())
    dense_mode = keep_layers >= len(blocks)
    if dense_mode:
        keep_p = {b: 1.0 for b in blocks}
    else:
        keep_p = {b: min(1.0, raw[b] / sraw * keep_layers) for b in blocks}
    if dense_mode or pin:
        keep_p[-1] = 1.0        # never drop the embedding/head group in these modes
    moments = {}

    def batch(i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        x = data[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def step():
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

    rows, auc, times, norms = [], 0.0, [], []
    kept_hist = []
    for i in range(steps):
        x, y = batch(i)
        t0 = time.time()
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        if dense_mode:
            allowed = set(grp)
        else:
            allowed = {b for b in blocks if float(torch.rand((), generator=gen)) < keep_p[b]}
            if pin:
                allowed |= {-1}
        kept_hist.append(len(allowed))
        for gid, names in grp.items():
            if gid in allowed:
                continue
            for n in names:
                if named[n].grad is not None:
                    named[n].grad = None
        nrm = step()
        if i >= 20:
            norms.append(nrm)
        for p in model.parameters():
            p.grad = None
        times.append(time.time() - t0)
        if i % 25 == 0 or i == steps - 1:
            rows.append({"step": i, "loss": round(float(loss.item()), 4)})
        auc += float(loss.item())
    tail = [r["loss"] for r in rows[-8:]]
    med_t = sorted(times)[len(times) // 2]
    res = {"mode": mode, "pin": pin, "keep_layers": round(keep_layers, 3),
           "mean_kept": round(sum(kept_hist) / len(kept_hist), 3),
           "auc": round(auc / steps, 4), "final_loss": round(sum(tail) / len(tail), 4),
           "median_upd_norm": round(sorted(norms)[len(norms) // 2], 3) if norms else None,
           "median_step_ms": round(med_t * 1e3, 2), "rows": rows}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def tune(mode, args, data, vocab, dev, gsens, target_ms, pin):
    lo, hi, best = 0.3, float(args.layers), None
    for _ in range(5):
        mid = (lo + hi) / 2
        r = run(mode, args, data, vocab, dev, gsens, mid, args.tune_steps, pin)
        ms = r["median_step_ms"]
        print(f"    tune {mode}{'_pin' if pin else ''} keep={mid:.2f} -> {ms:.1f} ms/step "
              f"(target {target_ms:.1f})", flush=True)
        if best is None or abs(ms - target_ms) < abs(best[2] - target_ms):
            best = (mid, None, ms)
        if ms > target_ms:
            hi = mid
        else:
            lo = mid
    return best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--tune-steps", type=int, default=50)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e17_equal_compute.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e17] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    data = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    print(f"[e17] corpus {len(data)/1e6:.1f}M | equal-compute comparison", flush=True)
    out = {"config": vars(args), "arms": {}}

    # gradient-norm sensitivity from a short dense warm-up
    torch.manual_seed(0)
    probe = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    pg = groups_of(probe)
    pn = dict(probe.named_parameters())
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    need = args.bs * (args.block + 1)
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
            acc[gid] = acc.get(gid, 0.0) + sum(float((pn[n].grad ** 2).sum())
                                               for n in names if pn[n].grad is not None)
        opt.step()
        opt.zero_grad(set_to_none=True)
    gsens = {k: math.sqrt(v / 30) for k, v in acc.items()}
    out["grad_sensitivity"] = gsens
    del probe, opt
    torch.cuda.empty_cache()
    print(f"[e17] sensitivity {({k: round(v,3) for k,v in sorted(gsens.items())})}", flush=True)

    dense = run("fully_random", args, data, vocab, dev, gsens, keep_layers=float(args.layers),
                steps=args.tune_steps, pin=False)
    target_ms = dense["median_step_ms"]
    print(f"[e17] dense reference: {target_ms:.1f} ms/step ({args.tune_steps} steps)", flush=True)

    plans = [("fully_random", False), ("grad_aware", False), ("grad_aware", True)]
    tuned = {}
    for mode, pin in plans:
        print(f"[e17] tuning {mode}{'_pin' if pin else ''}", flush=True)
        kl, ms = tune(mode, args, data, vocab, dev, gsens, target_ms, pin)
        tuned[(mode, pin)] = (kl, ms)
        print(f"  -> keep budget {kl:.2f}/{args.layers} gives {ms:.1f} ms/step", flush=True)

    print(f"\n[e17] final runs at equal compute ({args.steps} steps, ~{target_ms:.0f} ms/step)",
          flush=True)
    for (mode, pin), (kl, ms) in tuned.items():
        label = f"{mode}{'_pin' if pin else ''}"
        r = run(mode, args, data, vocab, dev, gsens, kl, args.steps, pin)
        r["label"] = label
        out["arms"][label] = r
        print(f"  {label:<18} auc {r['auc']:.4f} final {r['final_loss']:.4f} "
              f"kept {r['mean_kept']:.2f}/{args.layers} {r['median_step_ms']:.0f} ms "
              f"ratio {r['median_upd_norm']/dense['median_upd_norm'] if dense['median_upd_norm'] else 0:.3f}",
              flush=True)

    # dense at the same step count for reference
    dref = run("fully_random", args, data, vocab, dev, gsens, float(args.layers), args.steps, False)
    out["arms"]["dense"] = dref
    print(f"  {'dense':<18} auc {dref['auc']:.4f} final {dref['final_loss']:.4f} "
          f"{dref['median_step_ms']:.0f} ms", flush=True)

    print("\n[e17] EQUAL-COMPUTE SUMMARY")
    print(f"{'arm':<18}{'keep':>7}{'ms/step':>9}{'AUC':>9}{'cost':>9}{'norm ratio':>12}")
    dn = dref["median_upd_norm"] or 1.0
    for lbl, r in out["arms"].items():
        print(f"{lbl:<18}{r.get('mean_kept', args.layers):>7.2f}{r['median_step_ms']:>9.0f}"
              f"{r['auc']:>9.4f}{r['auc']-dref['auc']:>+9.4f}"
              f"{(r['median_upd_norm'] or 0)/dn:>12.3f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e17] wrote {args.out}")


if __name__ == "__main__":
    main()
