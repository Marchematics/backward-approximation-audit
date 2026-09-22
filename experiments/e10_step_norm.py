"""E10: is the cost of sparse backward an information cost or a step-size cost?

E9 found, on a compact transformer at fixed lr:
    5%  density: full 2.648 | topk_g 2.916 | topk_v 2.926 | topk_mv 3.035 | rand 3.087
    20% density: full 2.648 | topk_g 2.753 | topk_mv 2.778 | rand 2.785
    50% density: full 2.648 | topk_mv 2.696
    lr x5 for sparse arms at 5%: full 1.751 (collapses to memorisation) | topk_g 2.744 | topk_mv 2.791

Two facts: (i) the selection criterion barely matters at any density, (ii) the damage tracks the
DENSITY almost exactly. A third fact from the same runs: the sparse arms take a much smaller step.

Hypothesis. Sparse backward costs quality mainly because masking the gradient changes the *scale* of
Adam's update, not because the missing coordinates carry unique information. Under Adam the step is
m ~ EMA(g) and v ~ EMA(g^2); if only a fraction p of coordinates ever receives a gradient, those
coordinates' moments behave differently from the dense case, and the total update norm shrinks.

Test. For the same model/data/steps, record the L2 norm of the actual parameter update
    ||dtheta_t|| = || lr * m_hat / (sqrt(v_hat) + eps) ||
for the dense arm and each sparse arm at the same nominal lr. Then rerun the sparse arms with lr
rescaled by the measured ratio. If the gap closes, sparsity's cost at these densities is a
step-size artefact and the "which coordinates" question is not where the leverage is.

Run: python3 experiments/e10_step_norm.py --device cuda --steps 400
"""
import argparse, json, math, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text, build_blocks  # noqa: E402


def run(arm, args, data, vocab, dev, lr_scale=1.0, seed=0, measure_norms=False):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    lr = args.lr * lr_scale
    blk = build_blocks(model)
    state = {}
    gen = torch.Generator().manual_seed(seed + 7)

    def make_hook(name):
        def hook(grad):
            if arm == "full":
                return grad
            n_el = grad.numel()
            k = min(n_el, max(1, int(round(args.keep_frac * n_el))))
            gf = grad.detach().float().flatten()
            st = state.setdefault(name, {})
            st["calls"] = st.get("calls", 0) + 1
            cache = st.get("idx")
            if cache is None or st["calls"] % args.refresh_every == 1:
                if arm == "topk_g":
                    crit = gf.abs()
                elif arm == "topk_mv":
                    m, v = st.get("m"), st.get("v")
                    if m is None or v is None or m.numel() != n_el:
                        crit = gf.abs()
                    else:
                        crit = m.abs() / (v.sqrt() + 1e-8)
                elif arm == "rand":
                    crit = torch.rand(n_el, generator=gen)
                else:
                    raise ValueError(arm)
                cache = torch.topk(crit, k, sorted=False).indices
                st["idx"] = cache
            out = torch.zeros_like(grad)
            out.view(-1)[cache] = grad.view(-1)[cache]
            return out
        return hook

    handles = [p.register_hook(make_hook(n)) for n, p in blk.items()]
    moments = {}
    upd_norms = []

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
                if measure_norms:
                    sq += float((upd ** 2).sum())
                p.add_(upd, alpha=-lr)
            for name, st in moments.items():
                dst = state.setdefault(name, {})
                dst["m"] = st["m"].flatten()
                dst["v"] = st["v"].flatten()
        return math.sqrt(sq)

    def batch(i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        x = data[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    rows, auc = [], 0.0
    for i in range(args.steps):
        x, y = batch(i)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        nrm = step()
        if measure_norms and i >= 20:
            upd_norms.append(nrm)
        for p in model.parameters():
            p.grad = None
        if i % 25 == 0 or i == args.steps - 1:
            rows.append({"step": i, "loss": round(float(loss.item()), 4)})
        auc += float(loss.item())
    for h in handles:
        h.remove()
    tail = [r["loss"] for r in rows[-8:]]
    return {"arm": arm, "lr": lr, "auc": round(auc / args.steps, 4),
            "final_loss": round(sum(tail) / len(tail), 4),
            "median_upd_norm": round(sorted(upd_norms)[len(upd_norms) // 2], 6) if upd_norms else None,
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--keep-frac", type=float, default=0.05)
    ap.add_argument("--refresh-every", type=int, default=20)
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e10_step_norm.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        import time as _t
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e10] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            _t.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    data = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    print(f"[e10] corpus {len(data)/1e6:.1f}M bytes | {dev} | keep {args.keep_frac}", flush=True)

    out = {"config": vars(args), "phase1_norms": {}, "phase2_rescaled": {}}
    # phase 1: measure the actual update norm of each arm at the same nominal lr
    for arm in ["full", "topk_g", "topk_mv", "rand"]:
        r = run(arm, args, data, vocab, dev, 1.0, measure_norms=True)
        out["phase1_norms"][arm] = r
        print(f"[e10] {arm:<8} auc {r['auc']:.4f} final {r['final_loss']:.4f} "
              f"median||dtheta|| {r['median_upd_norm']}", flush=True)

    ref = out["phase1_norms"]["full"]["median_upd_norm"]
    print(f"\n[e10] update-norm ratios vs dense:")
    scales = {}
    for arm in ["topk_g", "topk_mv", "rand"]:
        m = out["phase1_norms"][arm]["median_upd_norm"]
        scales[arm] = ref / m if m else 1.0
        print(f"      {arm:<8} ||dtheta|| {m:.6f}  ratio {m/ref:.4f}  -> lr scale {scales[arm]:.3f}")

    # phase 2: rerun the sparse arms with the step norm matched to the dense arm
    for arm in ["topk_g", "topk_mv"]:
        r = run(arm, args, data, vocab, dev, scales[arm], measure_norms=True)
        out["phase2_rescaled"][arm] = r
        print(f"[e10] {arm:<8} lr x{scales[arm]:.2f}  auc {r['auc']:.4f} "
              f"final {r['final_loss']:.4f} median||dtheta|| {r['median_upd_norm']}", flush=True)
    out["reference"] = {"full_auc": out["phase1_norms"]["full"]["auc"],
                        "full_upd_norm": ref, "lr_scales": scales}
    print(f"\n[e10] dense auc {out['reference']['full_auc']:.4f}; "
          f"sparse at 5% without rescale: "
          f"{out['phase1_norms']['topk_g']['auc']:.4f} (topk_g) / "
          f"{out['phase1_norms']['topk_mv']['auc']:.4f} (topk_mv)")
    for arm in ["topk_g", "topk_mv"]:
        r = out["phase2_rescaled"][arm]
        closed = 1 - (r["auc"] - out["reference"]["full_auc"]) / \
                 (out["phase1_norms"][arm]["auc"] - out["reference"]["full_auc"])
        print(f"      {arm:<8} with matched step norm: auc {r['auc']:.4f} "
              f"({closed*100:.0f}% of the gap closed)")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e10] wrote {args.out}")


if __name__ == "__main__":
    main()
