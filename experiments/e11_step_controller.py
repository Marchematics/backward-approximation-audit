"""E11: make sparse backward quality-neutral by fixing the optimizer, not the criterion.

E10 established that at 5% density a masked backward divides the optimizer's step norm by ~3
(ratios 0.32-0.48 of dense), and that a one-shot lr rescale closes 52-55% of the quality gap.
This asks whether the rest can be closed by controlling the step norm *online*, and whether the
residual is a moment-consistency problem.

Arms (all at keep_frac, same data order and step count):
  full            : dense reference
  topk_g          : magnitude criterion, nominal lr (the SOTA baseline)
  topk_g_cal      : magnitude criterion, fixed lr rescaled by the E10 ratio
  topk_g_ctrl     : magnitude criterion, online controller holding ||dtheta|| at the dense level
  topk_g_ctrl_mom : as ctrl, plus moments frozen for coordinates outside the selected set
                    (i.e. m and v are only decayed/updated on coordinates that receive signal)

Reported per arm: AUC, final loss, median ||dtheta||, the lr trajectory of the controller, and the
fraction of the dense-vs-sparse AUC gap that is closed. If ctrl reaches the dense AUC, sparse
backward is quality-neutral at 5% and the speedup story is alive; if it plateaus, the remaining
cost is a genuine information cost and no scheduler will recover it.

Run: python3 experiments/e11_step_controller.py --device cuda --steps 400
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text, build_blocks  # noqa: E402


def make_hook(arm, state, keep_frac, refresh_every, gen):
    def hook_for(name):
        def hook(grad):
            if arm == "full":
                return grad
            n_el = grad.numel()
            k = min(n_el, max(1, int(round(keep_frac * n_el))))
            gf = grad.detach().float().flatten()
            st = state.setdefault(name, {})
            st["calls"] = st.get("calls", 0) + 1
            cache = st.get("idx")
            if cache is None or st["calls"] % refresh_every == 1:
                crit = gf.abs()
                cache = torch.topk(crit, k, sorted=False).indices
                st["idx"] = cache
            out = torch.zeros_like(grad)
            out.view(-1)[cache] = grad.view(-1)[cache]
            return out
        return hook
    return hook_for


def run(arm, args, data, vocab, dev, target_norm=None, seed=0, freeze_outside_moments=False):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    blk = build_blocks(model)
    state, moments = {}, {}
    gen = torch.Generator().manual_seed(seed + 7)
    hook_for = make_hook(arm, state, args.keep_frac, args.refresh_every, gen)
    handles = [p.register_hook(hook_for(n)) for n, p in blk.items()]

    lr_scale = args.cal_scale if arm == "topk_g_cal" else 1.0
    ctrl = {"scale": 1.0, "hist": []}
    rows, auc, norms = [], 0.0, []

    def step(i):
        cur_lr = args.lr * (ctrl["scale"] if arm == "topk_g_ctrl" else lr_scale)
        sq = 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                st = moments.setdefault(n, {"m": torch.zeros_like(p), "v": torch.zeros_like(p),
                                            "t": 0})
                st["t"] += 1
                m, v = st["m"], st["v"]
                mask = state.get(n, {}).get("idx") if freeze_outside_moments else None
                if mask is not None:
                    gflat = p.grad.flatten()
                    mf, vf = m.flatten(), v.flatten()
                    mf[mask] = args.b1 * mf[mask] + (1 - args.b1) * gflat[mask]
                    vf[mask] = args.b2 * vf[mask] + (1 - args.b2) * (gflat[mask] ** 2)
                    # coordinates outside the mask keep their moments (no decay)
                    m.copy_(mf.view_as(m))
                    v.copy_(vf.view_as(v))
                else:
                    m.mul_(args.b1).add_(p.grad, alpha=1 - args.b1)
                    v.mul_(args.b2).addcmul_(p.grad, p.grad, value=1 - args.b2)
                upd = (m / (1 - args.b1 ** st["t"])) / \
                      ((v / (1 - args.b2 ** st["t"])).sqrt() + 1e-8)
                sq += float((upd ** 2).sum())
                p.add_(upd, alpha=-cur_lr)
            for name, st in moments.items():
                dst = state.setdefault(name, {})
                dst["m"] = st["m"].flatten()
                dst["v"] = st["v"].flatten()
        norm = math.sqrt(sq)
        if arm == "topk_g_ctrl" and target_norm and norm > 0 and i >= args.ctrl_warmup:
            # proportional controller on the update norm
            err = target_norm / norm
            ctrl["scale"] = max(0.1, min(20.0, ctrl["scale"] * (err ** args.ctrl_gain)))
        if i % 5 == 0:
            ctrl["hist"].append(round(ctrl["scale"] if arm == "topk_g_ctrl" else lr_scale, 4))
        return norm

    def batch(i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        x = data[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    for i in range(args.steps):
        x, y = batch(i)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        nrm = step(i)
        if i >= 20:
            norms.append(nrm)
        for p in model.parameters():
            p.grad = None
        if i % 25 == 0 or i == args.steps - 1:
            rows.append({"step": i, "loss": round(float(loss.item()), 4)})
        auc += float(loss.item())
    for h in handles:
        h.remove()
    tail = [r["loss"] for r in rows[-8:]]
    return {"arm": arm, "auc": round(auc / args.steps, 4),
            "final_loss": round(sum(tail) / len(tail), 4),
            "median_upd_norm": round(sorted(norms)[len(norms) // 2], 3) if norms else None,
            "lr_hist": ctrl["hist"], "rows": rows}


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
    ap.add_argument("--cal-scale", type=float, default=2.76, help="E10 measured ratio for topk_g")
    ap.add_argument("--ctrl-warmup", type=int, default=15)
    ap.add_argument("--ctrl-gain", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e11_step_controller.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e11] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    data = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    print(f"[e11] corpus {len(data)/1e6:.1f}M | keep {args.keep_frac} | steps {args.steps}",
          flush=True)

    out = {"config": vars(args), "arms": {}}
    full = run("full", args, data, vocab, dev)
    out["arms"]["full"] = full
    print(f"[e11] full        auc {full['auc']:.4f} final {full['final_loss']:.4f} "
          f"||dtheta|| {full['median_upd_norm']}", flush=True)
    target = full["median_upd_norm"]

    specs = [("topk_g", False, None), ("topk_g_cal", False, None),
             ("topk_g_ctrl", False, target), ("topk_g_ctrl_mom", True, target)]
    for arm, freeze, tgt in specs:
        r = run(arm, args, data, vocab, dev, target_norm=tgt, freeze_outside_moments=freeze)
        out["arms"][arm] = r
        gap = out["arms"]["topk_g"]["auc"] - full["auc"] if "topk_g" in out["arms"] else None
        closed = None
        if gap:
            closed = 1 - (r["auc"] - full["auc"]) / gap
        print(f"[e11] {arm:<15} auc {r['auc']:.4f} final {r['final_loss']:.4f} "
              f"||dtheta|| {r['median_upd_norm']}"
              + (f"  gap closed {closed*100:.0f}%" if closed is not None else ""), flush=True)

    base = out["arms"]["topk_g"]["auc"]
    print(f"\n[e11] dense {full['auc']:.4f} | SOTA sparse {base:.4f} | gap {base-full['auc']:+.4f}")
    for arm, r in out["arms"].items():
        if arm in ("full", "topk_g"):
            continue
        print(f"      {arm:<15} {r['auc']:.4f}  -> {100*(1-(r['auc']-full['auc'])/(base-full['auc'])):.0f}%"
              f" of the gap closed")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e11] wrote {args.out}")


if __name__ == "__main__":
    main()
