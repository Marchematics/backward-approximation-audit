"""E14: does the step-norm ratio predict a scheme's quality cost? The diagnostic table.

E10/E11 established, for coordinate sparsification, that masking 95% of the gradient divides the
optimizer's step norm by ~3 and that this is worth about half the resulting quality loss. This asks
the general question for the whole family of backward-approximation mechanisms:

    ratio = ||dtheta_approx|| / ||dtheta_dense||

measured over the first few steps, versus the actual quality cost. If the ratio predicts the cost
across mechanisms, it is a pre-training diagnostic: score a proposed scheme in minutes instead of
training it to convergence.

Mechanisms at matched nominal budget:
    dense            reference
    sparse_block     drop every block outside a layer subset   (DropBP-style, real speedup)
    sparse_elem      mask all but 5% of coordinates per tensor  (no FLOP saving, control)
    lowrank_row      project each 2-D weight to a rank-r row space (no FLOP saving, control)
    lowrank_down     project the gradient through a fixed rank-r down-projection (GaLore-style)
    rand_elem        uniform random mask, 5%                    (floor)

Also reports E15 wall-clock for the mechanisms that genuinely remove backward work.

Run: python3 experiments/e14_diagnostic_table.py --device cuda --steps 400
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def build_names(model):
    groups, twod = {}, {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = n.split(".")
        gid = int(parts[1]) if parts[0] == "blocks" else -1
        groups.setdefault(gid, []).append(n)
        if p.dim() == 2:
            twod[n] = p
    return groups, twod


def make_grad_transform(mechanism, args, twod, gen):
    """Returns fn(name, grad) -> grad used by the training loop."""
    if mechanism in ("dense", "sparse_block"):
        return None   # sparse_block acts via allocation, not a mask
    if mechanism in ("sparse_elem", "rand_elem"):
        def fn(name, grad):
            n_el = grad.numel()
            k = min(n_el, max(1, int(round(args.keep_frac * n_el))))
            if mechanism == "rand_elem":
                idx = torch.randperm(n_el, generator=gen)[:k].to(grad.device)
            else:
                idx = torch.topk(grad.detach().abs().flatten(), k, sorted=False).indices
            out = torch.zeros_like(grad)
            out.view(-1)[idx] = grad.view(-1)[idx]
            return out
        return fn
    if mechanism == "lowrank_row":
        def fn(name, grad):
            if name not in twod:
                return grad
            g = grad.view(grad.shape[0], -1)
            r = min(args.rank, min(g.shape) - 1)
            u, s, v = torch.linalg.svd(g, full_matrices=False)
            rec = (u[:, :r] * s[:r]) @ v[:r, :]
            return rec.view_as(grad)
        return fn
    if mechanism == "lowrank_down":
        base = {}

        def fn(name, grad):
            if name not in twod:
                return grad
            g = grad.view(grad.shape[0], -1)
            r = min(args.rank, g.shape[1] - 1)
            key = (name, g.shape)
            if key not in base:
                base[key] = torch.randn(g.shape[1], r, generator=gen).to(grad.device) / math.sqrt(r)
            P = base[key]
            return ((g @ P) @ P.T).view_as(grad)
        return fn
    raise ValueError(mechanism)


def run(mechanism, args, data, vocab, dev, alloc=None, seed=0, measure_ratio=False):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    groups, twod = build_names(model)
    named = dict(model.named_parameters())
    gen = torch.Generator(device="cpu").manual_seed(seed + 3)
    transform = make_grad_transform(mechanism, args, twod, gen)
    moments = {}

    def batch(i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        x = data[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def step(i):
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
                p.add_(upd, alpha=-args.lr)
        return math.sqrt(sq)

    rows, auc, norms = [], 0.0, []
    for i in range(args.steps):
        x, y = batch(i)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        if transform is not None:
            for n, p in list(model.named_parameters()):
                if p.grad is not None:
                    p.grad = transform(n, p.grad)
        if alloc is not None:
            for gid, names in groups.items():
                if gid in alloc:
                    continue
                for n in names:
                    if named[n].grad is not None:
                        named[n].grad = None
        nrm = step(i)
        if i >= args.warmup:
            norms.append(nrm)
        for p in model.parameters():
            p.grad = None
        if i % 25 == 0 or i == args.steps - 1:
            rows.append({"step": i, "loss": round(float(loss.item()), 4)})
        auc += float(loss.item())
    tail = [r["loss"] for r in rows[-8:]]
    res = {"mechanism": mechanism, "auc": round(auc / args.steps, 4),
           "final_loss": round(sum(tail) / len(tail), 4),
           "median_upd_norm": round(sorted(norms)[len(norms) // 2], 3) if norms else None,
           "rows": rows}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def speed_benchmark(args, data, vocab, dev):
    torch.manual_seed(0)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    L = args.layers
    blocks = list(model.blocks)
    need = args.bs * (args.block + 1)
    x = data[:need].view(args.bs, args.block + 1)[:, :-1].to(dev)
    y = data[1:need + 1].view(args.bs, args.block + 1)[:, 1:].to(dev)

    def timed(fn, reps=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            fn()
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en) / reps

    res = {}
    def full():
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).backward()
    res["full"] = timed(full)
    for k in (L // 4, L // 2):
        def trunc(k=k):
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                b, t = x.shape
                h = model.tok(x) + model.pos(torch.arange(t, device=dev))[None]
                for blk in blocks[:L - k]:
                    h = blk(h)
            h = h.detach().requires_grad_(True)
            for blk in blocks[L - k:]:
                h = blk(h)
            out = model.head(model.lnf(h))
            loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
            torch.autograd.grad(loss, [h] + [p for blk in blocks[L - k:] for p in blk.parameters()])
        res[f"trunc_last{k}"] = timed(trunc)
    del model
    torch.cuda.empty_cache()
    return res


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
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--keep-layers", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e14_diagnostic_table.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e14] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    data = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    print(f"[e14] corpus {len(data)/1e6:.1f}M | keep {args.keep_frac} | rank {args.rank}", flush=True)

    out = {"config": vars(args)}
    print("[e14] wall-clock:", flush=True)
    sp = speed_benchmark(args, data, vocab, dev)
    for k, v in sp.items():
        extra = f"  {sp['full']/v:5.2f}x" if k != "full" else ""
        print(f"  {k:<16} {v:8.2f} ms{extra}", flush=True)
    out["speed"] = sp

    # the layer subset used by the DropBP-style mechanism: gradient-norm allocation from E12
    layers = list(range(args.layers))
    alloc = set(sorted(layers, key=lambda i: -(args.layers - i))[:args.keep_layers]) | {-1}
    out["sparse_block_alloc"] = sorted(alloc)

    arms = []
    dense = run("dense", args, data, vocab, dev)
    arms.append(dense)
    print(f"[e14] {'dense':<14} auc {dense['auc']:.4f} final {dense['final_loss']:.4f} "
          f"||dtheta|| {dense['median_upd_norm']}", flush=True)
    ref = dense["median_upd_norm"]

    for mech, alloc_kw in [("sparse_block", {"alloc": alloc}),
                           ("sparse_elem", {}), ("rand_elem", {}),
                           ("lowrank_row", {}), ("lowrank_down", {})]:
        r = run(mech, args, data, vocab, dev, seed=0, **alloc_kw)
        r["ratio"] = round(r["median_upd_norm"] / ref, 4) if ref else None
        arms.append(r)
        print(f"[e14] {mech:<14} auc {r['auc']:.4f} final {r['final_loss']:.4f} "
              f"||dtheta|| {r['median_upd_norm']} ratio {r['ratio']}", flush=True)

    out["arms"] = arms
    print("\n[e14] DIAGNOSTIC TABLE (sorted by quality cost)")
    print(f"{'mechanism':<15}{'||dtheta|| ratio':>18}{'AUC':>9}{'cost':>9}{'wall-clock':>12}")
    speed_note = {"sparse_block": f"{sp['full']/sp.get('trunc_last3', sp['full']):.2f}x",
                  "sparse_elem": "1.00x (mask)",
                  "rand_elem": "1.00x (mask)",
                  "lowrank_row": "1.00x (mask)",
                  "lowrank_down": "1.00x (mask)"}
    for r in sorted(arms, key=lambda r: r["auc"]):
        cost = r["auc"] - dense["auc"]
        print(f"{r['mechanism']:<15}{str(r.get('ratio', 1.0)):>18}{r['auc']:>9.4f}"
              f"{cost:>+9.4f}{speed_note.get(r['mechanism'], '-'):>12}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e14] wrote {args.out}")


if __name__ == "__main__":
    main()
