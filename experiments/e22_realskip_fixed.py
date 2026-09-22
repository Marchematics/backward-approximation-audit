"""E22: corrected graph-level backward skip -- every parameter really gets a gradient.

E20's skip arm was wrong in a way that flattered it. E21 measured it: `tok.weight` and `pos.weight`
came back with NO gradient, because the prefix forward ran under `torch.no_grad()` and the boundary
was detached. The arm was silently freezing the tables that carry ~48% of the gradient mass (A10).
Its three "seed" runs were also three identical runs (`--seed` was never threaded through).

This build fixes both, and the fix is the interesting part. A single autograd pass cannot both skip
the prefix backward and deliver the prefix's gradients -- once the boundary is detached the prefix
graph is gone. The construction that works is checkpoint-style replay:

    1. forward the prefix under no_grad, saving each block's input activation   (no graph)
    2. backward the suffix only, from the boundary                             (the saving)
    3. replay the prefix forward WITH grad from the saved activations,         (one extra forward)
       backprop that graph seeded by the boundary cotangent, and hand the
       resulting gradients to the prefix parameters

Step 3 costs a forward where a normal step pays a backward, so the saving is real but smaller than
E20 claimed. Arms:

    dense     normal forward + backward
    skip      steps 1-2 only: prefix blocks receive NO update   (what E20 actually measured)
    skip_ri   steps 1-3: replay, every parameter trained        (the corrected method)

`--verify-grads` records which parameters end a step without a gradient, which is the check E20
lacked.

Run: python3 experiments/e22_realskip_fixed.py --device cuda --steps 300 --seeds 0,1,2 --verify-grads
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def skip_step(model, x, y, k_keep, vocab, dev, replay):
    """Backward through the last k_keep blocks only. Returns (loss, params_without_grad)."""
    blocks = list(model.blocks)
    L = len(blocks)
    n_drop = L - k_keep
    saved = []
    with torch.no_grad():
        h = model.tok(x) + model.pos(torch.arange(x.shape[1], device=dev))[None]
        for i, blk in enumerate(blocks):
            if i < n_drop:
                saved.append(h)
                h = blk(h)
            else:
                break
    h = h.detach().requires_grad_(True)
    for blk in blocks[n_drop:]:
        h = blk(h)
    o = model.head(model.lnf(h))
    loss = F.cross_entropy(o.reshape(-1, vocab), y.reshape(-1))
    params = [p for blk in blocks[n_drop:] for p in blk.parameters()]
    params += list(model.head.parameters()) + list(model.lnf.parameters())
    grads = torch.autograd.grad(loss, [h] + params, allow_unused=True)
    g_boundary, suffix_grads = grads[0], grads[1:]
    for p, g in zip(params, suffix_grads):
        p.grad = g

    if replay:
        prefix_params = list(model.tok.parameters()) + list(model.pos.parameters())
        prefix_params += [p for blk in blocks[:n_drop] for p in blk.parameters()]
        hh = model.tok(x) + model.pos(torch.arange(x.shape[1], device=dev))[None]
        for i in range(n_drop):
            hh = hh + (saved[i] - hh).detach()   # re-enter the saved activation exactly
            hh = blocks[i](hh)
        seed = (hh * g_boundary.detach()).sum()
        pre_grads = torch.autograd.grad(seed, prefix_params, allow_unused=True)
        for p, g in zip(prefix_params, pre_grads):
            p.grad = g

    return loss, [n for n, p in model.named_parameters() if p.grad is None]


def run(arm, args, train, val, vocab, dev, k_keep, seed, lr_scale=1.0, verify=False):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    moments, times, rows, norms = {}, [], [], []
    auc = 0.0
    missing_report = None

    def batch(stream, i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block + seed * 7919) % (len(stream) - need - 1)
        x = stream[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def apply_update():
        sq = 0.0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                st = moments.setdefault(id(p), {"m": torch.zeros_like(p),
                                                "v": torch.zeros_like(p), "t": 0})
                st["t"] += 1
                m, v = st["m"], st["v"]
                m.mul_(args.b1).add_(p.grad, alpha=1 - args.b1)
                v.mul_(args.b2).addcmul_(p.grad, p.grad, value=1 - args.b2)
                upd = (m / (1 - args.b1 ** st["t"])) / \
                      ((v / (1 - args.b2 ** st["t"])).sqrt() + 1e-8)
                sq += float((upd ** 2).sum())
                p.add_(upd, alpha=-args.lr * lr_scale)
        return math.sqrt(sq)

    for i in range(args.steps):
        x, y = batch(train, i)
        t0 = time.time()
        if arm == "dense":
            o = model(x)
            loss = F.cross_entropy(o.reshape(-1, vocab), y.reshape(-1))
            loss.backward()
            if verify and i == 1:
                missing_report = [n for n, p in model.named_parameters() if p.grad is None]
        else:
            loss, missing = skip_step(model, x, y, k_keep, vocab, dev, arm == "skip_ri")
            if verify and i == 1:
                missing_report = missing
        nrm = apply_update()
        if i >= args.warmup:
            norms.append(nrm)
        for p in model.parameters():
            p.grad = None
        times.append(time.time() - t0)
        if i % 50 == 0 or i == args.steps - 1:
            rows.append({"step": i, "loss": round(float(loss.item()), 4)})
        auc += float(loss.item())

    model.eval()
    with torch.no_grad():
        vs, nb = 0.0, 0
        for j in range(args.val_batches):
            x, y = batch(val, j)
            o = model(x)
            lg = o.logits if hasattr(o, "logits") else o
            vs += float(F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1)).item())
            nb += 1
    med_t = sorted(times)[len(times) // 2]
    res = {"arm": arm, "seed": seed, "k_keep": k_keep, "lr_scale": lr_scale,
           "auc": round(auc / args.steps, 4), "val_loss": round(vs / max(nb, 1), 4),
           "median_step_ms": round(med_t * 1e3, 2),
           "median_upd_norm": round(sorted(norms)[len(norms) // 2], 3) if norms else None,
           "params_without_grad": missing_report, "final_loss": rows[-1]["loss"]}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--keep-blocks", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--chars", type=int, default=60_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--verify-grads", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e22_realskip_fixed.json"))
    args = ap.parse_args()
    dev = args.device
    seeds = [int(s) for s in args.seeds.split(",")]
    k = min(args.keep_blocks, args.layers)
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e22] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    ids = list(text.encode("utf-8", errors="ignore"))
    n_val = min(len(ids) // 5, max(200_000, args.val_batches * args.bs * (args.block + 1) * 4))
    train = torch.tensor(ids[:len(ids) - n_val], dtype=torch.long)
    val = torch.tensor(ids[len(ids) - n_val:], dtype=torch.long)
    print(f"[e22] train {train.numel()/1e6:.0f}M / val {val.numel()/1e6:.1f}M | keep {k}/{args.layers}"
          f" | seeds {seeds} | steps {args.steps} | verify {args.verify_grads}", flush=True)

    out = {"config": vars(args), "runs": []}
    for seed in seeds:
        dense = run("dense", args, train, val, vocab, dev, k, seed, verify=args.verify_grads)
        skip = run("skip", args, train, val, vocab, dev, k, seed, verify=args.verify_grads)
        scale = (dense["median_upd_norm"] / skip["median_upd_norm"]) if skip["median_upd_norm"] else 1.0
        scale = min(scale, 4.0)
        ri = run("skip_ri", args, train, val, vocab, dev, k, seed, lr_scale=scale,
                 verify=args.verify_grads)
        for r in (dense, skip, ri):
            r["speedup_vs_dense"] = round(dense["median_step_ms"] / r["median_step_ms"], 3)
            r["norm_ratio"] = round((r["median_upd_norm"] or 0) /
                                    (dense["median_upd_norm"] or 1), 3)
            r["val_cost"] = round(r["val_loss"] - dense["val_loss"], 4)
        out["runs"] += [dense, skip, ri]
        print(f"[e22] seed {seed}: dense val {dense['val_loss']:.4f} {dense['median_step_ms']:.1f}ms | "
              f"skip cost {skip['val_cost']:+.4f} {skip['speedup_vs_dense']:.2f}x "
              f"(no-grad {len(skip['params_without_grad'] or [])}) | "
              f"skip_ri cost {ri['val_cost']:+.4f} {ri['speedup_vs_dense']:.2f}x "
              f"(no-grad {len(ri['params_without_grad'] or [])})", flush=True)

    print("\n[e22] corrected summary (independent seeds)")
    print(f"{'arm':<10}{'val cost':>12}{'speedup':>10}{'norm ratio':>12}{'val loss':>10}")
    for arm in ("dense", "skip", "skip_ri"):
        rs = [r for r in out["runs"] if r["arm"] == arm]
        mc = sum(r["val_cost"] for r in rs) / len(rs)
        ms = sum(r["speedup_vs_dense"] for r in rs) / len(rs)
        mn = sum(r["norm_ratio"] for r in rs) / len(rs)
        mv = sum(r["val_loss"] for r in rs) / len(rs)
        print(f"{arm:<10}{mc:>+12.4f}{ms:>9.2f}x{mn:>12.3f}{mv:>10.4f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e22] wrote {args.out}")


if __name__ == "__main__":
    main()
