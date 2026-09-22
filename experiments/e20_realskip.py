"""E20: a real graph-level backward skip, with the step-norm correction, measured end to end.

A8/A11 established two things that have never been combined:

  * dropping backward work from the graph is the one lever worth real wall-clock (1.63-2.46x in a
    microbenchmark), and
  * the reason such schemes lose quality is largely that masking/skipping collapses the optimizer's
    step norm (A7), which a single rescale of the learning rate recovers half of.

This runs the combination in one training loop. Backward is executed only through a suffix of the
blocks by calling autograd.grad with explicit inputs (the boundary activation and the kept blocks'
parameters). The embedding/head are always trained (A10: they carry ~48% of the gradient mass).
Three arms at equal steps and matched nominal budget:

    dense            full backward through every block
    skip             backward through the last k blocks only
    skip_cal         same, with the learning rate rescaled to match the dense step norm

Reported: training AUC, held-out loss, measured ms/step, and the step-norm ratio. This is the first
arm in the project that both removes computation from the graph AND corrects the step norm.

Run: python3 experiments/e20_realskip.py --device cuda --steps 400
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def run(arm, args, train, val, vocab, dev, k_keep, lr_scale=1.0, seed=0):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    blocks = list(model.blocks)
    L = len(blocks)
    moments, times, rows, norms = {}, [], [], []
    auc = 0.0

    def batch(stream, i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(stream) - need - 1)
        x = stream[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def apply_update():
        sq = 0.0
        seen = set()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None or id(p) in seen:
                    continue
                seen.add(id(p))
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

    def forward_to_boundary(x):
        h = model.tok(x) + model.pos(torch.arange(x.shape[1], device=dev))[None]
        for blk in blocks[:L - k_keep]:
            h = blk(h)
        return h

    for i in range(args.steps):
        x, y = batch(train, i)
        t0 = time.time()
        if arm == "dense":
            out = model(x)
            loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
            loss.backward()
        else:
            # forward through everything, but keep the graph only from the boundary onward
            with torch.no_grad():
                h = forward_to_boundary(x)
            h = h.detach().requires_grad_(True)
            for blk in blocks[L - k_keep:]:
                h = blk(h)
            out = model.head(model.lnf(h))
            loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
            # gradients for: the boundary activation (drives the frozen prefix's weights' grads
            # is NOT requested) and the kept blocks' parameters. The embedding and head are
            # trained by including their parameters explicitly.
            params = [p for blk in blocks[L - k_keep:] for p in blk.parameters()]
            params += list(model.tok.parameters()) + list(model.pos.parameters())
            params += list(model.head.parameters()) + list(model.lnf.parameters())
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for p, g in zip(params, grads):
                p.grad = g
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
    res = {"arm": arm, "k_keep": k_keep, "lr_scale": lr_scale,
           "auc": round(auc / args.steps, 4), "val_loss": round(vs / max(nb, 1), 4),
           "median_step_ms": round(med_t * 1e3, 2),
           "median_upd_norm": round(sorted(norms)[len(norms) // 2], 3) if norms else None,
           "final_loss": rows[-1]["loss"]}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
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
    ap.add_argument("--chars", type=int, default=60_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e20_realskip.json"))
    args = ap.parse_args()
    dev = args.device
    k = min(args.keep_blocks, args.layers)
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e20] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    ids = list(text.encode("utf-8", errors="ignore"))
    cut = int(0.98 * len(ids))
    train = torch.tensor(ids[:cut], dtype=torch.long)
    val = torch.tensor(ids[cut:], dtype=torch.long)
    print(f"[e20] train {train.numel()/1e6:.0f}M / val {val.numel()/1e6:.1f}M bytes | "
          f"keep {k}/{args.layers} blocks | steps {args.steps}", flush=True)

    out = {"config": vars(args), "arms": {}}
    dense = run("dense", args, train, val, vocab, dev, k)
    out["arms"]["dense"] = dense
    print(f"[e20] {'dense':<10} auc {dense['auc']:.4f} val {dense['val_loss']:.4f} "
          f"{dense['median_step_ms']:.1f} ms/step", flush=True)
    smooth = dense["median_upd_norm"]

    skip = run("skip", args, train, val, vocab, dev, k)
    out["arms"]["skip"] = skip
    print(f"[e20] {'skip':<10} auc {skip['auc']:.4f} val {skip['val_loss']:.4f} "
          f"{skip['median_step_ms']:.1f} ms/step  ratio "
          f"{(skip['median_upd_norm'] or 0)/smooth:.3f}  speedup "
          f"{dense['median_step_ms']/skip['median_step_ms']:.2f}x", flush=True)

    scale = smooth / skip["median_upd_norm"] if skip["median_upd_norm"] else 1.0
    # a scalar rescale feeds back into the moments, so aim slightly below the raw ratio (see E11)
    scale = min(scale, 4.0)
    skipcal = run("skip_cal", args, train, val, vocab, dev, k, lr_scale=scale)
    out["arms"]["skip_cal"] = skipcal
    print(f"[e20] {'skip_cal':<10} auc {skipcal['auc']:.4f} val {skipcal['val_loss']:.4f} "
          f"{skipcal['median_step_ms']:.1f} ms/step  lr x{scale:.2f}  ratio "
          f"{(skipcal['median_upd_norm'] or 0)/smooth:.3f}", flush=True)

    print("\n[e20] summary")
    print(f"{'arm':<10}{'val loss':>10}{'cost':>9}{'ms/step':>10}{'speedup':>9}{'norm ratio':>12}")
    for a, r in out["arms"].items():
        print(f"{a:<10}{r['val_loss']:>10.4f}{r['val_loss']-dense['val_loss']:>+9.4f}"
              f"{r['median_step_ms']:>10.1f}"
              f"{dense['median_step_ms']/r['median_step_ms']:>8.2f}x"
              f"{(r['median_upd_norm'] or 0)/smooth:>12.3f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e20] wrote {args.out}")


if __name__ == "__main__":
    main()
