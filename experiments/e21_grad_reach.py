"""E21: does the E20 skip path actually update the embedding and position tables?

E20 builds the skip arm like this:

    with torch.no_grad():
        h = tok(x) + pos(arange)          # embedding forward with grad DISABLED
        for blk in blocks[:L-k]:
            h = blk(h)
    h = h.detach().requires_grad_(True)   # boundary
    ...
    torch.autograd.grad(loss, [boundary] + kept_params + emb_params + head_params)

The embedding and position forward happened inside `no_grad`, and `h` is detached afterwards.
If that severs them from the graph, then passing their parameters to autograd.grad yields None
(swallowed by allow_unused=True), and the arm is silently *freezing* the token/position tables
rather than training them -- which invalidates the claim "embedding/head are always trained".

This measures it directly instead of reasoning about it: per-parameter gradient norms on a single
step, dense path vs skip path. A parameter whose gradient is None or exactly zero in the skip path
is not being trained.

Run: python3 experiments/e21_grad_reach.py --device cuda
"""
import argparse, json, math, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def grad_report(model, loss, params, tag):
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
    rows = {}
    for p, g in zip(params, grads):
        name = next(n for n, q in model.named_parameters() if q is p)
        rows[name] = None if g is None else float((g.float() ** 2).sum()) ** 0.5
    n_none = sum(1 for v in rows.values() if v is None)
    n_zero = sum(1 for v in rows.values() if v == 0.0)
    print(f"[{tag}] params {len(rows)}  None {n_none}  exactly-zero {n_zero}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--keep-blocks", type=int, default=1)
    ap.add_argument("--chars", type=int, default=2_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e21_grad_reach.json"))
    args = ap.parse_args()
    dev = args.device
    text = load_text(args.chars)
    vocab = 256
    data = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    need = args.bs * (args.block + 1)
    x = data[:need].view(args.bs, args.block + 1)
    x, y = x[:, :-1].to(dev), x[:, 1:].to(dev)

    out = {"config": vars(args)}
    for tag in ("dense", "skip"):
        torch.manual_seed(0)
        model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
        blocks = list(model.blocks)
        L = len(blocks)
        k = args.keep_blocks
        if tag == "dense":
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
            params = [p for _, p in model.named_parameters()]
            rows = grad_report(model, loss, params, tag)
        else:
            with torch.no_grad():
                h = model.tok(x) + model.pos(torch.arange(x.shape[1], device=dev))[None]
                for blk in blocks[:L - k]:
                    h = blk(h)
            h = h.detach().requires_grad_(True)
            for blk in blocks[L - k:]:
                h = blk(h)
            o = model.head(model.lnf(h))
            loss = F.cross_entropy(o.reshape(-1, vocab), y.reshape(-1))
            params = [p for blk in blocks[L - k:] for p in blk.parameters()]
            params += list(model.tok.parameters()) + list(model.pos.parameters())
            params += list(model.head.parameters()) + list(model.lnf.parameters())
            rows = grad_report(model, loss, params, tag)
            missing = [n for n, v in rows.items() if v is None]
            print(f"  parameters with NO gradient: {missing}")
        out[tag] = {"grad_norms": rows,
                    "none": [n for n, v in rows.items() if v is None],
                    "zero": [n for n, v in rows.items() if v == 0.0]}
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("\n[e21] verdict")
    skip_none = set(out["skip"]["none"])
    emb = {n for n in out["skip"]["grad_norms"] if "tok." in n or "pos." in n}
    emb_none = emb & skip_none
    print(f"  embedding/position params requested: {sorted(emb)}")
    print(f"  of those, receiving NO gradient in the skip path: {sorted(emb_none)}")
    if emb_none:
        print("  -> the E20 skip arm was FREEZING the token/position tables: the claim")
        print("     'embedding/head are always trained' does not hold for that build.")
    else:
        print("  -> the embedding receives gradients; the implementation is fine on this point.")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e21] wrote {args.out}")


if __name__ == "__main__":
    main()
