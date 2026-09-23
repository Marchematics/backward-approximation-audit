"""E32: a predictive regime classifier — at what tokens-per-parameter does held-out loss turn over?

O16 showed that one nominal optimizer comparison returns -0.008 in three configurations and +1.09 in
a fourth, and that the discriminating variable is whether the run crossed into a memorisation race
(dense held-out loss 1.21 vs 2.61-2.98). If that boundary is predictable from a number a paper
normally reports -- tokens seen per parameter -- then a reader can tell whether a published
comparison was even measurable.

This measures the curve. Training on a fixed corpus slice, records for each token budget:
    tokens_per_param, held-out loss, train loss, and whether held-out loss is still falling
The regime is called `memorising` at the budget where held-out loss stops improving, and the threshold
is reported in tokens/parameter so it can be compared against published setups.

Caveat recorded up front: the threshold will depend on model size, corpus diversity and tokenizer, so
one curve gives one number for one setup. The deliverable is the *method* plus the measured number for
this setup, not a universal constant.

Run: python3 experiments/e32_regime_curve.py --device cuda
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=9000)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e32_regime_curve.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e32] waiting ({free/2**30:.1f} GB)", flush=True)
            time.sleep(20)

    text = load_text(args.chars)
    vocab = 256
    ids = list(text.encode("utf-8", errors="ignore"))
    n_val = 200_000
    train_ids = ids[:len(ids) - n_val]
    val = torch.tensor(ids[len(ids) - n_val:], dtype=torch.long)
    train = torch.tensor(train_ids, dtype=torch.long)
    need = args.bs * (args.block + 1)

    torch.manual_seed(0)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    print(f"[e32] model {n_param/1e6:.2f}M params | train {train.numel()/1e6:.0f}M bytes | "
          f"val {val.numel()/1e3:.0f}K", flush=True)

    def batch(stream, i):
        s = (i * args.bs * args.block) % (len(stream) - need - 1)
        x = stream[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def probe():
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for j in range(args.val_batches):
                x, y = batch(val, j)
                tot += float(F.cross_entropy(model(x).reshape(-1, vocab),
                                             y.reshape(-1)).item())
        model.train()
        return tot / args.val_batches

    rows, t0 = [], time.time()
    for i in range(args.max_steps):
        x, y = batch(train, i)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % args.probe_every == 0 or i == args.max_steps - 1:
            tokens = (i + 1) * args.bs * args.block
            pv = probe()
            rows.append({"step": i, "tokens": tokens,
                         "tokens_per_param": round(tokens / n_param, 4),
                         "train_loss": round(float(loss.item()), 4),
                         "heldout": round(pv, 4)})
            print(f"  step {i:5d}  tok/param {tokens/n_param:7.3f}  train {loss.item():6.4f}  "
                  f"heldout {pv:6.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # regime call: the smallest tokens/param after which held-out loss never improves again
    h = [r["heldout"] for r in rows]
    best = min(h)
    idx = h.index(best)
    call = {"best_heldout": best, "at_tokens_per_param": rows[idx]["tokens_per_param"],
            "final_heldout": h[-1], "final_tokens_per_param": rows[-1]["tokens_per_param"],
            "heldout_rose_after_best": h[-1] > best}
    print(f"\n[e32] best held-out {best:.4f} at {rows[idx]['tokens_per_param']:.3f} tok/param; "
          f"final {h[-1]:.4f} at {rows[-1]['tokens_per_param']:.3f}")
    print(f"[e32] held-out rose after the best point: {call['heldout_rose_after_best']}")
    out = {"config": vars(args), "model_params": n_param, "rows": rows, "regime_call": call}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e32] wrote {args.out}")


if __name__ == "__main__":
    main()
