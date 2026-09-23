"""E25: does the near-sign-boundary fraction change with model width?

The review's step 2 asked for a predictor derived from each optimizer's update rule. E23 delivered
that within an optimizer:

    Lion risk ~ P[sign(m_i + delta_i) != sign(m_i)]

That probability is a property of the distribution of |m_i| near zero, not of the model's loss. So a
scaling question follows, and it is answerable without any training-run comparison:

    as models get wider, does a larger or smaller fraction of coordinates sit close enough to the
    sign boundary that a fixed *relative* gradient error flips them?

If the fraction grows with width, sign-based optimizers become systematically more fragile to
backward approximation as models scale, which is a prediction no sparsity paper currently makes and
which the 1B comparison was supposed to measure directly.

This measures the dimensionless quantity |g_i| / sqrt(v_i) (which is what a sign flip depends on)
across model sizes, reporting the fraction of coordinates below several thresholds, in the same
training regime. No end-to-end runs, so the measurement is exactly reproducible and needs no
learning-rate calibration.

Run: python3 experiments/e25_boundary_fraction.py --device cuda
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402

THRESHOLDS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


def measure(model, batches, vocab, dev, steps, lr, b1, b2, tag):
    """Train briefly with Adam-style moments, then histogram |g|/sqrt(v) over 2-D weights."""
    hist = {t: 0 for t in THRESHOLDS}
    n_el = 0
    v_state = {}   # only the second moment is needed: the ratio |g|/sqrt(v) is what a sign
                   # flip depends on, and the first moment is never used in the statistic

    def logits(o):
        return o.logits if hasattr(o, "logits") else o

    for i in range(steps):
        x, y = batches(i)
        o = model(x)
        loss = F.cross_entropy(logits(o).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach().float()
                vs = v_state.get(id(p))
                vs = torch.zeros_like(g) if vs is None else vs
                v_state[id(p)] = b2 * vs + (1 - b2) * g * g
        for p in model.parameters():
            p.grad = None
        if i == steps - 1:
            # measure on the last batch's state
            break

    # recompute one more backward to have a gradient aligned with the state
    x, y = batches(steps)
    o = model(x)
    loss = F.cross_entropy(logits(o).reshape(-1, vocab), y.reshape(-1))
    loss.backward()
    for p in model.parameters():
        if p.grad is None or p.dim() != 2:
            continue
        g = p.grad.detach().float()
        vs = v_state.get(id(p))
        if vs is None:
            continue
        ratio = (g.abs() / (vs.sqrt() + 1e-8)).flatten()
        n_el += ratio.numel()
        for t in THRESHOLDS:
            hist[t] += int((ratio < t).sum())
        del ratio
    for p in model.parameters():
        p.grad = None
    out = {"tag": tag, "numel_m": n_el / 1e6,
           "frac_below": {str(t): round(hist[t] / max(n_el, 1), 6) for t in THRESHOLDS}}
    print(f"[e25] {tag:<28} " + "  ".join(f"P(|g|/sqrt(v)<{t})={hist[t]/max(n_el,1):.4f}"
                                          for t in THRESHOLDS), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--chars", type=int, default=20_000_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=3.0)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--model", default=os.environ.get("AUDIT_MODEL",
                                                      "/root/qcc/models/Llama-3.2-1B-Instruct"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e25_boundary_fraction.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e25] waiting ({free/2**30:.1f} GB free)", flush=True)
            time.sleep(20)

    out = {"config": vars(args), "scales": []}
    if args.pretrained:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        tok = AutoTokenizer.from_pretrained(args.model)
        vocab = AutoConfig.from_pretrained(args.model).vocab_size
        ids = tok(load_text(args.chars), add_special_tokens=False)["input_ids"]
        data = torch.tensor(ids, dtype=torch.long)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
        model.gradient_checkpointing_enable()
        nparam = sum(p.numel() for p in model.parameters()) / 1e6
        blk = args.bs * 128

        def batches(i):
            s = (i * blk) % (len(data) - blk - 1)
            x = data[s:s + blk].view(args.bs, -1)
            return x[:, :-1].to(dev), x[:, 1:].to(dev)
        r = measure(model, batches, vocab, dev, args.steps, args.lr, args.b1, args.b2,
                    f"pretrained {nparam:.0f}M params")
    else:
        ids = list(load_text(args.chars).encode("utf-8", errors="ignore"))
        data = torch.tensor(ids, dtype=torch.long)
        for name, dim, layers, heads in (("compact 10M", 384, 6, 6), ("compact 40M", 768, 8, 8)):
            torch.manual_seed(0)
            model = GPT(256, dim, layers, heads, 256).to(dev)
            nparam = sum(p.numel() for p in model.parameters()) / 1e6
            blk = args.bs * 256

            def batches(i, blk=blk):
                s = (i * blk) % (len(data) - blk - 1)
                x = data[s:s + blk].view(args.bs, -1)
                return x[:, :-1].to(dev), x[:, 1:].to(dev)
            r = measure(model, batches, 256, dev, args.steps, args.lr, args.b1, args.b2,
                        f"{name} ({nparam:.1f}M)")
            out["scales"].append(r)
            del model
            torch.cuda.empty_cache()
    out["scales"].append(r if not args.pretrained else r)

    print("\n[e25] near-boundary fraction vs scale")
    print(f"{'scale':<30}" + "".join(f"{'<'+str(t):>9}" for t in THRESHOLDS))
    for r in out["scales"]:
        print(f"{r['tag']:<30}" + "".join(f"{r['frac_below'][str(t)]:>9.4f}" for t in THRESHOLDS))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e25] wrote {args.out}")


if __name__ == "__main__":
    main()
