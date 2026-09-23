"""E26: is a sign-based optimizer MORE tolerant of skipping backward steps than AdamW?

The project's story so far says Lion is the fragile one: its amplification of a gradient error is
2.95x against AdamW's 0.126 (A14), and its exposure to sign flips grows with design choice (A28).
Both of those are about *perturbing* the gradient. This asks the opposite question, and the answer
is not implied by them:

    if a block's gradient is simply not computed for a step, which optimizer degrades less?

The mechanism that suggests Lion should win: Lion's update is `sign(m)`, which depends only on the
*direction* of the accumulated momentum. A stale momentum still points roughly the right way, so
skipping a step leaves the update qualitatively similar. AdamW's update is `m/(sqrt(v)+eps)`, which
depends on the accumulated *magnitude*; a stale `v` makes the step size wrong in a way that grows
with how much the gradient's scale has moved. So the prediction is:

    H_Lion  : Lion tolerates a 1-in-N backward schedule far better than AdamW
    H_AdamW : AdamW's damage grows faster with N

This is directly testable and it is a method-layer question: if true, the right backward schedule is
optimizer-dependent, which is the concrete form of "optimizer-conditioned backward execution" and
the thing E22 could not deliver by recomputation.

Arms: optimizer in {adamw, lion} x backward every {1, 2, 4, 8} steps. At every forward step the
optimizer still applies its update from the moments it has; the gradient is simply refreshed on a
schedule. Held-out loss is measured with a fixed probe, which is the only trustworthy signal (per
batch loss cannot resolve a trend -- see the E24 lesson).

Run: python3 experiments/e26_staleness.py --device cuda --steps 1200
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


class Opt:
    def __init__(self, name, params, lr, b1=0.9, b2=0.95, eps=1e-8):
        self.name, self.params, self.lr = name, list(params), lr
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, fresh=True):
        # `t` counts updates the optimizer actually applies; for AdamW a stale schedule otherwise
        # changes the bias-correction exponents and rescales the step for reasons unrelated to
        # staleness. Lion has no bias correction, so this only matters for AdamW.
        self.t += 1 if (fresh or self.name == "lion") else 0
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad
                st = getattr(p, "_st", None)
                if st is None:
                    st = {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
                    p._st = st
                m, v = st["m"], st["v"]
                if self.name == "lion":
                    upd = torch.sign(g) if self.t == 1 else torch.sign(self.b1 * m + (1 - self.b1) * g)
                    st["m"] = g.clone() if self.t == 1 else self.b1 * m + (1 - self.b1) * g
                else:
                    m = self.b1 * m + (1 - self.b1) * g
                    v = self.b2 * v + (1 - self.b2) * (g * g)
                    st["m"], st["v"] = m, v
                    bc1 = 1 - self.b1 ** self.t
                    bc2 = 1 - self.b2 ** self.t
                    upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
                p.add_(upd, alpha=-self.lr)


def run(opt_name, every, args, train, val, vocab, dev, seed=0):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    opt = Opt(opt_name, model.parameters(), args.lr, args.b1, args.b2, args.eps)
    probe_curve, t0 = [], time.time()

    def batch(stream, i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(stream) - need - 1)
        x = stream[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def fixed_probe():
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for j in range(args.val_batches):
                x, y = batch(val, j)
                tot += float(F.cross_entropy(model(x).reshape(-1, vocab),
                                             y.reshape(-1)).item())
        model.train()
        return tot / args.val_batches

    # Two distinct mechanisms, because they are different systems questions:
    #   "reuse"  -- the backward is skipped and the optimizer reuses the moments it built from the
    #               last gradient it saw. This is the real saving (no backward at all on that step)
    #               and it tests whether stale information still helps.
    #   "skip"   -- the backward is skipped and no update is applied at all. The control: it isolates
    #               how much of the "reuse" behaviour is just the extra progress from re-applying.
    reuse = args.mechanism == "reuse"
    for i in range(args.steps):
        x, y = batch(train, i)
        fresh = (i % every == 0)
        if fresh:
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p._cached = p.grad.detach().clone()
        if fresh or reuse:
            if not fresh:
                # restore the cached gradient so the optimizer applies its update from stale data
                for p in model.parameters():
                    c = getattr(p, "_cached", None)
                    if c is not None:
                        p.grad = c.clone()
            opt.step(fresh=fresh)
        opt.zero_grad()
        if i % args.probe_every == 0 or i == args.steps - 1:
            pv = fixed_probe()
            probe_curve.append(round(pv, 4))
            if i % (args.probe_every * 2) == 0 or i == args.steps - 1:
                tl = float(loss.item()) if fresh else float("nan")
                print(f"  [{opt_name} every {every} {args.mechanism}] step {i:5d} "
                      f"train {tl:.4f} probe {pv:.4f} ({time.time()-t0:.0f}s)", flush=True)
    res = {"optimizer": opt_name, "backward_every": every, "mechanism": args.mechanism,
           "probe_start": probe_curve[0], "probe_end": probe_curve[-1],
           "probe_curve": probe_curve,
           "sec": round(time.time() - t0, 1)}
    del model, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--lr-map", default="adamw=2e-4,lion=1e-4")
    ap.add_argument("--optimizers", default="adamw,lion")
    ap.add_argument("--schedules", default="1,2,4,8")
    ap.add_argument("--mechanism", default="reuse", choices=["reuse", "skip"])
    ap.add_argument("--chars", type=int, default=60_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e26_staleness.json"))
    args = ap.parse_args()
    dev = args.device
    lr_map = {}
    for kv in args.lr_map.split(","):
        if "=" in kv:
            k, v = kv.split("=")
            lr_map[k.strip()] = float(v)
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e26] waiting ({free/2**30:.1f} GB)", flush=True)
            time.sleep(20)

    text = load_text(args.chars)
    vocab = 256
    ids = list(text.encode("utf-8", errors="ignore"))
    n_val = min(len(ids) // 5, max(200_000, args.val_batches * args.bs * (args.block + 1) * 4))
    train = torch.tensor(ids[:len(ids) - n_val], dtype=torch.long)
    val = torch.tensor(ids[len(ids) - n_val:], dtype=torch.long)
    print(f"[e26] train {train.numel()/1e6:.0f}M / val {val.numel()/1e6:.1f}M | steps {args.steps}",
          flush=True)

    out = {"config": vars(args), "runs": []}
    for o in args.optimizers.split(","):
        if o in lr_map:
            args.lr = lr_map[o]
        for every in [int(x) for x in args.schedules.split(",")]:
            r = run(o, every, args, train, val, vocab, dev)
            out["runs"].append(r)
            print(f"[e26] {o:<6} backward every {every:<2} probe {r['probe_start']:.4f} -> "
                  f"{r['probe_end']:.4f}  ({r['sec']:.0f}s)", flush=True)
            with open(args.out + ".partial", "w") as fh:
                json.dump(out, fh, indent=2)

    print("\n[e26] damage vs backward schedule (probe_end - probe at every=1, same optimizer)")
    for o in args.optimizers.split(","):
        rs = [r for r in out["runs"] if r["optimizer"] == o]
        base = next((r["probe_end"] for r in rs if r["backward_every"] == 1), None)
        print(f"  {o}:")
        for r in sorted(rs, key=lambda r: r["backward_every"]):
            d = (r["probe_end"] - base) if base is not None else float("nan")
            print(f"    every {r['backward_every']:<2} probe {r['probe_end']:.4f}  damage {d:+.4f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e26] wrote {args.out}")


if __name__ == "__main__":
    main()
