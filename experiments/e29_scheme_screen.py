"""E29: does the risk score rank APPROXIMATION SCHEMES, not just sparsity densities?

A23 (now protocol-validated, E28) says an optimizer-specific risk score predicts the damage ordering
of a density sweep. If that extends to *different kinds of approximation*, the score becomes a
screening tool: before training, rank candidate backward-approximation schemes for the optimizer you
are actually using.

The schemes here all act on the gradient of a 2-D weight, at a named nominal budget:

    dense        no approximation
    topk50       keep the largest 50% of |g|
    topk05       keep the largest 5%
    rand05       keep a uniform random 5%          (same budget, no criterion)
    rank32       project g to its top-32 row space (low rank)
    quant4       per-tensor 4-bit quantisation of g (value error, same support)
    white        additive white noise at 5% relative norm (the pathological control)

For each scheme and each optimizer, two numbers are produced in the same run:
    risk     -- the optimizer-specific predictor (relative gradient error for SGD, preconditioned
                amplitude error for AdamW, sign-flip mass for Lion), from E23's definition
    damage   -- held-out loss minus that optimizer's dense held-out loss, at an identical step count

Then the question is simply whether `risk` orders `damage` across schemes, per optimizer, and whether
the ordering is the same for the different optimizers.

Run: python3 experiments/e29_scheme_screen.py --device cuda --steps 500
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
        self.state = {}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def upd(self, p):
        st = self.state.get(id(p), {})
        if self.name == "lion":
            m = st.get("m")
            return torch.sign(m) if m is not None else torch.sign(p.grad)
        if self.name == "sgd":
            return st.get("m", p.grad)
        m, v = st.get("m"), st.get("v")
        if m is None or v is None:
            return p.grad
        bc1 = 1 - self.b1 ** max(self.t, 1)
        bc2 = 1 - self.b2 ** max(self.t, 1)
        return (m / bc1) / ((v / bc2).sqrt() + self.eps)

    def step(self):
        self.t += 1
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state.setdefault(id(p), {})
                if self.name == "lion":
                    m = st.get("m")
                    upd = torch.sign(g) if m is None else torch.sign(self.b1 * m + (1 - self.b1) * g)
                    st["m"] = g.clone() if m is None else self.b1 * m + (1 - self.b1) * g
                elif self.name == "sgd":
                    m = st.get("m")
                    nm = g.clone() if m is None else 0.9 * m + g
                    st["m"] = nm
                    upd = nm
                else:
                    m, v = st.get("m"), st.get("v")
                    m = torch.zeros_like(p) if m is None else m
                    v = torch.zeros_like(p) if v is None else v
                    m = self.b1 * m + (1 - self.b1) * g
                    v = self.b2 * v + (1 - self.b2) * (g * g)
                    st["m"], st["v"] = m, v
                    bc1 = 1 - self.b1 ** self.t
                    bc2 = 1 - self.b2 ** self.t
                    upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
                p.add_(upd, alpha=-self.lr)


def approx(g, scheme, gen, rank=32):
    """Return the approximated gradient, at a nominal budget fixed by the scheme name."""
    if scheme == "dense":
        return g
    if scheme.startswith("topk") or scheme == "rand05":
        frac = {"topk50": 0.5, "topk05": 0.05, "rand05": 0.05}[scheme]
        k = max(1, int(round(frac * g.numel())))
        if scheme == "rand05":
            idx = torch.randperm(g.numel(), generator=gen)[:k].to(g.device)
        else:
            idx = torch.topk(g.detach().abs().flatten(), k, sorted=False).indices
        out = torch.zeros_like(g)
        out.view(-1)[idx] = g.view(-1)[idx]
        return out
    if scheme == "rank32":
        gm = g.view(g.shape[0], -1)
        r = min(rank, min(gm.shape) - 1)
        u, s, v = torch.linalg.svd(gm, full_matrices=False)
        rec = (u[:, :r] * s[:r]) @ v[:r, :]
        return rec.view_as(g)
    if scheme == "quant4":
        # per-tensor symmetric 4-bit: 15 levels, value error with full support
        amax = g.abs().max()
        if amax <= 0:
            return g
        step = amax / 7.0
        return torch.clamp(torch.round(g / step), -7, 7) * step
    if scheme == "white":
        r = torch.randn(g.numel(), generator=gen).to(g.device).view_as(g)
        r = r / (torch.linalg.vector_norm(r) + 1e-12)
        return g + r * (0.05 * torch.linalg.vector_norm(g))
    raise ValueError(scheme)


def risk_score(opt, model, scheme, gen, rank=32):
    """E23's optimizer-specific risk, evaluated on the error the given scheme actually makes."""
    tot_u2 = r_sgd = r_adam = r_lion = 0.0
    flip = n_drop = 0
    for p in model.parameters():
        if p.grad is None or p.dim() != 2:
            continue
        g = p.grad.detach().float()
        u = opt.upd(p).detach().float()
        tot_u2 += float((u * u).sum())
        gh = approx(g, scheme, gen, rank).detach().float()
        delta = gh - g
        r_sgd += float((delta ** 2).sum())
        ratio = torch.where(g.abs() > 1e-12, delta / g, torch.zeros_like(g))
        r_adam += float(((u * ratio) ** 2).sum())
        m = opt.state.get(id(p), {}).get("m")
        if m is not None:
            arg = m.float()
            fl = (torch.sign(arg + delta) != torch.sign(arg)) & (delta.abs() > 0)
            flip += float(fl.sum())
            n_drop += int((delta != 0).sum())
            r_lion += float((u.abs() * fl.float()).sum() * 2.0)
        del g, u, gh, delta, ratio
    d = max(tot_u2, 1e-30)
    return {"risk_sgd": r_sgd / d, "risk_adamw": r_adam / d, "risk_lion": r_lion / d,
            "frac_changed": flip / max(n_drop, 1)}


def run(opt_name, scheme, args, train, val, vocab, dev, seed=0):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    opt = Opt(opt_name, model.parameters(), args.lr, args.b1, args.b2, args.eps)
    gen = torch.Generator().manual_seed(seed + 31)
    risk = None
    probe_curve = []
    t0 = time.time()

    def batch(stream, i):
        need = args.bs * (args.block + 1)
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

    for i in range(args.steps):
        x, y = batch(train, i)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        if i == args.probe_at:
            risk = risk_score(opt, model, scheme, gen)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None and p.dim() == 2:
                    p.grad = approx(p.grad.detach().float(), scheme, gen).to(p.grad.dtype)
        opt.step()
        opt.zero_grad()
        if i % args.probe_every == 0 or i == args.steps - 1:
            probe_curve.append(round(probe(), 4))
            if i % (args.probe_every * 2) == 0 or i == args.steps - 1:
                print(f"  [{opt_name} {scheme}] step {i:4d} probe {probe_curve[-1]:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    res = {"optimizer": opt_name, "scheme": scheme,
           "probe_start": probe_curve[0], "probe_end": probe_curve[-1],
           "probe_curve": probe_curve, "sec": round(time.time() - t0, 1)}
    if risk:
        res.update({k: round(v, 6) for k, v in risk.items()})
    d = probe_curve[0]
    res["regime_generalising"] = bool(probe_curve[0] > 1.5)
    del model, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--probe-at", type=int, default=250)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--lr-map", default="sgd=3e-4,adamw=2e-4,lion=1e-4")
    ap.add_argument("--optimizers", default="adamw,lion,sgd")
    ap.add_argument("--schemes", default="dense,topk50,topk05,rand05,rank32,quant4,white")
    ap.add_argument("--chars", type=int, default=60_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e29_scheme_screen.json"))
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
            print(f"[e29] waiting ({free/2**30:.1f} GB)", flush=True)
            time.sleep(20)

    text = load_text(args.chars)
    vocab = 256
    ids = list(text.encode("utf-8", errors="ignore"))
    n_val = min(len(ids) // 5, max(200_000, args.val_batches * args.bs * (args.block + 1) * 4))
    train = torch.tensor(ids[:len(ids) - n_val], dtype=torch.long)
    val = torch.tensor(ids[len(ids) - n_val:], dtype=torch.long)
    print(f"[e29] train {train.numel()/1e6:.0f}M / val {val.numel()/1e6:.1f}M | steps {args.steps} "
          f"| schemes {args.schemes}", flush=True)

    out = {"config": vars(args), "runs": []}
    for o in args.optimizers.split(","):
        if o in lr_map:
            args.lr = lr_map[o]
        for s in args.schemes.split(","):
            r = run(o, s, args, train, val, vocab, dev)
            out["runs"].append(r)
            print(f"[e29] {o:<6} {s:<8} probe {r['probe_start']:.4f} -> {r['probe_end']:.4f} "
                  f"risk_lion {r.get('risk_lion',0):.4f} risk_adamw {r.get('risk_adamw',0):.4f}",
                  flush=True)
            with open(args.out + ".partial", "w") as fh:
                json.dump(out, fh, indent=2)

    print("\n[e29] scheme screening table (damage = probe_end - dense probe_end, per optimizer)")
    for o in args.optimizers.split(","):
        rs = [r for r in out["runs"] if r["optimizer"] == o]
        base = next((r["probe_end"] for r in rs if r["scheme"] == "dense"), None)
        if base is None:
            continue
        key = {"sgd": "risk_sgd", "adamw": "risk_adamw", "lion": "risk_lion"}[o]
        print(f"  {o}:  " + "  ".join(
            f"{r['scheme']} d={r['probe_end']-base:+.3f}/r={r.get(key,0):.3f}"
            for r in sorted(rs, key=lambda r: r.get(key, 0))))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e29] wrote {args.out}")


if __name__ == "__main__":
    main()
