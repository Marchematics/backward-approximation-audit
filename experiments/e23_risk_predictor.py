"""E23: a cheap, measurable predictor of what a gradient approximation will cost.

The measurement programme so far produced one diagnostic that works within an optimizer (the step
norm, A7/A12) and one that inverts across optimizers (A18/F8). The review's strongest suggestion was
to replace both with a quantity derived from each optimizer's own update rule:

    SGD      the update IS the gradient  -> risk ~ ||delta||/||g||
    AdamW    the update is m/(sqrt(v)+eps) -> risk ~ a preconditioned amplitude error,
             sum_i u_i^2 (delta_i/g_i)^2 / sum_i u_i^2
    Lion     the update is sign(.)       -> risk ~ P[sign(m_i + delta_i) != sign(m_i)],
             i.e. the probability that the perturbation crosses a sign boundary

This computes all three *from the same training states* and asks whether the optimizer-specific risk
predicts the held-out damage of a sparsity budget, across optimizers and densities.

To make the comparison meaningful every optimizer is trained at its own calibrated learning rate,
and the risk is evaluated on the same coordinate budget it is predicting.

Run: python3 experiments/e23_risk_predictor.py --device cuda --steps 400
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


class Opt:
    """AdamW / Lion / SGD-momentum with a common interface and exposed state."""

    def __init__(self, name, params, lr, b1=0.9, b2=0.95, eps=1e-8, momentum=0.9):
        self.name, self.params, self.lr = name, list(params), lr
        self.b1, self.b2, self.eps, self.momentum = b1, b2, eps, momentum
        self.state, self.t = {}, 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def update_of(self, p):
        st = self.state.get(p, {})
        if self.name == "adamw":
            m, v = st.get("m"), st.get("v")
            if m is None:
                return p.grad
            bc1 = 1 - self.b1 ** self.t
            bc2 = 1 - self.b2 ** self.t
            return (m / bc1) / ((v / bc2).sqrt() + self.eps)
        if self.name == "sgd":
            return st.get("m", p.grad)
        if self.name == "lion":
            m = st.get("m")
            return torch.sign(m) if m is not None else torch.sign(p.grad)
        raise ValueError(self.name)

    def step(self):
        self.t += 1
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state.setdefault(p, {})
                if self.name == "lion":
                    m = st.get("m")
                    upd = torch.sign(g) if m is None else torch.sign(self.b1 * m + (1 - self.b1) * g)
                    st["m"] = g.clone() if m is None else self.b1 * m + (1 - self.b1) * g
                elif self.name == "sgd":
                    m = st.get("m")
                    nm = g.clone() if m is None else self.momentum * m + g
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


def risk_scores(opt, model, c):
    """Optimizer-specific risk of masking all but the top-c fraction of |g|.

    Returns dict(risk, and the components used) summed over 2-D parameters.
    """
    tot_u2 = 0.0
    risk_pre = 0.0      # AdamW: preconditioned amplitude error
    risk_sgd = 0.0      # SGD: relative error of the retained set
    risk_lion = 0.0     # Lion: expected sign-flip mass
    flip_frac = 0.0
    n_el = 0
    for p in model.parameters():
        if p.grad is None or p.dim() != 2:
            continue
        g = p.grad.detach().float()
        u = opt.update_of(p).detach().float()
        u2 = float((u * u).sum())
        tot_u2 += u2
        k = max(1, int(round(c * g.numel())))
        # the mask a top-|g| budget would keep, and the error it leaves behind
        idx = torch.topk(g.abs().flatten(), k, sorted=False).indices
        keep = torch.zeros(g.numel(), dtype=torch.bool, device=g.device)
        keep[idx] = True
        keep = keep.view_as(g)
        delta = torch.where(keep, torch.zeros_like(g), g)      # dropped coordinates carry the error
        # component 1: relative gradient error (what every paper reports)
        risk_sgd += float((delta ** 2).sum())
        # component 2: update-space amplitude error, weighted by the coordinate's own sensitivity
        with torch.no_grad():
            ratio = torch.where(g.abs() > 1e-12, delta / g, torch.zeros_like(g))
        risk_pre += float(((u * ratio) ** 2).sum())
        # component 3: sign-flip mass -- how much of the update changes sign when delta is added
        st = opt.state.get(p, {})
        m = st.get("m")
        if m is not None:
            arg = m.float()
            flipped = (torch.sign(arg + delta) != torch.sign(arg)) & (delta.abs() > 0)
            n_el += int((delta != 0).sum())
            flip_frac += float(flipped.sum())
            risk_lion += float((u.abs() * flipped.float()).sum() * 2.0)  # a flipped sign costs 2|u|
    denom = max(tot_u2, 1e-30)
    return {"risk_sgd": risk_sgd / denom,
            "risk_adamw": risk_pre / denom,
            "risk_lion": risk_lion / denom,
            "flip_frac_of_dropped": flip_frac / max(n_el, 1)}


def run(opt_name, keep_frac, args, train, val, vocab, dev, seed=0, mask_scope="2d"):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    opt = Opt(opt_name, model.parameters(), args.lr, args.b1, args.b2, args.eps)
    auc, rows = 0.0, []
    risks = None

    def batch(stream, i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(stream) - need - 1)
        x = stream[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    for i in range(args.steps):
        x, y = batch(train, i)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        if i == args.probe_at:
            # measure the risk of the budget this arm is actually running, on live state
            risks = risk_scores(opt, model, keep_frac)
        maskable = [p for p in model.parameters() if p.grad is not None]
        if mask_scope == "2d":
            maskable = [p for p in maskable if p.dim() == 2]
        if keep_frac < 1.0:
            for p in maskable:
                g = p.grad
                k = max(1, int(round(keep_frac * g.numel())))
                idx = torch.topk(g.detach().abs().flatten(), k, sorted=False).indices
                out = torch.zeros_like(g)
                out.view(-1)[idx] = g.view(-1)[idx]
                p.grad = out
        opt.step()
        opt.zero_grad()
        auc += float(loss.item())
        if i % 50 == 0 or i == args.steps - 1:
            rows.append({"step": i, "loss": round(float(loss.item()), 4)})

    model.eval()
    with torch.no_grad():
        vs, nb = 0.0, 0
        for j in range(args.val_batches):
            x, y = batch(val, j)
            vs += float(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
            nb += 1
    res = {"optimizer": opt_name, "keep_frac": keep_frac, "auc": round(auc / args.steps, 4),
           "val_loss": round(vs / max(nb, 1), 4), "final_loss": rows[-1]["loss"]}
    if risks:
        res.update({k: round(v, 6) for k, v in risks.items()})
    del model, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--probe-at", type=int, default=200)
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
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--mask-scope", default="2d", choices=["2d", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-offset", type=int, default=0,
                    help="byte offset of the training slice, for configuration-dependence tests")
    ap.add_argument("--densities", default="1.0,0.5,0.2,0.05")
    ap.add_argument("--optimizers", default="sgd,adamw,lion")
    ap.add_argument("--lr-map", default="sgd=3e-4,adamw=2e-4,lion=1e-4")
    ap.add_argument("--chars", type=int, default=60_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e23_risk_predictor.json"))
    args = ap.parse_args()
    dev = args.device
    densities = [float(x) for x in args.densities.split(",")]
    opts = args.optimizers.split(",")
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
            print(f"[e23] waiting ({free/2**30:.1f} GB free)", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    ids = list(text.encode("utf-8", errors="ignore"))
    n_val = min(len(ids) // 5, max(200_000, args.val_batches * args.bs * (args.block + 1) * 4))
    off = min(args.train_offset, max(0, len(ids) - n_val - 1))
    train = torch.tensor(ids[off:len(ids) - n_val], dtype=torch.long)
    val = torch.tensor(ids[len(ids) - n_val:], dtype=torch.long)
    print(f"[e23] train {train.numel()/1e6:.0f}M val {val.numel()/1e6:.1f}M | opts {opts} "
          f"| densities {densities} | mask {args.mask_scope} | seed {args.seed} "
          f"| train_offset {args.train_offset}", flush=True)

    out = {"config": vars(args), "rows": []}
    for o in opts:
        if o in lr_map:
            args.lr = lr_map[o]
        for d in densities:
            r = run(o, d, args, train, val, vocab, dev, seed=args.seed, mask_scope=args.mask_scope)
            out["rows"].append(r)
            print(f"[e23] {o:<6} keep {d:<5} val {r['val_loss']:.4f} "
                  f"risk_sgd {r.get('risk_sgd',0):.4f} risk_adamw {r.get('risk_adamw',0):.4f} "
                  f"risk_lion {r.get('risk_lion',0):.4f} flip {r.get('flip_frac_of_dropped',0):.4f}",
                  flush=True)
            with open(args.out + ".partial", "w") as fh:
                json.dump(out, fh, indent=2)

    # does an optimizer-specific risk predict its own damage?
    print("\n[e23] damage vs risk, per optimizer (damage = val loss - that optimizer's dense)")
    for o in opts:
        rs = [r for r in out["rows"] if r["optimizer"] == o]
        dense = next((r for r in rs if r["keep_frac"] == 1.0), None)
        if not dense:
            continue
        print(f"  {o}:")
        for r in rs:
            dmg = r["val_loss"] - dense["val_loss"]
            key = {"sgd": "risk_sgd", "adamw": "risk_adamw", "lion": "risk_lion"}[o]
            print(f"    keep {r['keep_frac']:<5} damage {dmg:+.4f}  {key} {r.get(key,0):.4f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e23] wrote {args.out}")


if __name__ == "__main__":
    main()
