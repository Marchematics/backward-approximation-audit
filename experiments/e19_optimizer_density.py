"""E19: does the density-cost curve move with the optimizer?

E18 established that update-space amplification is a joint property of the optimizer and its state:
median at c=0.25 is 0.015 for SGD+momentum, 0.126 for AdamW and 2.95 for Lion, and Lion's grows 3.9x
as its momentum matures. Every sparsity and low-rank result in the literature is tuned and reported
on AdamW.

This measures the practical consequence. Same model, data, steps, sparsity mask and budget; only the
optimizer changes. Density is swept over {dense, 50%, 20%, 5%}.

Reported per (optimizer, density): training AUC, held-out loss on a disjoint slice, and the median
step-norm ratio against that optimizer's own dense run.

Prediction from E18: the cost of a fixed sparsity budget should be ordered SGD < AdamW < Lion, and
the ordering should widen as the budget shrinks. If that holds, compression ratios tuned on AdamW
are not transferable, and no paper in the sparsity literature currently reports the per-optimizer
number.

Run: python3 experiments/e19_optimizer_density.py --device cuda --steps 500
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


class Opt:
    """Minimal AdamW / Lion / SGD-momentum with an identical interface."""

    def __init__(self, name, params, lr, b1=0.9, b2=0.95, eps=1e-8, momentum=0.9):
        self.name, self.params, self.lr = name, list(params), lr
        self.b1, self.b2, self.eps, self.momentum = b1, b2, eps, momentum
        self.state, self.t = {}, 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

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
                    new_m = g.clone() if m is None else self.momentum * m + g
                    st["m"] = new_m
                    upd = new_m
                else:  # adamw
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


def run(opt_name, keep_frac, args, train, val, vocab, dev, seed=0):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    opt = Opt(opt_name, model.parameters(), args.lr, args.b1, args.b2, args.eps)
    norms, rows, auc = [], [], 0.0

    def batch(stream, i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(stream) - need - 1)
        x = stream[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    t0 = time.time()
    for i in range(args.steps):
        x, y = batch(train, i)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        if keep_frac < 1.0:
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad
                k = max(1, int(round(keep_frac * g.numel())))
                idx = torch.topk(g.detach().abs().flatten(), k, sorted=False).indices
                out = torch.zeros_like(g)
                out.view(-1)[idx] = g.view(-1)[idx]
                p.grad = out
        opt.step()
        opt.zero_grad()
        if i >= args.warmup:
            with torch.no_grad():
                s = 0.0
                for p in model.parameters():
                    st = opt.state.get(p, {})
                    if opt_name == "adamw" and "m" in st:
                        bc1 = 1 - args.b1 ** opt.t
                        bc2 = 1 - args.b2 ** opt.t
                        u = (st["m"] / bc1) / ((st["v"] / bc2).sqrt() + args.eps)
                    elif opt_name == "sgd" and "m" in st:
                        u = st["m"]
                    elif "m" in st:
                        u = torch.sign(st["m"])
                    else:
                        continue
                    s += float((u ** 2).sum())
                norms.append(math.sqrt(s))
        auc += float(loss.item())
        if i % 50 == 0 or i == args.steps - 1:
            rows.append({"step": i, "loss": round(float(loss.item()), 4)})

    model.eval()
    with torch.no_grad():
        vs = 0.0
        nb = 0
        for j in range(args.val_batches):
            x, y = batch(val, j)
            vs += float(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
            nb += 1
    res = {"optimizer": opt_name, "keep_frac": keep_frac,
           "auc": round(auc / args.steps, 4),
           "val_loss": round(vs / max(nb, 1), 4),
           "median_upd_norm": round(sorted(norms)[len(norms) // 2], 3) if norms else None,
           "final_loss": rows[-1]["loss"], "sec": round(time.time() - t0, 1)}
    del model, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
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
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--densities", default="1.0,0.5,0.2,0.05")
    ap.add_argument("--optimizers", default="sgd,adamw,lion")
    ap.add_argument("--lr-map", default="",
                    help="per-optimizer lr, e.g. sgd=1e-3,adamw=2e-4,lion=1e-4")
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e19_optimizer_density.json"))
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
            print(f"[e19] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    all_ids = list(text.encode("utf-8", errors="ignore"))
    cut = int(0.98 * len(all_ids))
    train = torch.tensor(all_ids[:cut], dtype=torch.long)
    val = torch.tensor(all_ids[cut:], dtype=torch.long)
    print(f"[e19] train {train.numel()/1e6:.1f}M / val {val.numel()/1e6:.1f}M bytes | "
          f"optimizers {opts} | densities {densities}", flush=True)

    out = {"config": vars(args), "grid": {}}
    for o in opts:
        out["grid"][o] = {}
        if o in lr_map:
            args.lr = lr_map[o]
            print(f"[e19] {o}: lr {args.lr}", flush=True)
        for d in densities:
            r = run(o, d, args, train, val, vocab, dev)
            out["grid"][o][str(d)] = r
            print(f"[e19] {o:<6} keep {d:<5} auc {r['auc']:.4f} val {r['val_loss']:.4f} "
                  f"||dtheta|| {r['median_upd_norm']}", flush=True)

    print("\n[e19] held-out loss by optimizer and density (lower is better)")
    header = "optimizer".ljust(10) + "".join(f"{('keep '+str(d)):>12}" for d in densities)
    print(header)
    for o in opts:
        dense = out["grid"][o][str(densities[0])]
        row = f"{o:<10}"
        for d in densities:
            r = out["grid"][o][str(d)]
            row += f"{r['val_loss']:>12.4f}"
        print(row)
        row2 = f"{'  cost':<10}"
        for d in densities:
            r = out["grid"][o][str(d)]
            row2 += f"{r['val_loss'] - dense['val_loss']:>+12.4f}"
        print(row2)
    print("\n[e19] step-norm ratio vs that optimizer's own dense run")
    for o in opts:
        dense = out["grid"][o][str(densities[0])]["median_upd_norm"] or 1.0
        row = f"{o:<10}"
        for d in densities:
            n = out["grid"][o][str(d)]["median_upd_norm"] or 0
            row += f"{n/dense:>12.3f}"
        print(row)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e19] wrote {args.out}")


if __name__ == "__main__":
    main()
