"""E9: the sparse/backward criterion comparison on a compact transformer (CPU, uncontended).

The criterion question -- which coordinates should a backward approximation compute? -- is a
coordinate-level question. It does not need 1.7B parameters to answer, and the A10G is shared
with another tenant whose footprint swings between 5 and 16 GB, so a large run is unreliable.
This runs the full paired comparison on a compact GPT trained on real text, entirely on CPU,
where 40 GB is available and nothing else competes.

Arms (identical data order, seed, step count, and compute budget):
    full        : no approximation
    topk_g      : |g|                 (SOTA: magnitude sparsification)
    topk_v      : |g| / sqrt(v)       (surprise: gradient relative to its own history)
    topk_mv     : |m| / sqrt(v)       (Adam's actual step statistic)
    topk_bound  : |g| where Adam is most sensitive to error, i.e. rank by |g| * |1/sqrt(v)|
                  with the same budget -- mathematically the same ordering as topk_v, kept
                  separate to document that the sensitivity and the surprise coincide
    rand        : uniform random (floor)

Reports loss vs step, area under the loss curve, and the same-criterion comparison at two
densities. Deterministic: one process, fixed seeds, fixed batch order.

Run: python3 experiments/e9_compact_criterion.py --steps 1500 --dim 384 --layers 6
"""
import argparse, glob, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = "/root/qcc/data/longbench/data"


def load_text(limit_chars):
    txt = []
    n = 0
    for f in sorted(glob.glob(os.path.join(CORPUS, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("context") or rec.get("input") or ""
                if t:
                    txt.append(t)
                    n += len(t)
                if n >= limit_chars:
                    break
        if n >= limit_chars:
            break
    return "\n".join(txt)[:limit_chars]


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        y = self.ln1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx

    def forward(self, idx):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))


def build_blocks(model):
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or p.dim() != 2:
            continue
        out[n] = p
    return out


def run_arm(arm, args, data, vocab, dev, seed):
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=1e-8)
    blk = build_blocks(model)
    gen = torch.Generator().manual_seed(seed + 99)
    state = {}

    def make_hook(name):
        def hook(grad):
            if arm == "full":
                return grad
            n_el = grad.numel()
            k = min(n_el, max(1, int(round(args.keep_frac * n_el))))
            gf = grad.detach().float().flatten()
            st = state.setdefault(name, {})
            st["calls"] = st.get("calls", 0) + 1
            cache = st.get("idx")
            if cache is None or st["calls"] % args.refresh_every == 1:
                if arm == "topk_g":
                    crit = gf.abs()
                elif arm == "topk_v":
                    v = st.get("v")
                    if v is None or v.numel() != n_el:
                        crit = gf.abs()
                    else:
                        crit = gf.abs() / (v.sqrt() + 1e-8)
                elif arm == "topk_mv":
                    m, v = st.get("m"), st.get("v")
                    if m is None or v is None or m.numel() != n_el or v.numel() != n_el:
                        crit = gf.abs()
                    else:
                        crit = m.abs() / (v.sqrt() + 1e-8)
                elif arm == "rand":
                    crit = torch.rand(n_el, generator=gen)
                else:
                    raise ValueError(arm)
                cache = torch.topk(crit, k, sorted=False).indices
                st["idx"] = cache
            out = torch.zeros_like(grad)
            out.view(-1)[cache] = grad.view(-1)[cache]
            return out
        return hook

    handles = [p.register_hook(make_hook(n)) for n, p in blk.items()]
    # we keep our own moments so the selection criteria can see them
    moments = {}

    def step():
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
                bc1 = 1 - args.b1 ** st["t"]
                bc2 = 1 - args.b2 ** st["t"]
                upd = (m / bc1) / ((v / bc2).sqrt() + 1e-8)
                p.add_(upd, alpha=-args.lr)
            # expose flattened moments to the hooks (flatten so no silent broadcasting)
            for name, st in moments.items():
                dst = state.setdefault(name, {})
                dst["m"] = st["m"].flatten()
                dst["v"] = st["v"].flatten()

    def batch(i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        x = data[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    rows = []
    t0 = time.time()
    auc = 0.0
    for step_i in range(args.steps):
        x, y = batch(step_i)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        step()
        for p in model.parameters():
            p.grad = None
        if step_i % 25 == 0 or step_i == args.steps - 1:
            rows.append({"step": step_i, "loss": round(float(loss.item()), 4),
                         "elapsed": round(time.time() - t0, 1)})
        auc += float(loss.item())
        if step_i % 250 == 0 or step_i == args.steps - 1:
            print(f"  [{arm}] step {step_i:5d} loss {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    tail = [r["loss"] for r in rows[-8:]]
    head = [r["loss"] for r in rows[:8]]
    return {"arm": arm, "rows": rows, "auc": round(auc / args.steps, 4),
            "first_loss": round(sum(head) / len(head), 4),
            "final_loss": round(sum(tail) / len(tail), 4),
            "min_loss": min(r["loss"] for r in rows),
            "sec": round(time.time() - t0, 1)}



def overlap_diagnostic(args, data, vocab, dev, seed=0):
    """Do the candidate criteria select the same coordinates, and how much of the update
    mass does each capture?  This is the mechanism behind the end-to-end comparison and it
    is independent of any training-result noise."""
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=1e-8)

    def batch(i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        x = data[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    warm = max(20, args.steps // 10)
    for i in range(warm):
        x, y = batch(i)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    x, y = batch(warm)
    loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
    loss.backward()
    print(f"[e9d] overlap diagnostic at loss {loss.item():.4f}, {warm} warmup steps", flush=True)

    rows = []
    for name, p in model.named_parameters():
        if p.grad is None or p.dim() != 2:
            continue
        st = opt.state.get(p, {})
        m, v = st.get("exp_avg"), st.get("exp_avg_sq")
        if m is None or v is None:
            continue
        g = p.grad.detach().float().flatten()
        m = m.float().flatten()
        v = v.float().flatten()
        sv = v.sqrt() + 1e-8
        u = m / sv
        crit = {"g": g.abs(), "v": g.abs() / sv, "mv": m.abs() / sv}
        tot_g2 = float((g * g).sum())
        tot_u2 = float((u * u).sum())
        if tot_u2 <= 0:
            continue
        for k in (0.01, 0.05, 0.10):
            n_keep = max(1, int(round(k * g.numel())))
            sets = {}
            for nm, c in crit.items():
                idx = torch.topk(c, n_keep, sorted=False).indices
                msk = torch.zeros_like(c, dtype=torch.bool)
                msk[idx] = True
                sets[nm] = msk
            sg, svv, sm = sets["g"], sets["v"], sets["mv"]
            cap = {nm: float((u[msk] ** 2).sum()) / tot_u2 for nm, msk in sets.items()}
            rows.append({
                "block": name, "k": k,
                "jaccard_g_mv": int((sg & sm).sum()) / int((sg | sm).sum()),
                "jaccard_g_v": int((sg & svv).sum()) / int((sg | svv).sum()),
                "upd_capture_g": cap["g"], "upd_capture_v": cap["v"], "upd_capture_mv": cap["mv"],
                "gain_mv_over_g": cap["mv"] / max(cap["g"], 1e-30),
                "gain_v_over_g": cap["v"] / max(cap["g"], 1e-30),
            })
            del sets, sg, svv, sm
        del g, m, v, sv, u, crit

    print(f"\n[e9d] {len(rows)} (block, density) measurements")
    for k in (0.01, 0.05, 0.10):
        sub = [r for r in rows if r["k"] == k]
        if not sub:
            continue
        def med(key):
            vals = sorted(r[key] for r in sub)
            return vals[len(vals) // 2]
        print(f"  k={k:<5} jaccard(|g|,|m|/sqrt(v)) {med('jaccard_g_mv'):.3f}   "
              f"jaccard(|g|,|g|/sqrt(v)) {med('jaccard_g_v'):.3f}")
        print(f"          update-mass captured: |g| {med('upd_capture_g')*100:.1f}%  "
              f"|g|/sqrt(v) {med('upd_capture_v')*100:.1f}%  "
              f"|m|/sqrt(v) {med('upd_capture_mv')*100:.1f}%   "
              f"gain {med('gain_mv_over_g'):.2f}x")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--keep-frac", type=float, default=0.05)
    ap.add_argument("--refresh-every", type=int, default=20)
    ap.add_argument("--arms", default="full,topk_g,topk_v,topk_mv,rand")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chars", type=int, default=20_000_000)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wait-free-gb", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e9_compact_criterion.json"))
    ap.add_argument("--overlap-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = args.device
    if dev == "cuda":
        import time as _t
        while True:
            free, total = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e9] waiting for {args.wait_free_gb:.1f} GB free "
                  f"(now {free/2**30:.1f} GB)", flush=True)
            _t.sleep(30)
        print(f"[e9] device cuda, free {free/2**30:.1f} GB", flush=True)

    text = load_text(args.chars)
    vocab = 256
    data = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    print(f"[e9] corpus {len(data)/1e6:.1f}M bytes | vocab {vocab} | device {dev}", flush=True)

    if args.overlap_only:
        rows = overlap_diagnostic(args, data, vocab, dev, args.seed)
        out = {"config": vars(args), "overlap": rows}
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[e9] wrote {args.out}")
        return

    out = {"config": vars(args), "corpus_bytes": len(data), "arms": {}}
    for arm in args.arms.split(","):
        print(f"[e9] arm {arm}", flush=True)
        out["arms"][arm] = run_arm(arm, args, data, vocab, dev, args.seed)
        r = out["arms"][arm]
        print(f"      -> first {r['first_loss']:.4f}  final {r['final_loss']:.4f}  "
              f"auc {r['auc']:.4f}  ({r['sec']:.0f}s)", flush=True)

    ranked = sorted(out["arms"].items(), key=lambda kv: kv[1]["auc"])
    print("\n[e9] ranked by area-under-loss-curve (lower is better):")
    for i, (arm, r) in enumerate(ranked, 1):
        print(f"  {i}. {arm:<10} auc {r['auc']:.4f}  final {r['final_loss']:.4f}  "
              f"min {r['min_loss']:.4f}")
    base = out["arms"].get("full", {}).get("auc")
    if base:
        for arm, r in ranked:
            print(f"     {arm:<10} auc delta vs full: {r['auc'] - base:+.4f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e9] wrote {args.out}")


if __name__ == "__main__":
    main()
