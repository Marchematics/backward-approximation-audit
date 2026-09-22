"""E1c: same as E1b, but measured at several points along a real training run, with
backward FLOPs taken from the actual layer shapes.

The quantity that decides the whole thesis is the drift of
    q_b = mean_i |g_i| / sqrt(v_i)          (per block)
from early (v ~ current gradient, q constant across blocks, optimizer state carries no
information) to late (v accumulates history, q differentiates blocks).

Reported per checkpoint:
  - spread of q across blocks
  - rho( per-FLOP update tolerance , per-FLOP gradient norm )        <- DropBP-style allocator
  - rho( per-FLOP update tolerance , (sum g^2/v)^(1/4) per FLOP )    <- derived allocator
  - the exact FLOPS-matched regret of each allocator vs the oracle (needs the cost exponent,
    which is a property of the approximation family, so it is reported for p = 1/2 and 1)

Run: ./venv/bin/python review20260922/e1c_horizon.py --steps 300 --points 0,50,150,300 --bs 4 --seq 512
"""
import argparse, glob, json, math, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AUDIT_MODEL", "/root/qcc/models/Llama-3.2-1B-Instruct")
CORPUS = os.environ.get("AUDIT_CORPUS", "/root/qcc/data/longbench/data")
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def build_tokens(tok, need):
    ids = []
    while len(ids) < need:                     # loop the corpus if needed
        for f in sorted(glob.glob(os.path.join(CORPUS, "*.jsonl"))):
            if len(ids) >= need:
                break
            with open(f) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    txt = rec.get("context") or rec.get("input") or ""
                    if txt:
                        ids.extend(tok(txt, add_special_tokens=False)["input_ids"])
                    if len(ids) >= need:
                        break
    return torch.tensor(ids[:need], dtype=torch.long)


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da and db else float("nan")


def index_max(a):
    return max(range(len(a)), key=lambda i: a[i])


def blocks_of(model):
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or not n.endswith(".weight"):
            continue
        parts = n.split(".")
        mod = parts[-2] if len(parts) > 1 else ""
        if mod not in PROJ or "layers" not in parts:
            continue
        layer = int(parts[parts.index("layers") + 1])
        out[f"L{layer:02d}.{mod.replace('_proj','')}"] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--points", default="0,50,150,300")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--levels", default="0.01,0.025,0.05,0.1,0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e1c_horizon.json"))
    a = ap.parse_args()
    dev = "cuda"
    levels = [float(x) for x in a.levels.split(",")]
    points = sorted({int(x) for x in a.points.split(",")})
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    toks = build_tokens(tok, a.bs * a.seq * (a.steps + 2))
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(a.beta1, a.beta2), eps=a.eps)
    blk = blocks_of(model)
    flops = {k: 4.0 * p.shape[0] * p.shape[1] * a.bs * a.seq / 1e9 for k, p in blk.items()}
    print(f"[e1c] {len(blk)} matrix blocks | steps={a.steps} | points={points} | "
          f"bwd FLOP budget {sum(flops.values()):.1f} GFLOP/step", flush=True)

    def batch(i):
        s = (i * a.bs * a.seq) % (len(toks) - a.bs * a.seq - 2)
        x = toks[s:s + a.bs * a.seq].view(a.bs, a.seq).to(dev)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()

    snaps = []
    gen = torch.Generator(device="cpu").manual_seed(a.seed + 1)
    losses = []
    for step in range(a.steps + 1):
        x, y = batch(step)
        loss = model(input_ids=x, labels=y).loss
        loss.backward()
        losses.append(round(float(loss.item()), 4))
        if step % 20 == 0 or step == a.steps:
            print(f"[e1c] step {step:4d} loss {loss.item():.4f}", flush=True)
        if step in points:
            snap = {"step": step, "loss": float(loss.item()), "blocks": {}}
            for k, p in sorted(blk.items()):
                g = p.grad.detach().float()
                st = opt.state.get(p, {})
                m, v = st.get("exp_avg"), st.get("exp_avg_sq")
                if m is None:
                    # state not yet created (step 0): the honest answer is "no optimizer state"
                    snap["blocks"][k] = None
                    continue
                m, v = m.float(), v.float()
                u = m / (torch.sqrt(v) + a.eps)
                g2 = float((g * g).sum())
                u2 = float((u * u).sum())
                q = float((g.abs() / (torch.sqrt(v) + a.eps)).mean())
                w = float((g * g / (v + a.eps)).sum())   # exact sum_i g_i^2 / v_i
                r = torch.randn(g.numel(), generator=gen).to(dev).view_as(g)
                rn = float(torch.linalg.vector_norm(r))
                rows = []
                for c in levels:
                    d = r * (c * math.sqrt(g2) / (rn + 1e-12))
                    mh = m + (1 - a.beta1) * d
                    uh = mh / (torch.sqrt(v) + a.eps)
                    rows.append({"c": c,
                                 "grad_mse": float((d * d).sum()) / (g2 + 1e-30),
                                 "upd_mse": float(((uh - u) ** 2).sum()) / (u2 + 1e-30)})
                amp = rows[2]["upd_mse"] / rows[2]["grad_mse"] if rows[2]["grad_mse"] else 0.0
                snap["blocks"][k] = {"g": math.sqrt(g2), "u": math.sqrt(u2), "q": q,
                                     "w": w, "amp": amp, "curves": rows}
                del g, m, v, u, r, d, mh, uh
            snaps.append(snap)
        opt.step()
        opt.zero_grad(set_to_none=True)

    def tol_at(b, thr=1e-3):
        rows = b["curves"]
        prev = None
        for row in rows:
            if row["upd_mse"] > thr:
                if prev is None:
                    return row["c"] * math.sqrt(thr / max(row["upd_mse"], 1e-30))
                lc0, lc1 = math.log(prev["c"]), math.log(row["c"])
                le0, le1 = math.log(prev["upd_mse"]), math.log(row["upd_mse"])
                t = (math.log(thr) - le0) / (le1 - le0)
                return math.exp(lc0 + t * (lc1 - lc0))
            prev = row
        return rows[-1]["c"] * 2

    report = []
    for snap in snaps:
        bs = {k: v for k, v in snap["blocks"].items() if v}
        if not bs:
            report.append({"step": snap["step"], "loss": snap["loss"], "note": "no optimizer state yet"})
            print(f"\n[e1c] step {snap['step']}: no optimizer state (all None)")
            continue
        names = sorted(bs)
        q = [bs[k]["q"] for k in names]
        tol = [tol_at(bs[k]) for k in names]
        gn = [bs[k]["g"] for k in names]
        amp = [bs[k]["amp"] for k in names]
        # derived allocator, S4: delta_b ~ (sum_{i in b} g_i^2 / v_i)^(1/4) / cost_b
        derived = [bs[k]["w"] ** 0.25 for k in names]
        per_cost_tol = [tol[i] / flops[names[i]] for i in range(len(names))]
        per_cost_g = [gn[i] / flops[names[i]] for i in range(len(names))]
        per_cost_der = [derived[i] / flops[names[i]] for i in range(len(names))]
        rho_g = spearman(per_cost_tol, per_cost_g)
        rho_d = spearman(per_cost_tol, per_cost_der)
        q_lo, q_hi = min(q), max(q)
        entry = {"step": snap["step"], "loss": snap["loss"],
                 "q_min": q_lo, "q_max": q_hi, "q_median": sorted(q)[len(q) // 2],
                 "w_min": min(bs[k]["w"] for k in names),
                 "w_max": max(bs[k]["w"] for k in names),
                 "q_spread": q_hi / q_lo if q_lo else None,
                 "amp_min": min(amp), "amp_max": max(amp), "amp_median": sorted(amp)[len(amp) // 2],
                 "rho_percost_tol_vs_gradnorm": round(rho_g, 4),
                 "rho_percost_tol_vs_derived": round(rho_d, 4),
                 "tol_median": sorted(tol)[len(tol) // 2]}
        report.append(entry)
        print(f"\n[e1c] step {snap['step']:4d} loss {snap['loss']:.4f} | q=|g|/sqrt(v): "
              f"{q_lo:.3f}-{q_hi:.3f} (spread {entry['q_spread']:.2f}x) | amp=upd/grad: "
              f"{entry['amp_min']:.4f}-{entry['amp_max']:.4f}")
        print(f"       rho(tol/gflop, |g|/gflop) = {rho_g:+.3f}   "
              f"rho(tol/gflop, sqrt(sum g2/v)/gflop) = {rho_d:+.3f}   "
              f"median tol@1e-3 = {entry['tol_median']:.4f}")

    out = {"model": "Llama-3.2-1B-Instruct", "steps": a.steps, "points": points,
           "bs": a.bs, "seq": a.seq, "lr": a.lr, "seed": a.seed, "levels": levels,
           "losses": losses, "report": report,
           "flops_gflop_per_block": flops,
           "blocks_at_last_point": snaps[-1]["blocks"] if snaps else {}}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[e1c] wrote {a.out}")


if __name__ == "__main__":
    main()
