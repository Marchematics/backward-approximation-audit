"""E3: which optimizer-state statistic predicts a block's fragility to gradient error?

E2 showed that the update-space amplification A_b = upd_mse/grad_mse depends overwhelmingly
on the *geometry* of the error, not just its size:
    proportional error (quantization-like)  A_b ~ 0.3, bounded
    rank-truncation error                   A_b ~ 0.8, block-dependent
    white / orthogonal error                A_b up to 1e4, dominated by a few blocks
and that the mean sensitivity q_b = mean_i |g_i|/sqrt(v_i) predicts A_b only for the
proportional model (rho +0.76), not for white noise (rho +0.14).

Hypothesis tested here: the fragile coordinates are not the *average* ones but the
low-sensitivity tail. A block is fragile if a non-negligible fraction of its coordinates
satisfies |g_i| < delta_i, i.e. if the quantile q_b(p) = quantile_p(|g_i|/sqrt(v_i))
for small p is below the injected relative error c.

So this run measures, per block:
    q_mean, q_p50, q_p10, q_p01        distribution of |g_i|/sqrt(v_i)
    w     = sum_i g_i^2 / v_i          the derived allocator statistic (mean-like)
    wq    = sum_i g_i^2 / v_i^2        a tail-sensitive variant
and reports Spearman rho(A_b, statistic) for each error model at c = 0.1, 0.25.

Whichever statistic wins is the one the runtime should schedule on; if none wins for the
realistic models, the honest conclusion is that per-block allocation is noise-limited.

Run: ./venv/bin/python review20260922/e3_statistic.py --steps 60 --bs 4 --seq 512
"""
import argparse, glob, json, math, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AUDIT_MODEL", "/root/qcc/models/Llama-3.2-1B-Instruct")
CORPUS = "/root/qcc/data/longbench/data"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
QS = (0.001, 0.01, 0.05, 0.10, 0.25, 0.50)


def build_tokens(tok, need):
    ids = []
    while len(ids) < need:
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


def blocks_of(model):
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or not n.endswith(".weight"):
            continue
        parts = n.split(".")
        mod = parts[-2] if len(parts) > 1 else ""
        if mod not in PROJ or "layers" not in parts:
            continue
        out[f"L{int(parts[parts.index('layers')+1]):02d}.{mod.replace('_proj','')}"] = (n, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--c", default="0.05,0.1,0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e3_statistic.json"))
    a = ap.parse_args()
    dev = "cuda"
    cs = [float(x) for x in a.c.split(",")]
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    toks = build_tokens(tok, a.bs * a.seq * (a.steps + 2))
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(a.beta1, a.beta2), eps=a.eps)
    blk = blocks_of(model)

    def batch(i):
        s = (i * a.bs * a.seq) % (len(toks) - a.bs * a.seq - 2)
        x = toks[s:s + a.bs * a.seq].view(a.bs, a.seq).to(dev)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()

    losses = []
    for step in range(a.steps):
        x, y = batch(step)
        loss = model(input_ids=x, labels=y).loss
        loss.backward()
        losses.append(round(float(loss.item()), 4))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or step == a.steps - 1:
            print(f"[e3] step {step:3d} loss {loss.item():.4f}", flush=True)

    x, y = batch(a.steps)
    loss = model(input_ids=x, labels=y).loss
    loss.backward()
    print(f"[e3] measurement loss {loss.item():.4f} | {len(blk)} blocks | c={cs}", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(a.seed + 7)
    res = {}
    for key, (name, p) in sorted(blk.items()):
        st = opt.state.get(p, {})
        m, v = st.get("exp_avg"), st.get("exp_avg_sq")
        if m is None or v is None:
            continue
        g = p.grad.detach().float()
        m, v = m.float(), v.float()
        sv = torch.sqrt(v) + a.eps
        u = m / sv
        g2 = float((g * g).sum())
        u2 = float((u * u).sum())
        if g2 <= 0 or u2 <= 0:
            continue
        ratio = (g.abs() / sv).flatten()
        qq = torch.quantile(ratio, torch.tensor(QS, device=dev)).tolist()
        stats = {"g": math.sqrt(g2),
                 "q_mean": float(ratio.mean()),
                 "w": float((g * g / (v + a.eps)).sum()),
                 "wq": float((g * g / (v + a.eps) ** 2).sum()),
                 "q_med": float((g.abs() / sv).median())}
        for q, val in zip(QS, qq):
            stats[f"q_p{int(q*1000):03d}"] = float(val)

        white = torch.randn(g.numel(), generator=gen).to(dev).view_as(g)
        white = white / (torch.linalg.vector_norm(white) + 1e-12)
        unit_g = g / (math.sqrt(g2) + 1e-30)
        jitter = torch.randn(g.numel(), generator=gen).to(dev).view_as(g)
        jitter = jitter / (torch.linalg.vector_norm(jitter) + 1e-12)
        jitter = jitter - float((jitter * unit_g).sum()) * unit_g

        def upd_err(d):
            mh = m + (1 - a.beta1) * d
            uh = mh / sv
            return float(((uh - u) ** 2).sum()) / u2

        sat = float((ratio < cs[0]).float().mean())     # fraction already below the smallest c
        stats["frac_below_cmin"] = sat
        stats["models"] = {}
        for c in cs:
            for mdl, d in (("white", white * (c * math.sqrt(g2))), ("prop", c * g),
                           ("orth", jitter * (c * math.sqrt(g2)))):
                ge = float((d * d).sum()) / g2
                ue = upd_err(d)
                stats["models"].setdefault(mdl, {})[str(c)] = {
                    "grad_mse": ge, "upd_mse": ue, "amp": ue / max(ge, 1e-30)}
            gm = g.view(g.shape[0], -1)
            try:
                uu, ss, vv = torch.linalg.svd(gm, full_matrices=False)
                r_used = max(1, int(round(c * min(gm.shape))))
                approx = (uu[:, :r_used] * ss[:r_used]) @ vv[:r_used, :]
                d = g - approx.view_as(g)
                ge = float((d * d).sum()) / g2
                ue = upd_err(d)
                stats["models"].setdefault("rank", {})[str(c)] = {
                    "grad_mse": ge, "upd_mse": ue, "amp": ue / max(ge, 1e-30)}
                del uu, ss, vv, approx
            except Exception:
                pass
            del d
        res[key] = stats
        del g, m, v, sv, u, white, jitter, unit_g, ratio

    names = sorted(res)
    c0 = str(cs[-1])
    cands = ["g", "q_mean", "w", "wq", "q_med", "frac_below_cmin"] + \
            [f"q_p{int(q*1000):03d}" for q in QS]
    print(f"\n[e3] Spearman rho(statistic, amplification A_b) over {len(names)} blocks, c={c0}")
    print(f"{'statistic':<18}" + "".join(f"{m:>10}" for m in ["white", "prop", "orth", "rank"]))
    table = {}
    for s in cands:
        vals = [res[k][s] for k in names]
        row = {}
        for mdl in ["white", "prop", "orth", "rank"]:
            amp = [res[k]["models"][mdl][c0]["amp"] for k in names if mdl in res[k]["models"]]
            kk = [k for k in names if mdl in res[k]["models"]]
            row[mdl] = spearman([res[k][s] for k in kk], amp)
        table[s] = row
        print(f"{s:<18}" + "".join(f"{row[m]:>10.3f}" for m in ["white", "prop", "orth", "rank"]))

    print(f"\n[e3] amplification at c={c0} (median / p90 / max):")
    for mdl in ["white", "prop", "orth", "rank"]:
        amp = sorted(res[k]["models"][mdl][c0]["amp"] for k in names if mdl in res[k]["models"])
        print(f"  {mdl:<6} {amp[len(amp)//2]:10.4f} {amp[int(0.9*len(amp))]:10.4f} {amp[-1]:12.3f}")

    out = {"model": "Llama-3.2-1B-Instruct", "steps": a.steps, "bs": a.bs, "seq": a.seq,
           "lr": a.lr, "seed": a.seed, "c": cs, "losses": losses,
           "measure_loss": float(loss.item()),
           "rho_table": table, "blocks": res}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e3] wrote {a.out}")


if __name__ == "__main__":
    main()
