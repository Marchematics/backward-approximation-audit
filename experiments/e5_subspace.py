"""E5: which subspace should a low-rank backward keep?

SOTA setting. GaLore / Q-GaLore / SLowMo and every low-rank-backward method pick a rank-r
subspace by the SVD of the gradient (or of the optimizer state), with a rank that is uniform
across layers or chosen by spectral energy. That objective is gradient-space:
    minimise  ||g - P_r g||_F .
But the runtime does not apply g; it applies U(g) = m/(sqrt(v)+eps). The measured quantity that
predicts damage is the update-space error
    minimise  sum_i (delta_i)^2 / v_i .
These two objectives have the same solution only when v is isotropic. In a real AdamW run v is
strongly heterogeneous across coordinates, and the SVD has no structural reason to align with it.

Prediction (counterintuitive): scaling the gradient by 1/sqrt(v) *before* the SVD -- i.e. choosing
the subspace that a *preconditioned* backward would choose -- yields strictly smaller update-space
error at identical rank and identical FLOPs, and the gain is largest exactly where the plain SVD
looks best (blocks whose raw spectrum is steep).

Measured per block, at identical rank r:
    svd_g    : top-r left/right subspace of g            (SOTA)
    svd_p    : top-r of g / sqrt(v)                      (preconditioned objective)
    svd_blk  : top-r of g * (block mean 1/sqrt(v))       (cheap scalar variant)
    rand     : random rank-r subspace                    (control/floor)

Run: ./venv/bin/python review20260922/e5_subspace.py --steps 60 --bs 4 --seq 512 --ranks 16,32,64
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
        out[f"L{int(parts[parts.index('layers')+1]):02d}.{mod.replace('_proj','')}"] = p
    return out


def proj_err(g, m, sv, r, mode, gen):
    """Update-space and gradient-space error of a rank-r projection at fixed r.

    The runtime applies U(g_hat) = (m + (1-b1)*delta)/sv, so the update-space error of
    replacing g by its rank-r projection is exactly ||delta/sv||^2 / ||m/sv||^2 with
    delta = g - P_r g.  Scores the *subspace choice*, holding rank and cost fixed.
    """
    gm = g.view(g.shape[0], -1)
    if mode == "rand":
        k = min(gm.shape)
        q, _ = torch.linalg.qr(torch.randn(gm.shape[0], r, generator=gen).to(g.device))
        p2, _ = torch.linalg.qr(torch.randn(gm.shape[1], r, generator=gen).to(g.device))
        approx = (q.T @ gm @ p2)  # r x r core
        rec = q @ approx @ p2.T
    else:
        if mode == "svd_g":
            a = gm
        elif mode == "svd_p":
            a = gm / sv.view_as(gm)
        elif mode == "svd_blk":
            a = gm * float((1.0 / sv).mean())
        else:
            raise ValueError(mode)
        uu, ss, vv = torch.linalg.svd(a, full_matrices=False)
        rr = min(r, ss.numel())
        # reconstruct the ORIGINAL gradient from the subspace selected on `a`
        # (left/right subspace of g/sqrt(v) is not orthonormal in g's geometry: project properly)
        basis = vv[:rr, :]                                   # rr x cols, right subspace
        rec = (gm @ basis.T) @ basis
    d = g - rec.view_as(g)
    upd = float(((d / sv) ** 2).sum())
    eg = float((d * d).sum())
    return eg, upd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--ranks", default="16,32,64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e5_subspace.json"))
    a = ap.parse_args()
    dev = "cuda"
    ranks = [int(x) for x in a.ranks.split(",")]
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
            print(f"[e5] step {step:3d} loss {loss.item():.4f}", flush=True)

    x, y = batch(a.steps)
    loss = model(input_ids=x, labels=y).loss
    loss.backward()
    print(f"[e5] measurement loss {loss.item():.4f} | {len(blk)} blocks | ranks {ranks}", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(a.seed + 3)
    res = {}
    for key, p in sorted(blk.items()):
        st = opt.state.get(p, {})
        m, v = st.get("exp_avg"), st.get("exp_avg_sq")
        if m is None or v is None:
            continue
        g = p.grad.detach().float()
        m, v = m.float(), v.float()
        sv = torch.sqrt(v) + a.eps
        u = m / sv
        u2 = float((u * u).sum())
        if u2 <= 0:
            continue
        gm = g.view(g.shape[0], -1)
        uu, ss, vv = torch.linalg.svd(gm, full_matrices=False)
        tail = (ss ** 2).cumsum(0) / (ss ** 2).sum()
        # preconditioned spectrum, for the theory check
        ap_ = gm / sv.view_as(gm)
        spp = torch.linalg.svdvals(ap_)
        tail_p = (spp ** 2).cumsum(0) / (spp ** 2).sum()
        entry = {"g": float(torch.linalg.vector_norm(g)), "u2": u2,
                 "q_mean": float((g.abs() / sv).mean()),
                 "v_ratio": float(v.max() / (v.min() + 1e-30)),
                 "s_eff_g": float(((ss ** 2).sum() ** 2) / ((ss ** 4).sum() + 1e-30)),
                 "s_eff_p": float(((spp ** 2).sum() ** 2) / ((spp ** 4).sum() + 1e-30)),
                 "rank_for_90_g": int((tail < 0.9).sum().item()) + 1,
                 "rank_for_90_p": int((tail_p < 0.9).sum().item()) + 1,
                 "results": {}}
        for r in ranks:
            r = min(r, min(gm.shape) - 1)
            row = {}
            for mdl in ["svd_g", "svd_p", "svd_blk", "rand"]:
                eg, eu = proj_err(g, m, sv, r, mdl, gen)
                row[mdl] = {"grad_mse": eg / float((g * g).sum()),
                            "upd_mse": eu / u2}
            entry["results"][str(r)] = row
        res[key] = entry

    names = sorted(res)
    print(f"\n[e5] update-space error at fixed rank (median over {len(names)} blocks)")
    print(f"{'rank':>6}" + "".join(f"{m:>12}" for m in ["svd_g", "svd_p", "svd_blk", "rand"])
          + "   gain(svd_g/svd_p)")
    summary = {}
    for r in ranks:
        rr = str(min(r, 511))
        if rr not in res[names[0]]["results"]:
            continue
        med = {}
        gains = []
        for mdl in ["svd_g", "svd_p", "svd_blk", "rand"]:
            vals = sorted(res[k]["results"][rr][mdl]["upd_mse"] for k in names)
            med[mdl] = vals[len(vals) // 2]
        for k in names:
            gg = res[k]["results"][rr]["svd_g"]["upd_mse"]
            pp = res[k]["results"][rr]["svd_p"]["upd_mse"]
            if pp > 0:
                gains.append(gg / pp)
        gains.sort()
        summary[rr] = {"median": med,
                       "gain_median": gains[len(gains) // 2],
                       "gain_p10": gains[int(0.1 * len(gains))],
                       "gain_p90": gains[int(0.9 * len(gains))],
                       "frac_blocks_improved": sum(1 for x in gains if x > 1) / len(gains)}
        print(f"{rr:>6}" + "".join(f"{med[m]:>12.4f}" for m in ["svd_g", "svd_p", "svd_blk", "rand"])
              + f"   {summary[rr]['gain_median']:.3f}x  "
              f"(p10 {summary[rr]['gain_p10']:.2f}, p90 {summary[rr]['gain_p90']:.2f}, "
              f"{summary[rr]['frac_blocks_improved']*100:.0f}% blocks better)")

    print(f"\n[e5] effective rank (participation ratio) and 90%-energy rank, per block:")
    print(f"{'block':<12}{'seff_g':>9}{'seff_p':>9}{'r90_g':>7}{'r90_p':>7}{'v_max/min':>12}")
    for k in sorted(names, key=lambda k: -res[k]["s_eff_g"])[:12]:
        r = res[k]
        print(f"{k:<12}{r['s_eff_g']:>9.1f}{r['s_eff_p']:>9.1f}{r['rank_for_90_g']:>7}"
              f"{r['rank_for_90_p']:>7}{r['v_ratio']:>12.2e}")
    seff = [res[k]["s_eff_g"] for k in names]
    seffp = [res[k]["s_eff_p"] for k in names]
    print(f"[e5] median effective rank: raw {sorted(seff)[len(seff)//2]:.1f} -> "
          f"preconditioned {sorted(seffp)[len(seffp)//2]:.1f}")

    out = {"model": "Llama-3.2-1B-Instruct", "steps": a.steps, "bs": a.bs, "seq": a.seq,
           "lr": a.lr, "seed": a.seed, "ranks": ranks, "losses": losses,
           "measure_loss": float(loss.item()), "summary": summary, "blocks": res}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e5] wrote {a.out}")


if __name__ == "__main__":
    main()
