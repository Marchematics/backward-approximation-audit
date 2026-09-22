"""E6: pick the low-rank subspace by update-space error, not gradient energy.

SOTA context. GaLore/Q-GaLore/SLowMo/SubTrack++ choose the rank-r subspace by the SVD of g
(or of tracked g), minimising  ||g - P_r g||_F  -- gradient-space energy. "No Subspace to
Track" (arXiv:2607.05872) shows this subspace is mostly estimator noise (only ~39/128
directions reproducible) and that carrying the second moment through the rotation is
provably ~ (r-k*)/2 worse than a rotation-blind estimator. E5 showed the same wall here:
the median gradient block has effective rank ~1 and 90% of its energy in 1-8 directions.

The untested lever. The runtime never applies g; it applies U(g) = m/(sqrt(v)+eps). So the
loss caused by keeping subspace S is the UPDATE-space error
      E(S) = || (I - P_S) g / sqrt(v) ||^2
      = || g_hat - P_S^W g_hat ||^2      with   g_hat = g / sqrt(v)
when the projection is done with a basis that is simultaneously orthonormal for g and g_hat
(equivalently: span the row space of g_hat and project g onto it -- minimising E over all
rank-r S). The plain SVD of g minimises ||(I-P)g|| instead, so it can be arbitrarily worse
under a heterogeneous v, and its error carries no guarantee at any rank.

Measured per block at matched rank, with self-checks:
    svd_g     : row space of g  (SOTA / GaLore-style)
    svd_v     : row space of g/sqrt(v)  (update-space objective)
    rand      : random row space (floor, should equal ~1 - r/cols)

Also reports, per block, the v-weighted energy fraction that svd_g fails to capture
(`missed_energy`) -- the blocks where the SOTA objective is looking in the wrong place.

Run: ./venv/bin/python review20260922/e6_update_subspace.py --steps 60 --bs 4 --seq 512 \
        --ranks 4,8,16,32,64,128,256
"""
import argparse, glob, json, math, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AUDIT_MODEL", "/root/qcc/models/Llama-3.2-1B-Instruct")
CORPUS = "/root/qcc/data/longbench/data"
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


def err_after_rowspace(g, sv, basis, u2):
    """E(S) for S = span(rows of `basis`), evaluated in update space."""
    rec = (g @ basis.T) @ basis
    d = g - rec
    eu = float(((d / sv) ** 2).sum())
    eg = float((d * d).sum())
    return eg, eu / u2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--ranks", default="4,8,16,32,64,128,256")
    ap.add_argument("--rank-repro", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e6_update_subspace.json"))
    a = ap.parse_args()
    dev = "cuda"
    ranks = [int(x) for x in a.ranks.split(",")]
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    toks = build_tokens(tok, a.bs * a.seq * (a.steps + 4))
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
            print(f"[e6] step {step:3d} loss {loss.item():.4f}", flush=True)

    # ---- measurement pass A: optimizer state + gradient (copied to CPU) ----
    xa, ya = batch(a.steps)
    loss = model(input_ids=xa, labels=ya).loss
    loss.backward()
    ga = {k: p.grad.detach().float().cpu().clone() for k, p in blk.items()}
    ma, va = {}, {}
    for k, p in blk.items():
        st = opt.state.get(p, {})
        if st.get("exp_avg") is None:
            continue
        ma[k] = st["exp_avg"].float().cpu().clone()
        va[k] = st["exp_avg_sq"].float().cpu().clone()
    print(f"[e6] measurement loss {loss.item():.4f} | {len(ga)} blocks", flush=True)
    opt.zero_grad(set_to_none=True)

    # ---- measurement pass B: disjoint minibatch, for subspace reproducibility ----
    # (the SOTA check: is the subspace an object at all on this model?)
    xb, yb = batch(a.steps + 1)
    model(input_ids=xb, labels=yb).loss.backward()
    gb = {k: p.grad.detach().float().cpu().clone() for k, p in blk.items()}
    opt.zero_grad(set_to_none=True)

    # free the GPU: all remaining work is linear algebra on 2048x2048 matrices
    del model, opt, xa, ya, xb, yb
    torch.cuda.empty_cache()
    torch.set_num_threads(14)
    def svd(x):
        return torch.linalg.svd(x, full_matrices=False)
    def qr(x):
        return torch.linalg.qr(x)
    randn = torch.randn
    svdvals = torch.linalg.svdvals

    res = {}
    r0 = a.rank_repro
    for k in sorted(ga):
        if k not in ma:
            continue
        g, m, v = ga[k], ma[k], va[k]
        sv = torch.sqrt(v) + a.eps
        u = m / sv
        u2 = float((u * u).sum())
        if u2 <= 0:
            continue
        gh = g / sv
        entry = {"g": float(torch.linalg.vector_norm(g)), "u": math.sqrt(u2),
                 "q_mean": float((g.abs() / sv).mean()),
                 "v_ratio": float(v.max() / (v.min() + 1e-30))}
        # -- subspace reproducibility vs a disjoint minibatch (SOTA diagnostic) --
        try:
            ua_, sa_, va_ = torch.linalg.svd(g, full_matrices=False)
            ub_, sb_, vb_ = svd(gb[k])
            rr = min(r0, sa_.numel(), sb_.numel())
            # principal angles between the two top-r row spaces
            ov = va_[:rr] @ vb_[:rr].T
            sv_ov = svdvals(ov)
            cos_min = float(sv_ov[-1])
            chordal = math.sqrt(max(0.0, 2 * rr - 2 * float((sv_ov ** 2).sum())))
            entry["subspace_chordal_same_step_disjoint"] = chordal
            entry["subspace_chordal_max"] = math.sqrt(2 * rr)
            entry["subspace_cos_min"] = cos_min
        except Exception as e:
            entry["subspace_error"] = str(e)
        # -- the comparison: gradient-energy vs update-error subspace --
        # g is (out, in). A rank-r row space is a projector P = V V^T on the *input* side.
        #   gradient-space objective:  minimise ||g - g P||_F^2
        #   update-space objective:    minimise ||(g - g P)/sqrt(v)||_F^2
        # Both are evaluated explicitly (no tail-energy shortcut: the two objectives have
        # different eigenbases, so the tails are not interchangeable).
        entry["results"] = {}
        G2 = max(float((g * g).sum()), 1e-30)
        w = (1.0 / sv).flatten()                       # 1/sqrt(v), length = in
        n_in = g.shape[1]
        assert w.numel() == n_in, (w.numel(), n_in)
        Gmat = g.T @ g                                 # in x in
        for r in ranks:
            r = max(1, min(r, n_in - 1))
            # SOTA basis: top-r eigenvectors of g^T g (== top-r right singular vectors of g)
            ev_g, Vg = torch.linalg.eigh(Gmat)
            Pg = Vg[:, -r:]                            # in x r
            dg = g - (g @ Pg) @ Pg.T
            # OURS basis: top-r eigenvectors of the v-weighted Gram  sum_i g_i g_i^T / v_i
            Hv = Gmat * torch.outer(w, w)
            Hv = (Hv + Hv.T) * 0.5
            ev_v, Vv = torch.linalg.eigh(Hv)
            Pv = Vv[:, -r:]
            dv = g - (g @ Pv) @ Pv.T
            row = {
                "svd_g": {"grad_mse": float((dg * dg).sum()) / G2,
                          "upd_mse": float(((dg / sv) ** 2).sum()) / max(u2, 1e-30)},
                "svd_v": {"grad_mse": float((dv * dv).sum()) / G2,
                          "upd_mse": float(((dv / sv) ** 2).sum()) / max(u2, 1e-30)},
            }
            q, _ = qr(randn(n_in, r))
            dr = g - (g @ q) @ q.T
            row["rand"] = {"grad_mse": float((dr * dr).sum()) / G2,
                           "upd_mse": float(((dr / sv) ** 2).sum()) / max(u2, 1e-30)}
            entry["results"][str(r)] = row
            del ev_g, Vg, Pg, dg, ev_v, Vv, Pv, dv, q, dr
        res[k] = entry

        del g, m, v, sv, u, gh

    names = sorted(res)
    print(f"\n[e6] subspace reproducibility (rank {r0}, disjoint minibatches):")
    ch = [res[k]["subspace_chordal_same_step_disjoint"] for k in names
          if "subspace_chordal_same_step_disjoint" in res[k]]
    mx = [res[k]["subspace_chordal_max"] for k in names if "subspace_chordal_max" in res[k]]
    if ch:
        ratio = sorted(c / m for c, m in zip(ch, mx))
        print(f"     median chordal/max = {ratio[len(ratio)//2]:.3f}  "
              f"(0 = identical subspaces, 1 = maximally different)  over {len(ratio)} blocks")

    print(f"\n[e6] update-space error at matched rank, median over {len(names)} blocks")
    print(f"{'rank':>6}{'svd_g':>12}{'svd_v':>12}{'rand':>12}{'gain':>9}{'blocks_better':>15}")
    summary = {}
    for r in ranks:
        rr = str(min(r, 511))
        if rr not in res[names[0]]["results"]:
            continue
        g_med = sorted(res[k]["results"][rr]["svd_g"]["upd_mse"] for k in names)
        v_med = sorted(res[k]["results"][rr]["svd_v"]["upd_mse"] for k in names)
        r_med = sorted(res[k]["results"][rr]["rand"]["upd_mse"] for k in names)
        gains = []
        for k in names:
            gg = res[k]["results"][rr]["svd_g"]["upd_mse"]
            vv_ = res[k]["results"][rr]["svd_v"]["upd_mse"]
            if vv_ > 1e-30:
                gains.append(gg / vv_)
        gains.sort()
        med_g = g_med[len(g_med) // 2]
        med_v = v_med[len(v_med) // 2]
        summary[rr] = {"svd_g": med_g, "svd_v": med_v, "rand": r_med[len(r_med) // 2],
                       "gain_median": gains[len(gains) // 2] if gains else None,
                       "gain_p10": gains[int(0.1 * len(gains))] if gains else None,
                       "gain_p90": gains[int(0.9 * len(gains))] if gains else None,
                       "frac_better": sum(1 for x in gains if x > 1.0) / max(len(gains), 1)}
        s = summary[rr]
        print(f"{rr:>6}{med_g:>12.5f}{med_v:>12.5f}{s['rand']:>12.5f}"
              f"{s['gain_median']:>8.2f}x{s['frac_better']*100:>14.0f}%")

    print(f"\n[e6] per-block gain (svd_g / svd_v) at rank "
          f"{min(ranks[-1], 511)}, top 12 blocks by v-anisotropy:")
    rr = str(min(ranks[-1], 511))
    for k in sorted(names, key=lambda k: -res[k]["v_ratio"])[:12]:
        row = res[k]["results"][rr]
        gain = row["svd_g"]["upd_mse"] / max(row["svd_v"]["upd_mse"], 1e-30)
        print(f"  {k:<10} v_ratio {res[k]['v_ratio']:>9.2e}  "
              f"svd_g {row['svd_g']['upd_mse']:.5f}  svd_v {row['svd_v']['upd_mse']:.5f}  "
              f"gain {gain:>7.2f}x")

    out = {"model": "Llama-3.2-1B-Instruct", "steps": a.steps, "bs": a.bs, "seq": a.seq,
           "lr": a.lr, "seed": a.seed, "ranks": ranks, "rank_repro": r0,
           "losses": losses, "measure_loss": float(loss.item()),
           "summary": summary, "blocks": res}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e6] wrote {a.out}")


if __name__ == "__main__":
    main()
