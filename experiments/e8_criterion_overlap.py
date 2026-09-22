"""E8: do the SOTA criterion and the update-space criterion actually select different coordinates?

This is the mechanism check behind E7, and it is independent of any end-to-end result.
For every matrix block, with a real AdamW state after a short run:

    SOTA set   = top-k by |g_i|                      (magnitude sparsification, DropBP-style)
    OURS set   = top-k by |g_i| / sqrt(v_i)          (update-space impact)

Reports, per block and per density k:
    jaccard(SOTA, OURS), overlap fraction, and -- the number that matters --
    the update-space error of each selection, normalised, plus the tail statistics
    (fraction of |g| mass in the SOTA set vs fraction of update-space mass).

If the two criteria pick essentially the same coordinates, E7 cannot differ and the whole
"corrected criterion" claim is void. If they diverge, the divergence is the mechanism.

Run: ./venv/bin/python review20260922/e8_criterion_overlap.py --steps 40 --bs 4 --seq 512
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--ks", default="0.01,0.05,0.10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e8_criterion_overlap.json"))
    a = ap.parse_args()
    dev = "cuda"
    ks = [float(x) for x in a.ks.split(",")]
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
        if step % 10 == 0 or step == a.steps - 1:
            print(f"[e8] step {step:3d} loss {loss.item():.4f}", flush=True)

    x, y = batch(a.steps)
    loss = model(input_ids=x, labels=y).loss
    loss.backward()
    print(f"[e8] measurement loss {loss.item():.4f}", flush=True)

    res = {}
    for key, p in sorted(blk.items()):
        st = opt.state.get(p, {})
        m, v = st.get("exp_avg"), st.get("exp_avg_sq")
        if m is None or v is None:
            continue
        g = p.grad.detach().float().cpu().flatten()
        m = m.float().cpu().flatten()
        v = v.float().cpu().flatten()
        sv = torch.sqrt(v) + a.eps
        u = m / sv
        upd_mass = (u * u)
        tot_upd = float(upd_mass.sum())
        tot_g2 = float((g * g).sum())
        if tot_upd <= 0 or tot_g2 <= 0:
            continue
        crit_g = g.abs()                    # SOTA: gradient magnitude
        crit_v = g.abs() / sv               # surprise: |g| relative to its own history
        crit_mv = m.abs() / sv              # Adam's actual step statistic
        entry = {"numel": g.numel(), "tot_g2": tot_g2, "tot_upd": tot_upd,
                 "v_ratio": float(v.max() / (v.min() + 1e-30)), "ks": {}}
        for k in ks:
            n_keep = max(1, int(round(k * g.numel())))
            sets = {}
            for nm, crit in (("g", crit_g), ("v", crit_v), ("mv", crit_mv)):
                idx = torch.topk(crit, n_keep, sorted=False).indices
                msk = torch.zeros_like(crit_g, dtype=torch.bool)
                msk[idx] = True
                sets[nm] = msk
            sg, sv_, sm = sets["g"], sets["v"], sets["mv"]
            cap = {nm: {"g_mass": float((crit_g[msk] ** 2).sum()) / tot_g2,
                        "upd_mass": float(upd_mass[msk].sum()) / tot_upd}
                   for nm, msk in sets.items()}
            entry["ks"][str(k)] = {
                "jaccard_g_v": int((sg & sv_).sum()) / int((sg | sv_).sum()),
                "jaccard_g_mv": int((sg & sm).sum()) / int((sg | sm).sum()),
                "overlap_g_v_frac_of_k": int((sg & sv_).sum()) / n_keep,
                "overlap_g_mv_frac_of_k": int((sg & sm).sum()) / n_keep,
                "capture_upd_mass_by_g": cap["g"]["upd_mass"],
                "capture_upd_mass_by_v": cap["v"]["upd_mass"],
                "capture_upd_mass_by_mv": cap["mv"]["upd_mass"],
                "capture_g_mass_by_g": cap["g"]["g_mass"],
                "upd_mass_gain_mv_over_g": cap["mv"]["upd_mass"] / max(cap["g"]["upd_mass"], 1e-30),
                "upd_mass_gain_v_over_g": cap["v"]["upd_mass"] / max(cap["g"]["upd_mass"], 1e-30),
            }
        res[key] = entry
        del g, m, v, sv, u, upd_mass, crit_g, crit_v

    names = sorted(res)
    print(f"\n[e8] criterion overlap over {len(names)} blocks "
          f"(jaccard=1 means the two criteria are identical)")
    for k in ks:
        def med(key):
            v = sorted(res[n]["ks"][str(k)][key] for n in names)
            return v[len(v) // 2]
        print(f"\n  k={k:<5} jaccard(|g|,|m|/sqrt(v)) median {med('jaccard_g_mv'):.3f}   "
              f"jaccard(|g|,|g|/sqrt(v)) {med('jaccard_g_v'):.3f}")
        print(f"          overlap with SOTA set: |m|/sqrt(v) keeps "
              f"{med('overlap_g_mv_frac_of_k')*100:.1f}% of the same coords, "
              f"|g|/sqrt(v) keeps {med('overlap_g_v_frac_of_k')*100:.1f}%")
        print(f"          update-mass captured at this density: "
              f"|g| {med('capture_upd_mass_by_g')*100:.1f}%  "
              f"|g|/sqrt(v) {med('capture_upd_mass_by_v')*100:.1f}%  "
              f"|m|/sqrt(v) {med('capture_upd_mass_by_mv')*100:.1f}%  "
              f"(gain {med('upd_mass_gain_mv_over_g'):.2f}x)")

    print(f"\n[e8] lowest-overlap blocks at k={ks[0]} (|g| vs |m|/sqrt(v)):")
    print(f"{'block':<12}{'jac_g_mv':>10}{'keep%':>8}{'upd_by_g%':>11}{'upd_by_mv%':>12}{'gain':>8}")
    for n in sorted(names, key=lambda n: res[n]["ks"][str(ks[0])]["jaccard_g_mv"])[:14]:
        r = res[n]["ks"][str(ks[0])]
        print(f"{n:<12}{r['jaccard_g_mv']:>10.3f}{r['overlap_g_mv_frac_of_k']*100:>8.1f}"
              f"{r['capture_upd_mass_by_g']*100:>11.1f}"
              f"{r['capture_upd_mass_by_mv']*100:>12.1f}"
              f"{r['upd_mass_gain_mv_over_g']:>8.2f}")

    out = {"model": "Llama-3.2-1B-Instruct", "steps": a.steps, "bs": a.bs, "seq": a.seq,
           "lr": a.lr, "seed": a.seed, "ks": ks, "losses": losses,
           "measure_loss": float(loss.item()), "blocks": res}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e8] wrote {a.out}")


if __name__ == "__main__":
    main()
