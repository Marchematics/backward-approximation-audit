"""E2: the amplification factor is a property of the error *geometry*, and it is large.

Holds the approximation budget fixed (relative gradient error c) and varies only the
structure of the error, which is what actually differs between real backward-approximation
families:

  white      delta ~ iid N(0,1) scaled to ||delta|| = c||g||      (random projection / generic noise)
  prop       delta = eps * g  (iid sign/magnitude jitter)         (magnitude-preserving: quantization)
  orth       delta = component of white noise orthogonal to g     (rank removal)
  input      delta from perturbing the block *input* activation   (sampling / token drop: the
             most realistic model, and the one that naturally yields rank-structured error)

For each block and model, reports
    A_b = upd_mse / grad_mse        at c = 0.1, 0.25
    rho(A_b, |g|_b / sqrt(v)_b)     does optimizer state predict which blocks are fragile?
    rho(A_b, |g|_b)                 does gradient magnitude?
and the per-block relative update error at fixed budget, which is the quantity an allocator
would actually minimise.

Only blocks with a non-degenerate second moment are ranked (v must carry history; the
near-zero-v tail is a numerical artifact of the first steps, not a mechanism).

Run: ./venv/bin/python review20260922/e2_error_geometry.py --steps 60 --bs 4 --seq 512
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
    ap.add_argument("--c", default="0.1,0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e2_error_geometry.json"))
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
            print(f"[e2] step {step:3d} loss {loss.item():.4f}", flush=True)

    x, y = batch(a.steps)
    loss = model(input_ids=x, labels=y).loss
    loss.backward()
    print(f"[e2] measurement loss {loss.item():.4f} | {len(blk)} blocks | c={cs}", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(a.seed + 7)
    res = {}
    for k, (name, p) in sorted(blk.items()):
        st = opt.state.get(p, {})
        m, v = st.get("exp_avg"), st.get("exp_avg_sq")
        if m is None or v is None:
            continue
        g = p.grad.detach().float()
        m, v = m.float(), v.float()
        sv = torch.sqrt(v) + a.eps
        u = m / sv
        g2, u2 = float((g * g).sum()), float((u * u).sum())
        if g2 <= 0 or u2 <= 0:
            continue
        w = float((g * g / (v + a.eps)).sum())
        q = float((g.abs() / sv).mean())
        entry = {"g": math.sqrt(g2), "u": math.sqrt(u2), "q": q, "w": w,
                 "mean_inv_s": float((1.0 / sv).mean()), "models": {}}

        # --- the four error geometries, all at the same relative gradient error c ---
        white = torch.randn(g.numel(), generator=gen).to(dev).view_as(g)
        white = white / (torch.linalg.vector_norm(white) + 1e-12)
        unit_g = g / (math.sqrt(g2) + 1e-30)
        jitter = torch.randn(g.numel(), generator=gen).to(dev).view_as(g)
        jitter = jitter / (torch.linalg.vector_norm(jitter) + 1e-12)
        jitter = jitter - float((jitter * unit_g).sum()) * unit_g     # orthogonalise

        # input-space perturbation: dx for a linear block, mapped back through W

        def upd_err(d):
            mh = m + (1 - a.beta1) * d
            uh = mh / sv
            return float(((uh - u) ** 2).sum()) / u2

        for c in cs:
            # white: ||delta|| = c||g||
            d = white * (c * math.sqrt(g2))
            ge = float((d * d).sum()) / g2
            entry["models"].setdefault("white", {})[str(c)] = {
                "grad_mse": ge, "upd_mse": upd_err(d), "amp": upd_err(d) / ge}
            # prop: delta = c * g (relative perturbation, direction preserved)
            d = c * g
            ge = float((d * d).sum()) / g2
            entry["models"].setdefault("prop", {})[str(c)] = {
                "grad_mse": ge, "upd_mse": upd_err(d), "amp": upd_err(d) / ge}
            # orth: noise orthogonal to g, same budget
            d = jitter * (c * math.sqrt(g2))
            ge = float((d * d).sum()) / g2
            entry["models"].setdefault("orth", {})[str(c)] = {
                "grad_mse": ge, "upd_mse": upd_err(d), "amp": upd_err(d) / ge}
            # rank: truncate the gradient to a fraction c of its rank (what a rank-r
            # backward approximation does), then score the residual at the same budget
            # (what a rank-r backward approximation does)
            gm = g.view(g.shape[0], -1)
            try:
                uu, ss, vv = torch.linalg.svd(gm, full_matrices=False)
            except Exception:
                uu = ss = vv = None
            if uu is not None:
                r_used = max(1, int(round(c * min(gm.shape))))
                approx = (uu[:, :r_used] * ss[:r_used]) @ vv[:r_used, :]
                d = (g - approx.view_as(g))
                ge = float((d * d).sum()) / g2
                entry["models"].setdefault("rank", {})[str(c)] = {
                    "grad_mse": ge, "upd_mse": upd_err(d), "amp": upd_err(d) / max(ge, 1e-30)}
                del uu, ss, vv, approx, d
        res[k] = entry
        del g, m, v, sv, u, white, jitter, unit_g

    names = sorted(res)
    # rank only well-conditioned blocks: second moment must carry real history
    good = [k for k in names if res[k]["q"] < 5.0]
    print(f"\n[e2] {len(names)} blocks measured, {len(good)} well-conditioned (q<5)")
    c0 = str(cs[-1])
    for mdl in ["white", "prop", "orth", "rank"]:
        amps = [(k, res[k]["models"][mdl][c0]["amp"]) for k in good]
        qs = [res[k]["q"] for k, _ in amps]
        amp = [v for _, v in amps]
        gs = [res[k]["g"] for k, _ in amps]
        med = sorted(amp)[len(amp) // 2]
        print(f"[e2] {mdl:<6} c={c0}: amp median {med:8.4f}  min {min(amp):8.4f} "
              f"max {max(amp):9.3f}  rho(amp,q) {spearman(amp, qs):+.3f}  "
              f"rho(amp,|g|) {spearman(amp, gs):+.3f}")

    print(f"\n{'block':<12}{'|g|':>9}{'q':>8}{'w':>12}"
          + "".join(f"{m+'_amp':>12}" for m in ["white", "prop", "orth", "rank"]))
    for k in sorted(good, key=lambda k: -res[k]["models"]["white"][c0]["amp"])[:14]:
        r = res[k]
        print(f"{k:<12}{r['g']:>9.3f}{r['q']:>8.3f}{r['w']:>12.2f}"
              + "".join(f"{r['models'][m][c0]['amp']:>12.4f}"
                        for m in ["white", "prop", "orth", "rank"]))

    out = {"model": "Llama-3.2-1B-Instruct", "steps": a.steps, "bs": a.bs, "seq": a.seq,
           "lr": a.lr, "seed": a.seed, "c": cs, "losses": losses,
           "measure_loss": float(loss.item()), "blocks": res,
           "well_conditioned": good}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e2] wrote {a.out}")


if __name__ == "__main__":
    main()
