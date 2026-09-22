"""E1b: gradient-space vs update-space error, measured directly, per block.

Reviewer-proof framing: hold the *approximation output* fixed and only change how the
error is scored. For each matrix block and each injected relative error level c:

    g_hat = g * (1 + c * r),  r ~ N(0,1) elementwise, scaled to ||delta|| = c ||g||
    grad_err  = ||g_hat - g||^2 / ||g||^2                      (gradient-space MSE)
    upd_err   = ||U(g_hat,m,v) - U(g,m,v)||^2 / ||U(g,m,v)||^2 (update-space MSE)

and the empirical tolerance c* that keeps update error under a threshold.

This is the honest version of the "gradient MSE 49x while update MSE improves 97x" claim:
the user-reported divergence has no artifact in this repo. Either this reproduces the
mechanism on a real 1B checkpoint or the mechanism is not there.

Cost is the true backward FLOPs of the block (linear layers only), so the allocation
question is asked in the units a runtime would actually spend.

Run: ./venv/bin/python review20260922/e1b_update_vs_grad.py --steps 24 --bs 4 --seq 512
"""
import argparse, glob, json, math, os
import torch
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AUDIT_MODEL", "/root/qcc/models/Llama-3.2-1B-Instruct")
CORPUS = "/root/qcc/data/longbench/data"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def build_tokens(tok, need):
    ids = []
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
    """Linear-layer blocks: name -> (layer, module, weight, backward-FLOPs weight)."""
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or not n.endswith(".weight"):
            continue
        parts = n.split(".")
        mod = parts[-2] if len(parts) > 1 else ""
        if mod not in PROJ or "layers" not in parts:
            continue
        layer = int(parts[parts.index("layers") + 1])
        # backward FLOPs ~ 2 * (in_features x out_features) per token for dX and dW
        out[f"L{layer:02d}.{mod.replace('_proj','')}"] = (layer, mod, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--levels", default="0.02,0.05,0.1,0.25,0.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e1b_update_vs_grad.json"))
    a = ap.parse_args()
    dev = "cuda"
    levels = [float(x) for x in a.levels.split(",")]
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    toks = build_tokens(tok, a.bs * a.seq * (a.steps + 2))
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(a.beta1, a.beta2), eps=a.eps)
    blocks = blocks_of(model)
    print(f"[e1b] {torch.cuda.get_device_name(0)} | {len(blocks)} matrix blocks | "
          f"steps={a.steps} levels={levels}", flush=True)

    def batch(i):
        s = i * a.bs * a.seq
        x = toks[s:s + a.bs * a.seq].view(a.bs, a.seq).to(dev)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()   # next-token labels

    losses = []
    for step in range(a.steps):
        x, y = batch(step)
        loss = model(input_ids=x, labels=y).loss
        loss.backward()
        losses.append(round(float(loss.item()), 4))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 6 == 0 or step == a.steps - 1:
            print(f"[e1b] step {step:3d} loss {loss.item():.4f} "
                  f"alloc {torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)

    # ---- measurement pass with the trained optimizer state ----
    x, y = batch(a.steps)
    loss = model(input_ids=x, labels=y).loss
    loss.backward()
    print(f"[e1b] measurement loss {loss.item():.4f}", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(a.seed + 1)
    results = {}
    for name, (layer, mod, p) in sorted(blocks.items()):
        g = p.grad.detach().float()
        st = opt.state.get(p, {})
        m = st.get("exp_avg")
        v = st.get("exp_avg_sq")
        if m is None or v is None:
            continue
        m, v = m.float(), v.float()
        u = m / (torch.sqrt(v) + a.eps)
        u_norm2 = float((u * u).sum())
        g_norm2 = float((g * g).sum())
        numel = g.numel()
        flops = 4 * p.numel()          # dX and dW, 2 FLOPs each per element per token
        r = torch.randn(numel, generator=gen).to(dev).view_as(g)
        rows = []
        for c in levels:
            # scale r so that ||c*r|| = c * ||g||  => relative grad error == c
            rn = torch.linalg.vector_norm(r)
            scale = c * math.sqrt(g_norm2) / (float(rn) + 1e-12)
            d = r * scale
            gh = g + d
            mh = m + (1 - a.beta1) * d
            uh = mh / (torch.sqrt(v) + a.eps)
            gerr = float(((gh - g) ** 2).sum()) / (g_norm2 + 1e-30)
            uerr = float(((uh - u) ** 2).sum()) / (u_norm2 + 1e-30)
            rows.append({"c": c, "grad_mse": gerr, "upd_mse": uerr})
            del d, gh, mh, uh
        # empirical tolerance: largest c with upd_mse <= 1e-3, interpolated in log-log
        def tol(thr):
            prev = None
            for row in rows:
                if row["upd_mse"] > thr:
                    if prev is None:
                        return row["c"] * math.sqrt(thr / row["upd_mse"])
                    lc0, lc1 = math.log(prev["c"]), math.log(row["c"])
                    le0, le1 = math.log(prev["upd_mse"]), math.log(row["upd_mse"])
                    t = (math.log(thr) - le0) / (le1 - le0)
                    return math.exp(lc0 + t * (lc1 - lc0))
                prev = row
            return rows[-1]["c"] * 2
        results[name] = {
            "layer": layer, "module": mod, "params_m": numel / 1e6, "backward_gflops": flops / 1e9,
            "grad_norm": math.sqrt(g_norm2), "upd_norm": math.sqrt(u_norm2),
            "mean_abs_g_over_sqrt_v": float((g.abs() / (torch.sqrt(v) + a.eps)).mean()),
            "curves": rows, "tol_1e-2": tol(1e-2), "tol_1e-3": tol(1e-3), "tol_1e-4": tol(1e-4),
        }
        del g, m, v, u, r

    names = list(results)
    tol1 = [results[n]["tol_1e-3"] for n in names]
    gn = [results[n]["grad_norm"] for n in names]
    fl = [results[n]["backward_gflops"] for n in names]
    per_cost_tol = [results[n]["tol_1e-3"] / results[n]["backward_gflops"] for n in names]
    per_cost_g = [results[n]["grad_norm"] / results[n]["backward_gflops"] for n in names]
    olaw = [(results[n]["tol_1e-3"] ** 4 * results[n]["backward_gflops"]) for n in names]

    rho_gn = spearman(per_cost_tol, per_cost_g)
    rho_olaw = spearman(per_cost_tol, olaw)
    print(f"\n[e1b] median update tolerance (upd_mse<=1e-3): "
          f"{sorted(tol1)[len(tol1)//2]:.4f}")
    for lv in levels:
        gm = sorted(r["grad_mse"] for r in results[names[0]]["curves"])
    ex = None
    for n in names:
        for row in results[n]["curves"]:
            if row["c"] == 0.25:
                ex = (n, row)
    print(f"[e1b] example block {ex[0]}: at c=0.25 grad_mse={ex[1]['grad_mse']:.4f} "
          f"upd_mse={ex[1]['upd_mse']:.6f}")
    print(f"[e1b] rho(per-cost tolerance, per-cost gradient norm) = {rho_gn:.3f}")
    print(f"[e1b] rho(per-cost tolerance, per-cost |g|^2/v proxy) = {rho_olaw:.3f}")
    print(f"\n{'block':<12}{'Mpar':>7}{'bwd GFLOP':>11}{'|g|':>9}{'tol@1e-3':>10}"
          f"{'tol/GF':>10}{'|g|/GF':>10}")
    for n in sorted(names, key=lambda n: -results[n]["tol_1e-3"] / results[n]["backward_gflops"])[:16]:
        r = results[n]
        print(f"{n:<12}{r['params_m']:>7.3f}{r['backward_gflops']:>11.3f}{r['grad_norm']:>9.3f}"
              f"{r['tol_1e-3']:>10.4f}{r['tol_1e-3']/r['backward_gflops']:>10.5f}"
              f"{r['grad_norm']/r['backward_gflops']:>10.3f}")

    out = {"model": "Llama-3.2-1B-Instruct", "steps": a.steps, "bs": a.bs, "seq": a.seq,
           "seed": a.seed, "lr": a.lr, "levels": levels, "losses": losses,
           "loss_final": losses[-1], "measure_loss": float(loss.item()),
           "rho_percost_tol_vs_gradnorm": round(rho_gn, 4),
           "rho_percost_tol_vs_sumg2_over_v": round(rho_olaw, 4),
           "blocks": results}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e1b] wrote {a.out}")


if __name__ == "__main__":
    main()
