"""E1: does optimizer-state sensitivity allocate backward compute differently
from gradient magnitude, on a real pretrained LM?

Per parameter tensor, with the optimizer state that would condition the next backward:
  grad_norm = ||g||_F                        (the DropBP-class statistic)
  sens      = sum_i g_i^2 / v_i              (Adam's diagonal preconditioner: the derived weight)
  tol_grad  = grad_norm / cost               (tolerance if magnitude is the allocator)
  tol_sens  = sens^(1/4) / cost              (tolerance under the derived design rule)

Design rule (docs/NEXT_BACKWARD_DIRECTION.md S4): with block-uniform relative error,
    delta_b ~ ( sum_{i in b} g_i^2 / v_i )^(1/4) / cost_b
so two blocks with equal gradient norm but different second moment must get different compute.

Also reports the regime split of |g| vs sqrt(v), which is where the sign-flip channel lives.

Memory note: this container shares the A10G with another job (~9.3 GB resident), so the
optimizer state is 8-bit (bitsandbytes) and stats are streamed per tensor.

Run: ./venv/bin/python review20260922/e1_sensitivity.py --steps 12 --bs 2 --seq 512
"""
import argparse, json, glob, math, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AUDIT_MODEL", "/root/qcc/models/Llama-3.2-1B-Instruct")
CORPUS = "/root/qcc/data/longbench/data"


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
                if not txt:
                    continue
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


def block_name(name):
    parts = name.split(".")
    layer = -1
    if "layers" in parts:
        layer = int(parts[parts.index("layers") + 1])
    if name.endswith("embed_tokens.weight"):
        return "L-1.embed"
    if name.endswith("lm_head.weight"):
        return "L99.lm_head"
    if name.endswith("norm.weight"):
        return f"L{layer:02d}.norm"
    for k, short in {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
                     "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}.items():
        if k in parts:
            return f"L{layer:02d}.{short}"
    return f"L{layer:02d}.{parts[-2]}"


def state_sq(opt, p, want8bit):
    """One-dimensional second-moment estimate for p, from a real optimizer state."""
    st = opt.state.get(p, {})
    v = st.get("exp_avg_sq")
    if v is None:
        return torch.zeros_like(p, dtype=torch.float32)
    if want8bit:
        # bitsandbytes stores quantised state; dequantise via its own codec
        try:
            import bitsandbytes.functional as F
            code = st["state2"]
            absmax = st["absmax2"]
            v = F.dequantize_blockwise(code, absmax, blocksize=st.get("blocksize", 256))
        except Exception:
            v = v.float()
    return v.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--opt", choices=["8bit", "fp32"], default="8bit")
    ap.add_argument("--skip-embed", action="store_true", default=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e1_sensitivity.json"))
    a = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(MODEL)
    toks = build_tokens(tok, a.bs * a.seq * (a.steps + 1))
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if a.opt == "8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr,
                                  betas=(a.beta1, a.beta2), eps=a.eps)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                betas=(a.beta1, a.beta2), eps=a.eps)
    want8 = a.opt == "8bit"
    used = torch.cuda.memory_allocated() / 2**30
    print(f"[e1] {torch.cuda.get_device_name(0)} | opt={a.opt} | layers={model.config.num_hidden_layers}"
          f" | after setup {used:.2f} GB allocated", flush=True)

    losses = []
    rows = {}
    for step in range(a.steps):
        s = step * a.bs * a.seq
        x = toks[s:s + a.bs * a.seq].view(a.bs, a.seq).to(dev)
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        losses.append(round(float(loss.item()), 4))

        measure = step in (0, a.steps - 1)   # the only two snapshots we need
        for n, p in named:
            if p.grad is None:
                continue
            if a.skip_embed and (n.endswith("embed_tokens.weight") or n.endswith("lm_head.weight")):
                continue
            b = block_name(n)
            r = rows.setdefault(b, {"g2": 0.0, "w": 0.0, "numel": 0,
                                    "big": 0, "small": 0, "w_num": 0})
            if not measure:
                continue
            with torch.no_grad():
                g = p.grad.detach().float()
                if step == a.steps - 1:
                    v = state_sq(opt, p, want8)
                    g2sum = float((g * g).sum())
                    wsum = float((g * g / (v + a.eps)).sum())
                    sv = torch.sqrt(v)
                    r["big"] += int((g.abs() > 2 * sv).sum())
                    r["small"] += int((g.abs() < sv).sum())
                    r["numel"] += g.numel()
                    # record only from the final snapshot, tagged by step
                    r["g2"] = g2sum
                    r["w"] = wsum
                    r["w_num"] = g.numel()
                    del v, sv
                del g
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 4 == 0 or step == a.steps - 1:
            print(f"[e1] step {step:3d} loss {loss.item():.4f} "
                  f"alloc {torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)

    blocks = sorted(rows.keys())
    gn, sens, tol_g, tol_s, cost = [], [], [], [], []
    big = small = numel = 0
    for b in blocks:
        r = rows[b]
        n = r["w_num"] or 1
        gn.append(math.sqrt(r["g2"]))
        sens.append(r["w"])
        cost.append(n / 1e6)
        tol_g.append(math.sqrt(r["g2"]) / (n / 1e6))
        tol_s.append((r["w"] ** 0.25) / (n / 1e6) if r["w"] > 0 else 0.0)
        big += r["big"]; small += r["small"]; numel += n

    rho = spearman(tol_g, tol_s)
    k = max(1, len(blocks) // 4)
    top_g = set(sorted(range(len(blocks)), key=lambda i: -tol_g[i])[:k])
    top_s = set(sorted(range(len(blocks)), key=lambda i: -tol_s[i])[:k])
    jac = len(top_g & top_s) / len(top_g | top_s)

    print(f"\n[e1] blocks={len(blocks)}  rho(tol_grad, tol_sens) = {rho:.3f}  "
          f"top-{k} Jaccard = {jac:.3f}")
    print(f"[e1] |g|>2sqrt(v): {big/numel*100:.1f}%   |g|<sqrt(v): {small/numel*100:.1f}%"
          f"   (of {numel/1e6:.1f}M measured coords)")
    print(f"\n{'block':<14}{'|g|':>11}{'sum g^2/v':>13}{'tol_grad':>11}{'tol_sens':>11}{'ratio':>9}")
    for i in sorted(range(len(blocks)), key=lambda i: -tol_s[i])[:15]:
        ratio = tol_s[i] / tol_g[i] if tol_g[i] else float("inf")
        print(f"{blocks[i]:<14}{gn[i]:>11.3f}{sens[i]:>13.2f}{tol_g[i]:>11.2f}"
              f"{tol_s[i]:>11.2f}{ratio:>9.2f}")

    gnorm_sorted = sorted(range(len(blocks)), key=lambda i: -gn[i])[:k]
    print(f"\n[e1] top-{k} blocks by gradient norm: {[blocks[i] for i in gnorm_sorted]}")
    print(f"[e1] top-{k} blocks by sens^(1/4)/cost: {[blocks[i] for i in sorted(range(len(blocks)), key=lambda i: -tol_s[i])[:k]]}")

    out = {
        "model": "Llama-3.2-1B-Instruct", "steps": a.steps, "bs": a.bs, "seq": a.seq,
        "optimizer": a.opt, "adamw": {"lr": a.lr, "b1": a.beta1, "b2": a.beta2, "eps": a.eps},
        "losses": losses, "loss_final": losses[-1],
        "rho_tol_grad_vs_tol_sens": round(rho, 4),
        "top_quartile_jaccard": round(jac, 4),
        "frac_abs_g_gt_2sqrt_v": round(big / numel, 4),
        "frac_abs_g_lt_sqrt_v": round(small / numel, 4),
        "blocks": [{"block": blocks[i], "grad_norm": gn[i], "sens": sens[i],
                    "tol_grad": tol_g[i], "tol_sens": tol_s[i], "params_m": cost[i]}
                   for i in range(len(blocks))],
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e1] wrote {a.out}")


if __name__ == "__main__":
    main()
