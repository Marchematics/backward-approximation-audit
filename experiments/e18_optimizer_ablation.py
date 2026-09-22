"""E18: is the amplification Adam-specific, or does the error geometry dominate for any optimizer?

The project's framing assumed the optimizer is the causal object: Adam divides by sqrt(v), so an
error in a high-v coordinate should be cheap and an error in a low-g coordinate expensive. If that
is right, changing the optimizer should change the whole amplification picture. If the picture is
the same for SGD and Lion, then the effect belongs to numerical linear algebra (the error direction
relative to the update direction) and the word "optimizer-conditioned" has to come out of the title.

Three optimizers, identical model/data/steps/budget, two error models injected into the gradient:

    proportional   delta = c * g                      (magnitude-preserving: quantization)
    orthogonal     delta = component of noise perp. to g, scaled to ||delta|| = c||g||

Measured per block:  A_b = upd_mse / grad_mse, and the fraction of coordinates whose update SIGN
flips under the perturbation (the mechanism E2 proposed).

Predictions to test:
    H1  SGD: A = 1 for proportional error (the update is the gradient, so relative error passes
        through unchanged) and A ~ 1/c^2 for orthogonal error (the update is replaced by noise).
    H2  Adam: A < 1 for proportional error (the 1/sqrt(v) preconditioner absorbs it) -- E2 measured
        0.13 at c=0.25.
    H3  Lion: sign-based, so any coordinate with |g_i| < delta_i flips a full-size step. A should be
        large for both models, and the sign-flip fraction should predict it.

Run: python3 experiments/e18_optimizer_ablation.py --device cuda --steps 60
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def update_of(opt_name, g, m, v, step_i, args):
    """The update each optimizer would apply, given the current gradient and its own state."""
    if opt_name == "sgd":
        # momentum SGD: m is the momentum buffer
        return m + (1 - args.momentum) * g if step_i > 0 else g
    if opt_name == "adam":
        if m is None:
            return g
        bc1 = 1 - args.b1 ** step_i
        bc2 = 1 - args.b2 ** step_i
        return (m / bc1) / ((v / bc2).sqrt() + args.eps)
    if opt_name == "lion":
        # Lion: update = sign(b1*m + (1-b1)*g); state m is the EMA of the gradient
        if m is None:
            return torch.sign(g)
        return torch.sign(args.b1 * m + (1 - args.b1) * g)
    raise ValueError(opt_name)


def build_model(args, vocab, dev, seed=0):
    """A compact GPT, or a real pretrained checkpoint when --pretrained is given."""
    torch.manual_seed(seed)
    if not args.pretrained:
        return GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    from transformers import AutoModelForCausalLM
    model_dir = os.environ.get("AUDIT_MODEL", "/root/qcc/models/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    return model


def train_state(opt_name, args, data, vocab, dev, seed=0):
    """Run a short training pass and return the model plus each optimizer's state arrays."""
    model = build_model(args, vocab, dev, seed)
    named = dict(model.named_parameters())
    m_state, v_state = {}, {}

    def batch(i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        x = data[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def step(i):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad
                m = m_state.get(n)
                v = v_state.get(n)
                if opt_name == "sgd":
                    new_m = g.clone() if m is None else args.momentum * m + g
                    m_state[n] = new_m
                    upd = new_m
                    p.add_(upd, alpha=-args.lr)
                elif opt_name == "adam":
                    new_m = torch.zeros_like(p) if m is None else m
                    new_v = torch.zeros_like(p) if v is None else v
                    new_m = args.b1 * new_m + (1 - args.b1) * g
                    new_v = args.b2 * new_v + (1 - args.b2) * (g * g)
                    m_state[n], v_state[n] = new_m, new_v
                    upd = update_of("adam", g, new_m, new_v, i + 1, args)
                    p.add_(upd, alpha=-args.lr)
                elif opt_name == "lion":
                    new_m = g.clone() if m is None else args.b1 * m + (1 - args.b1) * g
                    upd = torch.sign(new_m)
                    m_state[n] = new_m
                    p.add_(upd, alpha=-args.lr)
    def logits_of(x):
        out = model(x)
        return out.logits if hasattr(out, "logits") else out

    for i in range(args.warmup):
        x, y = batch(i)
        loss = F.cross_entropy(logits_of(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        step(i)
        for p in model.parameters():
            p.grad = None
    # keep the state for the measurement pass, but off the GPU: the A10G here is shared and a
    # full-size fp32 Adam state on a 1B model does not coexist with another tenant's 22 GB
    m_cpu = {k: (v.detach().to("cpu", torch.float32) if v is not None else None)
             for k, v in m_state.items()}
    v_cpu = {k: (v.detach().to("cpu", torch.float32) if v is not None else None)
             for k, v in v_state.items()}
    del m_state, v_state
    return model, m_cpu, v_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--cs", default="0.05,0.25")
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--pretrained", action="store_true",
                    help="use AUDIT_MODEL (a HuggingFace checkpoint) instead of a random-init GPT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e18_optimizer_ablation.json"))
    args = ap.parse_args()
    dev = args.device
    cs = [float(x) for x in args.cs.split(",")]
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e18] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    if args.pretrained:
        from transformers import AutoTokenizer
        model_dir = os.environ.get("AUDIT_MODEL", "/root/qcc/models/Llama-3.2-1B-Instruct")
        tok = AutoTokenizer.from_pretrained(model_dir)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        from transformers import AutoConfig
        vocab = AutoConfig.from_pretrained(model_dir).vocab_size
        assert max(ids) < vocab, f"token id {max(ids)} >= vocab {vocab}"
        data = torch.tensor(ids, dtype=torch.long)
    else:
        vocab = 256
        data = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    print(f"[e18] corpus {len(data)/1e6:.1f}M | optimizers: sgd, adam, lion | c={cs}", flush=True)

    out = {"config": vars(args), "optimizers": {}}
    for opt_name in ["sgd", "adam", "lion"]:
        model, m_state, v_state = train_state(opt_name, args, data, vocab, dev)
        # one clean measurement batch
        need = args.bs * (args.block + 1)
        x = data[:need].view(args.bs, args.block + 1)
        _o = model(x[:, :-1].to(dev))
        _lg = _o.logits if hasattr(_o, "logits") else _o
        loss = F.cross_entropy(_lg.reshape(-1, vocab), x[:, 1:].to(dev).reshape(-1))
        loss.backward()
        print(f"[e18] {opt_name}: loss {loss.item():.4f}", flush=True)
        # free the GPU completely: the amplification arithmetic is linear algebra and runs
        # comfortably on the CPU (40 GB) while the A10G is held by another tenant
        grads_cpu = {n: p.grad.detach().to("cpu", torch.float32)
                     for n, p in model.named_parameters() if p.grad is not None and p.dim() == 2}
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        torch.set_num_threads(14)
        cdev = "cpu"
        gen = torch.Generator(device=cdev).manual_seed(11)
        rows = {}
        for name, g in grads_cpu.items():
            m = m_state.get(name)
            v = v_state.get(name)
            g = g.to(cdev)
            u = update_of(opt_name, g, m, v, args.warmup, args).to(cdev)
            u_norm2 = float((u * u).sum())
            g_norm2 = float((g * g).sum())
            if u_norm2 <= 0 or g_norm2 <= 0:
                continue
            entry = {"g_norm": math.sqrt(g_norm2), "u_norm": math.sqrt(u_norm2)}
            # state ratio diagnostics
            if v is not None:
                entry["mean_g_over_sqrt_v"] = float((g.abs() / (v.sqrt() + args.eps)).mean())
            if m is not None and opt_name in ("adam", "lion"):
                entry["zeros_in_sign_arg"] = float((m.abs() < 1e-12).float().mean())
            white = torch.randn(g.numel(), generator=gen).view_as(g)
            white = white / (torch.linalg.vector_norm(white) + 1e-12)
            unit_g = g / (math.sqrt(g_norm2) + 1e-30)
            orth = torch.randn(g.numel(), generator=gen).view_as(g)
            orth = orth - float((orth * unit_g).sum()) * unit_g
            orth = orth / (torch.linalg.vector_norm(orth) + 1e-12)
            entry["models"] = {}
            for c in cs:
                for tag, d in (("proportional", c * g), ("orthogonal", orth * (c * math.sqrt(g_norm2)))):
                    gh = g + d
                    if opt_name == "adam":
                        mh = m + (1 - args.b1) * d
                        uh = update_of("adam", gh, mh, v, args.warmup, args)
                    elif opt_name == "sgd":
                        uh = update_of("sgd", gh, m + d if m is not None else None, None,
                                       args.warmup, args)
                    else:  # lion: perturb the EMA argument the same way
                        mh = m + (1 - args.b1) * d
                        uh = update_of("lion", gh, mh, None, args.warmup, args)
                    gerr = float(((gh - g) ** 2).sum()) / g_norm2
                    uerr = float(((uh - u) ** 2).sum()) / u_norm2
                    flips = float((torch.sign(uh) != torch.sign(u)).float().mean())
                    entry["models"].setdefault(tag, {})[str(c)] = {
                        "grad_mse": gerr, "upd_mse": uerr,
                        "amp": uerr / max(gerr, 1e-30), "sign_flip_frac": flips}
                    del d, gh, uh
            rows[name] = entry
            del g, m, v, u, white, orth, unit_g
        del grads_cpu
        names = sorted(rows)
        summary = {}
        for tag in ("proportional", "orthogonal"):
            for c in cs:
                amps = sorted(rows[n]["models"][tag][str(c)]["amp"] for n in names)
                flips = sorted(rows[n]["models"][tag][str(c)]["sign_flip_frac"] for n in names)
                summary[f"{tag}@{c}"] = {
                    "amp_median": amps[len(amps) // 2], "amp_p10": amps[int(0.1 * len(amps))],
                    "amp_p90": amps[int(0.9 * len(amps))], "amp_max": amps[-1],
                    "flip_median": flips[len(flips) // 2]}
        out["optimizers"][opt_name] = {"summary": summary, "blocks": rows,
                                       "measure_loss": float(loss.item())}
        print(f"  {'error model':<16}{'amp p10':>10}{'amp med':>10}{'amp p90':>10}"
              f"{'flip med':>10}", flush=True)
        for tag in ("proportional", "orthogonal"):
            for c in cs:
                s = summary[f"{tag}@{c}"]
                print(f"  {tag+'@'+str(c):<16}{s['amp_p10']:>10.4f}{s['amp_median']:>10.4f}"
                      f"{s['amp_p90']:>10.4f}{s['flip_median']:>10.4f}", flush=True)
    print("\n[e18] cross-optimizer comparison (median amplification at c=0.25)")
    print(f"{'optimizer':<12}{'proportional':>14}{'orthogonal':>13}{'sign-flip(prop)':>18}")
    for opt_name, d in out["optimizers"].items():
        p = d["summary"][f"proportional@{cs[-1]}"]
        o = d["summary"][f"orthogonal@{cs[-1]}"]
        print(f"{opt_name:<12}{p['amp_median']:>14.4f}{o['amp_median']:>13.4f}"
              f"{p['flip_median']:>18.4f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e18] wrote {args.out}")


if __name__ == "__main__":
    main()
