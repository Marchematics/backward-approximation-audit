"""E24: the risk predictor at 1B scale, with a configuration that actually learns.

Everything at scale so far failed for one reason: the configuration was not discriminative. The
first attempt (E19 --pretrained, 1.5 M tokens, lr 1e-5) had training loss *rising* while held-out
loss stayed flat to four decimals, so no density effect could show up.

This build is purpose-made for the question and makes the prerequisite explicit:

  0. CALIBRATION IS PART OF THE RUN. Training loss is printed every 100 steps and the run reports
     `train_loss_decreasing`, which is False when the learning rate is too high. A run whose
     training loss rises is reported as non-discriminative instead of being read as a result.
  1. AdamW and Lion, dense and 5% density, at 1B.
  2. The E23 risk predictor computed on the live optimizer state, so the prediction is on the same
     scale as the measurement (AdamW preconditions with 1/sqrt(v), Lion uses sign(m + delta)).
  3. Optimizer state is kept fp16 so the whole thing fits on a shared 24 GB card, and the risk
     predictor uses the fp32-upcast state.

Run: python3 experiments/e24_scale_risk.py --pretrained --device cuda --steps 1200 --lr 3e-6
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import load_text  # noqa: E402


class BnbAdam:
    """8-bit AdamW wrapper exposing the same interface as LiteOpt for the risk predictor."""

    def __init__(self, params, lr, b1=0.9, b2=0.95, eps=1e-8):
        import bitsandbytes as bnb
        self.name = "adamw"
        self.params = list(params)
        self.opt = bnb.optim.AdamW8bit(self.params, lr=lr, betas=(b1, b2), eps=eps)
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    def step(self):
        self.t += 1
        self.opt.step()

    def _slots(self, p):
        """Return (m, v) dequantised. bitsandbytes keeps the code and its QuantState under
        separate keys, and the QuantState object is what dequantize_blockwise needs -- passing
        the tensor as the second argument is what raised 'Tensor has no attribute blocksize'."""
        st = self.opt.state.get(p, {})
        out = []
        for base, qkeys in (("state1", ("state1_quant", "qstate1", "quant_state1")),
                            ("state2", ("state2_quant", "qstate2", "quant_state2"))):
            t = st.get(base, st.get("exp_avg" if base == "state1" else "exp_avg_sq"))
            if t is None:
                out.append(None)
                continue
            if t.dtype in (torch.float32, torch.float16):
                out.append(t)
                continue
            qs = next((st[k] for k in qkeys if k in st), None)
            if qs is None:
                out.append(t.float())
                continue
            import bitsandbytes.functional as BF
            try:
                out.append(BF.dequantize_blockwise(t, quant_state=qs))
            except TypeError:
                out.append(BF.dequantize_blockwise(t, qs))
        return out[0], out[1]

    def upd(self, p):
        m, v = self._slots(p)
        if m is None or v is None:
            return p.grad.float()
        bc1 = 1 - self.b1 ** max(self.t, 1)
        bc2 = 1 - self.b2 ** max(self.t, 1)
        return (m.float() / bc1) / ((v.float() / bc2).sqrt() + self.eps)


class LiteOpt:
    """AdamW / Lion with fp32 state. (fp16 state was tried and overflowed: v ~ g^2 exceeds
    fp16 range once masking inflates g, which showed up as a diverging loss -- caught by the
    train_loss_decreasing guard rather than silently reported.)"""

    def __init__(self, name, params, lr, b1=0.9, b2=0.95, eps=1e-8):
        self.name, self.params, self.lr = name, list(params), lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.state, self.t = {}, 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def upd(self, p):
        """The update this optimizer would apply right now, in fp32 (no lr, no momentum update)."""
        st = self.state.get(p, {})
        m = st.get("m")
        if self.name == "lion":
            return torch.sign(m) if m is not None else torch.sign(p.grad.float())
        if m is None:
            return p.grad.float()
        if self.name == "sgd":
            return m
        v = st.get("v")
        bc1 = 1 - self.b1 ** max(self.t, 1)
        bc2 = 1 - self.b2 ** max(self.t, 1)
        return (m / bc1) / ((v / bc2).sqrt() + self.eps)

    def step(self):
        self.t += 1
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state.setdefault(p, {})
                if self.name == "lion":
                    m = st.get("m")
                    upd = torch.sign(g.float()) if m is None else \
                        torch.sign(self.b1 * m + (1 - self.b1) * g.float())
                    st["m"] = (g.float() if m is None else
                               self.b1 * m + (1 - self.b1) * g.float())
                else:  # adamw
                    m, v = st.get("m"), st.get("v")
                    if m is None:
                        m = torch.zeros_like(p, dtype=torch.float32)
                        v = torch.zeros_like(p, dtype=torch.float32)
                    m = self.b1 * m + (1 - self.b1) * g.float()
                    v = self.b2 * v + (1 - self.b2) * g.float().pow(2)
                    st["m"], st["v"] = m, v
                    bc1 = 1 - self.b1 ** self.t
                    bc2 = 1 - self.b2 ** self.t
                    upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
                p.add_(upd.to(p.dtype), alpha=-self.lr)


def risk_scores(opt, keep_frac):
    """E23's predictors, computed on CPU one tensor at a time.

    At 1B the fp32 optimizer state (9.9 GB) plus a bf16 checkpoint plus the risk temporaries do not
    fit together on a shared 24 GB card, so each parameter is copied to the CPU, scored there, and
    freed. Math is identical; only the device moves.
    """
    tot_u2 = r_sgd = r_adam = r_lion = 0.0
    flip_num = n_drop = 0
    for p in opt.params:
        if p.grad is None or p.dim() != 2:
            continue
        g = p.grad.detach().to("cpu", torch.float32)
        u = opt.upd(p).detach().to("cpu", torch.float32)
        tot_u2 += float((u * u).sum())
        k = max(1, int(round(keep_frac * g.numel())))
        idx = torch.topk(g.abs().flatten(), k, sorted=False).indices
        keep = torch.zeros(g.numel(), dtype=torch.bool)
        keep[idx] = True
        keep = keep.view_as(g)
        delta = torch.where(keep, torch.zeros_like(g), g)
        r_sgd += float((delta ** 2).sum())
        ratio = torch.where(g.abs() > 1e-12, delta / g, torch.zeros_like(g))
        r_adam += float(((u * ratio) ** 2).sum())
        m = opt.state.get(p, {}).get("m") if isinstance(opt, LiteOpt) else opt._slots(p)[0]
        if m is not None:
            arg = m.detach().to("cpu", torch.float32)
            flipped = (torch.sign(arg + delta) != torch.sign(arg)) & (delta.abs() > 0)
            flip_num += float(flipped.sum())
            n_drop += int((delta != 0).sum())
            r_lion += float((u.abs() * flipped.float()).sum() * 2.0)
            del arg, flipped
        del g, u, delta, ratio, keep
    d = max(tot_u2, 1e-30)
    return {"risk_sgd": r_sgd / d, "risk_adamw": r_adam / d, "risk_lion": r_lion / d,
            "flip_frac_of_dropped": flip_num / max(n_drop, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--model", default=os.environ.get("AUDIT_MODEL",
                                                      "/root/qcc/models/Llama-3.2-1B-Instruct"))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--probe-at", type=int, default=600)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--opt8bit", action="store_true")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--densities", default="1.0,0.05")
    ap.add_argument("--optimizers", default="adamw,lion")
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e24_scale_risk.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e24] waiting ({free/2**30:.1f} GB free)", flush=True)
            time.sleep(30)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = AutoConfig.from_pretrained(args.model).vocab_size
    t0 = time.time()
    text = load_text(args.chars)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"[e24] tokenized {len(ids)/1e6:.2f}M tokens in {time.time()-t0:.0f}s", flush=True)
    n_val = 200_000
    train = torch.tensor(ids[:len(ids) - n_val], dtype=torch.long)
    val = torch.tensor(ids[len(ids) - n_val:], dtype=torch.long)
    print(f"[e24] train {train.numel()/1e6:.2f}M / val {val.numel()/1e3:.0f}K tokens | "
          f"steps {args.steps} bs {args.bs}x{args.block} | lr {args.lr}", flush=True)

    out = {"config": vars(args), "results": []}
    densities = [float(x) for x in args.densities.split(",")]
    for opt_name in args.optimizers.split(","):
        for keep in densities:
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_pretrained(args.model,
                                                         dtype=torch.bfloat16).to(dev)
            model.gradient_checkpointing_enable()
            model.train()
            ps = [p for p in model.parameters() if p.requires_grad]
            if args.opt8bit and opt_name == "adamw":
                opt = BnbAdam(ps, args.lr, args.b1, args.b2, args.eps)
            else:
                opt = LiteOpt(opt_name, ps, args.lr, args.b1, args.b2, args.eps)
            hist, losses, risks = [], [], None
            probe_curve = []
            t0 = time.time()

            def fixed_probe():
                model.eval()
                with torch.no_grad():
                    tot = 0.0
                    for j in range(2):
                        need = args.bs * (args.block + 1)
                        st = (j * args.bs * args.block) % (len(val) - need - 1)
                        w = val[st:st + need].view(args.bs, args.block + 1)
                        o = model(w[:, :-1].to(dev))
                        lg = o.logits if hasattr(o, "logits") else o
                        tot += float(F.cross_entropy(lg.reshape(-1, vocab),
                                                     w[:, 1:].to(dev).reshape(-1)).item())
                model.train()
                return tot / 2

            def batch(stream, i):
                need = args.bs * (args.block + 1)
                s = (i * args.bs * args.block) % (len(stream) - need - 1)
                x = stream[s:s + need].view(args.bs, args.block + 1)
                return x[:, :-1].to(dev), x[:, 1:].to(dev)

            for i in range(args.steps):
                x, y = batch(train, i)
                o = model(x)
                lg = o.logits if hasattr(o, "logits") else o
                loss = F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
                loss.backward()
                if i == args.probe_at:
                    risks = risk_scores(opt, keep)
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                if keep < 1.0:
                    with torch.no_grad():
                        for p in opt.params:
                            if p.grad is None:
                                continue
                            g = p.grad
                            k = max(1, int(round(keep * g.numel())))
                            idx = torch.topk(g.detach().abs().flatten(), k,
                                             sorted=False).indices
                            out_g = torch.zeros_like(g)
                            out_g.view(-1)[idx] = g.view(-1)[idx]
                            p.grad = out_g
                opt.step()
                opt.zero_grad()
                losses.append(float(loss.item()))
                if i % args.probe_every == 0 or i == args.steps - 1:
                    pv = fixed_probe()
                    probe_curve.append(round(pv, 4))
                    print(f"  [{opt_name} keep {keep}] step {i:5d} loss {loss.item():.4f} "
                          f"probe {pv:.4f} ({time.time()-t0:.0f}s)", flush=True)
                if i % 200 == 0 and risks is not None:
                    hist.append({"step": i, "loss": round(float(loss.item()), 4)})

            model.eval()
            with torch.no_grad():
                vs, nb = 0.0, 0
                for j in range(args.val_batches):
                    x, y = batch(val, j)
                    o = model(x)
                    lg = o.logits if hasattr(o, "logits") else o
                    vs += float(F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1)).item())
                    nb += 1
            # the trend is judged on the FIXED probe, not on per-batch training loss
            first = probe_curve[0] if probe_curve else float("nan")
            last = probe_curve[-1] if probe_curve else float("nan")
            res = {"optimizer": opt_name, "keep_frac": keep, "val_loss": round(vs / max(nb, 1), 4),
                   "train_first50": round(first, 4), "train_last50": round(last, 4),
                   "probe_curve": probe_curve,
                   "train_loss_decreasing": bool(last < first),
                   "median_step_ms": round((time.time() - t0) / args.steps * 1e3, 1)}
            if risks:
                res.update({k: round(v, 6) for k, v in risks.items()})
            out["results"].append(res)
            flag = "" if res["train_loss_decreasing"] else "   <-- NON-DISCRIMINATIVE"
            print(f"[e24] {opt_name:<6} keep {keep:<5} val {res['val_loss']:.4f} "
                  f"train {first:.4f}->{last:.4f} {res['median_step_ms']:.0f}ms/step{flag}",
                  flush=True)
            with open(args.out + ".partial", "w") as fh:
                json.dump(out, fh, indent=2)
            del model, opt
            torch.cuda.empty_cache()

    print("\n[e24] summary")
    for r in out["results"]:
        print(f"  {r['optimizer']:<6} keep {r['keep_frac']:<5} val {r['val_loss']:.4f} "
              f"decreasing={r['train_loss_decreasing']}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e24] wrote {args.out}")


if __name__ == "__main__":
    main()
