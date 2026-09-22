"""E7: which coordinates should a sparse backward compute? (end-to-end, paired arms)

SOTA setting. Sparse / low-rank backward and gradient-compression methods choose what to keep
by gradient magnitude: top-k by |g_i|, or rank-r by spectral energy. Both are gradient-space
criteria. But the runtime applies U(g) = m/(sqrt(v) + eps), and the update-space contribution
of coordinate i is proportional to g_i / sqrt(v_i), not to g_i. Under Adam, coordinates with a
large accumulated second moment are *cheap to get wrong* -- their update is divided down --
and coordinates with |g_i| << sqrt(v_i) are *expensive to get wrong* -- a small error flips the
sign of a full-size step.

Measured here (E2/E3): the same relative gradient error costs 0.13x to 1.1e5x as much in
update space depending only on where the error sits.

Arms, all at identical compute budget (same k per tensor, same steps, same data order, same seed):
    full      : no approximation (reference)
    topk_g    : keep the k largest |g_i|                       (SOTA criterion)
    topk_v    : keep the k largest |g_i| / sqrt(v_i)           (update-space criterion)
    topk_rand : keep k uniformly random coordinates            (floor / control)

Everything else is held fixed: AdamW on the sparse coordinates only, identical schedule.
Reported: training loss vs step, wall-clock per step, and steps-to-reach a matched loss target.

Run: ./venv/bin/python review20260922/e7_sparse_criterion.py --steps 400 --bs 4 --seq 512 \
        --keep-frac 0.05 --arms full,topk_g,topk_v
"""
import argparse, glob, json, math, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AUDIT_MODEL", "/root/qcc/models/Llama-3.2-1B-Instruct")
CORPUS = "/root/qcc/data/longbench/data"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def build_tokens(tok, need, start_frac=0.0):
    ids = []
    for f in sorted(glob.glob(os.path.join(CORPUS, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = rec.get("context") or rec.get("input") or ""
                if txt:
                    ids.extend(tok(txt, add_special_tokens=False)["input_ids"])
    n = len(ids)
    s = int(start_frac * n)
    out = ids[s:s + need]
    while len(out) < need:                       # wrap if the split is short
        out = out + ids[:need - len(out)]
    return torch.tensor(out, dtype=torch.long)


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


class SparseAdamW:
    """AdamW where only a mask of coordinates receives a gradient (sparse backward)."""

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 state_dtype=torch.float16):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.state = {}
        self.state_dtype = state_dtype
        self.t = 0
        self.refresh_sec = 0.0

    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                st = self.state.setdefault(p, {})
                m = st.get("m")
                v = st.get("v")
                if m is None:
                    m = torch.zeros_like(p, dtype=self.state_dtype)
                    v = torch.zeros_like(p, dtype=self.state_dtype)
                    st["m"], st["v"] = m, v
                mask = st.get("mask")
                g = p.grad
                if mask is not None:
                    gm = g[mask].to(self.state_dtype)
                    m[mask] = self.b1 * m[mask] + (1 - self.b1) * gm
                    v[mask] = self.b2 * v[mask] + (1 - self.b2) * (gm * gm)
                else:
                    gf = g.to(self.state_dtype)
                    m.mul_(self.b1).add_(gf, alpha=1 - self.b1)
                    v.mul_(self.b2).addcmul_(gf, gf, value=1 - self.b2)
                upd = (m.to(torch.float32) / bc1) / ((v.to(torch.float32) / bc2).sqrt() + self.eps)
                if self.wd:
                    upd = upd + self.wd * p.float()
                p.add_(upd.to(p.dtype), alpha=-self.lr)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()


def make_hook(mode, st, keep_frac, gen, block_seed=0, refresh_every=20):
    """Backward hook: zero all but the selected coordinates of this tensor's gradient.

    The selection is refreshed every `refresh_every` steps (the mask is otherwise reused),
    which is both cheaper and closer to what a real planner would do.
    """
    def hook(grad):
        numel = grad.numel()
        k = max(1, int(round(keep_frac * numel)))
        st["calls"] = st.get("calls", 0) + 1
        cached = st.get("mask")
        if cached is not None and st["calls"] % refresh_every != 1:
            out = torch.zeros_like(grad)
            out.view(-1)[cached.view(-1)] = grad.view(-1)[cached.view(-1)]
            return out
        gf = grad.detach().float().flatten()
        if mode == "topk_g":
            idx = torch.topk(gf.abs(), k, sorted=False).indices
        elif mode == "topk_v":
            v = st.get("v")
            if v is None or st.get("t", 0) == 0:
                idx = torch.topk(gf.abs(), k, sorted=False).indices
            else:
                ratio = gf.abs() / (v.float().flatten().sqrt() + 1e-8)
                idx = torch.topk(ratio, k, sorted=False).indices
        elif mode == "topk_mv":
            # the corrected criterion: Adam's own step at this coordinate, |m|/sqrt(v).
            # |g|/sqrt(v) is the coordinate's *surprise* (v is an EMA of g^2), not the size
            # of the update the optimizer will take; m/sqrt(v) is the update.
            m = st.get("m")
            v = st.get("v")
            if m is None or v is None or st.get("t", 0) == 0:
                idx = torch.topk(gf.abs(), k, sorted=False).indices
            else:
                crit = (m.float().flatten().abs() / (v.float().flatten().sqrt() + 1e-8))
                idx = torch.topk(crit, k, sorted=False).indices
        elif mode == "topk_rand":
            idx = torch.randperm(numel, generator=gen)[:k]
        else:
            return grad
        mask = torch.zeros(numel, dtype=torch.bool, device=grad.device)
        mask[idx] = True
        out = torch.zeros_like(grad)
        out.view(-1)[mask] = grad.view(-1)[mask]
        st["mask"] = mask.view_as(grad)
        st["t"] = st.get("t", 0) + 1
        return out
    return hook


def run_arm(arm, a, dev, tok, toks, seed):
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    blk = blocks_of(model)
    params = list(model.parameters())
    opt = SparseAdamW(params, lr=a.lr, betas=(a.beta1, a.beta2), eps=a.eps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    handles = []
    if arm != "full":
        for name, p in blk.items():
            st = opt.state.setdefault(p, {})
            handles.append(p.register_hook(
                make_hook(arm, st, a.keep_frac, gen, refresh_every=a.refresh_every)))

    def batch(i):
        s = (i * a.bs * a.seq) % (len(toks) - a.bs * a.seq - 2)
        x = toks[s:s + a.bs * a.seq].view(a.bs, a.seq).to(dev)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()

    rows = []
    t0 = time.time()
    for step in range(a.steps):
        x, y = batch(step)
        loss = model(input_ids=x, labels=y).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == a.steps - 1:
            torch.cuda.synchronize()
            rows.append({"step": step, "loss": round(float(loss.item()), 4),
                         "elapsed": round(time.time() - t0, 1)})
            if step % 50 == 0 or step == a.steps - 1:
                print(f"  [{arm}] step {step:4d} loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    torch.cuda.synchronize()
    total = time.time() - t0
    del model, opt
    torch.cuda.empty_cache()
    return {"arm": arm, "rows": rows, "total_sec": round(total, 1),
            "sec_per_step": round(total / a.steps, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--keep-frac", type=float, default=0.05)
    ap.add_argument("--refresh-every", type=int, default=20)
    ap.add_argument("--arms", default="full,topk_g,topk_v")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e7_sparse_criterion.json"))
    a = ap.parse_args()
    dev = "cuda"
    arms = a.arms.split(",")

    tok = AutoTokenizer.from_pretrained(MODEL)
    toks = build_tokens(tok, a.bs * a.seq * (a.steps + 2))
    print(f"[e7] {torch.cuda.get_device_name(0)} | steps={a.steps} bs={a.bs} seq={a.seq} "
          f"keep={a.keep_frac} | arms={arms}", flush=True)

    out = {"model": "Llama-3.2-1B-Instruct", "steps": a.steps, "bs": a.bs, "seq": a.seq,
           "lr": a.lr, "keep_frac": a.keep_frac, "seed": a.seed, "arms": {}}
    for arm in arms:
        print(f"[e7] arm {arm}", flush=True)
        out["arms"][arm] = run_arm(arm, a, dev, tok, toks, a.seed)

    # steps-to-target: use the full arm's final loss as the target
    if "full" in out["arms"]:
        target = out["arms"]["full"]["rows"][-1]["loss"]
        out["target_loss"] = target
        print(f"\n[e7] target loss (full arm, final) = {target:.4f}")
        for arm in arms:
            rows = out["arms"][arm]["rows"]
            hit = next((r for r in rows if r["loss"] <= target), None)
            out["arms"][arm]["steps_to_target"] = hit["step"] if hit else None
            out["arms"][arm]["time_to_target"] = hit["elapsed"] if hit else None
            print(f"  {arm:<10} steps_to_target {out['arms'][arm]['steps_to_target']}  "
                  f"time {out['arms'][arm]['time_to_target']}s  "
                  f"sec/step {out['arms'][arm]['sec_per_step']}")

    path = a.out if not a.tag else a.out.replace(".json", f"_{a.tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e7] wrote {path}")


if __name__ == "__main__":
    main()
