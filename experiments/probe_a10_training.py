"""A10G feasibility probe: can we train Llama-3.2-1B full-parameter on one 24 GB card,
and what are the real step times / memory / gradient+optimizer-state structure?

No claims; this only prints measurements. Run:
    python3 experiments/probe_a10_training.py --steps 15 --bs 4 --seq 512
"""
import argparse, json, math, os, time, glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AUDIT_MODEL", "/root/qcc/models/Llama-3.2-1B-Instruct")
CORPUS = os.environ.get("AUDIT_CORPUS", "/root/qcc/data/longbench/data")


def build_tokens(tok, need, seq):
    """Concatenate real LongBench document text into one token stream."""
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
    if len(ids) < need:
        raise SystemExit(f"corpus too small: {len(ids)} < {need}")
    return torch.tensor(ids[:need], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "probe_a10_training.json"))
    a = ap.parse_args()
    dev = "cuda"

    print(f"[probe] torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"[probe] free before load: {free/2**30:.2f} / {total/2**30:.2f} GB", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    need = a.bs * a.seq * (a.steps + 2)
    toks = build_tokens(tok, need, a.seq)
    print(f"[probe] token stream: {len(toks)} real tokens from LongBench", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    print(f"[probe] model loaded in {time.time()-t0:.1f}s, "
          f"params {sum(p.numel() for p in model.parameters())/1e6:.1f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), eps=1e-8)

    def batch(i):
        s = i * a.bs * a.seq
        x = toks[s:s + a.bs * a.seq].view(a.bs, a.seq).to(dev)
        return x, x.clone()

    rows = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        t = time.time()
        x, y = batch(step)
        out = model(input_ids=x, labels=y)
        loss = out.loss
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append({"step": step, "loss": loss.item(), "gnorm": gnorm,
                     "sec": dt, "tok_per_s": a.bs * a.seq / dt, "peak_gb": peak})
        if step == a.warmup_steps:
            rows = [r for r in rows[a.warmup_steps:]]  # keep warmup out of the mean
        if step % 5 == 0 or step == a.steps - 1:
            print(f"[step {step:3d}] loss {loss.item():.4f} gnorm {gnorm:.3f} "
                  f"{dt*1e3:7.1f} ms {a.bs*a.seq/dt:7.0f} tok/s peak {peak:.2f} GB", flush=True)

    steady = [r for r in rows if r["step"] >= a.warmup_steps]
    summary = {
        "device": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "free_before_load_gb": round(free / 2**30, 2),
        "model_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "batch": a.bs, "seq": a.seq, "steps": a.steps, "grad_ckpt": True,
        "opt": "AdamW fp32 states", "dtype": "bf16",
        "median_ms": round(sorted(r["sec"] for r in steady)[len(steady)//2] * 1e3, 1),
        "median_tok_per_s": round(sorted(r["tok_per_s"] for r in steady)[len(steady)//2]),
        "peak_gb": round(max(r["peak_gb"] for r in rows), 2),
        "loss_first": round(steady[0]["loss"], 4), "loss_last": round(steady[-1]["loss"], 4),
        "rows": rows,
    }
    print("\n[summary]", json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[probe] wrote {a.out}")


if __name__ == "__main__":
    main()
