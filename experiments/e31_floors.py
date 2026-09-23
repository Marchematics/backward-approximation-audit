"""E31: measure the two noise floors directly, then compute what effect sizes are detectable.

Six results in this project were withdrawn because their effect was smaller than the variation of
their own measurement (CLAIMS F10-F14). Those withdrawals cite floors that were estimated from the
casualties. This measures the floors on purpose:

  FLOOR-1 (quality / regime). Run the *identical* configuration N times with different seeds, no
    treatment difference, and record the spread of (a) dense held-out loss and (b) the 5%-density
    damage. The seed-to-seed standard deviation of the damage is the smallest damage difference that
    can be attributed to a treatment rather than to the seed.

  FLOOR-2 (wall clock / device). Time the *identical* step N times, interleaved, and record the
    spread. On a shared GPU the contention enters here, so the floor must be measured next to the
    experiment rather than assumed.

Then the useful output: for each of the six withdrawn comparisons, the claimed effect and the
measured floor, as a ratio. Anything with effect/floor < ~2 was never measurable on this hardware,
which is a statement about the measurement setup and not about the science.

Run: python3 experiments/e31_floors.py --device cuda
"""
import argparse, json, math, os, statistics as st, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def quality_floor(args, train, val, vocab, dev, n_seeds):
    """Identical config, different seeds: spread of dense loss and of 5% damage."""
    rows = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
        need = args.bs * (args.block + 1)

        def batch(stream, i):
            s = (i * args.bs * args.block + seed * 7919) % (len(stream) - need - 1)
            x = stream[s:s + need].view(args.bs, args.block + 1)
            return x[:, :-1].to(dev), x[:, 1:].to(dev)

        def probe():
            model.eval()
            with torch.no_grad():
                tot = 0.0
                for j in range(args.val_batches):
                    x, y = batch(val, j)
                    tot += float(F.cross_entropy(model(x).reshape(-1, vocab),
                                                 y.reshape(-1)).item())
            model.train()
            return tot / args.val_batches

        for i in range(args.steps):
            x, y = batch(train, i)
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()
        rows.append({"seed": seed, "dense": round(probe(), 4)})

        # same seed, but with the 5% mask -- the "damage" a sparsity experiment reports
        torch.manual_seed(seed)
        model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
        for i in range(args.steps):
            x, y = batch(train, i)
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None or p.dim() != 2:
                        continue
                    g = p.grad
                    k = max(1, int(round(0.05 * g.numel())))
                    idx = torch.topk(g.detach().abs().flatten(), k, sorted=False).indices
                    out = torch.zeros_like(g)
                    out.view(-1)[idx] = g.view(-1)[idx]
                    p.grad = out
            opt.step()
            opt.zero_grad()
        rows[-1]["sparse05"] = round(probe(), 4)
        rows[-1]["damage"] = round(rows[-1]["sparse05"] - rows[-1]["dense"], 4)
        print(f"  seed {seed}: dense {rows[-1]['dense']:.4f} sparse05 {rows[-1]['sparse05']:.4f} "
              f"damage {rows[-1]['damage']:+.4f}", flush=True)
        del model, opt
        if dev == "cuda":
            torch.cuda.empty_cache()

    dense = [r["dense"] for r in rows]
    dmg = [r["damage"] for r in rows]
    out = {"runs": rows,
           "dense_mean": round(st.mean(dense), 4), "dense_sd": round(st.stdev(dense), 4),
           "dense_spread": round(max(dense) - min(dense), 4),
           "damage_mean": round(st.mean(dmg), 4), "damage_sd": round(st.stdev(dmg), 4),
           "damage_spread": round(max(dmg) - min(dmg), 4)}
    print(f"  -> dense spread {out['dense_spread']:.4f} (sd {out['dense_sd']:.4f}); "
          f"damage spread {out['damage_spread']:.4f} (sd {out['damage_sd']:.4f})", flush=True)
    return out


def wallclock_floor(args, data, vocab, dev, n_reps):
    """Time the identical step n_reps times, interleaved, and report the spread."""
    torch.manual_seed(0)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    need = args.bs * (args.block + 1)
    times = []
    for rep in range(n_reps):
        s = (rep * args.bs * args.block) % (len(data) - need - 1)
        w = data[s:s + need].view(args.bs, args.block + 1)
        x, y = w[:, :-1].to(dev), w[:, 1:].to(dev)
        for _ in range(2):                      # warm-up
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.timing_steps):
            loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        times.append((time.time() - t0) / args.timing_steps * 1e3)
        print(f"  rep {rep}: {times[-1]:.2f} ms/step", flush=True)
    out = {"times_ms": [round(t, 2) for t in times],
           "median_ms": round(st.median(times), 2),
           "min_ms": round(min(times), 2), "max_ms": round(max(times), 2),
           "spread": round(max(times) / min(times), 3)}
    print(f"  -> median {out['median_ms']} ms, spread {out['spread']:.2f}x", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--timing-steps", type=int, default=10)
    ap.add_argument("--timing-reps", type=int, default=6)
    ap.add_argument("--chars", type=int, default=60_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "results", "e31_floors.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e31] waiting ({free/2**30:.1f} GB)", flush=True)
            time.sleep(20)

    text = load_text(args.chars)
    vocab = 256
    ids = list(text.encode("utf-8", errors="ignore"))
    n_val = min(len(ids) // 5, max(200_000, args.val_batches * args.bs * (args.block + 1) * 4))
    train = torch.tensor(ids[:len(ids) - n_val], dtype=torch.long)
    val = torch.tensor(ids[len(ids) - n_val:], dtype=torch.long)
    print(f"[e31] train {train.numel()/1e6:.0f}M val {val.numel()/1e6:.1f}M | {args.seeds} seeds",
          flush=True)

    out = {"config": vars(args)}
    print("[e31] FLOOR-1 quality (identical config, different seeds)", flush=True)
    out["quality"] = quality_floor(args, train, val, vocab, dev, args.seeds)
    print("[e31] FLOOR-2 wall clock (identical step, interleaved)", flush=True)
    out["wallclock"] = wallclock_floor(args, train, vocab, dev, args.timing_reps)

    # what effect sizes are detectable, given the floors
    q_sd = out["quality"]["damage_sd"]
    w_spread = out["wallclock"]["spread"]
    out["detectable"] = {
        "min_damage_difference (2 sd)": round(2 * q_sd, 4),
        "min_speedup (max/min)": round(w_spread, 3),
        "note": ("a damage difference below 2 sd, or a speedup below the observed spread, cannot be "
                 "distinguished from the measurement on this setup"),
    }
    print(f"\n[e31] DETECTABLE: damage differences >= {2*q_sd:.4f}, speedups >= {w_spread:.2f}x")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e31] wrote {args.out}")


if __name__ == "__main__":
    main()
