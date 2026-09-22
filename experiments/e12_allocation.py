"""E12/E15: the allocation axis, and what layer-drop actually costs in wall-clock.

Motivation. E9/E11 killed the selection-criterion axis: at 5% density, magnitude / v-weighted /
momentum-weighted / random all land within 0.17 AUC, while the kept fraction moves quality by
0.27 AUC, and no scalar recalibration recovers the rest. Two questions remain, and they are the
only places a runtime can still win:

  E12  ALLOCATION: given a fixed global budget of layers to backprop through, does it matter WHICH
       layers? Three policies at an identical budget:
         upd   -- allocate by each layer's contribution to ||dtheta||   (update-space)
         grad  -- allocate by each layer's gradient norm                (DropBP-style)
         rand  -- uniformly at random                                   (floor)
       If `upd` beats `grad` at equal budget, the allocation axis has real leverage even though
       the per-coordinate criterion does not.

  E15  SPEED: layer-drop saves wall-clock only if the backward through dropped layers is actually
       skipped. This measures it -- truncated backward via torch.autograd.grad on a boundary
       activation -- for k of L layers, on the same model and batch, with CUDA events. No masking
       (masking after a full backward saves optimizer work but zero FLOPs, which is why the earlier
       sparse arms showed NO speedup: 1.0 s/step vs 0.64 s/step dense).

Run: python3 experiments/e12_allocation.py --device cuda --steps 500
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_compact_criterion import GPT, load_text  # noqa: E402


def layer_groups(model):
    """(block_index, [parameter names]) for each transformer block, plus head/tail groups."""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = n.split(".")
        if parts[0] == "blocks":
            groups.setdefault(int(parts[1]), []).append(n)
        else:
            groups.setdefault(-1, []).append(n)   # embeddings / final norm / head
    return groups


def train(arm, args, data, vocab, dev, seed=0, allocation=None, probe_layers=None):
    """One training run. `allocation` = set of block ids allowed to receive an update."""
    torch.manual_seed(seed)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    groups = layer_groups(model)
    named = dict(model.named_parameters())
    moments = {}
    gen = torch.Generator().manual_seed(seed + 11)

    def batch(i):
        need = args.bs * (args.block + 1)
        s = (i * args.bs * args.block) % (len(data) - need - 1)
        x = data[s:s + need].view(args.bs, args.block + 1)
        return x[:, :-1].to(dev), x[:, 1:].to(dev)

    def step(i, update_norms=None):
        sq_total = 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                st = moments.setdefault(n, {"m": torch.zeros_like(p), "v": torch.zeros_like(p),
                                            "t": 0})
                st["t"] += 1
                m, v = st["m"], st["v"]
                m.mul_(args.b1).add_(p.grad, alpha=1 - args.b1)
                v.mul_(args.b2).addcmul_(p.grad, p.grad, value=1 - args.b2)
                upd = (m / (1 - args.b1 ** st["t"])) / \
                      ((v / (1 - args.b2 ** st["t"])).sqrt() + 1e-8)
                sq_total += float((upd ** 2).sum())
                if update_norms is not None:
                    for gid, names in groups.items():
                        if n in names:
                            update_norms[gid] = update_norms.get(gid, 0.0) + \
                                float((upd ** 2).sum())
                            break
                p.add_(upd, alpha=-args.lr)
        return math.sqrt(sq_total)

    # per-group gradient norms, used by the `grad` policy
    def group_grad_norms():
        out = {}
        for gid, names in groups.items():
            s = 0.0
            for n in names:
                g = named[n].grad
                if g is not None:
                    s += float((g ** 2).sum())
            out[gid] = math.sqrt(s)
        return out

    rows, auc = [], 0.0
    probe = []
    for i in range(args.steps):
        x, y = batch(i)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()

        if probe_layers is not None and i == args.probe_at:
            un = {}
            step(i, update_norms=un)
            gn = group_grad_norms()
            probe.append({"step": i, "update_norm_by_group": un, "grad_norm_by_group": gn})
            for p in model.parameters():
                p.grad = None
            continue

        if allocation is not None:
            # zero the gradient of every parameter outside the allowed blocks, so those
            # blocks receive no update (equivalent, for quality, to not backpropagating them)
            for gid, names in groups.items():
                if gid in allocation:
                    continue
                for n in names:
                    if named[n].grad is not None:
                        named[n].grad = None
        step(i)
        for p in model.parameters():
            p.grad = None
        if i % 25 == 0 or i == args.steps - 1:
            rows.append({"step": i, "loss": round(float(loss.item()), 4)})
        auc += float(loss.item())

    tail = [r["loss"] for r in rows[-8:]]
    out = {"arm": arm, "auc": round(auc / args.steps, 4),
           "final_loss": round(sum(tail) / len(tail), 4), "rows": rows}
    if probe:
        out["probe"] = probe
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def speed_benchmark(args, data, vocab, dev):
    """Honest wall-clock: truncated backward through only the last k blocks."""
    torch.manual_seed(0)
    model = GPT(vocab, args.dim, args.layers, args.heads, args.ctx).to(dev)
    L = args.layers
    blocks = list(model.blocks)
    need = args.bs * (args.block + 1)
    x = data[:need].view(args.bs, args.block + 1)[:, :-1].to(dev)
    y = data[1:need + 1].view(args.bs, args.block + 1)[:, 1:].to(dev)

    def timed(fn, reps=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            fn()
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en) / reps

    results = {}
    # full backward
    def full():
        model.zero_grad(set_to_none=True)
        out = model(x)
        F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1)).backward()
    results["full"] = timed(full)

    # truncated: keep the graph only through the last k blocks by re-running the forward
    # without grad for the frozen prefix, then autograd.grad on the boundary activation
    for k in (L // 2, L // 4, 2, 1):
        if k < 1:
            continue

        def trunc(k=k):
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                b, t = x.shape
                h = model.tok(x) + model.pos(torch.arange(t, device=dev))[None]
                for blk in blocks[:L - k]:
                    h = blk(h)
            h = h.detach().requires_grad_(True)
            for blk in blocks[L - k:]:
                h = blk(h)
            out = model.head(model.lnf(h))
            loss = F.cross_entropy(out.reshape(-1, vocab), y.reshape(-1))
            torch.autograd.grad(loss, [h] + [p for blk in blocks[L - k:]
                                             for p in blk.parameters()])
        results[f"trunc_last{k}"] = timed(trunc)

    del model
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--keep-layers", type=int, default=3, help="how many blocks may update")
    ap.add_argument("--always-include", default="", help="group ids always allowed to update, e.g. -1")
    ap.add_argument("--compare-layers-only", action="store_true")
    ap.add_argument("--probe-at", type=int, default=150)
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wait-free-gb", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e12_allocation.json"))
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda":
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 >= args.wait_free_gb:
                break
            print(f"[e12] waiting for {args.wait_free_gb} GB free (now {free/2**30:.1f})", flush=True)
            time.sleep(30)

    text = load_text(args.chars)
    vocab = 256
    data = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    print(f"[e12] corpus {len(data)/1e6:.1f}M | {args.layers} blocks | keep {args.keep_layers}",
          flush=True)
    out = {"config": vars(args)}

    # ---- E15 first: what does skipping actually save in wall-clock? ----
    print("[e15] wall-clock benchmark (truncated backward)", flush=True)
    sp = speed_benchmark(args, data, vocab, dev)
    full_ms = sp["full"]
    print(f"  full backward            {full_ms:8.2f} ms")
    for k, v in sp.items():
        if k == "full":
            continue
        print(f"  {k:<24} {v:8.2f} ms   speedup {full_ms/v:5.2f}x")
    out["speed"] = sp
    out["speed_speedup"] = {k: full_ms / v for k, v in sp.items() if k != "full"}

    # ---- E12: measure per-layer contribution, then compare allocation policies ----
    print("[e12] dense reference + per-layer probe", flush=True)
    dense = train("full", args, data, vocab, dev, probe_layers=True)
    probe = dense["probe"][0]
    un, gn = probe["update_norm_by_group"], probe["grad_norm_by_group"]
    out["probe"] = probe
    out["dense"] = {k: v for k, v in dense.items() if k != "probe"}
    print(f"  dense auc {dense['auc']:.4f}  final {dense['final_loss']:.4f}", flush=True)
    print(f"  per-group update-norm share and gradient-norm share:")
    tot_u = sum(un.values()) or 1.0
    tot_g = sum(gn.values()) or 1.0
    for gid in sorted(un):
        print(f"    block {gid:>2}: upd {un[gid]/tot_u*100:5.1f}%   grad {gn[gid]/tot_g*100:5.1f}%")

    k = args.keep_layers
    always = {int(x) for x in args.always_include.split(",") if x.strip()}
    cand = [g for g in sorted(un) if g not in always]
    policies = {
        "upd": sorted(cand, key=lambda g: -un[g])[:k],
        "grad": sorted(cand, key=lambda g: -gn[g])[:k],
        "rand": sorted(cand, key=lambda g: torch.rand(()).item() + hash(str(g)) % 100 / 1e6)[:k],
    }
    policies = {n: sorted(set(v) | always) for n, v in policies.items()}
    out["policies"] = {n: v for n, v in policies.items()}
    print(f"  policies: " + "  ".join(f"{n}={v}" for n, v in policies.items()), flush=True)

    out["arms"] = {"full": out["dense"]}
    for name, alloc in policies.items():
        r = train(f"alloc_{name}", args, data, vocab, dev, allocation=set(alloc))
        out["arms"][f"alloc_{name}"] = r
        gap = r["auc"] - dense["auc"]
        print(f"  alloc {name:<5} ({alloc}) auc {r['auc']:.4f} final {r['final_loss']:.4f} "
              f"gap vs dense {gap:+.4f}", flush=True)

    print(f"\n[e12] dense {dense['auc']:.4f}; budget = {k}/{args.layers} blocks")
    for name in policies:
        a = out["arms"][f"alloc_{name}"]["auc"]
        print(f"      {name:<5} auc {a:.4f}  cost {a - dense['auc']:+.4f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[e12] wrote {args.out}")


if __name__ == "__main__":
    main()
