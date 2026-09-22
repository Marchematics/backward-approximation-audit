# What should a backward approximation compute?

A measurement study of gradient approximation for LLM training. Every number here was produced by
the scripts in `experiments/`, and every claim is tagged **[measured]**, **[falsified]** or
**[open]** in `CLAIMS.md`.

The study started from a specific hypothesis and then falsified it five times. What is left is a
small set of measurements that are, as far as this project could determine, not in the literature.

---

## The two numbers that motivate everything

With AdamW, the runtime does not apply the gradient `g`. It applies `U(g) = m / (sqrt(v) + eps)`.
Every approximation method in the literature is scored by how well it reproduces `g` — gradient MSE,
V norm, cosine, spectral energy. Those scores turn out not to determine the training outcome.

**1. The same relative gradient error costs between 0.13x and 1.1e5x as much in update space,
depending only on the *shape* of the error** (112 matrix blocks of Llama-3.2-1B, identical injected
budget):

| error model | what it represents | update-space amplification `A_b` (median / p90 / max) |
|---|---|---:|
| `delta ∝ g` | magnitude-preserving: quantization, scaled low rank | **0.13** / 0.17 / 0.22 |
| rank truncation | rank-`r` backward, token/channel sampling | **1.07** / 9.4 / 48.6 |
| white / orthogonal | random projection | **2.79** / 833 / **1.09e5** |

Reporting "gradient error 0.25" therefore says almost nothing about the optimizer. This is not a
statement about a mysterious optimizer: it is `(I - gg^T/||g||^2)` acting on the low-`g` tail, where
`|g_i| << sqrt(v_i)` and a small error flips the sign of a full-size step.

**2. The optimiser's step norm collapses when the backward is masked, and half the quality cost of
sparsification is exactly that.** At 5% density a masked backward takes a step whose norm is
0.32-0.36x the dense one. Rescaling the learning rate to match closes **55%** of the AUC gap. An
online controller cannot do better (53%) — it saturates without ever reaching the dense norm.

---

**3. The amplification is a joint property of the optimizer and its state.** Identical gradients,
identical injected budget, three optimizers: median amplification 0.015 (SGD+momentum), 0.126
(AdamW), **2.95 (Lion)** — a 200x spread. And within Lion it **grows 3.9x as the run gets longer**
(0.78 at 5 warmup steps -> 3.06 at 120), because a mature momentum puts more coordinates near a sign
boundary where a small error flips a full-size step. The sparsity literature is evaluated almost
entirely on AdamW; these numbers say the ratios do not transfer to sign-based optimizers.

## The four levers, all measured at the same scale

| lever | measured effect size | who works on it |
|---|---:|---|
| per-coordinate selection criterion | 0.17 AUC spread across 4 criteria, 2 seeds | **the sparsity / importance-sampling literature** |
| optimizer step-norm consistency | 0.19 AUC (half of the 5%-density cost) | essentially nobody |
| always updating the embedding/head block | **0.49 AUC** | nobody |
| removing backward work from the graph | **1.5-2.4x wall-clock** | DropBP, SLowMo and friends |
| which optimizer the run uses | **200x change in amplification** (SGD 0.015 -> Lion 2.95) | nobody, in this context |

The most crowded lever is the smallest one. Choosing coordinates cleverly is worth less than half of
what keeping the embedding block updated is worth, it buys zero wall-clock by itself, and the one
lever worth 1.5-2.4x — deleting backward computation from the graph — makes the per-coordinate
criterion irrelevant by construction.

Supporting measurements:

* **At 5% density, magnitude / v-weighted / momentum-weighted / uniform-random selection are within
  0.17 AUC of each other** over two seeds, while the kept fraction moves quality by 0.27 AUC
  (5% -> +0.27, 20% -> +0.11, 50% -> +0.05 AUC against dense).
* **The obvious fix inverts the ranking.** `|m|/sqrt(v)` captures 5.8x more of the update mass than
  the SOTA `|g|` criterion and trains *worse* than all of them. A single-step proxy for update
  importance predicted the opposite of the measured ordering.
* **Truncating the backward graph is a real speedup**: full backward 28.3 ms, last 3 of 6 blocks
  17.4 ms (1.63x), last 1 of 6 blocks 11.5 ms (2.46x), measured with CUDA events.
* **Block dropping is nearly free at this scale, and the layer choice is worth nothing.** At a
  fixed budget of 4 of 6 blocks with the embedding kept, random / gradient-sensitive / pinned
  allocation differ by **0.002 AUC**, and dropping a third of the blocks entirely costs **+0.026
  AUC** (`results/e17_fixedk.json`). The same null result as the coordinate axis (A5), on a second
  independent axis.

## What was falsified

Recorded because negative results are the reason to trust the rest:

0. "The amplification is optimizer-independent numerical linear algebra." Falsified the other way:
   the structure differs by 200x across SGD / AdamW / Lion, so "optimizer-conditioned" stays in the
   title — by measurement, not preference.
1. "Allocate backward compute by optimizer-state sensitivity." The derived statistic `sum g^2/v`
   loses to plain gradient norm at predicting fragility (Spearman 0.39 vs 0.60 over 112 blocks,
   four error geometries), and in per-coordinate form it ranks *worst* of four criteria.
2. `|g|/sqrt(v)` as the selection criterion. `v` is an EMA of `g^2`, so this measures the
   coordinate's *surprise*, not the size of the update. It lost at every logged checkpoint.
3. An online step-norm controller can recover the sparse quality gap. It saturates at its clip.
4. Freezing the moments outside the mask restores quality. It restores the *norm* (726 vs 872) and
   makes quality worse.
5. A scalar step-norm ratio ranks mechanism quality. `rand_elem` (ratio 0.645) costs 3x what
   `sparse_block` (0.747) costs; `lowrank_down` (1.131) costs 25x what `lowrank_row` (0.999) costs.

## Reproduction

Requires a CUDA GPU (an A10G with ~4 GB free was enough for the compact experiments; the Llama-3.2-1B
diagnostics want ~12 GB) and a local copy of a corpus. Paths are configurable:

```bash
export AUDIT_MODEL=/path/to/Llama-3.2-1B-Instruct        # default /root/qcc/models/...
# E2/E3 (amplification and statistic comparison):
python3 experiments/e2_error_geometry.py --steps 60 --bs 4 --seq 512
python3 experiments/e3_statistic.py      --steps 60 --bs 4 --seq 512
# E9/E10/E11 (criterion, density, step norm) -- compact model, ~1 min per arm:
python3 experiments/e9_compact_criterion.py  --device cuda --steps 500 --chars 200000000
python3 experiments/e10_step_norm.py         --device cuda --steps 400
python3 experiments/e11_step_controller.py   --device cuda --steps 400
# E12/E14/E17 (allocation, diagnostic table, equal-compute comparison):
python3 experiments/e12_allocation.py        --device cuda --steps 500
python3 experiments/e14_diagnostic_table.py  --device cuda --steps 400
python3 experiments/e17_equal_compute.py     --device cuda --steps 400
# E18 (optimizer ablation; --pretrained runs a real checkpoint via AUDIT_MODEL):
python3 experiments/e18_optimizer_ablation.py --device cuda --steps 60 --warmup 40
python3 experiments/e18_optimizer_ablation.py --pretrained --device cuda --bs 1 --block 512 --lr 1e-5
```

Raw outputs land in `results/`. The corpus is not bundled — point `CORPUS` in the scripts at any
JSONL/plain-text directory with a `context`-ish field, or adapt `load_text`.

## Layout

```
docs/NEXT_BACKWARD_DIRECTION.md  the full write-up: theory, SOTA survey, all measurements per
                                 experiment, and the 23-row claim ledger (V1-V23)
docs/WHY_MEASUREMENTS_DISAGREE.md  the comparison-frame trap, with the bug that caused it
experiments/                     one script per experiment, each self-contained
results/                         raw JSON, one file per run, never edited by hand
CLAIMS.md                        claim ledger with regeneration commands
```

## Caveats

* Mechanism measurements, not throughput claims. The A10G here is shared with other tenants and
  step time drifted by up to 3x across runs of one identical configuration (dense measured at 83, 93
  and 238 ms/step), so ms/step comparisons are reported but not relied on. The quality comparisons
  are made at equal blocks-kept per step and equal steps, which is timing-independent.
* The only wall-clock numbers treated as trustworthy are the truncated-backward microbenchmarks in
  `results/e12_allocation.json` and `results/e14_diagnostic_table.json` (CUDA events, 5 reps each,
  reproduced in two independent runs).
* The criterion and density results are on one 40 M-parameter transformer, one corpus, ~4 seeds,
  AdamW. The amplification results are on Llama-3.2-1B, 112 matrix blocks, 60 steps.
* Scale has not been checked above 1.2 B parameters, and no published method was re-run
  head-to-head on this hardware beyond a faithful DropBP-style block-dropping reimplementation.
