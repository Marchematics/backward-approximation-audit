# Claim ledger

Every row names the shipped script that produced the number and, where the number is load-bearing,
a regeneration command. A claim without a matched measurement is listed as open rather than
estimated. `[falsified]` rows are kept deliberately: they are the reason to trust the others.

All measurements are mechanism measurements on a shared A10G, not throughput claims. Wall-clock
numbers are the exception and are labelled as such.

## A. Established

| # | claim | evidence | command |
|---|---|---|---|
| A1 | Llama-3.2-1B full-parameter bf16 training is feasible on one A10G: 671 ms/step, 1526 tok/s, 11.77 GB peak at bs 2 x seq 512 with gradient checkpointing | `results/probe_a10_training.json` | `python3 experiments/probe_a10_training.py --steps 15 --bs 2 --seq 512` |
| A2 | **Update-space amplification `A_b` spans 0.13-1.1e5 as a function of error *geometry*, at an identical injected budget**: proportional 0.13 / rank-truncation 1.07 / white 2.79 median (p90 833, max 1.09e5), over 112 matrix blocks | `results/e2_error_geometry.json`, `results/e3_statistic.json` | `python3 experiments/e2_error_geometry.py --steps 60 --bs 4 --seq 512` |
| A3 | The injected error is linear in the budget (`grad_mse = c^2` exactly, `upd_mse = A*c^2`), so A2 is not a large-perturbation artefact | `results/e1b_update_vs_grad.json` | `python3 experiments/e1b_update_vs_grad.py --steps 24 --bs 4 --seq 512` |
| A4 | Gradient norm predicts block fragility better than any `v`-derived statistic: Spearman 0.58-0.63 vs 0.38-0.41 for `sum g^2/v`, 0.40-0.51 for quantiles of `\|g\|/sqrt(v)` | `results/e3_statistic.json` | `python3 experiments/e3_statistic.py --steps 60 --bs 4 --seq 512` |
| A5 | At 5% coordinate density, four selection criteria (magnitude, `\|g\|/sqrt(v)`, `\|m\|/sqrt(v)`, uniform random) land within 0.17 AUC of each other, over two seeds | `results/e9_compact_gpu.json`, `results/e9_seed1.json` | `python3 experiments/e9_compact_criterion.py --device cuda --steps 700 --chars 200000000` |
| A6 | The kept fraction is first-order: cost against dense is +0.27 AUC at 5%, +0.11 at 20%, +0.05 at 50% | `results/e9_dens20.json`, `results/e9_dens50.json` | `... --keep-frac 0.2` / `0.5` |
| A7 | A 5% masked backward takes a step of norm 0.32-0.36x the dense one; rescaling lr to match closes **55%** of the AUC gap | `results/e10_step_norm.json`, `results/e11_step_controller.json` | `python3 experiments/e10_step_norm.py --device cuda --steps 400` |
| A8 | Truncating the backward graph is a real wall-clock saving: 1.63x (keep last 3 of 6 blocks), 2.46x (keep 1 of 6), CUDA events, reproduced in two independent runs | `results/e12_allocation.json`, `results/e14_diagnostic_table.json` | `python3 experiments/e14_diagnostic_table.py --device cuda --steps 400` |
| A9 | Update-norm allocation of a layer budget is indistinguishable from random (3.2230 vs 3.2242 AUC) and 10x worse than gradient-norm allocation (+0.527 vs +0.054 cost at 3/6 blocks) | `results/e12_allocation.json` | `python3 experiments/e12_allocation.py --device cuda --steps 500 --keep-layers 3` |
| A10 | The whole allocation effect is the embedding/head group: pinning it (3.2% of `\|\|dtheta\|\|`, 48.3% of gradient norm) collapses all policies to +0.039..+0.041 cost and makes update-norm and gradient-norm select the same set | `results/e12_layers_only.json` | `... --always-include=-1` |
| A11 | At a fixed 4-of-6 block budget with the embedding kept, random / gradient-sensitive / pinned allocation differ by 0.002 AUC, and the total cost of dropping a third of the blocks is +0.026 | `results/e17_fixedk.json` | `python3 experiments/e17_equal_compute.py --device cuda --steps 400 --keep-layers 4` |
| A12 | Quality-neutral requires the step-norm ratio near 1: the two mechanisms with ratio within 0.01 of dense are the two cheapest (cost 0.000 and 0.006); every other mechanism deviates and pays | `results/e14_diagnostic_table.json` | `python3 experiments/e14_diagnostic_table.py --device cuda --steps 400` |
| A13 | Masking after a full backward saves optimizer work and **zero FLOPs**: sparse arms ran 1.0 s/step against 0.64 s/step dense | `results/e7_sparse_criterion_main.json` | `python3 experiments/e7_sparse_criterion.py --steps 400 --bs 4 --seq 512` |

## B. Falsified (kept on purpose)

| # | hypothesis | how it died | evidence |
|---|---|---|---|
| F1 | Allocate backward compute by optimizer-state sensitivity, i.e. `sum g^2/v` beats gradient norm | loses at predicting fragility (A4) and ranks worst of four criteria end-to-end | `results/e3_statistic.json`, `results/e9_compact_gpu.json` |
| F2 | `\|g\|/sqrt(v)` is the right per-coordinate criterion | `v` is an EMA of `g^2`, so this measures surprise, not update size; it lost at all 40 logged checkpoints | `results/e7_sparse_criterion_main.json` |
| F3 | `\|m\|/sqrt(v)`, the "corrected" criterion, should win because it captures 5.8x more update mass | captures 5.8x more update mass and trains worse than all four alternatives | `results/e9_overlap.json`, `results/e9_compact_gpu.json` |
| F4 | An online step-norm controller recovers the sparse quality gap | saturates at its lr clip (20x) with the norm stuck at 210 against an 872 target; 53% closed vs 55% for a fixed rescale | `results/e11_step_controller.json` |
| F5 | Freezing the moments outside the mask restores quality | restores the norm (726 vs 872) and makes quality worse: 16% closed vs 55% | `results/e11_step_controller.json` |
| F6 | A scalar step-norm ratio ranks mechanism quality | `rand_elem` (0.645) costs 3x `sparse_block` (0.747); `lowrank_down` (1.131) costs 25x `lowrank_row` (0.999) | `results/e14_diagnostic_table.json` |
| F7 | Pinning the embedding is what makes block dropping work | an experimental bug had been dropping the embedding in the control arms; after the fix the effect mostly disappears (A11) | `docs/WHY_MEASUREMENTS_DISAGREE.md` |

## C. Open, with the run that would close it

| # | question | why it matters | the run |
|---|---|---|---|
| O1 | Does the 0.026 cost of dropping one third of the blocks survive at 1B+ scale? | every practical claim depends on the cost staying small as models grow | E17 at 1B inside a memory-gated window |
| O2 | Is DropBP's sensitivity criterion measurably better when the sensitivity spread is large? | here the spread was 25% across six blocks, which makes the test weak | a deeper model (24+ blocks) where sensitivity varies by >2x |
| O3 | Does any of this hold for SGD or Lion? | if the amplification structure is not Adam-specific, the framing becomes numerical linear algebra | rerun E2/E10 with SGD and Lion |
| O4 | Can a real kernel realise the 1.63x truncated-backward saving inside a training loop? | A8 measured the graph saving on a microbenchmark, not in a full step with a scheduler | fused block-dropping kernel + end-to-end step timing |
| O5 | Does the quality-neutral point move with model scale and task? | the 5%/20%/50% cost curve is one corpus and one architecture | repeat A6 at two model sizes and a second corpus |
