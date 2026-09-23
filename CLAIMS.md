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
| A14 | **Amplification is a joint property of the optimizer and its state.** Same gradients and budget, median at c=0.25: SGD 0.015, AdamW 0.126, Lion 2.95 -- a 200x spread | `results/e18_optimizer_ablation.json` | `python3 experiments/e18_optimizer_ablation.py --device cuda --steps 60 --warmup 40` |
| A15 | Within one optimizer it moves with state age: Lion proportional 0.78 (warmup 5) -> 2.95 (40) -> 3.06 (120); Adam 0.077 -> 0.126; SGD *falls* 0.081 -> 0.013 | `results/e18_warm5.json`, `results/e18_warm120.json` | same script with `--warmup 5` / `--warmup 120` |
| A16 | The ordering survives scale on a real checkpoint: Llama-3.2-1B gives SGD 0.263, AdamW 0.228 (orthogonal 1.48), Lion 1.264 (orthogonal 2.68) | `results/e18_1b_final.json` | `... --pretrained --bs 1 --block 512 --lr 1e-5` |

| A17 | **RETRACTED — see F10.** The "Lion pays 4.8x AdamW" number came from a single unseeded run whose dense baseline was the lowest of three seeds | `results/e19_repro_s0.json`, `results/e19_repro_s1.json`, `results/e19_repro_s2.json` | `python3 experiments/e19_optimizer_density.py --seed N ...` |
| A22 | **Seeded, five-run comparison of the 5%-density cost**: +0.086 (AdamW), +0.14..+0.17 (SGD), +0.065..+0.075 (Lion) — Lion is the *least* damaged, and the spread across runs is 1.17x | `results/e23_2d_s0.json` ... `results/e23_all_s1.json` | `python3 experiments/e23_risk_predictor.py --mask-scope 2d --seed N` |
| A23 | **An optimizer-specific risk score predicts that optimizer's damage ordering**: Spearman rho(risk, damage) over four densities is 1.00 for AdamW (5/5 runs), 1.00 for SGD, 0.80-1.00 for Lion. The scores are `\|\|delta\|\|^2`, the preconditioned amplitude error, and the sign-flip mass | same files | same command |
| A24 | **The dense run is the noisy arm at this scale**: across three seeds the dense held-out loss spans 0.29 while the 5%-sparse run spans 0.13 | `results/e19_repro_s*.json` | as A17 |
| A18 | The step-norm diagnostic does **not** explain damage across optimizers: at 5% density Lion's norm ratio is 0.833 and its cost +0.311, while AdamW's is 0.519 and its cost +0.065 | same file | same command |
| A19 | **RETRACTED — see F9.** The E20 speedup was measured on an arm that silently froze the token/position embeddings and whose three "seed" runs were identical | `results/e21_grad_reach.json` | `python3 experiments/e21_grad_reach.py --device cuda` |
| A20 | **Corrected speedup, honest scope**: backward through the last 3 of 6 blocks, giving 1.72x on three genuinely independent seeds (1.64/1.89/1.63) at +0.0087 held-out cost — but 38 parameters (the token/position tables and the dropped blocks) receive no update | `results/e22_realskip_fixed.json` | `python3 experiments/e22_realskip_fixed.py --device cuda --steps 300 --seeds 0,1,2 --verify-grads` |
| A21 | **Making the skip gradient-correct removes the speedup**: checkpoint-style replay trains every parameter (0 without gradient) but runs at 0.82x, slower than dense, because replay costs a forward where the baseline pays a backward | same file | same command |

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
| F8 | The step-norm ratio predicts a scheme's quality cost across settings | it ranks mechanisms within one optimizer but inverts across optimizers: Lion moves its norm least and loses the most (A18) | `results/e19_optimizer_density.json` |
| F10 | "At 5% density Lion pays 4.8x what AdamW pays" | not reproducible under seeding: the three dense baselines span 0.29, larger than the effect, and the published +0.311 used the lowest of them. Seeded five-run comparison reverses the ordering (A22) | `results/e19_repro_s*.json`, `results/e23_*.json` |
| F9 | "Graph skip + step-norm correction = 1.72x at +0.003 with the embedding always trained" | the embedding was never trained: the prefix forward ran under `no_grad` and the boundary was detached, so `tok.weight`/`pos.weight` had no gradient; and all three "seed" runs used `seed=0`. The corrected, gradient-complete construction (replay) runs at 0.82x | `results/e21_grad_reach.json`, `results/e22_realskip_fixed.json` |

## B2. Configuration failures (recorded so they are not repeated)

| # | attempt | how it failed | evidence |
|---|---|---|---|
| X1 | 1B optimizer x density grid, corpus 1.5 M tokens, 1000 steps, lr 1e-5 | **non-discriminative**: training loss *rose* (AdamW 2.489 -> 2.766, SGD -> 3.000) while held-out loss stayed flat to four decimals (SGD: 2.4014 at every density). Fine-tuning a pretrained 1B on too little data cannot resolve a sparsity effect, so the grid answers nothing | `results/e19_1b_adamw.json`, `results/e19_1b_sgd.json` |
| X2 | same grid, Lion arm | the process was killed mid-run by an external action, not by an in-code failure | partial only |
| X3 | per-seed replication of E19 | revealed that E19's dense arm moves 0.29 across seeds, larger than the reported effect (see F10) | `results/e19_repro_s*.json` |

The 1B question is therefore still open (O7), and the next attempt needs a corpus large enough that
held-out loss actually moves: >= 50 M characters and a learning rate calibrated until training loss
*decreases*.

## C. Open, with the run that would close it

| # | question | why it matters | the run |
|---|---|---|---|
| O1 | Does the 0.026 cost of dropping one third of the blocks survive at 1B+ scale? | every practical claim depends on the cost staying small as models grow | E17 at 1B inside a memory-gated window |
| O2 | Is DropBP's sensitivity criterion measurably better when the sensitivity spread is large? | here the spread was 25% across six blocks, which makes the test weak | a deeper model (24+ blocks) where sensitivity varies by >2x |
| O3 | ~~Does any of this hold for SGD or Lion?~~ **answered (A14-A16)**: the structure is optimizer- and state-dependent, so the framing survives | measured 200x spread across optimizers | done |
| O6 | Do the sparsity/density cost curves (A5, A6) differ under Lion? | A14 predicts they must: a 5% budget that is nearly free under SGD should be far more damaging under Lion, which no paper in the sparsity literature reports | rerun E9 with the optimizer as a variable |
| O4 | ~~Can the truncated-backward saving be realised inside a training loop?~~ **answered (A19)**: yes, 1.60-2.13x at a measured +0.003 to +0.009 held-out cost | done, on a 40 M model and a byte-level corpus | remaining: scale it to a real tokenizer and a 1B model |
| O7 | Does the Lion penalty (A17) hold at scale and on a real tokenizer? | this is the claim most likely to change practice, and it rests on one 40 M byte-level run | E19 on a 1B pretrained checkpoint with a real tokenizer is implemented and running (`--pretrained --opt8bit`); results not yet in |
| O8 | Can the dropped blocks' updates be kept current cheaply, without a replay forward? | E22 shows the obvious correction is slower than dense, so the system story depends on finding a compensation that is not recomputation | carry the dropped blocks' optimizer state with an estimate, then measure quality and step time |
| O5 | Does the quality-neutral point move with model scale and task? | the 5%/20%/50% cost curve is one corpus and one architecture | repeat A6 at two model sizes and a second corpus |
