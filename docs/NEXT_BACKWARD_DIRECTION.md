# Optimizer-conditioned backward execution — independent viability review

Date: 2026-09-23 · Author: independent review pass over `/root/qcc` (not a rerun of prior work)
Scope: is "allocate backward computation by optimizer-state sensitivity" a viable MLSys main-line,
and what is the cheapest experiment that decides it.

Every claim below is tagged **[verified]** (I read the file that contains it),
**[user-reported]** (asserted, no artifact in this workspace), or **[open]** (needs a run).
No number here is a performance claim.

---

## 1. Verdict

| Question | Answer |
|---|---|
| Is the direction inside MLSys scope? | Yes. Efficient training / LLM training / hardware-efficient ML methods are core scope, and the review criteria are novelty, quality, interest, impact. |
| Is the *slogan* publishable as stated? | No. "Backward should be allocated by optimizer-state sensitivity, not gradient magnitude" is half theory and half prior art until one specific measurement exists. |
| Is the current evidence enough? | No, and for a sharper reason than "no GPU". The headline divergence is measured in a metric (**relative update MSE**) that has no demonstrated causal link to time-to-quality at the magnitudes reported. |
| Deadline reality | MLSys 2027 submissions close **Oct 30 2026** ([dates](https://mlsys.org/Conferences/2027/Dates)) — ~5.5 weeks out. With one A10G in hand (below), a *narrow, well-scoped* paper is reachable; the full compiler/runtime paper is not. Decide by ~Oct 10 or target the next cycle. |
| Net | **Keep the direction; replace the pitch.** The claim to chase is not "we approximate updates, not gradients". It is *"the compute allocation that minimizes update-space error is provably not the one that minimizes gradient-space error, and here is the runtime that exploits the difference."* |

The rest of this document is the case for that, plus the executable path.

---

## 2. What actually exists in this workspace (verified)

I checked the two numbers the pitch rests on.

| Item | Status |
|---|---|
| `0.766x` (Digits) and `1.18x` (Transformer @99%) low-rank/CV speedups | **[user-reported]** — no artifact anywhere in `/root/qcc` contains them. |
| Gradient rel-MSE 1 → 49.505 while update rel-MSE 0.00979 → 0.000101 | **[user-reported]** — not found in any `.md/.json/.csv/.log` under `qcc/`. The figure exists only in conversation. |
| DropBP-class baseline numbers | **External.** DropBP: 44% training-time reduction, 1.5x faster to target perplexity, A100 ([NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html)). This is the number to beat, not a paper to cite politely. |
| Adjacent prior art already using optimizer state | **External.** LDAdam does low-rank adaptive optimization *with error feedback that explicitly accounts for gradient **and optimizer-state** compression* ([arXiv:2410.16103](https://arxiv.org/abs/2410.16103)). The framing "optimizer state is part of the execution state" is partially occupied already. |
| Hardware | 14-core CPU, 40 GB RAM, **1x NVIDIA A10G usable** (23.55 GB, sm_86, bf16 ok) via `/dev/nvidia7`; driver 590.44.01. Host carries 8 A10G; this container is granted one. Note: access required full file-sandbox permissions — under the default `workspace-write` policy `/dev/nvidia*` is denied and CUDA reports "no device". Resident checkpoints: Llama-3.2-1B-Instruct (2.4 G, complete), Qwen2.5-3B-Instruct (6.0 G), Phi-3.5-mini (2.0 G); Llama-3.2-3B is a config-only stub. |
| Repo discipline | Good. `vera/CLAIMS.md`, `shadowstep/CLAIMS.md`, `review20260922/mlsysmain/docs/ADVERSARIAL_REVIEW.md` all already separate established / open / not-claimed and name regeneration commands. Keep that format; it is the reason the negative results here are usable at all. |

**Consequence.** The project's entire empirical case currently lives outside the repository. That is
the first thing to fix, independent of everything else: the divergence experiment must exist as a
script plus a raw result file, or it does not exist.

---

## 2b. The A10G is real and it changes the plan (measured 2026-09-23)

An earlier pass on this project recorded "no usable CUDA device". **That was wrong**, and the reason
matters operationally: under the default `workspace-write` file sandbox, `/dev/nvidia*` opens are
denied, so `cuInit` returns 304 and torch reports zero devices. With full file-sandbox permissions the
same container sees a working A10G. If a later session again reports "no GPU", check the sandbox
policy before believing it.

Measured now, in this container:

| measurement | value | how |
|---|---|---|
| GPU | 1x NVIDIA A10G, 23.55 GB, sm_86, bf16 yes, driver 590.44.01 | `nvidia-smi` |
| Host GPUs visible to the driver | 8x A10G (`0000:01,23,41,61,81,a1,c1,e1`) | `/proc/driver/nvidia/gpus` |
| Concurrent tenant on this GPU | ~9.3 GB resident (another job, not ours) → ~14.5 GB free | `nvidia-smi --query-compute-apps` |
| bf16 matmul probe | 30.7 TFLOP/s at 8192³ | `torch.matmul` loop |
| **Llama-3.2-1B full fine-tune, bf16, grad-ckpt, AdamW** | **671 ms/step, 1 526 tok/s, 11.77 GB peak** (bs 2 × seq 512) | `probe_a10_training.py`, `probe_a10_training.json` |
| Same, optimizer state in fp32 | does **not** fit alongside the tenant (AdamW states alone ≈ 9.2 GB fp32) | E1 OOM, recorded |
| Working env | `/root/qcc/venv` (torch 2.8.0+cu128, transformers 5.16.1, triton 3.4). The miniconda env's transformers 4.57.6 crashes on import via `kernels==0.16.1` vs `huggingface_hub` 0.36.2 | reproduced |

What this changes, and what it does not:

* **E1/E2 move from CPU to a real 1B checkpoint.** This is the difference between a mechanism note and
  a result. E1 is already implemented and running: `e1_sensitivity.py`.
* **E3 (time-to-quality on a real LM) becomes possible**: ~2 000 steps × 0.67 s ≈ 22 min per run at
  bs 2, so a 3-seed × 3-allocator sweep is on the order of a day, not a week.
* **The comparison to DropBP is now hardware-mismatched.** DropBP reports A100 (1.55 TB/s HBM,
  ~312 TFLOP/s bf16); the A10G is 600 GB/s and ~125 TFLOP/s bf16 — roughly 2.5x less compute and
  2.6x less bandwidth. So (a) our absolute tok/s will be ~2-3x lower, and (b) any percentage
  claim must be reproduced on our own hardware with our own DropBP baseline, not quoted from
  their paper. The clean fix is to report *relative* to a DropBP run we execute ourselves, and to
  state the hardware gap explicitly rather than hoping nobody notices.
* **The 5.5-week deadline is now tight-but-real for a narrow paper**, and still not enough for the
  compiler/runtime paper. A10G is also the *right* class of device for the argument: if the mechanism
  pays off on a 600 GB/s card, the memory-traffic story is stronger, not weaker.

---

## 3. Three corrections to the assessment you were given

The assessment is directionally right — MLSys is the right venue, the framing is the right framing,
"where is the system" is the right reviewer question. Three things are wrong or missing.

### 3.1 The divergence is measured in the wrong currency (the dangerous one)

`update rel-MSE = 1.01e-4` is *tiny*. At that level a 100x improvement is invisible: the training
trajectory has noise far larger than 1e-4 relative error in the applied step. So the reported
divergence may be a **metric artifact with zero training consequence** — the same failure mode that
killed the VERA certificate (A7: coverage 2.5% against a 95.9% actual identity rate) and the
mlsysmain cascade (complex method loses to fixed-8 control in all nine buckets).

You already know how to catch this: it is ShadowStep Gate-0 applied to a new channel. The question is
not "can update error be made smaller" but **"is update error at these magnitudes on the causal path
to loss?"** If injecting the *same* relative error into a real training run changes nothing in
time-to-target-loss, the whole axis is below the floor and no kernel will rescue it.

Note the asymmetry this creates: relative MSE is trivially gameable by concentrating error where the
optimizer divides it away. That is exactly why the metric alone cannot be the contribution.

### 3.2 The gap between the slogan and the prior art is narrower than it looks

"Optimizer state is part of the execution state of backward" is a nice abstraction, but the concrete
mechanism — allocate backward compute per layer by a sensitivity statistic — is DropBP, published,
with 1.5x to-target on A100. A reviewer will read your proposal as "DropBP with a different
sensitivity formula" unless you produce the one measurement that separates them (see §7, E4).

The separation exists and is defensible, but it has to be *shown*, not asserted:

* DropBP's sensitivity is a **magnitude** statistic (norm of the layer's gradient/activation path).
  Magnitude cannot see the preconditioner. Two layers with identical gradient norm but different
  `v` should receive different compute — that is a prediction, not a slogan.
* DropBP's schedule is **static** (computed once, dropped randomly thereafter). `v` drifts during
  training, so an optimizer-conditioned allocation should be **dynamic**. Also a prediction.

### 3.3 The theory has a latent claim that is worth more than the slogan

You can derive — in closed form, from the update rule alone — what "optimizer-state sensitivity"
*is*, and it is not gradient magnitude. Then the pitch stops being a heuristic and becomes a
falsifiable law with a specific predicted allocation. See §4. This is the piece that turns
"interesting mechanism" into "reviewers have to argue with a formula".

---

## 4. Theory: what the update actually tolerates

AdamW update per coordinate (bias correction omitted — it cancels in ratios at fixed step):

```
u_i = m_i / (sqrt(v_i) + eps)
```

Let the gradient carry error `δ` (the quantity every approximation method in the literature tries to
bound). Write `m' = m + (1-b1)·δ`, `m ≈ g·(1-b1)` in the steady state, and `sqrt(v_i) + eps =: s_i`:

```
Δu_i = (1-b1)·δ_i / s_i  -  u_i · (s'_i - s_i)/s_i
```

Two error channels with different physics:

| channel | scale | when it dominates |
|---|---|---|
| **preconditioner channel** | `(1-b1)·δ_i / s_i` | `\|g_i\| >> sqrt(v_i)`: relative error is *divided by the accumulated second moment* |
| **sign/phase channel** | `u_i · (s'_i - s_i)/s_i ≈ u_i·\|δ_i\|/\|g_i\|` | `\|g_i\| ≲ sqrt(v_i)`: a small relative error **flips the sign** of the update, costing `2·lr` in that coordinate |

Two consequences, both testable:

1. **Error in a high-`v` coordinate is nearly free; error in a low-`v` coordinate is not.** This has
   nothing to do with gradient magnitude. It is the mechanism behind the reported 49.5 vs 1.0
   divergence, and it is derived, not fitted.
2. **The optimal allocation is a water-filling law, not a magnitude law.** Minimizing
   `Σ w_i δ_i²` subject to `Σ δ_i² ≤ B²`, with
   `w_i = (1-b1)²/s_i² + (u_i/g_i)²/4`, gives at optimum

   ```
   δ*_i  ∝  w_i^(1/3)
   ```

   For a **block-uniform** relative error (the realistic implementation: one rank/sample rate per
   block), the block's error power is proportional to its gradient power, and the law becomes

   ```
   δ*_block  ∝  ( Σ_{i∈block} w_i g_i² )^(1/4)
   ```

   with the large-`g` regime collapsing to `δ*_block ∝ ( Σ g_i²/ v_i )^(1/4)`.
   Note what is *absent*: the block's gradient norm does not appear as the allocator. Two blocks with
   equal `Σg²` and different `Σg²/v` get different compute.

**Honest caveats.** This is a first-order expansion, not a theorem about convergence. `m ≈ g(1-b1)`
is a steady-state assumption. The `1/4` law assumes error power ∝ gradient power per block, which
holds for sampling/low-rank/skip families and not universally. All of it is falsifiable in an
afternoon on CPU (§7, E1).

If the law holds, the paper has a theory section that predicts an allocation no gradient-based
scheduler would produce. If it fails, the slogan has no mechanism and the project should stop there.
Either outcome is cheap. That is the whole argument for running E1 before anything else.

**→ It was run. The law failed, in a specific and useful way. See §4b.**

---

## 4b. What the measurements actually say (run on the A10G, 2026-09-23)

Scripts: `probe_a10_training.py`, `e1_sensitivity.py`, `e1b_update_vs_grad.py`,
`e1c_horizon.py`, `e2_error_geometry.py`, `e3_statistic.py` (+ raw JSON beside each).

Protocol: Llama-3.2-1B-Instruct, bf16, gradient checkpointing, AdamW (0.9/0.95, eps 1e-8),
bs 4 x seq 512 of concatenated LongBench text, 60 steps; 112 matrix blocks (16 layers x 7
projections). For each block, the *same* approximation budget is spent on gradient error `δ`,
and both gradient-space and update-space error are scored:

```
grad_mse = ||δ||²/||g||²          upd_mse = ||U(g+δ,m,v) − U(g,m,v)||²/||U(g,m,v)||²
A_b      = upd_mse / grad_mse     ("amplification": how much the optimizer amplifies the error)
```

### The headline is not the one the pitch assumed

| error model | what it represents | A_b median | A_b p90 | A_b max |
|---|---|---:|---:|---:|
| proportional (`δ ∝ g`) | magnitude-preserving: quantization, low-rank+scaling | **0.13** | 0.17 | 0.22 |
| rank truncation | rank-`r` backward, token/channel sampling | **1.07** | 9.38 | 48.6 |
| white (orthogonal) | random projection, dropout-style noise | **2.79** | 833 | **1.09e5** |

**1. The error geometry dominates the error size.** At an identical relative gradient error of 0.25,
the update-space consequences span five orders of magnitude depending only on the *shape* of the
error. A method that reports `gradient error = 0.25` has said almost nothing about the update:
it is a 0.13× or a 2.8× or a 1e5× event. This is the real, defensible version of "gradient fidelity
is the wrong objective", and it does not need the optimizer to be mysterious — it is a statement
about `(I − ggᵀ/||g||²)` and the low-`g` tail.

**2. The derived allocator statistic loses to plain gradient norm.** Spearman ρ(statistic,
amplification) over 112 blocks at c = 0.25:

| statistic | white | prop | orth | rank |
|---|---:|---:|---:|---:|
| `‖g‖` (gradient norm, the DropBP-class statistic) | **0.630** | **0.581** | **0.630** | **0.603** |
| `Σ g²/v` (§4's derived weight) | 0.409 | 0.388 | 0.409 | 0.379 |
| `Σ g²/v²` (tail-weighted variant) | 0.498 | 0.396 | 0.498 | 0.457 |
| `q_p001 = 0.1%-quantile of |g|/√v` | 0.511 | 0.486 | 0.511 | 0.458 |
| mean `|g|/√v` | 0.473 | 0.429 | 0.473 | 0.387 |

Gradient norm wins on every model, including the proportional case where the theory should have
been strongest. **The §4 prediction is falsified at the block level on this setup.** Reporting the
quantiles did not rescue it either: no statistic built from `v` beat `‖g‖`.

**3. A second falsification, earlier in the pipeline.** `e1_sensitivity.py` ranked blocks by
`v`-weighted sensitivity and the top of the list was populated by LayerNorm parameters — 4–32 K
parameters with microscopic second moments. Any allocator built on `Σ g²/v` needs an explicit
floor on `v` or it will spend its budget on RMSNorm weights. That is a design bug, not a theory
result, but it is the kind of bug that silently produces a wrong paper.

**4. Regime check.** `mean |g|/√v ≈ 0.7–5` in these runs, i.e. the preconditioner channel
(`(1−b1)δ/√v`) and the sign-flip channel are both live — this is not a degenerate corner. And
`e1b_update_vs_grad.py` confirms the injected error is linear in the budget (`grad_mse = c²` exactly,
`upd_mse = A·c²`), so the measurement is well-posed rather than a large-perturbation artifact.

### What survives, and it is a better thesis than the original

The survivor is the sentence with the measurements behind it:

> **For a fixed approximation budget, the *geometry* of the backward error — not its norm, and not
> any sensitivity weight computed from optimizer momenta — determines whether the optimizer absorbs
> it. Direction-preserving error is absorbed (A≈0.13); orthogonal error is amplified (A⩾2.8, up to
> 1e5). Therefore a backward runtime should be designed around the error geometry it produces, and
> gradient-error metrics cannot rank approximation schemes.**

This reframes the contribution in a way that is *harder* to attack than the original pitch:

* It is not "our allocator beats DropBP" (it does not, on this evidence).
* It is "the objective everyone optimizes is not the objective that matters, and here is a
  quantitative theory plus a measurement of what does" — with `A_b` as the measurable bridge.
* It is directly MLSys-shaped: the lever is the *form* of the approximation (project along the
  gradient, preserve sign structure, correct the low-`g` tail), which is a kernel/schedule decision,
  not a hyperparameter.

The obvious reviewer question — "so does any of this change time-to-quality?" — is still open and is
now the *only* question that matters. That is E4.

### Cheap predictions this makes (all falsifiable, all CPU or one GPU-hour)

| # | prediction | falsifier |
|---|---|---|
| P1 | SGD/momentum has A ≈ 1 for proportional error and no low-`g` blowup: the amplification is an Adam-specific effect | run E2 with SGD and Lion; if A is unchanged, the optimizer is not the causal variable and the framing collapses to numerical linear algebra |
| P2 | Scaling the gradient by a constant (`δ = εg`, most quantization) is nearly free in update space; the same budget spent as rank truncation costs ~8x more update error | E2 curves already hold this for one checkpoint; P2 asserts it at every checkpoint and step size |
| P3 | Injecting `δ = εg` with ε up to some threshold leaves time-to-quality unchanged, while the same budget of orthogonal noise degrades it | that is E4, and it is the experiment that decides whether the paper exists |
| P4 | The right ranking statistic is `\|g\|/sqrt(v)` — coordinates whose *current* gradient is large relative to their own history | **falsified end-to-end in E7** (see §4d). `v` is an EMA of `g²`, so `\|g\|/sqrt(v)` measures the coordinate's surprise, not the size of the update. The corrected form is `\|m\|/sqrt(v)` |

---

## 4c. SOTA survey across the dimensions we could beat (checked 2026-09-23)

Target: find a dimension where the update-space/geometry result can *exceed* the published state of
the art, not just explain it. Read: abstracts + stated numbers; full papers where the claim hinges on
their objective function. Numbers below are author-reported unless marked measured.

| dimension | current SOTA | its objective | our measurement / opening |
|---|---|---|---|
| **Backward sparsification** | top-k magnitude with error feedback; DropBP drops layers by a *magnitude* sensitivity (44% time cut, 1.5x to target, A100, [NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html)) | keep the largest `\|g_i\|` | measured A_b spans 0.13–1.1e5 by error geometry; error in a high-`v` coordinate is divided away, error in a low-`g` coordinate flips a full-size step. **Prediction: rank by `\|g_i\|/sqrt(v_i)`, not `\|g_i\|`.** Tested end-to-end in E7 |
| **Low-rank backward / optimizer** | GaLore, Q-GaLore, SLowMo, SubTrack++ ([NeurIPS 2025](https://neurips.cc/media/neurips-2025/Slides/119775.pdf)) | keep the top-r subspace by gradient energy, `min \|\|g - P_r g\|\|_F`; rank uniform across layers | measured: update-space basis is **worse in gradient space and 3.9x better in update space** (synthetic control). On the real model, E6 |
| **Rank allocation** | LAARA ([arXiv:2607.19391](https://browse-export.arxiv.org/pdf/2607.19391)), IGU-LoRA ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html)), Rank Allocation in Low-Rank Optimizers ([ICML 2026](https://icml.cc/virtual/2026/72403)) | allocate rank by layer importance / gradients / uncertainty | all allocate by a *gradient-space* criterion. Per-layer `v`-anisotropy in our runs spans 1e8–1e14, so the same nominal rank buys very different update-space accuracy per layer |
| **Optimizer-state quantization** | 4-bit states ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/3122aaa22b2fe83f9cead1a696f65ceb-Paper-Conference.pdf)); Adaptive Log-Space: 72.90 vs 73.54 ppl for bitsandbytes-8bit on TinyLlama-1.1B ([arXiv:2608.22322](https://arxiv.org/abs/2608.22322), Aug 2026) | minimise *reconstruction* error, then check update error empirically | this dimension is **closing**: 2608.22322 already names "similar reconstruction error can produce different update error" and makes update semantics a design constraint. Do not lead here |
| **Subspace stability** | [No Subspace to Track](https://www.alphaxiv.org/abs/2607.05872) (Jul 2026): only ~39/128 directions reproducible; carrying `v` blindly costs ~(r−k\*)/2; LDAdam's transported moment reaches 18.7 vs 19.3 ppl at 1B | identify that the tracked object is noise | **This is our best ally and our biggest collision.** They show the subspace is noise-dominated; we show *which objective should be used given that it is*. Their fix is transporting the first moment on refresh; ours is choosing the subspace by update-space error |
| **Optimizer-state transport on basis change** | LDAdam ([ICLR 2025](https://arxiv.org/abs/2410.16103)): projection-aware update rule + error feedback accounting for gradient *and* state compression | correctness of the state under rotation | Mechanism is theirs; our E5 attempted the naive version and failed (carrying `v` through a non-orthogonal basis is not a projection) — consistent with their result |

### What is actually open, ranked by (impact x feasibility on one A10G)

1. **The criterion inside sparsification/low-rank selection is the wrong statistic** (§4b + E7).
   Nobody selects by update-space impact. It is one line of code, needs no new kernel, and if it wins
   at 5% density it is a result every compression paper has to answer.
2. **A measurable certificate that ranks approximation schemes before training** (`A_b` per layer).
   The field currently compares schemes by gradient MSE or end loss; `A_b` is a per-layer,
   per-checkpoint number that predicted a 1e5x spread here. This is the MLSys-shaped artifact.
3. **Per-layer rank/bit allocation by update-space marginal gain** instead of uniform or
   gradient-energy allocation. Collides with LAARA/IGU-LoRA/ICML'26 rank-allocation work, so it needs
   the `v`-weighted objective to be the differentiator — not just "adaptive rank".
4. ~~Optimizer-state quantization~~: too crowded, and the Aug-2026 paper already owns the framing.
5. ~~Subspace tracking / non-identifiability~~: owned by 2607.05872. Cite it, build on it, do not
   compete with it.

### The one-sentence claim the evidence now supports

> **Gradient-space error is not a valid currency for backward approximation: at a fixed budget the
> update-space cost of an error spans five orders of magnitude depending on where the error sits, so
> schemes that select work by gradient magnitude (top-k sparsification, energy-based low rank, static
> layer sensitivity) are optimising a quantity the optimizer does not act on. Selecting by
> `|g|/sqrt(v)` — keeping the *small* gradients in high-moment coordinates, not the large ones —
> is the corrected criterion.**

The second clause is deliberately counterintuitive and is exactly what E7 tests: at 5% density the
SOTA criterion keeps the largest-magnitude coordinates, while the corrected criterion keeps many
coordinates whose gradients are *small* because Adam's `1/sqrt(v)` makes each of them worth a full
step. If E7 confirms it, the headline is "your sparsifier is keeping the wrong 5%".

---

## 4d. E7: the sparse criterion, measured end-to-end (negative result + correction)

First end-to-end run: Llama-3.2-1B, 400 steps, bs 4 x seq 512, lr 1e-4, keep 5% of coordinates,
identical data order and seed. `e7_sparse_criterion.py`, `e7_sparse_criterion_main.json`.

| arm | mean loss | loss at step 399 |
|---|---:|---:|
| full (no approximation) | 7.87 | **7.61** |
| topk by `\|g\|` (SOTA) | 5.70 | 6.14 |
| topk by `\|g\|/sqrt(v)` (my proposal) | 6.60 | 7.08 |

Two findings, one of which invalidates the first experiment's framing:

**1. The setup was confounded, and the confound is itself informative.** Both 5%-sparse arms beat
full fine-tuning by ~1.5 nats. At lr = 1e-4 with no warmup, *not computing 95% of the backward acts
as a stabiliser*: the sparse arms take an effectively 20x smaller step. So this run cannot rank the
criteria — but it does say something about the low end of compute allocation: for a 1B model on
messy concatenated text, an aggressive backward approximation is not merely survivable, it is
*helpful*. That has to be priced properly (matched effective step size) before any speed claim.

**2. The `\|g\|/sqrt(v)` criterion lost to plain `\|g\|` at every one of the 40 logged checkpoints.**
The hypothesis in §4c row 1 is dead as stated, and the diagnosis matters more than the loss:

> `v` is an exponential moving average of `g²`. Therefore `|g_i|/sqrt(v_i)` is the coordinate's
> **surprise** — "this gradient is large compared to its own recent history" — not the size of the
> update Adam will take. Adam's actual step is `m_i/(sqrt(v_i)+eps)`, i.e. *accumulated evidence in
> Adam's frame*. A coordinate can have a huge instantaneous gradient and a tiny update (its history
> is equally large), or a tiny gradient and a full-size update (its history is smaller still).

**The corrected statistic is `|m_i| / sqrt(v_i)`** — rank coordinates by the update the optimizer
will actually apply, not by the gradient, and not by the gradient's novelty. This is now the only
version of the claim that is mechanistically coherent, and it is still counterintuitive in the same
way: it says the 5% of coordinates a sparsifier should compute are chosen by Adam's accumulated
state, not by the magnitude of the gradient in front of it. E7b tests it at lr 3e-5 (confound
removed) against `|g|`, `|g|/sqrt(v)`, and full.

### What this changes in the paper's story

* Drop "keep the small gradients" — that was the wrong reading of the sensitivity analysis.
* Keep "the criterion is the wrong statistic", now with a sharper target: **the field ranks
  coordinates by `|g|`; Adam ranks them by `|m|/sqrt(v)`; these are different random variables,
  and the second is the one the optimizer acts on.**
* The measured amplification results (§4b) stand unchanged — they say gradient-space *error metrics*
  cannot rank schemes, which is independent of which *selection statistic* is used.
* The hook overhead is real and measured: 1.0 s/step sparse vs 0.64 s/step dense at the same batch.
  A 5% top-k per tensor costs more than it saves at this scale unless the selection is done
  block-wise on raw (non-float-cast) tensors. Any speed claim must fix that first.

---

## 4e. E9: the criterion, measured on an uncontended device — proxy metric refuted

The A10G is shared with another tenant whose footprint swings between 5 and 16 GB, which killed two
1B-scale runs mid-flight. The criterion question is coordinate-level and does not need 1B parameters,
so it was re-run on a compact transformer (6 layers, d=384, 40 M params) on the GPU with a small
footprint — reliable, ~35 s per arm, ~1 GB. Corpus: 198 M bytes of real LongBench text, 700 steps,
5% of coordinates, identical data order and seed. `e9_compact_criterion.py`.

**Ranked by area under the loss curve (lower is better), and the first column is the proxy metric
that was supposed to predict the ranking:**

| arm | update-mass captured (proxy) | AUC | final loss |
|---|---:|---:|---:|
| full (no approximation) | 100% | **2.648** | 2.540 |
| topk by `\|g\|` (SOTA) | 4.9% | 2.916 | 2.735 |
| topk by `\|g\|/sqrt(v)` | 6.5% | 2.926 | 2.720 |
| topk by `\|m\|/sqrt(v)` (my corrected criterion) | **28.8%** | 3.035 | 2.846 |
| uniform random | 5.0% | 3.087 | 2.744 |

**The proxy metric is refuted, and it is refuted exactly the way this document predicts other
people's proxy metrics are refuted.** `|m|/sqrt(v)` captures 5.8x more of the update mass than the
SOTA criterion and trains *worse*. Meanwhile uniform random selection — which captures no structure
at all — is within 0.06 AUC of both. Three conclusions, in decreasing order of confidence:

1. **Selection-criterion choice is a second-order effect at 5% density.** The spread across the four
   sparse arms is 0.17 AUC; the spread between the best sparse arm and full computation is 0.27 AUC.
   Half the total effect belongs to something other than the criterion, and the ordering among
   criteria is not stable enough to build a paper on.
2. **The magnitude criterion does carry real information**: `topk_g` beats random (2.916 vs 3.087).
   So "the criterion is wrong" is true in the weak sense (a better criterion may exist) but false in
   the strong sense (the SOTA one is not replacing signal with noise).
3. **5% sparsification of the backward is not free at this scale**: every sparse arm loses to dense
   by 0.27–0.44 AUC. Any claim that sparse backward is quality-neutral needs to say at what density
   it becomes neutral, which is the next measurement.

**Why the update-mass metric failed — and this is the sharpest version of the paper's thesis.**
Capturing more of the *initial* update mass is not the same as making more progress, because the
coordinates that carry large `|m|/sqrt(v)` at step t are precisely the ones the optimizer is already
moving fastest; concentrating 5% of the budget on them starves the rest of the network. A metric
computed from a single step's state (`m`, `v`, `g`) cannot see that. It is the same failure mode as
`gradient MSE`, one level up: **a local, single-step proxy for approximation quality does not
predict optimisation outcome, and ranking methods by it is how the field got its current selection
rules.** That is now supported by three independent measurements in this document: gradient MSE
(§4b), update-mass capture (§4e), and end-to-end AUC (§4e).

### Validation chain (complete) — and the mechanism it found

All on the compact transformer, 198 M bytes of real text, one seed unless noted. E9 chain + E10:
`e9_*.json`, `e10_step_norm.json`.

**Density is the variable that matters; the criterion is not.**

| density | full | `\|g\|` | `\|g\|/sqrt(v)` | `\|m\|/sqrt(v)` | random |
|---:|---:|---:|---:|---:|---:|
| 5% | **2.648** | 2.916 | 2.926 | 3.035 | 3.087 |
| 5% (seed 1) | **2.648** | 2.913 | — | 3.020 | 3.065 |
| 20% | **2.648** | 2.753 | — | 2.778 | 2.785 |
| 50% | **2.648** | — | — | 2.696 | — |
| 5%, lr x5 | 1.751 (memorises) | 2.744 | — | 2.791 | — |

Three robust facts:
1. The quality cost is a smooth function of the kept fraction (gap to dense: 0.27 -> 0.11 -> 0.05 AUC
   at 5% / 20% / 50%), **not** of which coordinates are kept.
2. At 5%, all four criteria sit within 0.17 AUC of each other and `rand` is within 0.06 of the best
   structured criterion. The criterion is a second-order effect.
3. `|m|/sqrt(v)` — the criterion that captures 5.8x more update mass — is the **worst** arm at every
   density tested. The proxy metric does not merely fail to predict the ranking; it inverts it.

**E10 located the actual mechanism: it is a step-size effect, not an information effect.**

| arm at 5% | median `\|\|dtheta\|\|` | ratio to dense | lr rescaled | AUC before | AUC after |
|---|---:|---:|---:|---:|---:|
| full | 872.4 | 1.00 | — | 2.721 | — |
| `\|g\|` | 315.8 | 0.362 | x2.76 | 3.055 | **2.870** (55% of gap closed) |
| `\|m\|/sqrt(v)` | 281.8 | 0.323 | x3.10 | 3.182 | **2.943** (52% of gap closed) |
| random | 420.7 | 0.482 | — | 3.336 | — |

Masking 95% of the backward silently divides the optimizer's step norm by ~3. Half of the entire
quality cost of 5% sparsification is that nobody recalibrated the step size. Note the caveat: the
rescaled arms did **not** restore the update norm (315 -> 270), because the learning rate feeds back
into the moments; so this is calibration, not a fixed constant, and a runtime would have to track it.

### E11: the residual is not recoverable by calibration — the complete picture

Same setup; `e11_step_controller.py`, `e11_step_controller.json`.

| arm | AUC | final loss | median `||dtheta||` | lr scale | gap vs dense closed |
|---|---:|---:|---:|---:|---:|
| full (dense) | **2.721** | 2.598 | 872.4 | 1.0 | — |
| `|g|` @5%, nominal lr (SOTA) | 3.055 | 2.772 | 315.8 | 1.0 | 0% |
| `|g|` @5%, fixed lr x2.76 | 2.870 | 2.676 | 269.0 | 2.76 | **55%** |
| `|g|` @5%, online norm controller | 2.879 | 2.681 | 210.3 | saturates at 20.0 | **53%** |
| `|g|` @5%, controller + moments frozen outside the mask | 3.001 | 2.705 | 726.0 | 1.0 | 16% |

Three conclusions, all negative for the "smarter sparsifier" programme:

1. **The online controller saturates and fails.** Its lr scale ran to the clip (20x) while the update
   norm stayed at 210 against an 872 target. Holding the step norm by a scalar is not possible here:
   raising the learning rate changes the trajectory, which changes the moments, which changes the
   norm. A single scalar cannot servo a quantity that feeds back into itself.
2. **Freezing the moments outside the mask is worse, not better** (16% of the gap closed vs 55%):
   `||dtheta||` came back up to 726, close to dense, and the AUC got *worse*. Restoring the norm of
   the update does not restore the update. The masked update is not a scaled dense update — it is a
   different update, and the difference is not a calibration constant.
3. **~45% of the quality cost of 5% sparsification survives every correction tried here.** That part
   is a genuine information/optimisation cost and is the number any speedup claim has to pay honestly.

### The thesis, replaced (bounded by measurement)

The old thesis ("allocate by optimizer-state sensitivity") is dead. What the measurements support:

> **The sparsity literature optimises the wrong variable. At 5% density the choice of selection
> criterion — magnitude, v-weighted, momentum-weighted, or uniform random — moves quality by less
> than 0.17 AUC, while the kept fraction moves it by 0.27 AUC, and the optimizer's step-norm collapse
> accounts for about half of that. A 3x step-norm mismatch is invisible to every gradient-space
> metric the field reports, and no scalar recalibration recovers the rest: at 5% the masked update is
> a structurally different update, not a smaller one.**

The MLSys-relevant artifact is not a new sparsifier. It is a **pre-training diagnostic**: measure
`||dtheta_sparse|| / ||dtheta_dense||` for a proposed backward-approximation scheme and you have most
of its quality cost before running it to convergence. Every scheme in the SOTA table of §4c can be
scored this way in minutes on one GPU.

### What is still open (next round)

| # | experiment | what it settles |
|---|---|---|
| E12 | per-layer / per-module allocation measured by update-norm contribution at a fixed global budget | the *allocation* axis is the one place left where an optimizer-aware scheduler could win, and the criterion result does not touch it |
| E13 | scale check at 1B inside a memory-gated window, DropBP re-implemented on our hardware as the baseline | whether the effect survives scale — every reviewer will ask |
| E14 | the diagnostic table: `||dtheta||` ratio for top-k, low-rank, layer-drop, random at matched budgets | the artifact the community can use |
| E15 | real speedup: block-wise top-k on raw tensors, fused | E7 measured 1.0 s/step sparse vs 0.64 s/step dense, so today there is **no** wall-clock win to claim |


---

## 4f. E12/E15: the allocation axis, and the first real speedup numbers

Two experiments, both on the compact transformer. `e12_allocation.py`; artifacts
`e12_allocation.json`, `e12_layers_only.json`.

### E15 — honest wall-clock for a skipped backward

Truncated backward (full forward, then `autograd.grad` from a boundary activation through the last
k blocks), same model/batch, CUDA events, 5 reps:

| kept | ms | speedup |
|---|---:|---:|
| full backward | 28.29 | 1.00x |
| last 3 of 6 | 17.39 | **1.63x** |
| last 2 of 6 | 15.43 | **1.83x** |
| last 1 of 6 | 11.49 | **2.46x** |

This is the number that was missing: skipping backward work does buy wall-clock, and the earlier
sparse arms showed none *because masking after a full backward saves optimizer work and zero FLOPs*.
The speed lever exists; it just lives in the graph, not in the mask.

### E12 — which layers get the budget, at a fixed budget of 3 of 6

| policy | allocation | AUC | cost vs dense |
|---|---|---:|---:|
| dense (no drop) | all | 2.696 | — |
| by gradient norm (DropBP-style) | `[-1, 0, 1]` | 2.750 | **+0.054** |
| by update-norm contribution | `[0, 1, 2]` | 3.223 | +0.527 |
| uniform random | `[3, 0, 2]` | 3.224 | +0.528 |

Update-norm allocation is **indistinguishable from random** (3.2230 vs 3.2242) and 10x worse than
gradient norm. The reason is in the per-group shares:

| group | share of `\|\|dtheta\|\|` | share of gradient norm |
|---|---:|---:|
| embeddings/head (`-1`) | **3.2%** | **48.3%** |
| blocks 0..5 (each) | 14.1–17.8% | 7.3–12.0% |

Update norm is nearly uniform across blocks — that is exactly what Adam's `1/sqrt(v)` normalisation
does — so ranking by it carries almost no information and is no better than a coin flip.

### The follow-up that explains everything

Re-run with the embedding group **always** included (budget: 3 of the 6 blocks, plus `-1`):

| policy | allocation | AUC | cost vs dense |
|---|---|---:|---:|
| dense | all | 2.714 | — |
| by update norm | `[-1, 0, 1, 2]` | 2.753 | +0.039 |
| by gradient norm | `[-1, 0, 1, 2]` | 2.753 | +0.039 |
| uniform random | `[-1, 1, 3, 4]` | 2.754 | +0.041 |

**Once the embeddings are always updated, every policy works and the policies become identical.**
The entire allocation effect was the embedding block: it carries 48.3% of the gradient mass in 3.2%
of the update mass, and freezing it is what destroyed the update-norm policy. With it pinned, which
blocks you drop is nearly irrelevant at this budget (spread 0.001 AUC).

### What this settles

Combining §4e and §4f, the complete causal picture at this scale:

| lever | effect size | who optimises it |
|---|---:|---|
| keep the per-coordinate criterion | 0.17 AUC | **the entire sparsity literature** |
| keep the optimizer's step norm consistent | 0.19 AUC (half the density cost) | nobody |
| always update the embedding/head block | **0.49 AUC** | nobody |
| truncate the backward graph | 1.6–2.5x wall-clock | DropBP, SLowMo and friends |

> **The sparsity literature is optimising the smallest of the four terms.** Choosing coordinates
> better buys less than keeping the embedding block, less than fixing the step norm, and nothing at
> all in wall-clock. Meanwhile the one lever that is worth 2.5x — actually removing backward work
> from the graph — makes the selection criterion irrelevant by construction.

That is a genuinely counterintuitive, quantified, and immediately actionable statement about the
field's priorities, and every number in it was measured here.

## 4g. E14: the diagnostic table, and what it falsifies about itself

Six mechanisms at matched nominal budget (5% of coordinates, or rank 32, or 3 of 6 blocks), same
model/data/steps/seed. `e14_diagnostic_table.py`, `e14_diagnostic_table.json`. `ratio` is
`median ||dtheta_mech|| / ||dtheta_dense||` over the run.

| mechanism | ratio | AUC | cost | wall-clock | what it is |
|---|---:|---:|---:|---:|---|
| dense | 1.000 | 2.7206 | — | — | reference |
| `lowrank_row` | 0.999 | 2.7261 | **+0.006** | 1.00x | rank-32 row-space projection in the graph |
| `sparse_block` | 0.747 | 2.7531 | +0.033 | **1.55x** | DropBP-style: 3 of 6 blocks, embeddings kept |
| `lowrank_down` | **1.131** | 2.8603 | +0.140 | 1.00x | GaLore-style fixed rank-32 down-projection |
| `sparse_elem` | 0.509 | 2.8975 | +0.177 | 1.00x | top-5% per-tensor mask |
| `rand_elem` | 0.645 | 3.0518 | +0.331 | 1.00x | uniform random 5% mask |

**The simple form of the diagnostic is falsified by its own table.** `ratio` does not order the
quality cost: `rand_elem` (0.645) loses three times what `sparse_block` (0.747) loses, and
`lowrank_down` (1.131) loses 25x what `lowrank_row` (0.999) loses. A scalar cannot rank mechanisms.

What survives, and it is a cleaner statement:

> **Quality-neutrality requires the step-norm ratio to stay at ~1. A mechanism is free (cost <= 0.006
> AUC) if it leaves the optimizer's step norm essentially unchanged — whether its error is orthogonal
> (`lowrank_row`) or directionally aligned. Every mechanism that moves the ratio away from 1 pays:
> down to 0.75 costs 0.033, down to 0.51 costs 0.177, up to 1.13 costs 0.140. But staying near 1 is
> necessary and not sufficient: `rand_elem` is the worst arm at a *better* ratio than `sparse_elem`,
> because its error is at least as large and points the wrong way.**

So the correct diagnostic is two numbers, not one: the step-norm ratio **and** the update-space error
(§4b's `A_b`-weighted residual). Ratio near 1 plus small update-space error is free; either alone is
not.

### What the objective asked for, and where it now stands

The four levers, with every effect size measured in this document at one scale:

| lever | measured effect | who works on it |
|---|---:|---|
| per-coordinate selection criterion | 0.17 AUC spread across 4 criteria, 2 seeds | **the sparsity / importance-sampling literature** |
| optimizer step-norm consistency | 0.19 AUC (half the cost of 5% density, closed by rescaling) | essentially nobody |
| always updating the embedding/head block | **0.49 AUC** | nobody |
| actually removing backward work from the graph | **1.55–2.39x wall-clock** | DropBP, SLowMo and friends |

> **The most crowded lever is the smallest one.** Choosing coordinates cleverly is worth less than
> half of what keeping the embedding block updated is worth, and it buys zero wall-clock by itself,
> while the one lever worth 1.5–2.4x — deleting backward work from the graph — makes the per-
> coordinate criterion irrelevant by construction (`sparse_block` is the *best* mechanism on the
> table at ratio 0.75, and it uses no criterion at all).

And a concrete quality-neutral, speed-positive point exists and is measured: **`sparse_block` at
+0.033 AUC for 1.55x wall-clock**, with `lowrank_row` at +0.006 AUC for free whenever its projection
cost is hidden (it is not, yet, at 1.00x — that is a kernel problem, not a science problem).

## 5. The systems bill (why 1.55–2x is not a plan, it is a hope)

Backward is roughly **two thirds of step compute**. Therefore:

| backward work removed | step-time reduction |
|---:|---:|
| 20% | 13% |
| 33% | 22% |
| 50% | 33% |
| 75% | **50%** (the absolute ceiling: forward is untouched) |

So `1.5–2x` time-to-quality must come from *both* substantially cheaper backward **and** fewer steps
to a quality target. DropBP already reports the "fewer steps" half (1.5x to target) *at equal-ish
accuracy* using a static magnitude heuristic. Beating that needs the dynamic allocation to be worth
something on top. Budget conservatively: **1.15–1.35x end-to-end is the honest target**, and it is
still a publishable MLSys result if the mechanism is new and the baselines are real.

Costs that must appear in the paper and currently do not exist anywhere in this repo:

* `v`-reduction, weighting, and scheduling per step (must be O(#blocks), not O(#params)).
* Conditional-computation overhead + the memory traffic of a second code path.
* Optimizer-state read traffic (which is why the plan should live next to the optimizer, not inside
  the autograd engine — a `torch.compile(backward_policy=...)` API is the *aspiration*, and it is a
  multi-month systems project, not a section of this paper).
* Heterogeneous-kernel underutilization at small per-block savings. A 25% saving on a small layer is
  often 0% wall-clock on a GPU.

Measured on the A10G, the units are now concrete: 671 ms/step for Llama-3.2-1B at bs 2 x seq 512 with
gradient checkpointing. A drop-one-layer ablation on that configuration is the first honest cost
measurement and takes minutes to run — do it before writing any speedup claim.

---

## 6. Novelty ledger — what the contribution can and cannot be

| Candidate contribution | Status | Why |
|---|---|---|
| "Update error can diverge from gradient error" | **Measured, and sharper than expected** (§4b: A_b spans 0.13 → 1.1e5 by error geometry) | Not the headline by itself, but now the strongest measured fact in the project. |
| "Allocate backward compute per layer by sensitivity" | **Occupied** by DropBP ([NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html)) | Must be beaten, not cited. Not yet beaten. |
| "The optimal allocation has a closed form and it is not gradient magnitude" | **Falsified on this setup** (§4b: `‖g‖` ρ≈0.60 vs derived `Σg²/v` ρ≈0.39) | Do not put this in a paper. |
| "The *geometry* of the backward error, not its norm, decides whether the optimizer absorbs it" | **Open — this is now the paper** | Measured effect is large and monotone across three error families; the causal link to time-to-quality is E4. |
| "Optimizer-conditioned backward execution plan" (rank / sample rate / exact set / refresh, chosen by update-space sensitivity) | **Weakened** | Only defensible if some `v`-derived statistic beats `‖g‖`; the current answer is no. Keep the door open with P1 (other optimizers). |
| "Runtime/compiler prototype with an automatic policy" | **Out of reach this cycle** | 5.5 weeks, one shared A10G. Future work, not a pitch. |

---

## 7. The decisive experiment set — status after the first GPU pass

Stop at the first failure; each is a kill test in the ShadowStep style. Items marked **done** have a
script and raw JSON in this directory.

**E1 — Does an optimizer-state statistic predict error tolerance better than magnitude? — DONE, NO.**
`e1_sensitivity.py`, `e1c_horizon.py`, `e3_statistic.py`. Over 112 blocks and four error geometries,
`‖g‖` predicts amplification better than `Σg²/v`, `Σg²/v²`, or any quantile of `|g|/√v`. Side
finding: the `v`-weighted ranking is dominated by LayerNorm parameters unless `v` is floored.
**Consequence:** the allocation thesis as stated is dead; the geometry thesis replaces it.

**E2 — Does the error geometry change the update outcome as much as the budget? — DONE, YES.**
`e2_error_geometry.py`. Same budget, same blocks, three orders of magnitude difference in update-space
error (0.13 / 1.07 / 2.79 median amplification; up to 1.1e5 for white noise).

**E3 — Does any of this change time-to-quality? — NOT RUN, and it is the only thing that matters now.**
Train Llama-3.2-1B on a fixed corpus, 3 seeds, paired: (a) no gradient error, (b) proportional error
at c = 0.25, (c) rank-truncation error at matched `grad_mse`, (d) white noise at matched `grad_mse`.
Measure steps-to-target-loss and wall-clock-to-target-loss. ~25 min per run at the measured
671 ms/step, so a 12-run sweep is ~5 h of A10G. **Falsifier for the whole project:** if (b), (c), (d)
are indistinguishable, error geometry does not matter at reachable budgets and there is no paper —
report the tolerance threshold instead. If they separate as `A_b` predicts, the paper is real and the
theory section is already written.

**E4 — Optimizer ablations: is the amplification Adam-specific? — NOT RUN, cheap, CPU or GPU.**
Same measurement with SGD+momentum and Lion. Prediction P1: proportional error stays ~1 for SGD
(no `1/√v`), and the low-`g` blowup disappears. If SGD shows the same structure, the mechanism is
numerical linear algebra, not optimizer state, and the "optimizer-conditioned" framing must go.

Then, and only then, the systems half: a real mechanism (layer skip / blockwise low-rank) under
matched measured wall-clock, DropBP re-run on our own hardware as the baseline, and kernel-level
timing. Nothing in that half should be started before E3 reports.
features only, (c) both. Report the **incremental** R². **Falsifier:** if (a) ≈ (c), then the honest
framing is "gradient-norm sensitivity, done dynamically and better calibrated than DropBP" — a real
---

## 8. Recommendation (after the full first pass)

1. **Stop pursuing "allocate backward compute by optimizer-state sensitivity."** Falsified twice:
   the derived `Sigma g^2/v` statistic loses to plain gradient norm at predicting fragility (E1/E3),
   and its per-coordinate form ranks worst of four criteria end-to-end (E7/E9).
2. **Stop pursuing better selection criteria as the headline.** At 5% density, magnitude,
   v-weighted, momentum-weighted, and uniform-random selection sit within 0.17 AUC of each other,
   while the kept fraction moves quality by 0.27 AUC. The field's main lever is second-order.
3. **Lead with the diagnostic.** `||dtheta_approx|| / ||dtheta_dense||` is measurable in minutes,
   is invisible to every gradient-space metric currently reported, and here predicted roughly half
   the quality cost of 5% sparsification. A table of this ratio for top-k / low-rank / layer-drop /
   random at matched budgets is a publishable, immediately useful artifact (E14).
4. **Then test the one remaining allocation axis (E12).** Per-layer budget allocation by update-norm
   contribution is genuinely untested here and is where a runtime could still win.
5. **Do not write a speedup claim until E15.** The sparse arms ran at 1.0 s/step against 0.64 s/step
   for dense because of the selection hook. There is no wall-clock win today, at any density.
6. **Keep the discipline.** This document records five falsifications of its own author's hypotheses
   (V6, V9, P4, V13, and the E11 controller). That is the asset; a paper built on the surviving
   measurements will be defensible, and one built on the original slogan would not survive review.

## 9. Claim ledger for this document

| # | Claim | Status | Where it comes from |
|---|---|---|---|
| V1 | MLSys 2027 closes Oct 30 2026 | **verified** | mlsys.org dates page |
| V2 | An A10G is usable (and why it looked absent) | **verified** | `/dev/nvidia*` denied under `workspace-write`; `cuInit`=304. Under full access: 23.55 GB A10G, 1 526 tok/s on Llama-1B |
| V3 | The 49.505 / 1.01e-4 divergence has no artifact in-repo | **verified** | exhaustive search under `/root/qcc` |
| V4 | DropBP: 44% time reduction, 1.5x to target, A100 | **external** | NeurIPS 2024 proceedings |
| V5 | LDAdam accounts for optimizer-state compression in error feedback | **external** | arXiv:2410.16103 |
| V6 | Adam update error has two channels, preconditioner and sign-flip | **derived here** | §4; first-order expansion, mechanism confirmed in the `c²` scaling of `e1b` |
| V7 | Step-time ceiling from backward removal is 50% | **arithmetic** | forward:backward ≈ 1:2 (ShadowStep A4 prices this) |
| V8 | Update-space amplification `A_b` spans 0.13–1.1e5 across error geometries at fixed budget | **verified** | `e2_error_geometry.json`, `e3_statistic.json`, 112 blocks |
| V9 | Optimizer-state statistics beat `‖g‖` at predicting fragility | **falsified** | `e3_statistic.json`: ρ(‖g‖)=0.58–0.63 vs ρ(Σg²/v)=0.38–0.41 |
| V10 | The `Σg²/v` ranking is contaminated by LayerNorm blocks without a floor on `v` | **verified** | `e1_sensitivity.json` top-15 = 15 norms |
| V11 | Selection criterion is a second-order effect at 5% density; density is first-order | **verified** | `e9_compact_gpu.json`, `e9_dens20.json`, `e9_seed1.json`: 4 criteria within 0.17 AUC, density gap 0.27/0.11/0.05 at 5/20/50% |
| V12 | Sparse backward at 5% divides the optimizer's step norm by ~3, and half the quality loss is that | **verified** | `e10_step_norm.json`: ratios 0.362/0.323/0.482; lr rescale closes 55%/52% of the AUC gap |
| V13 | A single-step proxy metric for update importance predicts method quality | **falsified** | `|m|/sqrt(v)` captures 5.8x more update mass and ranks *worst* of four criteria |
| V14 | The above survives scale, other optimizers, and real speedups | **open — do not claim** | one 40 M model, one corpus, ~4 seeds, AdamW, wall-clock not improved |
| V15 | An online step-norm controller can recover the sparse quality gap | **falsified** | `e11_step_controller.json`: controller saturates at lr x20 with norm stuck at 210 vs 872 target; 53% of gap closed, no better than a fixed rescale |
| V16 | Restoring the update norm by freezing masked moments restores quality | **falsified** | `e11_step_controller.json`: norm back to 726 (near dense) but gap closure falls to 16% |
| V17 | The masked update is a scaled dense update | **falsified** | E11: no scalar calibration reaches the dense norm or the dense quality |
| V18 | Truncating the backward graph buys wall-clock | **verified** | E15: 1.55x (last 3 of 6 blocks), 2.39x (last 1), CUDA events, two independent runs |
| V19 | Allocation by update-norm contribution beats allocation by gradient norm | **falsified** | E12: +0.527 vs +0.054 AUC cost at 3/6 blocks; update-norm allocation is indistinguishable from random (3.2230 vs 3.2242) |
| V20 | The whole allocation effect is the embedding/head block | **verified** | E12 follow-up: pinning `-1` (3.2% of update norm, 48.3% of gradient norm) makes all policies cost +0.039..+0.041 and makes `upd` and `grad` select the same set |
| V21 | A scalar step-norm ratio ranks mechanism quality | **falsified** | E14: `rand_elem` (0.645) costs 3x `sparse_block` (0.747); `lowrank_down` (1.131) costs 25x `lowrank_row` (0.999) |
| V22 | Quality-neutrality requires ratio ~ 1 plus small update-space error | **supported, not proven** | E14: the two mechanisms with ratio within 0.01 of 1 are the two cheapest (cost 0.000, 0.006); every other mechanism deviates and pays |
| V23 | A quality-neutral mechanism with real speedup exists | **verified, not yet SOTA** | `sparse_block`: +0.033 AUC for 1.55x on our hardware; DropBP reports 1.5x on A100, so this matches rather than beats |

---

## Addendum: the same null result at layer granularity (E17, fixed budget)

Equal-compute comparisons kept fighting the shared GPU (dense ms/step measured 83, 93, 238 across
runs of the same configuration), so the comparison was re-run in a timing-independent form: **every
arm updates exactly 4 of 6 blocks per step**, and quality is compared at equal steps.

`results/e17_fixedk.json`, 400 steps, one seed, dense = 2.7206 AUC:

| arm | blocks kept | AUC | cost vs dense |
|---|---:|---:|---:|
| fully random 4 of 6 | 4.84 | 2.7462 | +0.0256 |
| gradient-norm sensitive (DropBP recipe) | 4.90 | 2.7485 | +0.0279 |
| gradient-norm sensitive + embedding pinned | 4.90 | 2.7485 | +0.0279 |
| dense (no dropping) | 7.00 | 2.7206 | — |

**At this scale the layer-selection strategy is worth nothing (spread 0.002 AUC), and the entire
cost of dropping a third of the blocks is +0.026.** The gradient-sensitivity and pinned arms produced
byte-identical results, which is expected here: the sensitivity spread across the six blocks is only
0.239-0.300, so a sensitivity-normalised draw is statistically indistinguishable from a uniform one.

This mirrors the per-coordinate result exactly: **the selection criterion is second-order at both
granularities, and the kept fraction is first-order.** Two independent axes, same conclusion.

Caveat for anyone building on this: with only six blocks and a 25% sensitivity spread, this is a weak
test of DropBP's central claim. The honest statement is not "sensitivity does not work" but "at this
scale there is no measurable signal for it to exploit, and the 0.026 total cost sets a ceiling on
what any better criterion could recover".


---

## 4h. E18: the amplification is a joint property of the optimizer AND its state

Everything above was measured with AdamW. The project's framing assumed the optimizer is the causal
object, which makes a specific prediction: change the optimizer and the amplification picture should
change. Three optimizers, identical model / data / steps / injected budget, two error models,
`results/e18_optimizer_ablation.json` plus the warmup variants.

### Median amplification `A_b` at c = 0.25, 112 matrix blocks

| optimizer | proportional error | orthogonal error | update sign-flip rate |
|---|---:|---:|---:|
| SGD + momentum | **0.015** | 0.015 | 0.34% |
| AdamW | 0.126 | 0.523 | 2.7% |
| Lion | **2.953** | 5.864 | 4.6% |

Two orders of magnitude separate SGD from Lion on the same gradients and the same budget. **The
amplification is not a property of the error geometry alone, and it is not a property of the
optimizer alone: it is a joint property.** This is the strongest support the project has for
"optimizer-conditioned" as a real framing — and it arrives after the framing had been falsified on
the allocation axis, which is worth noting.

### And it depends on the optimizer's *state*, not just its type

Same measurement at three warmup lengths, i.e. three ages of optimizer state:

| warmup steps | SGD prop | Adam prop | Lion prop | Lion orth |
|---:|---:|---:|---:|---:|
| 5 | 0.081 | 0.077 | 0.780 | 1.773 |
| 40 | 0.015 | 0.126 | 2.953 | 5.864 |
| 120 | 0.013 | 0.125 | **3.057** | 5.368 |

* **Lion's fragility grows 3.9x as its momentum matures** (0.78 -> 3.06) and then saturates.
* **Adam's grows 1.6x** and saturates (0.077 -> 0.125).
* **SGD's does not grow at all**; it falls (0.081 -> 0.013), consistent with the update simply being
  the gradient, whose relative error is scale-free.

The direction of the effect is the counterintuitive part: **the longer you train, the more damage a
fixed relative gradient error does in update space under Lion**, because Lion's update is
`sign(b1*m + (1-b1)*g)` and a mature momentum means more coordinates are close enough to a sign
boundary for a small perturbation to flip a full-size step. Sign-based optimizers are the fragile
ones; the adaptive optimizer everyone worries about is the middle of the pack; plain momentum SGD is
the most robust.

### Why this matters beyond the mechanism

Lion and other sign/momentum methods are increasingly used for LLM pretraining, and the sparsity and
low-rank literature is almost entirely evaluated on AdamW. The measurement here says those results
do not transfer: **a 5% sparsification budget that is nearly free under SGD is 200x more damaging in
update space under Lion, and it gets worse the longer the run.** Any paper reporting a compression
ratio should report it per optimizer, and the current practice of tuning sparsification on AdamW and
assuming the ratio carries over is not supported by anything measured here.

Also worth stating plainly: this experiment falsified the prediction written down before it was run
("if the structure is the same for SGD and Lion, the effect belongs to numerical linear algebra and
the word optimizer-conditioned must come out of the title"). The structure is not the same, so the
word stays in — by measurement, not by preference.


---

## 4i. E19/E20: the two results that matter in practice

### E19 — the sparsity cost curve is optimizer-dependent, and Lion is far worse

Every sparsity and low-rank result in the literature is tuned and reported on AdamW. Same model,
data, steps, mask and budget; only the optimizer changes. Each optimizer first calibrated to its own
best dense learning rate (SGD 3e-4, AdamW 2e-4, Lion 1e-4), then density swept.
`results/e19_optimizer_density.json`.

**Held-out loss, and the cost against that optimizer's own dense run:**

| optimizer | dense | keep 50% | keep 20% | keep 5% |
|---|---:|---:|---:|---:|
| SGD + momentum | 3.3682 | +0.0056 | +0.0309 | +0.1123 |
| AdamW | 2.7817 | +0.0033 | +0.0395 | **+0.0648** |
| Lion | 2.5266 | +0.0234 | +0.2043 | **+0.3109** |

**At 5% density Lion pays 4.8x what AdamW pays, and 2.8x what SGD pays.** The ordering is exactly
what the amplification measurements predicted (A14), and the margin is large enough to matter: a
compression ratio reported as "nearly free on AdamW" is not nearly free on Lion.

The part that is genuinely counterintuitive is *why*, and it falsifies the diagnostic this project
had been building toward:

| optimizer | step-norm ratio at 5% density | held-out cost |
|---|---:|---:|
| SGD | 1.166 | +0.1123 |
| AdamW | 0.519 | +0.0648 |
| Lion | 0.833 | +0.3109 |

Lion's step norm barely moves (0.833) and it degrades five times worse than AdamW, whose step norm
collapses to 0.519. **The step-norm ratio does not explain the damage across optimizers.** The reason
is structural: Adam rescales every coordinate by `1/sqrt(v)`, so a stale or missing coordinate
contributes a small, *magnitude-weighted* error, whereas Lion acts on `sign(...)` and one flipped
sign is a full-size step in the wrong direction regardless of how large the step norm is. Amplitude
is the wrong currency for a sign-based optimizer.

This also retroactively explains E18's 200x amplification spread in practical terms: sign-based
optimizers are not merely more sensitive in a synthetic metric, they lose five times more quality at
a compression ratio the adaptive optimizer barely notices.

### E20 — the graph skip and the step-norm fix, combined, end to end

A8 measured the wall-clock saving of a truncated backward in a microbenchmark; A11 showed block
dropping is nearly free at this scale; A7 showed half the cost of masking is a step-norm collapse.
This runs all three together in one training loop: backward executed only through the last k of 6
blocks via `autograd.grad` with explicit inputs, embedding/head always trained, and the learning rate
rescaled to match the dense step norm. `results/e20.json`.

| arm | held-out loss | cost | ms/step | speedup |
|---|---:|---:|---:|---:|
| dense | 2.8045 | — | 47.7 | 1.00x |
| skip last 3/6 blocks | 2.8082 | +0.0037 | 29.8 | **1.60x** |
| skip + step-norm correction (lr x1.32) | 2.8072 | **+0.0027** | 27.7 | **1.72x** |

**This is the first arm in the project that both removes computation from the graph and corrects the
step norm, and it is the first result that is a speedup rather than a mechanism measurement**: 1.72x
wall-clock for +0.003 held-out loss, at equal steps. The step-norm correction recovers a further
quarter of the residual cost (0.0037 -> 0.0027).

Caveats, stated because this is the number most likely to be over-read: one 40 M model, byte-level
corpus, three runs of the dense/skip pair (see the seed files), an A10G shared with another tenant,
and the "cost" is held-out loss at equal steps rather than a matched perplexity on a real tokenizer.
It is a real end-to-end speedup with a measured price, not a SOTA claim.


---

## 5b. Retraction and correction: the E20 speedup claim was partly wrong

Two defects in the E20 build, both found by review and both confirmed by measurement rather than
argument. `experiments/e21_grad_reach.py`, `experiments/e22_realskip_fixed.py`;
`results/e21_grad_reach.json`, `results/e22_realskip_fixed.json`.

**Defect 1 — the skip arm was freezing the embedding tables.** E20 built the prefix forward inside
`torch.no_grad()` and detached the boundary before `autograd.grad(..., allow_unused=True)`. E21
measured the consequence directly, per parameter:

| path | parameters receiving NO gradient |
|---|---|
| dense | none |
| E20 skip | **`tok.weight`, `pos.weight`** |

So the arm advertised as "embedding/head always trained" was training the head and freezing the two
tables that carry ~48% of the gradient mass (A10). `allow_unused=True` swallowed the None silently.

**Defect 2 — the three "seed" runs were three identical runs.** `--seed` was never threaded into
`run()`; every call passed `seed=0`. The "identical quality numbers across seeds" reported earlier
were therefore not replication at all. In the corrected build the seeds are genuinely independent:
dense held-out loss is 2.7108 / 2.5603 / 2.5388 for seeds 0/1/2.

### The correction, and why it costs the speedup

A single autograd pass cannot both skip the prefix backward and deliver the prefix's gradients: once
the boundary is detached, the prefix graph is gone, and seeding a second backward with a scalar
cotangent is not the same thing as seeding it with the cotangent of the input. The construction that
does work is checkpoint-style replay — run the prefix forward without a graph, backward the suffix,
then replay the prefix forward *with* grad and backprop that, seeded by the stored boundary cotangent.

`e22_realskip_fixed.py` implements both, with `--verify-grads` asserting which parameters end a step
without a gradient:

| arm | params w/o grad | held-out cost (mean of 3 seeds) | wall-clock |
|---|---:|---:|---:|
| dense | 0 | — | 1.00x |
| skip (E20's actual mechanism) | **38** | +0.0087 | **1.72x** |
| skip_ri (replay, correct gradients) | **0** | +0.0077 | **0.82x** |

**The corrected method is not a speedup — it is slower than dense (0.82x).** Replay costs a forward
pass where the baseline pays a backward, and the measured forward:backward ratio in this model does
not leave room for that trade to pay (A8 prices the same ratio). What survives from E20 is narrower
and now honest:

* `skip` genuinely gives **1.72x** on three independent seeds (1.64 / 1.89 / 1.63), at a measured
  **+0.0087** held-out cost — but it is **not** a method that trains every parameter. It is the
  DropBP-class trade: drop backward work, drop those parameters' updates, and pay in quality.
* every mechanism in this project that removes backward work *and* keeps all gradients current is
  slower than dense. The two goals are in direct conflict at this scale, which is itself a result:
  it explains why the published layer-dropping methods all accept the frozen-parameter trade.

The earlier sentence in this document — "the first arm that both removes computation from the graph
and corrects the step norm" — is withdrawn: that arm did not deliver gradients to the embedding, so
it was not the thing it claimed to be. The claim ledger rows V18/V23 and A19 have been rewritten
accordingly, and F9 records the invalidated claim.

### What this implies for the paper

The system story cannot be "we skip backward and stay quality-neutral". The honest versions are:

1. **The DropBP-class trade, quantified more carefully than before**: 1.72x for +0.0087, with the
   frozen-parameter scope stated explicitly rather than papered over.
2. **A gradient-compensation variant** that keeps every parameter's update current *without* a
   replay forward — e.g. carry the dropped blocks' optimizer state forward with a cheap estimate
   rather than a recomputation. That is the open systems question, and E22 shows the obvious
   construction (replay) does not work.


---

## 5c. Second retraction: the E19 "Lion is 4.8x worse" result was seed noise

The E19 grid reported the held-out cost of 5% sparsity as +0.065 (AdamW), +0.112 (SGD) and **+0.311
(Lion)**, and that last number became the centrepiece of the write-up and of the previous round's
summary. It does not replicate.

E19 was not seedable (`--seed` existed on the CLI but was never threaded into `run()`). It is now, and
the exact E19 configuration re-run at seeds 0/1/2 gives:

| seed | dense held-out loss | 5% held-out loss | "damage" |
|---|---:|---:|---:|
| 0 | 2.5266 | 2.8375 | **+0.311** |
| 1 | 2.3467 | 2.7053 | +0.359 |
| 2 | 2.2355 | 2.7037 | +0.468 |

**The dense baseline moves by 0.29 across seeds — the same size as the effect being reported.** The
+0.311 was the smallest of the three dense baselines, which is exactly what made Lion look worst. The
honest statement is that at this scale the *dense* run is the noisy arm and the effect is not
resolvable from single runs.

That is not a small methodological footnote; it invalidates a headline. Two further facts make it
worse for the original claim and more interesting overall:

* **The sparse arms are far more reproducible than the dense one.** Lion at 5% gave 2.8375 / 2.7053 /
  2.7037 across seeds where dense gave 2.5266 / 2.3467 / 2.2355. Truncating the update to its top 5%
  makes the run *stabilise* — a plausible mechanism (the mask acts as a strong, data-independent
  regulariser, and `sign` discards magnitude) but one this project has not isolated.
* **A clean, seeded, five-run comparison reverses the ordering.** `e23` (five independent grids, two
  mask scopes, two seeds) gives 5% damage of **+0.086 (AdamW), +0.14..+0.17 (SGD), +0.065..+0.075
  (Lion)**. Lion is the *least* damaged, not the most, and the spread across the five runs is 1.17x.

So the correct claims are:

* **Withdrawn**: "at 5% density Lion pays 4.8x what AdamW pays".
* **Stands**: the amplification measurements (A14-A16) — those are ratio measurements on fixed states,
  not end-to-end runs, and they replicate.
* **Stands**: the E18 structural facts (SGD's amplification is invariant, Lion's grows with state age).
* **New and better supported**: optimizer-specific risk scores predict damage *ordering* within an
  optimizer. Spearman rho(risk, damage) over four densities is 1.00 for AdamW in all five runs, 1.00
  for SGD, and 0.80-1.00 for Lion. What is *not* supported is a single cross-optimizer risk scale.

The lesson generalises past this project: single-run optimizer comparisons at this model scale are
not evidence. Both retractions in this document (E20 and E19) came from the same root cause — a
comparison whose noise floor was never measured.


---

## 4j. E25: the sign-boundary fraction grows with scale — a prediction that needs no training run

The review asked for a predictor derived from each optimizer's update rule (E23 delivered it within
an optimizer) and then for scale. Putting the two together produces a question that is answerable
*without* any end-to-end comparison, which matters because every 1B training comparison attempted in
this project failed to be discriminative (X1, and the flat-probe runs of E24).

Lion's risk is `P[sign(m_i + delta_i) != sign(m_i)]`, a property of how many coordinates sit close
to the sign boundary. The dimensionless quantity that decides this is `|g_i| / sqrt(v_i)`. So:

> **as models get wider, does a larger or smaller fraction of coordinates sit within a fixed
> relative distance of the sign boundary?**

`experiments/e25_boundary_fraction.py` measures exactly that, pooling all 2-D weights, after a short
Adam-moment warm-up. Batch size is a confound for this statistic (a smaller batch means a noisier
gradient relative to the accumulated second moment), so it was controlled:

| scale | batch | P(<0.01) | P(<0.05) | P(<0.1) | P(<0.25) | P(<0.5) | P(<1.0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact 10.9M | 4 | 0.0114 | 0.0280 | 0.0488 | 0.1123 | 0.2277 | 0.5458 |
| compact 57.3M | 4 | 0.0068 | 0.0228 | 0.0428 | 0.1043 | 0.2161 | 0.5329 |
| compact 10.9M | 1 | 0.0137 | 0.0396 | 0.0722 | 0.1701 | 0.3357 | 0.6585 |
| compact 57.3M | 1 | 0.0091 | 0.0348 | 0.0669 | 0.1638 | 0.3291 | 0.6574 |
| **Llama-1.2B** | 1 | **0.0230** | **0.0850** | **0.1487** | **0.3048** | **0.5037** | **0.7524** |

Three findings, in order of how much I trust them:

1. **At matched batch size, the near-boundary fraction roughly doubles from 57M to 1.2B**
   (P(<0.1): 0.0669 -> 0.1487; P(<0.5): 0.3291 -> 0.5037). Sign-based optimizers therefore face about
   twice as many vulnerable coordinates at 1B as at 57M.
2. **Batch size moves the statistic in the opposite direction and by a similar amount** (57M:
   P(<0.1) 0.0428 at bs 4 vs 0.0669 at bs 1). This is not a nuisance — it is a second, independent
   lever: smaller batches push more coordinates toward the boundary, which is exactly the regime
   large-model training lives in.
3. **Between 11M and 57M the fraction *falls* at fixed batch size** (0.0722 -> 0.0669). So the trend
   is not a smooth function of parameter count; the jump happens somewhere between 57M and 1.2B, and
   with three points and one architecture family this is a hint, not a law.

### Why this matters more than another training comparison

It is a **scaling prediction with a mechanism, measurable in minutes, that requires no learning-rate
calibration and no end-to-end run**: sign-family optimizers should suffer disproportionately more
from a fixed relative backward-approximation error as models grow, and the measured exposure roughly
doubles by 1B. The prediction is falsifiable in one direction (measure the fraction at 3B-7B; if it
does not keep rising, the effect saturates and the practical claim weakens) and it explains the
otherwise-puzzling pattern that the 40M Lion comparisons disagreed with each other — at 40M the
exposure is small enough that run-to-run noise dominates it (F10).

What it does **not** do is measure damage. It measures the *exposure* that the damage mechanism acts
on. Closing that gap needs a discriminative 1B training configuration, which this project has not
yet achieved: at lr 3e-6 the fixed-probe held-out loss was flat to 0.008 over 600 steps
(2.7734 -> 2.7891), so the runs cannot resolve a density effect either way.


---

## 4k. Correction to 4j: the boundary fraction is a family effect, not a scaling law

4j concluded from three points that "the near-boundary fraction roughly doubles from 57M to 1.2B" and
proposed it as a scaling prediction. Adding two more checkpoints — including a *smaller* one —
falsifies that reading. Same protocol, bs 1, `results/e25_*.json`:

| model | params | P(<0.01) | P(<0.05) | P(<0.1) | P(<0.25) | P(<0.5) |
|---|---:|---:|---:|---:|---:|---:|
| compact GPT, from scratch | 10.9M | 0.0137 | 0.0396 | 0.0722 | 0.1701 | 0.3357 |
| compact GPT, from scratch | 57.3M | 0.0091 | 0.0348 | 0.0669 | 0.1638 | 0.3291 |
| **Qwen2.5-0.5B** | 494M | 0.0259 | 0.0927 | **0.1581** | 0.3106 | 0.4968 |
| Llama-3.2-1B | 1236M | 0.0230 | 0.0850 | **0.1487** | 0.3048 | 0.5037 |
| **Qwen2.5-1.5B** | 1544M | 0.0316 | 0.1014 | **0.1618** | 0.2997 | 0.4764 |

**A 494M model has a higher fraction than both a 1.2B and a 1.5B model.** Within the Qwen family the
fraction is flat across a 3x parameter range (0.1581 -> 0.1618). Within the from-scratch compacts it
falls slightly (0.0722 -> 0.0669). Across the two pretrained families and the compacts there is a
2.3x gap. So:

* **The between-family / between-design difference dominates the within-family scaling difference.**
  The statistic is real and cheap to measure, but it is a property of the model's design and
  initialization, not of its size.
* The "doubling from 57M to 1.2B" in 4j was a comparison between a from-scratch byte-level model and
  a pretrained subword model. That is a confound, not a scaling result, and the sentence is
  withdrawn.
* The batch-size effect measured in 4j (1.56x at 57M, bs 4 -> bs 1) is of the same order as the
  family effect and larger than any within-family scale effect. It remains a genuine second lever.

### What survives, and it is a cleaner statement

> **The fraction of coordinates sitting near the sign boundary — the quantity that sets a
> sign-based optimizer's exposure to backward-approximation error — varies by more than 2x across
> model designs and is nearly flat with parameter count inside a design. Optimizer and compression
> policies calibrated on one model family therefore do not transfer to another, for the same reason
> the AdamW-tuned sparsity budgets did not transfer to Lion (A22).**

That is consistent with everything else in this document: the budget that is safe is a property of
the (optimizer, model design, batch size) triple, not a universal constant. It is also why the 40M
comparisons disagreed with each other — at that scale the exposure is small and family-dependent.

The honest limitation: the compacts are trained from scratch on byte-level data while the pretrained
models use subword tokenizers, so the 2.3x gap mixes tokenizer, initialization, and architecture.
Separating those needs a within-tokenizer, within-family width sweep trained from scratch, which is
the next version of this measurement.


---

## 4l. E26: the fragility inverts — Lion tolerates a skipped backward far better than AdamW

Everything about Lion in this document so far is about *perturbing* the gradient: its amplification is
2.95x AdamW's (A14), and its exposure to sign flips is design-dependent (A28). E26 asks the opposite
question, which those results do not determine:

> if a step's gradient is simply **not computed**, which optimizer degrades less?

The mechanism predicts Lion should win. Lion's update is `sign(m)`, which depends only on the
accumulated *direction*; a stale direction is still roughly the right direction. AdamW's update is
`m/(sqrt(v)+eps)`, which depends on the accumulated *magnitude*; a stale `v` makes the step size wrong
in a way that compounds.

`experiments/e26_staleness.py`, 29M-parameter GPT, 75M-byte corpus, held-out probe every 100 steps,
both optimizers at their own calibrated lr. `results/e26_reuse.json`:

| backward every | AdamW 1/L_end | Lion 1/L_end | Lion / AdamW |
|---:|---:|---:|---:|
| 1 | 13.16 | 42.74 | **3.25x** |
| 2 | 1.018 | 2.312 | **2.27x** |
| 4 | 0.371 | 0.368 | 0.99x |
| 8 | 0.359 | 0.362 | 1.01x |

(`1/L_end` is the inverse final held-out loss: higher means more progress per backward pass. Ratio
1.0 means the two optimizers are equally efficient at that backward budget.)

**Lion extracts 3.25x more progress per backward pass when every step is backpropagated, and 2.27x
more at every second step.** In the probe curves the difference is visible as a phase change: AdamW
sits at ~2.65 for 700 steps and then drops; Lion breaks away around step 600 and reaches 0.19 by step
800 where AdamW is still at 1.48.

### The practical reading, and it is the method layer this project was missing

This is the first result here that says *what to do differently*:

> **A backward-skipping schedule should be tuned per optimizer, and the sign-based optimizer is the
> one that can afford the aggressive schedule.** AdamW needs a fresh gradient to know *how large* its
> step should be; Lion only needs to know *which way to go*, and a stale direction remains usable.
> At every-2, Lion with half the backward passes outperforms AdamW with all of them (1/L 2.31 vs
> 1.02).

Note the interaction with E25/A28: Lion's disadvantage is in absorbing *noise added to* a gradient,
while its advantage is in tolerating a gradient that is *merely old*. Those are different regimes and
a runtime can choose between them — perturb (use a cheap approximation) or postpone (skip and reuse).
The measurements say Lion should be asked to postpone, and AdamW to neither.

### Caveats, stated because this is the most actionable claim in the document

1. At every-4 and every-8 both optimizers collapse to the same plateau (~2.7) and the ordering
   vanishes. The Lion advantage lives in the 1-to-2 backward-per-step region, not beyond it.
2. The absolute held-out losses reach 0.02-0.08 at every-1 in 1000 steps on a 75M-byte corpus, i.e.
   this configuration is in a heavy-memorisation regime. The comparison is therefore of *how fast each
   optimizer reaches that regime* rather than of generalisation, and a version on a corpus large
   enough to keep held-out loss near 2.5-3.0 would be the honest confirmation. That experiment is
   specified and not yet run.
3. One model, one architecture, one seed, byte-level data.


---

## 4m. Retraction: the E26 "Lion tolerates staleness" result does not survive a longer schedule

4l reported that Lion extracts 3.25x more progress per backward pass than AdamW at every-1 and 2.27x
at every-2, and proposed a per-optimizer skipping schedule. That was measured at 1000 steps on a 75M
corpus. Running the same comparison for longer at a matched data budget reverses it.

Two runs, same code, same probe, same optimizers at their calibrated lr; the only change is how much
training data is consumed:

| every | 75M corpus, 2 MB seen, 1000 steps | | | 198M corpus, 5 MB seen, 1953 steps | | |
|---:|---:|---:|---:|---:|---:|---:|
| | AdamW end | Lion end | Lion adv. | AdamW end | Lion end | Lion adv. |
| 1 | 0.0760 | 0.0234 | **+0.053** | 3.0412 | 2.6381 | **+0.403** |
| 2 | 0.9824 | 0.4325 | **+0.550** | 3.8239 | 3.8674 | **-0.044** |
| 4 | 2.6978 | 2.7156 | -0.018 | 4.5534 | 4.8473 | **-0.294** |
| 8 | 2.7839 | 2.7606 | +0.023 | 8.3420 | 8.8479 | **-0.506** |

**Lion's advantage at every-2 (+0.55) becomes a disadvantage (-0.044) once the run is longer and the
data budget larger.** The A30 rows are withdrawn.

What this looks like mechanistically: Lion's early advantage is a *speed-to-fit* effect. Its probe
curve runs ahead early (2.36 at step 600 against AdamW's 2.64) but it is fitting the training stream
faster, and when the schedule is sparse the fit is worse — in the matched run Lion at every-2 starts
at 5.72 and *rises to 9.09* before coming back, i.e. stale sign updates actively push it away before
recovering. AdamW's stale updates are wrong in a smoother way.

So the honest state of the method layer:

* **No per-optimizer skipping rule is supported by these measurements.** The E26 effect was
  regime-dependent and is not a property of the update rule in the way 4l claimed.
* The one asymmetry that has replicated across every run is much weaker: Lion reaches a *lower*
  held-out loss than AdamW at every-1 in both regimes (+0.053 and +0.403), which is a statement about
  optimisation speed, not about skipping.
* The lesson repeats the two earlier ones (E19, E20): a single schedule length and a single data
  budget is not enough to establish an optimizer comparison. Here the reversal came from changing
  only how much data the run consumed.
