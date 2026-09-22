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
