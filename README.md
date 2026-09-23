# When Backward-Approximation Results Reverse

A measurement-methodology study of approximate backpropagation for LLM training.

This repository began as an attempt to build a better backward-approximation method. It produced six
counterintuitive positive results, and then falsified all six of them with its own follow-up
experiments. What is left is a quantified account of **why** they reversed, a small set of claims that
survived replication across configurations, and a protocol for checking whether a comparison is
measurable at all.

**Read `CLAIMS.md` for the ledger: every claim tagged measured / falsified / open, with the script that
regenerates it.** Six rows are marked RETRACTED; they are kept deliberately.

---

## The headline finding

Take one comparison — Lion's held-out damage minus AdamW's at 5% backward sparsity, both at their own
calibrated learning rate — and run it under four configurations (2 run lengths x 2 training-slice
offsets), 3 seeds each, 24 runs (`results/o16_*.json`):

| configuration | dense held-out loss | Lion − AdamW damage |
|---|---:|---:|
| 300 steps, offset 0 | 2.6096 | **−0.0079** |
| 300 steps, offset 40M | 2.9849 | +0.0074 |
| **800 steps, offset 0** | **1.2122** | **+1.0924** |
| 800 steps, offset 40M | 2.9187 | +0.0086 |

Between-configuration SD **0.5449** vs within-configuration (seed) SD **0.1939**. The grand mean
(+0.2751) is not distinguishable from zero and the sign is not stable across configurations.

The one large effect occurs in the configuration whose **dense** held-out loss has fallen to 1.2122 —
a memorisation regime. The three generalising configurations agree to within **0.009** of zero effect.

> **The same nominal comparison ranges from "no effect" to "Lion is far worse" depending on
> configuration choices most papers never report. More seeds do not rescue a comparison performed in
> the wrong training regime.**

This is not an argument that published work is wrong. Published LLM training sees 10²–10⁴ tokens per
parameter on corpora with 10⁹–10¹² distinct tokens; the turnover points measured here (5–25 bytes per
parameter) are orders of magnitude away. The claim is narrower and checkable: **on a small,
low-diversity corpus, a comparison run past the turnover point measures the memorisation race, not the
treatment.**

---

## What survived replication

| claim | evidence | strength |
|---|---|---|
| **Update-space amplification depends on error geometry**: at a fixed gradient-error budget, `A_b` spans **0.13 → 1.1e5** across proportional / rank-truncation / orthogonal error (112 blocks, Llama-3.2-1B) | `results/e2_error_geometry.json`, `e3_statistic.json` | ★★★★★ |
| **An optimizer-specific risk score predicts that optimizer's damage ordering**: Spearman ρ over four densities = **0.80–1.00** (1.00 in 15 of 18) across **3 seeds × 2 run lengths × 3 optimizers** | `results/e28_s*_seed*.json` | ★★★★★ |
| **Approximation damage is regime-dependent by an order of magnitude**: the same arms give +0.03…+0.20 at 5% when dense held-out is 2.69–3.54, and +2.23…+2.51 when it is 0.03–0.46 | `results/e28_*.json` | ★★★★★ |
| **Configuration variation exceeds seed variation** (O16 above) | `results/o16_*.json` | ★★★★★ |
| **The memorisation turnover point is measurable and capacity-dependent**: 1.12 tok/param (10.94M model), 6.38 (2.20M), not reached by 12.4 (0.35M) on a 75.4 MB corpus | `results/e32*.json` | ★★★★★ |
| **Measured floors**: seed floor for damage differences **0.0244**; wall-clock spread **1.008×** within one interleaved process vs **1.78×** across processes | `results/e31.json` | ★★★★ |
| **Naive graph-level backward skip is 3.6–4.5× faster than dense** (keep = L/12, L=24/48, 6 runs) — but it freezes the dropped parameters' updates | `results/o14_*.json` | ★★★★ |

### Known failures, kept on purpose

| withdrawn claim | why |
|---|---|
| "Lion pays 4.8× what AdamW pays at 5% density" | a memorisation-race artifact (O16 above) |
| "200× optimizer amplification asymmetry (SGD 0.015 / AdamW 0.126 / Lion 2.95)" | cross-optimizer ratio is 0.93–1.11× under 2 seeds × 2 run lengths |
| "1.6–2.1× from graph skip + step-norm correction, embedding trained" | the arm silently froze `tok`/`pos` embeddings (38 parameters with no gradient) and its three "seeds" were identical |
| "The near-boundary fraction doubles from 57M to 1.2B, so sign optimizers get worse with scale" | a 494M model sits above a 1.5B model; it is a design effect, not scale |
| "Lion tolerates a skipped backward 2.27× better than AdamW" | reverses at a longer schedule and larger data budget |
| "There is a depth crossover where gradient-correct replay beats dense (1.02–1.23×)" | one seed produced it; 3 seeds × 2 depths give 0.58–1.05× |

Six withdrawals, one cause: an effect smaller than the variation between configurations, established
by varying one axis at a time.

---

## The protocol this produced

Before trusting any approximate-training comparison:

1. **Calibrate the regime.** Measure the held-out-loss turnover point in tokens/parameter for your
   model–corpus pair and report it. Below it a comparison is measurable; above it, it is a
   memorisation race. (`experiments/e32_regime_curve.py`)
2. **Replicate across configurations, not just seeds.** Vary at least run length and data slice.
   Seeds cannot substitute: within-config SD was 0.19 against between-config SD 0.54 here.
   (`experiments/e23_risk_predictor.py --steps N --train-offset M --seed S`)
3. **Report the dense baseline's held-out loss with every damage number**, since the same arms give
   ±0.2 or ±2.4 depending on it.
4. **Time pairs interleaved in one process.** The wall-clock floor is 1.008× there versus up to 1.78×
   across processes; speedups below ~1.8× cannot be resolved sequentially on a shared device.
5. **Verify gradient reach.** Masking or detaching a subgraph can silently starve parameters; assert
   that no parameter ends a step without a gradient. (`experiments/e21_grad_reach.py`, and
   `--verify-grads` on `e22`)

---

## Reproduce

A CUDA GPU is required; the compact experiments fit in ~2 GB. Paths are configurable:

```bash
export AUDIT_MODEL=/path/to/Llama-3.2-1B-Instruct      # for the pretrained diagnostics
export AUDIT_CORPUS=/path/to/jsonl_or_text_dir        # any JSONL with a context/input field
```

```bash
# the configuration-dependence result (O16)
python3 experiments/e23_risk_predictor.py --device cuda --steps 800 --train-offset 0 \
        --densities 1.0,0.05 --optimizers adamw,lion --seed 0
# the regime classifier (E32)
python3 experiments/e32_regime_curve.py --device cuda --dim 384 --layers 6 --max-steps 9000
# the measured floors (E31)
python3 experiments/e31_floors.py --device cuda --seeds 5
# gradient-reach check (E21)
python3 experiments/e21_grad_reach.py --device cuda
```

Raw outputs land in `results/`, one file per run, never edited by hand. `MANIFEST.sha256` covers
`results/`, `experiments/` and `docs/`.

---

## Layout

```
CLAIMS.md                        claim ledger: measured / falsified / open / configuration failures
docs/NEXT_BACKWARD_DIRECTION.md  the full record, including all six retractions and the theory
docs/WHY_MEASUREMENTS_DISAGREE.md  post-mortem of the bug that produced a plausible, wrong result
experiments/                     one script per experiment, self-contained
results/                         raw JSON, one file per run
```

## Scope and limitations

* Mechanism and methodology measurements, not throughput claims. Absolute numbers come from a single
  shared A10G, a byte-level corpus, and small models; only the *relative* and *protocol* results are
  offered as transferable.
* The one prediction that survived (the optimizer-specific risk score) is validated within an
  optimizer and within one approximation family; it does **not** rank different error structures
  (`results/e29_scheme_screen.partial.json`).
* The regime classifier's turnover points are measured for one corpus and three model sizes; the
  numbers will differ elsewhere, which is why the *measurement* is the deliverable rather than the
  constant.
