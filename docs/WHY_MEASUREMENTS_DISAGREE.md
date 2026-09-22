# Why two runs of the same experiment disagreed

This is a short post-mortem, kept because the bug it describes produced a *plausible and completely
wrong* experimental conclusion, and because the lesson generalises to any "compare two training
regimes" setup.

## The symptom

Two scripts trained the same model on the same data with the same seed:

```
E16 dense       losses at step 25: 2.9458   final 2.8971
E17 fully_random losses at step 25: 3.3703   final 3.3634
```

The first logged loss was identical (5.6976). The model initialisation hash was identical. The
number of forward calls per run was identical. The batch offsets were identical. Only the loss
trajectory differed, and it differed from the very first update.

## The bisection

Each step narrowed it, and each step is a technique worth reusing:

1. **Same init?** Hash all parameters right after construction. Identical -> the RNG seeding is not
   the problem.
2. **Same data?** Hash the batch tensors and print the offsets. Identical.
3. **Same forward count?** Instrument `GPT.forward` with a counter. 40 calls in both.
4. **Same update?** Print `||dtheta||` after the first step: **3294.39 vs 3260.95**. Different.

That last number localises the bug to the update, not the data or the model. Since both scripts had
just been verified to use the same "dense" configuration, the difference had to be in which
parameters were allowed to update.

## The bug

Both scripts implement layer-dropping by *masking gradients*: for a dropped block, set
`param.grad = None` so the optimizer never touches it. The set of kept groups was built like this:

```python
keep_p = {b: min(1.0, raw[b] / sraw * keep_layers) for b in blocks}   # blocks = 0..5
if pin:
    keep_p[-1] = 1.0
```

`blocks` is `range(n_layers)` — the transformer blocks. The embedding and LM head live in a
separate group, `-1`. Two consequences:

* With `pin=False`, **the embedding group was not in `keep_p` at all**, so it was silently dropped:
  `allowed` never contained `-1`, and the mask zeroed the embedding and head gradients.
* With `keep_layers == n_layers` ("dense"), the formula gave `keep_p = 6/7 = 0.857` for each block,
  so the "dense" reference was itself dropping a block ~14% of the time.

The verification that pinned it down was direct: compute the update norm over all parameters
(3294.39) and over only the parameters in `allowed` (3260.95). The second number matched the
"wrong" run exactly.

## Why this mattered scientifically, not just numerically

The buggy version produced a clean, believable, and inverted result:

```
by steps   : dropbp +0.554 cost, dropbp_pin +0.053 cost   -> "pinning the embedding looks great"
by AUC/sec : dropbp 30.56,      dropbp_pin 18.50          -> "plain dropping looks great"
```

Neither frame was the intended comparison. Equal *steps* lets the cheap arm do less work; equal
wall-clock lets it do more steps; and in this case the two arms also differed in a second variable
(whether the embedding block participated) that had nothing to do with the question being asked.
Three variables were moving — layers dropped, steps taken, and which parameter groups were eligible
— and the headline flipped depending on which one you normalised by.

## The rules this suggests

1. **Normalise on the quantity the claim is about.** "Better at equal compute" needs equal measured
   compute, not equal steps. Report both and say which one supports the claim.
2. **Give every arm a shared floor.** If one arm gets an always-on component (here, the embedding),
   every arm must get it, or the comparison is about that component.
3. **Verify the reference arm.** The "dense" baseline was quietly approximate. Always assert that
   the control is actually the control (`sum(keep_p) == n_groups`, drop count == 0).
4. **Bisect on hashes and norms, not on intuition.** Init hash -> batch hash -> forward count ->
   update norm localises almost any "same code, different result" report in four steps.
5. **A plausible result is not evidence of a correct experiment.** The buggy run agreed with the
   hypothesis beautifully; that is exactly when to check the mask.
