# Toy study: which loop-transformer strategy trains most stably?

A controlled, small-GPU (single RTX 3090) comparison of loop/recurrent
Transformer strategies, to test the claim behind
[docs/TRAINABILITY.md](../TRAINABILITY.md): that the two-pass additive-routing
design is **more stable to train** than traditional looped alternatives.

Run it yourself: `python studies/loop_study.py --steps 1500 --seeds 3` then
`python studies/analyze.py studies/results.json`.

## Setup (held identical across every strategy — the fair-comparison contract)

- **Task:** modular cumulative sum — predict `(Σ inputs[0..t]) mod 7` at each
  position. A long associative scan: depth/iteration *should* help, and it does
  not saturate at toy scale, so strategies actually separate.
- Same `d=128`, 4 heads, 6 layers (or 1 weight-tied block looped), seqlen 48,
  batch 256, AdamW, 1500 steps, same data stream.
- **9 strategies**, **4 learning rates** `{1e-3, 3e-3, 1e-2, 3e-2}` (incl. an
  aggressive LR to stress stability), **3 seeds** each = 108 runs.
- We report **forward-pass count** so quality is read per-compute.

Strategies (grounded in the literature reviewed in `docs/TRAINABILITY.md`):
`baseline` (1×), `ut2`/`ut4` (Universal-Transformer weight-tied loop, full BPTT),
`ut4_tbptt` (truncated BPTT, grad through last iter), `ut4_inject` (recurrent-depth,
re-inject embeddings), `act` (ACT halting), `deq` (DEQ-style, grad through last
step only), **`ours`** (two-pass additive routing, gate init ≈ 0), **`ours_hot`**
(same but gate init large — the warm-start ablation).

## Results

Accuracy is at the best LR, averaged over 3 seeds. We report **mean** (not the
single best seed) — for a stability study, rewarding one lucky run would flatter
the high-variance methods. Seed range shows the min–max spread.

| strategy | passes | **mean acc** | seed range | grad-max p90 | **seed σ** |
|---|---|---|---|---|---|
| ut2 (UT, full BPTT) | 2× | **0.736** | 0.43–0.90 | **107** | 0.220 |
| ours_hot (gate large) | 2× | 0.495 | 0.39–0.62 | 31 | 0.094 |
| **ours** (gate≈0, warm start) | 2× | **0.455** | 0.43–0.48 | 24 | **0.020** |
| baseline | 1× | 0.403 | 0.29–0.60 | 24 | 0.141 |
| ut4 (UT, full BPTT) | 4× | 0.392 | 0.34–0.46 | 30 | 0.051 |
| act (ACT halting) | 4× | 0.325 | 0.27–0.44 | 15 | 0.079 |
| ut4_inject (recurrent-depth) | 4× | 0.314 | 0.27–0.35 | 9 | 0.032 |
| deq (grad last step only) | 6× | 0.164 † | 0.16–0.17 | 6 | 0.003 |
| ut4_tbptt (trunc. BPTT) | 4× | 0.157 † | 0.15–0.16 | 8 | 0.005 |

Sorted by mean accuracy. **ours beats baseline on mean (0.455 vs 0.403)** and has
the lowest seed variance of any strategy that learns — its worst seed (0.432) still
tops baseline's average. Baseline's 0.60 "best seed" was one lucky run (its other
two were ~0.30).

† at chance (1/7 ≈ 0.143) — did **not** learn the task.

## What it shows

1. **The near-zero-gate warm start is the decisive stability lever.** `ours` and
   `ours_hot` are the *same architecture* differing only in gate init. Warm start
   (`ours`) gives **seed σ = 0.020** vs `ours_hot`'s **0.094** at the best LR —
   ~5× lower run-to-run variance — and a much smoother LR profile. This is the
   cleanest, most direct confirmation of the trainability thesis.

2. **`ours` beats the single-pass baseline on mean accuracy AND is far more stable.**
   Mean 0.455 vs baseline 0.403 — and `ours`'s *worst* seed (0.432) still tops
   baseline's *average* (0.403). It has the lowest seed variance of any strategy
   that learns (0.020) — below baseline (0.141), ut2 (0.220), ut4 (0.051), act
   (0.079). (ut4_tbptt and deq have ~0 variance only because they are stuck at chance.)
   Baseline's higher *best seed* (0.60) is a single lucky run; its other two were ~0.30.

3. **"Detach most of the loop" strategies fail to learn the scan.** `ut4_tbptt`
   (truncated BPTT) and `deq` (grad through last step only) sit at chance — cutting
   the gradient through the weight-tied iterations starves the earlier steps of
   learning signal, exactly the truncated-BPTT-bias failure mode in the literature
   (Tallec & Ollivier 2017; Bai et al. 2019).

4. **Full-BPTT weight-tied UT is powerful but risky.** `ut2` reaches the highest
   mean accuracy (0.74, best seed 0.90) — genuine capability from the loop — but
   with the largest gradient spikes (grad-max p90 = 107 vs ours' 24) and the
   **highest seed variance (0.220)**: its three seeds at the best LR spanned
   0.43 → 0.90. High ceiling, unreliable floor.

## Honest caveats (this is a toy study, not the last word)

- **On this synthetic scan, `ours` beats baseline but does not reach the top mean**
  — `ut2` (deep full BPTT) is higher on mean quality (at much higher variance/cost).
  The two-pass design's headline *quality* advantage was measured on **language
  modeling** (the d24 run: +9% CORE), not on this task; a
  pure associative scan is arguably the home turf of deep recurrence. What this
  toy study isolates is **stability**, which is task-transferable and is where
  `ours` clearly leads.
- No strategy *diverged* (0% everywhere) in this LR range — the stability signal
  here is **seed variance and gradient spikes**, not outright NaNs. A wider LR
  sweep or larger models would likely surface hard divergences (esp. for full-BPTT
  UT).
- Single task, single scale, 3 seeds. Directional, not definitive.

## Bottom line

For a fixed, small number of passes, the **two-pass additive-routing design with a
near-zero-gate warm start beats the single-pass baseline on mean accuracy and is
the most stable strategy that learns** — lowest seed variance, moderate gradient
norms, smooth LR profile — and the warm-start ablation shows *why*. Deep full-BPTT recurrence (`ut2`) can reach a higher ceiling
on this depth-hungry toy task, but does so with large gradient spikes and 10×
higher run-to-run variance; the gradient-truncating strategies (`ut4_tbptt`,
`deq`) fail to learn it at all. This matches the trainability argument: the design
trades the ceiling of deep recurrence for a well-conditioned, reliable optimization.
