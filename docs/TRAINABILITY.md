# Why the two-pass routing loop is easier to train than a traditional looped Transformer

Looped, recurrent, and weight-tied Transformers are attractive because they add
depth/compute without adding parameters — but they are **notoriously hard to
optimize**. This note collects the documented, citable training pathologies of
that family and argues, failure-mode by failure-mode, why the **two-pass
additive-routing** design in this repo sidesteps them by construction.

> **Scope & honesty.** The literature claims below are from primary sources
> (original papers or authors' own repos) and were adversarially fact-checked.
> The two-pass design's advantages are argued by **mapping its construction onto
> those documented failure modes** — it was not independently benchmarked against
> each baseline here. Where deep recurrence has a real advantage this design
> gives up, that is stated plainly (§7).

## The design, in one paragraph

Run the shared-weight trunk **exactly twice**. Pass 1 is an ordinary causal
Transformer; it caches the normalized, RoPE'd `(q₁, k₁)` at a few source layers.
Each pass-2 layer adds a causal **additive attention-logit prior** from one
source layer, `attn_logits = q₂k₂ᵀ/√D + g·(q₁k₁ᵀ/√D)`, with a learned per-layer
gate `g = softplus(gate_logit)` **initialized near zero**. It is fully
differentiable, trained with ordinary backprop — **no BPTT through many steps, no
fixed-point solve, no halting mechanism** — and the prior is realized via
width-`2D` concatenated Q/K through ordinary/FlexAttention, so it stays **fully
parallel** over the sequence.

---

## The traditional difficulties (cited)

### 1. Variable-depth recurrence needs a halting mechanism — and halting is unstable

Universal Transformers apply the same weight-tied self-attention block over a
variable number of refinement steps, "combin[ing] the parallelizability … of the
Transformer with the recurrent inductive bias of RNNs" [UT, Dehghani et al. 2019,
[arXiv:1807.03819](https://arxiv.org/abs/1807.03819)]. To decide *how many* steps
per position, they use **Adaptive Computation Time (ACT)** — a halting unit
trained with a time penalty `L̂ = L + τ·P(x)`. Graves states plainly that "the
behaviour of the network is quite sensitive to the value of τ, and it is not
obvious how to choose a good value" [Graves 2016,
[arXiv:1603.08983](https://arxiv.org/abs/1603.08983)]. CoTFormer restates ACT's
"instability and its sensitivity to hyperparameters"
[[arXiv:2310.10845](https://arxiv.org/abs/2310.10845)].

The proposed fixes are themselves hard to train. CoTFormer reports that "training
with [the] PonderNet mechanism [was] challenging and sensitive to the choice of
hyperparameter, specifically the weight of the KL divergence" — best-tuned
PonderNet perplexity **41.37 vs 33.08** for a non-halting Mixture-of-Repeats — and
that adaptive token-wise halting **starves deep iterations of gradient**,
underperforming a fixed-depth model at equal inference compute (**23.83 vs
23.19**) because "a good portion of tokens will halt before reaching those
repeats" [[arXiv:2310.10845](https://arxiv.org/abs/2310.10845)].

### 2. BPTT through many unrolled steps: vanishing/exploding gradients

Training unrolled recurrence has "two widely known issues … the vanishing and the
exploding gradient problems," requiring explicit interventions — gradient-norm
clipping for explosions and a soft constraint for vanishing — that a shallow graph
does not need [Pascanu, Mikolov & Bengio 2013,
[arXiv:1211.5063](https://arxiv.org/abs/1211.5063)].

### 3. Activation memory forces truncated BPTT → biased gradients

Because storing activations across many steps is expensive, recurrent-depth
models fall back to **truncated BPTT**, whose "gradient estimate … is biased, so
that it does not benefit from the convergence guarantees from stochastic gradient
theory," and which "displays unreliable performance, and in worst case scenarios,
divergence" [Tallec & Ollivier 2017,
[arXiv:1705.08209](https://arxiv.org/abs/1705.08209)]. The truncation length is a
hard hyperparameter — "choosing the optimal truncation length is difficult"
[Aicher et al. 2019, [arXiv:1905.07473](https://arxiv.org/abs/1905.07473)]. A
recent recurrent-depth LLM confirms the compromise in practice: "we backpropagate
through only the last k iterations … we fix k=8," specifically so that "maximum
activation memory and backward compute is now independent of r" (the sampled
recurrence count) [Geiping et al. 2025,
[arXiv:2502.05171](https://arxiv.org/abs/2502.05171)].

### 4. DEQ / implicit models: fixed-point instability, no existence guarantee

Deep Equilibrium models replace unrolling with direct fixed-point root-finding
(an "infinite-depth" weight-tied network) and get O(1) memory via implicit
differentiation [Bai et al. 2019,
[arXiv:1909.01377](https://arxiv.org/abs/1909.01377)] — but they are "slower,
brittle to architectural choices, and introduce potential instability" [Bai et
al. 2021, [arXiv:2106.14342](https://arxiv.org/abs/2106.14342)]. The instability
is structural: DEQs "suffer from unstable convergence to a solution and lack
guarantees that a solution exists" [Winston & Kolter 2020,
[arXiv:2006.08591](https://arxiv.org/abs/2006.08591)]. Stability is "directly
characterized by the Jacobian matrix at the equilibrium point"
[[locuslab/deq](https://github.com/locuslab/deq)] and must be **engineered** —
via Jacobian regularization that penalizes "the upper bound of [the] Jacobian
spectral radius" [TorchDEQ 2023,
[arXiv:2310.18605](https://arxiv.org/abs/2310.18605)], or a constrained
monotone-operator parameterization to guarantee "the existence of a unique
equilibrium point" [Winston & Kolter 2020].

---

## Why the two-pass routing loop avoids each one

| Traditional failure mode | Two-pass routing loop |
|---|---|
| **Halting/ACT instability** (§1) — τ, KL-weight tuning; gradient-starved deep steps | **No halting decision at all** — the loop count is a fixed constant of 2. Every pass receives full gradient signal. |
| **Vanishing/exploding gradients from deep BPTT** (§2) | Backward graph spans **only 2 passes**; with the gate near zero at init, the model starts essentially identical to a standard Transformer — a well-conditioned landscape. |
| **Truncated-BPTT bias** (§3) | **Fixed depth 2 ⇒ bounded activation memory ⇒ no truncation.** The full (2-pass) graph is differentiated exactly; no `k`-step truncation, no biased gradient, no truncation-length hyperparameter. |
| **DEQ fixed-point instability / Jacobian regularization** (§4) | **No fixed point is ever solved.** It is a plain twice-run forward pass, so there is no equilibrium to converge, no existence question, and nothing to spectrally regularize. |
| **Sequential recurrence bottleneck** | The prior is an **additive logit bias via width-`2D` concatenated Q/K** through ordinary/FlexAttention — **fully parallel over the sequence**, no step-by-step recurrence. |
| **Full recurrent-state feedback (hard to condition)** | A **single low-rank source-per-layer prior** — bounded, well-conditioned extra signal, not a full hidden-state fed back through depth. |

### The key trainability lever: near-zero-gate warm start (a homotopy)

The gate `g = softplus(gate_logit)` is initialized near zero, so **at
initialization the routing term is off and the model is ~identical to a standard
Transformer**. Training therefore starts from the *same* well-conditioned
optimization landscape that ordinary Transformers already train well on, and the
routing prior is *ramped in smoothly* as the gate grows — a homotopy from
"standard Transformer" to "looped model," rather than optimizing the full looped
objective cold. This is the same principle that makes near-zero-initialized gated
residuals train stably elsewhere: ReZero initializes a residual gate to zero for
stable deep training [Bachlechner et al. 2020,
[arXiv:2003.04887](https://arxiv.org/abs/2003.04887)], and Flamingo uses a
`tanh`-gating initialized at 0 so a new cross-attention pathway starts as an
identity and is eased in [Alayrac et al. 2022,
[arXiv:2204.14198](https://arxiv.org/abs/2204.14198)]. None of the deep-recurrence
failure modes above have an analogous free "start as a standard Transformer"
initialization.

---

## The honest tradeoff

This trainability is bought by giving up the main payoff of deep recurrence:
**test-time compute scaling**. Recurrent-depth models can extrapolate to *more*
iterations at inference than they saw in training — Geiping et al. (2025) report
improvements "up to a computation load equivalent to 50 billion parameters" by
unrolling further at test time [[arXiv:2502.05171](https://arxiv.org/abs/2502.05171)].
The two-pass model's loop count is **fixed at 2**, so it gets none of that — and
we verified this directly on our own model: running >2 passes at inference only
degrades quality (see [REPORT.md](../REPORT.md), "Test-time pass scaling"). It
also costs **~2.08× the training FLOPs** of the single-pass baseline and is, by
design, **less expressive than a deep loop**.

So the claim is narrow and specific: *for a fixed, small number of passes, the
two-pass additive-routing design captures a useful slice of "looped" behavior
while remaining as trainable as a standard Transformer* — trading the ceiling of
deep recurrence for the floor of its optimization difficulty.

---

## References

- Dehghani et al. **Universal Transformers**. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
- Graves. **Adaptive Computation Time for RNNs**. [arXiv:1603.08983](https://arxiv.org/abs/1603.08983)
- Mohtashami et al. **CoTFormer**. [arXiv:2310.10845](https://arxiv.org/abs/2310.10845)
- Pascanu, Mikolov & Bengio. **On the difficulty of training RNNs**. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
- Tallec & Ollivier. **Unbiasing Truncated BPTT**. [arXiv:1705.08209](https://arxiv.org/abs/1705.08209)
- Aicher et al. **Adaptively Truncating BPTT**. [arXiv:1905.07473](https://arxiv.org/abs/1905.07473)
- Geiping et al. **Scaling up Test-Time Compute with Latent Reasoning (recurrent depth)**. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
- Bai, Kolter & Koltun. **Deep Equilibrium Models**. [arXiv:1909.01377](https://arxiv.org/abs/1909.01377)
- Bai et al. **Stabilizing Equilibrium Models by Jacobian Regularization**. [arXiv:2106.14342](https://arxiv.org/abs/2106.14342)
- Winston & Kolter. **Monotone Operator Equilibrium Networks**. [arXiv:2006.08591](https://arxiv.org/abs/2006.08591)
- Gao & Kolter. **TorchDEQ**. [arXiv:2310.18605](https://arxiv.org/abs/2310.18605)
- locuslab. **DEQ reference implementation**. [github.com/locuslab/deq](https://github.com/locuslab/deq)
- Bachlechner et al. **ReZero is All You Need**. [arXiv:2003.04887](https://arxiv.org/abs/2003.04887)
- Alayrac et al. **Flamingo** (tanh-gated cross-attention). [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)

*Open questions this note does not settle: whether the near-zero-gate warm start
empirically beats a cold-started two-pass model on matched budget; how much
accuracy is left on the table versus a truncated-BPTT recurrent-depth model at
matched training compute; and whether 3+ fixed passes recover depth benefits
before trainability degrades.*
