# looped_nanochat

Experimental two-pass, shared-weight causal Transformer based on
`karpathy/nanochat` commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

## Idea

Pass 1 runs the ordinary causal Transformer and retains the normalized, RoPE-applied
Q/K features from selected deeper layers. Pass 2 reruns the same Transformer from
the token embeddings. Each pass-2 layer receives a causal attention-logit prior
from one selected pass-1 layer.

For a pass-2 layer, the desired logits are

```text
q2 @ k2.T / sqrt(D) + g * q1 @ k1.T / sqrt(D)
```

The implementation does **not** store either attention matrix. It constructs

```text
q_aug = concat(2^(1/4) q2, 2^(1/4) sqrt(g) q1)
k_aug = concat(2^(1/4) k2, 2^(1/4) sqrt(g) k1)
```

and runs ordinary scaled-dot-product attention at width `2D`. Because its scale is
`1/sqrt(2D)`, this is exactly the additive formula above. V remains width `D`.
The routing gate `g = softplus(gate_logit)` is learned separately for every
pass-2 layer.

Everything is causal and differentiable. The pass-2 language-model loss
backpropagates through the routing Q/K into pass 1 unless `--routing-detach` is
set. A small pass-1 auxiliary LM loss is enabled by default.

## Files changed

- `nanochat/gpt.py`: two-pass model, routing connectivity, loss, FLOP/cache estimates
- `nanochat/flash_attention.py`: SDPA helper allowing Q/K width to differ from V
- `nanochat/engine.py`: paired pass-1/pass-2 KV caches for incremental generation
- `scripts/base_train.py`: routing-loop command-line options
- `tests/test_routing_loop.py`: math, gradient, prefill, and decode tests

## Install over the pinned nanochat revision

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
git checkout 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
unzip -o /path/to/looped_nanochat.zip
uv sync --extra gpu --group dev
uv run pytest tests/test_routing_loop.py
```

The ZIP contains only the replacement/additional files. Alternatively, apply
`looped_nanochat.patch` from the repository root:

```bash
git apply /path/to/looped_nanochat.patch
```

## Small smoke training run

CPU/MPS-sized smoke test:

```bash
uv run python -m scripts.base_train \
  --routing-loop \
  --routing-pattern=progressive \
  --window-pattern=L \
  --depth=4 \
  --max-seq-len=128 \
  --device-batch-size=1 \
  --total-batch-size=128 \
  --num-iterations=20 \
  --eval-tokens=128 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --model-tag=routing-d4-smoke
```

A more meaningful GPU experiment can use the normal nanochat training command and
add:

```bash
--routing-loop \
--routing-pattern=progressive \
--routing-gate-init=0.05 \
--first-pass-loss-weight=0.1 \
--window-pattern=L
```

`--window-pattern=L` is recommended initially. Pass 2 uses PyTorch SDPA because
its Q/K feature width is larger than its V width; alternating sliding windows
require explicit masks in nanochat's SDPA fallback and will be slower.

## Connectivity patterns

`--routing-pattern` controls which pass-1 layer supplies Q/K to each pass-2 layer:

- `last`: final pass-1 layer feeds every pass-2 layer.
- `same`: depth-aligned connections, `m -> m`.
- `reverse`: U-Net-like reverse-depth connections.
- `progressive`: early/middle/late pass-2 thirds receive middle/upper/final pass-1 anchors. This is the default and only retains three source-layer Q/K tensors during training.
- `offset:0.5`: target layer `l` reads approximately `l + L/2`, clipped at the final layer.
- `offset:4`: target layer `l` reads `l + 4`, clipped at the final layer.

`same`, `reverse`, and unquantized offsets can retain routing Q/K from many source
layers during training. `progressive` or `last` is preferable for the first run.

## Loss

During training, the returned scalar is

```text
loss = loss_pass2 + first_pass_loss_weight * loss_pass1
```

In evaluation mode, the model returns only `loss_pass2`, so nanochat validation BPB remains directly comparable to an ordinary model.

The auxiliary loss keeps pass 1 independently predictive. Set
`--first-pass-loss-weight=0` to train only through the final pass. Set
`--routing-detach` for an ablation where pass 2 cannot train pass-1 routing
features through the feedback path.

## Inference

The existing `Engine` interface works unchanged. It detects `routing_loop=True`
and allocates two ordinary KV caches. Per generated token:

1. increment pass 1 and collect the current token's selected Q/K;
2. increment pass 2 using the selected pass-1 K cache as the routing prior;
3. sample from pass-2 logits.

The model has one parameter set but executes it twice. Transformer/MLP compute is
therefore close to 2x. Pass-2 attention uses doubled Q/K width, so its attention
portion is about 2x a normal attention operation. KV-cache storage is exactly 2x
because the two passes keep separate normal-width K/V caches; routing keys are
read directly from pass 1 rather than copied into a third cache.

## Recommended ablations

Compare at matched training tokens and, separately, matched FLOPs:

1. ordinary one-pass nanochat;
2. two shared-weight passes with routing gates fixed to zero;
3. routing loop with `last`;
4. routing loop with `progressive`;
5. routing loop with `--routing-detach`;
6. an ordinary deeper model with similar executed-layer FLOPs.

The key test is (2) versus (3)/(4). That isolates whether pass-1 routing improves
pass 2 beyond merely executing the model twice.

## Current limitations

- This is a research prototype, not an optimized long-context kernel.
- No hard top-k pruning is included. The current soft prior is fully differentiable.
- Pass-2 routing attention uses PyTorch SDPA rather than nanochat's external FA3
  wrapper. A custom fused kernel could concatenate pass-1/pass-2 Q/K logically
  without transient concatenation.
- Checkpoints trained before this patch load as ordinary `routing_loop=False`
  models. A routing-loop checkpoint is not architecture-compatible with unpatched
  nanochat.
