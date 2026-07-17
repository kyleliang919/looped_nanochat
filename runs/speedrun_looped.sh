#!/bin/bash

# LOOPED nanochat speedrun (two-pass causal routing loop), adapted for 4xH100.
# Same pipeline and same total_batch_size / tokens as the baseline, so the two
# runs are directly comparable; the only difference is the routing-loop model.
#
# Notes:
#   - Routing flags are passed to base_train ONLY. The checkpoint stores the
#     full model config (asdict), so chat_sft / *_eval rebuild the looped model
#     automatically -- do NOT re-pass routing flags downstream.
#   - The looped model costs ~2x FLOPs/token (two passes; pass-2 attention at
#     2x Q/K width), so expect ~4.5-5h on 4xH100.
#   - VRAM: pass-2 runs wider, so we start at --device-batch-size=8 to be safe.
#     If you OOM, drop to 4 (grad-accum auto-compensates, trajectory unchanged).
#     If it fits comfortably, bump back to 16 to match the baseline's speed.
#
# Launch:
#   bash runs/speedrun_looped.sh
# Or in a screen session (recommended):
#   screen -L -Logfile runs/speedrun_looped.log -S looped bash runs/speedrun_looped.sh
# With wandb:
#   WANDB_RUN=looped bash runs/speedrun_looped.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

NGPU="${NGPU:-4}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
# Distinct checkpoint tag so this run does not collide with the baseline run
# (both are depth-24, which would otherwise share the default tag "d24").
MODEL_TAG="${MODEL_TAG:-looped-d24}"

# Routing-loop hyperparameters (defaults match scripts/base_train.py).
ROUTING_PATTERN="${ROUTING_PATTERN:-progressive}"
ROUTING_GATE_INIT="${ROUTING_GATE_INIT:-0.05}"

# -----------------------------------------------------------------------------
# Python venv setup with uv (reuses .venv if the baseline run already built it)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb (dummy = disabled)
if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tokenizer + data
# Skipped automatically if already present from a prior run (dataset shards and
# the trained tokenizer live under $NANOCHAT_BASE_DIR and are reused).
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) -- routing loop ENABLED here.
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 --device-batch-size=$DEVICE_BATCH_SIZE --fp8 \
    --routing-loop --routing-pattern=$ROUTING_PATTERN --routing-gate-init=$ROUTING_GATE_INIT \
    --model-tag=$MODEL_TAG --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_eval -- \
    --model-tag=$MODEL_TAG --device-batch-size=$DEVICE_BATCH_SIZE

# -----------------------------------------------------------------------------
# SFT + eval (routing config is inherited from the checkpoint; no flags needed)
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
    --model-tag=$MODEL_TAG --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_eval -- \
    -g $MODEL_TAG -i sft

echo "Looped speedrun complete (model tag: $MODEL_TAG)."
echo "Chat with: python -m scripts.chat_cli -g $MODEL_TAG -i sft -p 'Why is the sky blue?'"
