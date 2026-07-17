#!/bin/bash

# Baseline (NON-looped) nanochat speedrun, adapted for a 4xH100 node.
# Trains a GPT-2 grade LLM (pretraining + SFT + eval), no routing loop.
#
# Differences from the stock 8xH100 speedrun:
#   - --nproc_per_node=4 (gradient accumulation is auto-doubled to keep the
#     same total_batch_size in tokens; the training trajectory is unchanged).
#
# Launch:
#   bash runs/speedrun_baseline.sh
# Or in a screen session (recommended, run is ~2.5-3h on 4xH100):
#   screen -L -Logfile runs/speedrun_baseline.log -S baseline bash runs/speedrun_baseline.sh
# With wandb:
#   WANDB_RUN=baseline bash runs/speedrun_baseline.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

NGPU="${NGPU:-4}"
# Distinct checkpoint tag so this run does not collide with the looped run
# (both are depth-24, which would otherwise share the default tag "d24").
MODEL_TAG="${MODEL_TAG:-baseline-d24}"

# -----------------------------------------------------------------------------
# Python venv setup with uv
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
# Download ~2B chars (8 shards) for tokenizer training, then the rest in the bg.
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model, slightly undertrained (data:param ratio 8) to beat GPT-2.
torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 \
    --model-tag=$MODEL_TAG --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_eval -- \
    --model-tag=$MODEL_TAG --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT + eval
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
    --model-tag=$MODEL_TAG --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_eval -- \
    -g $MODEL_TAG -i sft

echo "Baseline speedrun complete (model tag: $MODEL_TAG)."
echo "Chat with: python -m scripts.chat_cli -g $MODEL_TAG -i sft -p 'Why is the sky blue?'"
