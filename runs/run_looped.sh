#!/bin/bash
# Full looped run, robust against transient HF API errors. Reuses cached
# pretraining data + tokenizer from the baseline run.
set -uo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PATH="$HOME/.local/bin:$PATH"
cd /root/looped_nanochat
source .venv/bin/activate
source /root/looped_nanochat/.env.secrets
export WANDB_RUN=looped
MODEL_TAG=looped-d24
NGPU=4
DBS=8   # device batch size (looped uses more VRAM; grad-accum auto-compensates)

# Prewarm SFT datasets (single process, retries) so a transient HF 400 cannot
# kill the distributed SFT job.
python - <<'PY'
import time, sys
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
specs = [lambda: SmolTalk(split="train"), lambda: SmolTalk(split="test"),
         lambda: MMLU(subset="all", split="auxiliary_train"),
         lambda: MMLU(subset="all", split="test", stop=5200),
         lambda: GSM8K(subset="main", split="train"),
         lambda: GSM8K(subset="main", split="test", stop=420)]
for f in specs:
    for attempt in range(8):
        try: f(); break
        except Exception as e:
            print(f"prewarm retry {attempt}: {e}", flush=True); time.sleep(10)
    else:
        print("PREWARM FAILED", file=sys.stderr); sys.exit(1)
print("datasets prewarmed OK", flush=True)
PY
[ $? -ne 0 ] && { echo "prewarm failed"; exit 1; }

# 1) Pretraining (routing loop ON). Retry once on failure.
for attempt in 1 2; do
  torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- \
      --depth=24 --target-param-data-ratio=8 --device-batch-size=$DBS --fp8 \
      --routing-loop --routing-pattern=progressive --routing-gate-init=0.05 \
      --model-tag=$MODEL_TAG --run=$WANDB_RUN && break
  echo "base_train attempt $attempt failed; retrying in 20s"; sleep 20
done

torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_eval -- \
    --model-tag=$MODEL_TAG --device-batch-size=$DBS

# 2) SFT + eval
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
    --model-tag=$MODEL_TAG --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_eval -- \
    -g $MODEL_TAG -i sft

echo "LOOPED RUN COMPLETE"
