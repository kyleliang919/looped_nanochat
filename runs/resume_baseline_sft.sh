#!/bin/bash
# Resume the baseline run from the SFT stage (base model already trained).
set -uo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PATH="$HOME/.local/bin:$PATH"
cd /root/looped_nanochat
source .venv/bin/activate
source /root/looped_nanochat/.env.secrets
export WANDB_RUN=baseline
MODEL_TAG=baseline-d24
NGPU=4

# Pre-warm the SFT datasets on a single process with retries, so a transient
# HuggingFace API hiccup (the HTTP 400 we saw) does not kill the distributed job.
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
        try:
            f(); break
        except Exception as e:
            print(f"prewarm retry {attempt}: {e}", flush=True); time.sleep(10)
    else:
        print("PREWARM FAILED", file=sys.stderr); sys.exit(1)
print("SFT datasets prewarmed OK", flush=True)
PY
if [ $? -ne 0 ]; then echo "dataset prewarm failed, aborting"; exit 1; fi

torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- --model-tag=$MODEL_TAG --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_eval -- -g $MODEL_TAG -i sft
echo "BASELINE SFT+EVAL COMPLETE"
