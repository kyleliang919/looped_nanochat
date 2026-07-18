#!/bin/bash
# Resume looped run from SFT (base model already trained + saved). Skips the
# redundant standalone base_eval (in-loop CORE=0.2783 already captured).
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

# Prewarm SFT datasets with retries (guards against transient HF 400s).
python - <<'PY'
import time,sys
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
specs=[lambda:SmolTalk(split="train"),lambda:SmolTalk(split="test"),
       lambda:MMLU(subset="all",split="auxiliary_train"),
       lambda:MMLU(subset="all",split="test",stop=5200),
       lambda:GSM8K(subset="main",split="train"),
       lambda:GSM8K(subset="main",split="test",stop=420)]
for f in specs:
    for a in range(8):
        try: f(); break
        except Exception as e: print("retry",a,e,flush=True); time.sleep(10)
    else: print("PREWARM FAILED",file=sys.stderr); sys.exit(1)
print("prewarmed OK",flush=True)
PY
[ $? -ne 0 ] && { echo "prewarm failed"; exit 1; }

torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- --model-tag=$MODEL_TAG --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_eval -- -g $MODEL_TAG -i sft
echo "LOOPED SFT+EVAL COMPLETE"
