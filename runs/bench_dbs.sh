#!/bin/bash
set -uo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PATH="$HOME/.local/bin:$PATH"
cd /root/looped_nanochat
source .venv/bin/activate
source /root/looped_nanochat/.env.secrets 2>/dev/null
for DBS in 8 16 24 32; do
  echo "########## DBS=$DBS ##########"
  # total_batch_size must be multiple of DBS*seq_len*ngpu = DBS*2048*4.
  TBS=$(( DBS * 2048 * 4 ))
  timeout 420 torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=24 --device-batch-size=$DBS --total-batch-size=$TBS --num-iterations=6 \
    --core-metric-every=-1 --eval-tokens=2048 --fp8 \
    --routing-loop --routing-pattern=progressive --model-tag=bench --run=dummy 2>&1 \
    | grep -E "^step 0000[3-5]|out of memory|OutOfMemory|Error" | tail -4
  rm -rf /root/.cache/nanochat/base_checkpoints/bench
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1
  echo ""
done
echo "BENCH DONE"
