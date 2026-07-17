#!/bin/bash
# Wait for the baseline sft screen session to finish, then launch the looped run.
echo "$(date) waiting for baseline sft screen to end..."
while screen -ls 2>/dev/null | grep -q "\.sft"; do sleep 30; done
echo "$(date) baseline sft screen gone; launching looped run"
cd /root/looped_nanochat
screen -dmS looped -L -Logfile /root/looped_nanochat/runs/looped.log bash runs/run_looped.sh
echo "$(date) looped run launched"
