#!/usr/bin/env bash
set -o pipefail
cd /home/andre/projects/PT_English_Tutor || exit 1
export HF_HOME=/home/andre/.cache/huggingface
export PYTHONUNBUFFERED=1
LOG=/mnt/c/Users/andre/Documents/PT_English_Tutor/training/training_live.log
printf '\n[step 117] Starting Qwen3.5-4B loading trial; weights may take a while to download.\n' >> "$LOG"
/home/andre/projects/PT_English_Tutor/.venv/bin/python -u /mnt/c/Users/andre/Documents/PT_English_Tutor/training/scripts/load_4b_smoke.py >> "$LOG" 2>&1
code=$?
printf '[step 117] Process exit code: %s\n' "$code" >> "$LOG"
exit "$code"
