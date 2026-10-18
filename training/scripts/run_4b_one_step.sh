#!/usr/bin/env bash
set -o pipefail
LOG=/mnt/c/Users/andre/Documents/PT_English_Tutor/training/training_live.log
printf '\n[step 120] Launching one real QLoRA 4B training step.\n' >> "$LOG"
/home/andre/projects/PT_English_Tutor/.venv/bin/python -u /mnt/c/Users/andre/Documents/PT_English_Tutor/training/scripts/train_4b_one_step.py >> "$LOG" 2>&1
code=$?
printf '[step 120] Exit code: %s\n' "$code" >> "$LOG"
exit "$code"
