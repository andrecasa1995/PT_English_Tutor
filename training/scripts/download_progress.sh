#!/usr/bin/env bash
LOG=/mnt/c/Users/andre/Documents/PT_English_Tutor/training/training_live.log
CACHE=/home/andre/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B
printf '\n[monitor] Download monitor activated (20-second updates)\n' >> "$LOG"
for i in $(seq 1 90); do
  if ! pgrep -f '^/home/andre/projects/PT_English_Tutor/.venv/bin/python -u /mnt/c/Users/andre/Documents/PT_English_Tutor/training/scripts/load_4b_smoke.py$' >/dev/null; then
    printf '[monitor] Model loading process ended. Check results above.\n' >> "$LOG"
    exit 0
  fi
  bytes=$(du -sb "$CACHE" 2>/dev/null | cut -f1)
  mib=$(( ${bytes:-0} / 1048576 ))
  printf '[monitor] Qwen3.5-4B cached directory: %s MiB (%s)\n' "$mib" "$(date +%H:%M:%S)" >> "$LOG"
  sleep 20
done
printf '[monitor] Automatic download size monitor finished after 30 minutes.\n' >> "$LOG"
