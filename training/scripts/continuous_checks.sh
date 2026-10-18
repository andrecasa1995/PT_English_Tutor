#!/usr/bin/env bash
# Safe recurring checks. Stop with: touch training/STOP_CONTINUOUS_CHECKS
set -u
ROOT=/mnt/c/Users/andre/Documents/PT_English_Tutor
PY=/home/andre/projects/PT_English_Tutor/.venv/bin/python
LOG="$ROOT/training/training_live.log"
STOP="$ROOT/training/STOP_CONTINUOUS_CHECKS"
INTERVAL_SECONDS=1800
printf '[workflow] Continuous checks started; interval=%s seconds. Stop file: %s\n' "$INTERVAL_SECONDS" "$STOP" >> "$LOG"
while [[ ! -e "$STOP" ]]; do
  "$PY" "$ROOT/training/scripts/local_workflow_runner.py" >> "$LOG" 2>&1
  result=$?
  if [[ $result -ne 0 ]]; then
    printf '[workflow] Validation failed (exit %s); monitor stopping.\n' "$result" >> "$LOG"
    exit "$result"
  fi
  sleep "$INTERVAL_SECONDS" & wait $!
done
printf '[workflow] Stop file detected; monitor exited.\n' >> "$LOG"
