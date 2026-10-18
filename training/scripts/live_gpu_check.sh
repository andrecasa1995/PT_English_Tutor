#!/usr/bin/env bash
set -e
LOG=/mnt/c/Users/andre/Documents/PT_English_Tutor/training/training_live.log
printf '[step 113] GPU check started\n' >> "$LOG"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader >> "$LOG"
printf '[step 113] GPU check finished. No training started.\n' >> "$LOG"
