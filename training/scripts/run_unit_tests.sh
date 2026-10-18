#!/usr/bin/env bash
set -o pipefail
ROOT=/mnt/c/Users/andre/Documents/PT_English_Tutor
cd "$ROOT/services/writing-engine" || exit 1
PYTHONPATH="$PWD" /home/andre/projects/PT_English_Tutor/.venv/bin/python -m unittest discover -s tests -v 2>&1 | tee -a "$ROOT/training/training_live.log"
