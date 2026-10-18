#!/usr/bin/env bash
cd /mnt/c/Users/andre/Documents/PT_English_Tutor || exit 1
printf '\nPT_English_Tutor - LIVE TRAINING LOG\n'
printf 'Monitoring: training/training_live.log\nPress Ctrl+C to stop monitoring only.\n\n'
touch training/training_live.log
tail -n 30 -F training/training_live.log
