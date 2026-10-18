# Synthetic training candidates — review queue

- Step 129: generated 38 additional *candidate* items (26 corrections plus 12 correct/no-edit examples) in `datasets/additional_synthetic_candidates.json`.
- Step 130: validator confirmed no collisions with 12 training seeds and 8 frozen evaluation examples. Candidate file is **not** included by `prepare_sft.py` and is **not yet part of model training**.
- Step 131: all 11 writing-engine unit tests pass under Ubuntu after making backend test dependencies available. Use `bash training/scripts/run_unit_tests.sh` from any directory in WSL.
- Human/pedagogical review is still required. Example `This results looks promising.` has **two** errors, but its synthetic label corrects only one; it must be fixed or excluded before use. `tarefa` -> `assignment` is context-sensitive and should be reviewed. Inspect every candidate for missed edits, ambiguous translation, awkward English, and explanation accuracy.
- Do not edit or train on the frozen `heldout_eval.json`. The initial two-step LoRA adapter remains experimental and is not suitable for deployment.
