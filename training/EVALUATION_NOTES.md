# Qwen3.5-4B — experimental LoRA evaluation

## Step 127: frozen 8-example holdout

- Baseline Qwen3.5-4B, NF4: **0/8** exact edit triplets; **6/8** structurally valid JSON `{edits: [...]}`.
- Two-step LoRA adapter: **2/8** exact edit triplets; **7/8** structurally valid JSON.
- Exact metric compares the complete set of `(original, replacement, category)` for each example. Explanation wording is not scored. JSON validity checks only top-level object with an `edits` list, not full explanation quality or API schema.
- These are *tiny, synthetic, held-out* tests with deterministic decoding. Treat differences as exploratory, not statistical evidence of generalization.
- `eval-01`: both model variants incorrectly changed `explain` -> `explained` instead of `explains`; adapter fixed category capitalization.
- The held-out examples were **not** used for training; do not use their detailed target labels to generate new training examples or to tune prompts.
- Full per-sample generations saved at `training/outputs/heldout_comparison_8samples.json`.
- Adapter checkpoint at `training/outputs/experimental_qwen35_4b_lora_2steps` is for experiments only; **do not deploy** it to Obsidian/Ollama.

## Next work

1. Expand **training-only** examples from 12 to a reviewed, diverse collection with independent writing scenarios and correct/no-edit sentences. Keep the frozen holdout untouched.
2. Strengthen validators (span boundaries, field schema, duplicates across train/holdout).
3. Build a repeatable, resumable training pipeline with checkpoints, reproducible hyperparameters, explicit train-only data loading, and resource limits for an 8 GB GPU.
4. Evaluate a new model only after its training methodology and dataset are frozen; use a fresh genuinely independent test set for final claims.
