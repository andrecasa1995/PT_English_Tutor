# PT English Tutor — local GPU experiments

## Current environment
Ubuntu 24.04 via WSL2, RTX 4060 Laptop (8 GiB), Python 3.12.3, torch 2.11.0+cu128, Transformers 5.20.1, PEFT 0.21.2, bitsandbytes 0.50.2.

## Verified results
- Qwen3-0.6B NF4 + LoRA: one successful optimizer step (initial compatibility test).
- Qwen3.5-4B NF4 + LoRA rank 8: one optimizer step successful, peak PyTorch reserved 5.402 GiB.
- Two-step Qwen3.5-4B LoRA adapter saved and reloaded at `outputs/experimental_qwen35_4b_lora_2steps` (64 tensors, 6,299,904 bytes).
- Eight-sample frozen heldout evaluation on the original model vs two-step adapter: baseline 0/8 exact edits, 6/8 valid JSON; adapter 2/8 exact edits, 7/8 valid JSON. Sample too small for broad claims. Details: `EVALUATION_NOTES.md`.
- Created 38 synthetic training candidates; excluded one ambiguous candidate `tarefa -> assignment`. Curated collection contains 37 candidates, **not yet independently human reviewed**.
- Experimental SFT JSONL has 49 rows (12 seed + 37 curated), and excludes the eight held-out examples.
- Qwen3.5-4B NF4 + LoRA eight-step SFT on 8 selected rows completed successfully in 19.88 s, peak PyTorch reserved 5.420 GiB. Loss per different sample is not a measure of improvement.
- Eight-step adapter saved at `outputs/experimental_qwen35_4b_curated_8steps`; verified 64 finite tensors, 6,299,904 bytes, plus metadata and config.
- Writing-engine unit tests: 11/11 pass under WSL2.

## Remaining limitations
- Both adapters are **experimental, NOT deployment-ready**. Do not replace Ollama or Obsidian model yet.
- Synthetic examples must undergo careful language and explanation review; original frozen eight held-out examples have been used for exploratory evaluation and should not be used for hyperparameter selection.
- Obtain a fresh independent evaluation set before making final model quality claims.
- No Git commit, push, or merge performed.
