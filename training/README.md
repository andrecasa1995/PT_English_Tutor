# PT English Tutor — Fine-tuning preparation

This is a small, synthetic, human-reviewable seed dataset, **not** enough to train a model or claim accuracy. No user notes were used.

Each example includes `id`, `group`, `text`, and `edits`. An edit includes `original`, `replacement`, `category`, and `explanation_pt`.

Before training: review each example, expand with diverse inputs and correct sentences, freeze a held-out evaluation set, and split by source `group` to avoid leakage. Keep existing `tests/spelling_cases.json` out of training because it was used to tune prompts.

Fine-tuning with QLoRA on an 8 GB GPU requires a separate compatibility/memory trial. Nothing here changes Ollama or Obsidian.
