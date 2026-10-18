# Implementation Plan

## Goal
Build a Grammarly-like Obsidian writing assistant that detects Portuguese words in otherwise English writing, proposes technical/contextual translations, flags grammar and spelling issues, and provides brief explanations in Brazilian Portuguese.

## Target environment
- Windows
- NVIDIA RTX 4060 Laptop GPU (8 GB dedicated VRAM), 32 GB RAM
- Ollama installed locally
- Obsidian and VS Code

## Milestone 1: Local model evaluation
- Confirm Ollama works and GPU acceleration is active.
- Benchmark Qwen3.5 4B and Qwen3.5 9B if available and suitable; keep an alternative small model ready.
- Measure startup, tokens per second, memory usage, and corrections on an evaluation set.
- Select an interactive model based on measured latency and correction accuracy.

## Milestone 2: Local engine
- Python + FastAPI.
- `POST /v1/analyze` accepts bounded text and context.
- Return validated text spans, suggestions, categories, and Portuguese explanations.
- Separate deterministic span detection from LLM recommendations.
- Reject stale or malformed responses and avoid logging private note text.
- Bind to loopback only (127.0.0.1).

## Milestone 3: Obsidian plugin
- TypeScript + CodeMirror 6.
- Debounce edits; skip code, markdown syntax, frontmatter and existing wikilinks.
- Display nonblocking underline decorations, clickable corrections, and explanations.
- Apply changes only after user approval; preserve undo history.
- Gracefully handle Ollama/service downtime.

## Milestone 4: Learning features
- Optional local vocabulary/history storage.
- Personal dictionary and ignore controls.
- Review recurring mistakes and AI-domain expressions.

## Milestone 5: Quality + GitHub readiness
- Unit/integration tests, format/lint commands and GitHub Actions CI.
- Documentation, screenshots and release packaging.
- Choose a license before making repository public.
- No credentials, private notes, models or databases in Git.

## Acceptance criteria (initial targets)
- No silent writing changes.
- Markdown and code protected from modifications.
- Normal typing stays responsive even when inference is slow.
- Correctly identify Portuguese/English mixed text in representative AI-study examples.
- Benchmarks and test results documented before claiming performance targets.

## Deployment notes
GitHub hosts source code and downloadable release packages. Ollama inference and the local service still run on each user's computer; publishing to GitHub does not deploy a cloud AI service.
