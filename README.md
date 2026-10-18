# PT English Tutor

An open-source, local-first English writing assistant for Brazilian Portuguese speakers, with an Obsidian desktop plugin and a Python/Ollama backend.

## What it does

- Underlines Portuguese-to-English translation suggestions in blue and English spelling/grammar suggestions in red.
- Displays contextual correction suggestions and explanations in Brazilian Portuguese.
- **Never edits a note automatically.** Each correction requires an explicit click on **Apply correction**.
- Provides a personal dictionary, ignored words and optional local history of corrections the user approved.
- Offers shuffled practice sessions mixing translations, spelling and grammar exercises, without storing note sentences.
- Skips fenced code blocks and YAML frontmatter during analysis.

## Getting started

1. Install Ollama and obtain the model described in `services/writing-engine/README.md`.
2. Set up the local FastAPI service according to that README. It listens on `127.0.0.1:8765`.
3. In `apps/obsidian-plugin`, run `npm ci` and `npm run build`.
4. Install `manifest.json`, `main.js` and `styles.css` into your vault's `.obsidian/plugins/pt-english-tutor/` directory, enable community plugins, and enable PT English Tutor.
5. Recommended: turn off Obsidian's built-in `Settings > Editor > Spellcheck` to avoid duplicate underlines.

See [plugin instructions](apps/obsidian-plugin/README.md) for additional details.

## Quality checks

Run `npm test` inside `apps/obsidian-plugin` and `python -m unittest discover -s tests -q` inside `services/writing-engine` (with dependencies installed). GitHub Actions runs both checks on push and pull requests.

Release status: **beta 0.1.0**. Model suggestions are not infallible; examine each proposal before applying it. The Obsidian plugin is desktop-only and requires locally running Ollama and Python services. **Installing the plugin does not install or start the local engine**; follow the backend setup instructions first. The plugin manifest and version compatibility map are also available at the repository root for Community Directory review.

## Repository layout

- `apps/obsidian-plugin/`: Obsidian plugin and regression checks.
- `services/writing-engine/`: FastAPI service and unit tests.
- `tests/`: evaluation and diagnostic scripts.
- `training/`: experimental synthetic examples and training utilities; no model weights are included.
- `docs/`: implementation plan and release checklist.

## Privacy

Note fragments are processed locally. The optional approved-correction history is stored inside the user's vault and is not uploaded to the repository. Never commit personal vaults, plugin `data.json`, tokens, model weights or other local artifacts.

## License

MIT License — see [LICENSE](LICENSE). Ollama and model weights are governed by their respective licenses.
