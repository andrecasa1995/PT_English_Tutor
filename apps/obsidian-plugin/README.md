# PT English Tutor — Obsidian

Local-first bilingual writing assistant for Portuguese speakers composing in English. The plugin connects to a FastAPI service at http://127.0.0.1:8765 backed by local Ollama inference.

**No correction is ever applied automatically**, including corrections previously approved. The user must click Apply correction for each individual change.

## Install

1. Install Ollama, download the selected model, and start the local writing engine (see services/writing-engine/README.md).
2. Run npm ci and npm run build in apps/obsidian-plugin.
3. Copy manifest.json, main.js and styles.css to the vault's .obsidian/plugins/pt-english-tutor/ directory.
4. Reload Obsidian and enable the community plugin.

Alternatively, extract the ZIP created by scripts/package-release.ps1 to that plugin directory.

### Recommended Obsidian editor setting

Disable Obsidian's built-in **Spellcheck** under **Settings → Editor → Behavior → Spellcheck** when using PT English Tutor. Otherwise the native spellchecker and PT English Tutor may underline the same text twice or show conflicting spelling suggestions. This is optional: it changes only the built-in spellcheck setting and does **not** disable PT English Tutor or grant permission to apply corrections automatically.

## Features

- Blue underlines for Portuguese translations, red for English spelling/grammar.
- Click a marked span to view a suggestion and request a Brazilian Portuguese explanation.
- Apply correction modifies the note only after explicit approval; Ignore once and Always ignore do not modify text.
- Persistent local ignored-word list and personal preferred-corrections dictionary.
- Optional approved-correction history (off by default): individual terms, categories, counters and dates. Never stores entire notes.
- Practice alternates and shuffles translation, spelling and grammar cards. Grammar uses synthetic example sentences rather than saved note content.
- Scans pasted notes up to 40 eligible prose lines and excludes Markdown fenced code and YAML frontmatter.

## Verify

From the command palette use PT English Tutor: Check local engine connection. This contacts the local health endpoint without sending note content.

## Privacy and limitations

The backend is bound to the local loopback interface. Note fragments are processed by the user's local model. Local plugin settings live in the vault's private plugin data file; never publish this file or private notes. LLM suggestions may be incorrect, so review each before applying. The grammar practice sentence templates cover common verbs and need expansion for unusual forms.

## Release

From the repository root run:

    powershell -ExecutionPolicy Bypass -File scripts/package-release.ps1

This builds and validates a versioned ZIP in release/ containing exactly manifest.json, main.js, and styles.css.

## Open-source license

PT English Tutor is distributed under the MIT License (see the repository-root LICENSE file). The plugin and backend require locally installed dependencies; model weights are separately licensed by their providers and are not bundled. Review privacy, accessibility, and release acceptance tests before public distribution.
