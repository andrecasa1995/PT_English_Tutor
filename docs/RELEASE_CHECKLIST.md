# PT English Tutor — Release readiness

## Passed locally (2026-10-18)
- Obsidian TypeScript project compiles.
- Python writing-engine unit test suite: 21 tests pass.
- Release ZIP contains only main.js, manifest.json, and styles.css.
- npm dependency audit during package build reported zero vulnerabilities.
- GitHub Actions workflow exists, but has not yet been run on GitHub.

## Manual acceptance required before public distribution
- Verify corrections always require the Apply correction button, including repeat mistakes.
- Confirm Ignore once, Always ignore, and Add to dictionary never change note text.
- Confirm code blocks, YAML frontmatter, and Markdown links are unaffected.
- Check keyboard-only operation, focus order and screen-reader labels in all modals.
- Test engine unavailable, model unloaded, delayed response and edited-note race conditions.
- Check history is disabled initially and no full note sentences are retained in data.json.
- Test Remove and Clear history, and ensure the practice list remains usable.
- Verify installation in a completely clean Obsidian vault and Windows account.
- Set author metadata and select an appropriate LICENSE before publishing.
- Confirm GitHub Actions passes on a remote push and avoid committing user notes.

## Release decision
Do not describe version 0.1.0 as publicly release-ready until all manual checks pass.
