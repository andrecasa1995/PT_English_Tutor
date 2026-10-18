# Writing engine (prototype)

This is a local Python/FastAPI API that forwards a short writing sample to Ollama.

## Setup (PowerShell)

From `services/writing-engine`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Visit http://127.0.0.1:8765/docs to inspect the local API.

The `POST /v1/correct` endpoint only suggests text; it never writes to disk or edits Obsidian notes.

> Prototype only: do not send passwords or sensitive text. More precise, span-based suggestions, validation, authentication and privacy checks are planned before connecting editor clients.
