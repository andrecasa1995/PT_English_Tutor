"""Local-only prototype API for PT_English_Tutor.

No note content is persisted and no corrections are automatically applied.
"""
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = os.getenv("TUTOR_MODEL", "qwen3.5:4b")

app = FastAPI(title="PT English Tutor", version="0.1.0")


class CorrectionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class CorrectionResponse(BaseModel):
    original: str
    suggestion: str
    model: str
    requires_approval: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL}


@app.post("/v1/correct", response_model=CorrectionResponse)
async def correct(request: CorrectionRequest) -> CorrectionResponse:
    """Proof of concept; structured, span-level corrections come later."""
    instruction = (
        "You are an American English tutor for a Brazilian Portuguese speaker. "
        "Translate any Portuguese words into contextual English and correct spelling "
        "and grammar. Preserve the author's meaning, grammatical number, and style. "
        "Make the fewest possible edits. Output ONLY the corrected sentence."
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "system": instruction,
                    "prompt": request.text,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 180},
                },
            )
            result.raise_for_status()
            suggestion = result.json().get("response", "").strip()
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Local Ollama is unavailable") from error

    if not suggestion:
        raise HTTPException(status_code=502, detail="Model returned an empty response")
    return CorrectionResponse(
        original=request.text, suggestion=suggestion, model=MODEL
    )


from .analysis import router as analysis_router

app.include_router(analysis_router)
