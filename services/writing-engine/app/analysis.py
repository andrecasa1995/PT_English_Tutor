"""Structured two-pass analysis against local Ollama."""
import asyncio
import json
import os
import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .markdown_guard import overlaps_protected

router = APIRouter()
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = os.getenv("TUTOR_MODEL", "qwen3.5:4b")
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class Suggestion(BaseModel):
    start: int
    end: int
    original: str
    replacement: str
    category: str
    explanation_pt: str


class AnalyzeResponse(BaseModel):
    original: str
    suggestions: list[Suggestion]
    requires_approval: bool = True


SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "replacement": {"type": "string"},
                    "category": {"type": "string", "enum": ["translation", "grammar", "spelling", "style"]},
                    "explanation_pt": {"type": "string"},
                },
                "required": ["original", "replacement", "category", "explanation_pt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}

TRANSLATE_PROMPT = (
    "You are a bilingual Brazilian Portuguese and American English writing tutor. "
    "Analyze English writing containing occasional Portuguese words. "
    "Identify ALL Portuguese words and phrases embedded in otherwise English text, "
    "including words at the END of the sentence and after English prepositions. "
    "Inspect each word independently, not only the first foreign term. "
    "Translate each into natural contextual American English, preserving technical AI terms. "
    "If the entire sentence is Portuguese, do not translate it. "
    "Return separate minimal edits, category=translation. "
    "Each original must be an EXACT substring of the input. "
    "Explain each translation briefly and accurately in Brazilian Portuguese. "
    "Never edit code, Markdown links, or identifiers. Return JSON edits; no issues means []."
)

GRAMMAR_PROMPT = (
    "You are a conservative American English proofreader. Correct only demonstrable errors; "
    "do not invent mistakes in already grammatical sentences. Do not suggest has->have "
    "when a singular subject such as This neural network already agrees with has. "
    "Review every adjective and noun modifier: in the phrase multiple layers, "
    "multiple is an adjective and must NOT become multiples. "
    "Review ALL English verbs, spelling and grammar in the complete sentence. "
    "Pay special attention to coordinated verbs "
    "that share the same subject, including verbs after objects or foreign words. "
    "For example: 'The algorithm use data and generate predictions' requires "
    "use->uses AND generate->generates, because BOTH verbs refer to algorithm. "
    "Portuguese words may occur in the sentence; leave them unchanged in this pass. "
    "Make only the smallest separate edits to English words (category=grammar or spelling). "
    "Each original must be an EXACT substring of input. Explain grammar in Brazilian "
    "Portuguese, correctly identifying the grammatical subject. Preserve technical "
    "terminology, code and Markdown; return JSON edits; no issues means []."
)


def validate_edits(text: str, edits: list[dict]) -> list[Suggestion]:
    """Retain only uniquely locatable, exact, nonoverlapping suggestions."""
    valid: list[Suggestion] = []
    for edit in edits[:20]:
        if not isinstance(edit, dict):
            continue
        original, replacement = edit.get("original"), edit.get("replacement")
        category, explanation = edit.get("category"), edit.get("explanation_pt")
        if not all(isinstance(v, str) for v in (original, replacement, category, explanation)):
            continue
        if not original or original == replacement or category not in {"translation", "grammar", "spelling", "style"}:
            continue
        start = text.find(original)
        if start < 0 or text.find(original, start + 1) != -1:
            continue
        end = start + len(original)
        if overlaps_protected(text, start, end):
            continue
        if any(start < prior.end and end > prior.start for prior in valid):
            continue
        valid.append(Suggestion(start=start, end=end, original=original,
                                replacement=replacement, category=category,
                                explanation_pt=explanation))
    return sorted(valid, key=lambda item: item.start)


def merge_edits(text: str, translations: list[dict], grammar: list[dict]) -> list[Suggestion]:
    """Translation takes precedence for overlaps; grammar fills remaining spans."""
    return validate_edits(text, translations + grammar)


async def _analyze_pass(client: httpx.AsyncClient, text: str, instructions: str) -> list[dict]:
    response = await client.post(OLLAMA_URL, json={
        "model": MODEL, "system": instructions, "prompt": text, "stream": False,
        "think": False, "format": SCHEMA,
        "options": {"temperature": 0, "num_predict": 1024},
    })
    response.raise_for_status()
    edits = json.loads(response.json()["response"])["edits"]
    if not isinstance(edits, list):
        raise ValueError("edits must be an array")
    return edits


@router.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            # One local model, two independent prompts. Concurrent calls are
            # intentionally avoided on an 8 GB GPU until benchmarked.
            translations = await _analyze_pass(client, request.text, TRANSLATE_PROMPT)
            grammar = await _analyze_pass(client, request.text, GRAMMAR_PROMPT)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        # Log only the error class, never note text or model output.
        logger.warning("AI analysis failed: %s", type(error).__name__)
        raise HTTPException(status_code=502, detail="Local AI analysis failed") from error
    return AnalyzeResponse(
        original=request.text, suggestions=merge_edits(request.text, translations, grammar)
    )
