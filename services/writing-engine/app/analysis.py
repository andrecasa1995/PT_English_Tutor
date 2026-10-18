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
    mode: str = Field(default="all", pattern="^(all|word|sentence|batch)$")


class ExplainRequest(BaseModel):
    original: str = Field(min_length=1, max_length=100)
    replacement: str = Field(min_length=1, max_length=100)
    category: str = Field(pattern="^(translation|spelling|grammar)$")
    context: str = Field(min_length=1, max_length=500)


class ExplainResponse(BaseModel):
    explanation_pt: str


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

COMPACT_SCHEMA = {
    "type": "object",
    "properties": {"edits": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "original": {"type": "string"},
            "replacement": {"type": "string"},
            "category": {"type": "string", "enum": ["translation", "grammar", "spelling"]},
        },
        "required": ["original", "replacement", "category"],
        "additionalProperties": False,
    }}},
    "required": ["edits"],
    "additionalProperties": False,
}

EXPLAIN_PROMPT = (
    "Explain ONLY the given English correction in natural Brazilian Portuguese, "
    "in one clear short sentence. Include the actual grammatical rule or translation "
    "reason; do not suggest other changes. Return plain text, not JSON."
)

TRANSLATE_PROMPT = (
    "You are a bilingual Brazilian Portuguese and American English writing tutor. "
    "Analyze English writing containing occasional Portuguese words. "
    "Identify ALL Portuguese words and phrases embedded in otherwise English text, "
    "including words at the END of the sentence and after English prepositions. "
    "Inspect each word independently, not only the first foreign term. "
    "Translate each into natural contextual American English, preserving technical AI terms. "
    "If the entire sentence is Portuguese, do not translate it. "
    "Do not fix English typos or grammar in this translation pass: have, multiples, "
    "algoritm and imput are NOT Portuguese words. Leave those to the grammar pass. "
    "Only flag genuinely Portuguese words, for example exemplos and previsoes, "
    "including Portuguese words at the very end of a sentence. "
    "Make a complete independent pass over every token, including the final word. "
    "Output one edit for each real Portuguese word, even if English grammar elsewhere is wrong. "
    "Return separate minimal edits, category=translation. "
    "Each original must be an EXACT substring of the input. "
    "Explain ONLY the original->replacement translation in Brazilian Portuguese; "
    "never claim another word was changed or that uncorrected grammar is correct. "
    "Never edit code, Markdown links, or identifiers. Return JSON edits; no issues means []."
)

WORD_PROMPT = (
    "You are a bilingual Brazilian Portuguese to American English tutor checking ONE word. "
    "STRICT PRIORITY: First determine whether the word is Portuguese. If it is Portuguese, "
    "translate it to natural English, category=translation; NEVER label Portuguese words as spelling. "
    "Only if the word is NOT Portuguese, look for English misspellings and use category=spelling. "
    "Examples: resultado->result (translation), camadas->layers (translation), "
    "algoritm->algorithm (spelling). "
    "Never translate an English typo into Portuguese; every replacement MUST be English. "
    "If the word is already correct English, return zero edits. Preserve technical terms and proper nouns. "
    "The original must exactly match the entire input; never output an empty replacement. "
    "Explain the change briefly in Brazilian Portuguese. Return JSON edits only."
)

BATCH_PROMPT = (
    "You are a bilingual Brazilian Portuguese to American English proofreader. "
    "Review the COMPLETE input from FIRST to LAST token; never stop at the first corrections. "
    "Return an independent edit for EACH mistake, including the FINAL word. "
    "Check EACH verb for agreement, including verbs after AND. "
    "Example: She go to school every day. => go->goes (grammar). "
    "Example: The algorithm use data and generate predictions. => use->uses AND generate->generates (grammar). "
    "Example: My sistema have a algoritm that use dados to generate resultado. "
    "=> sistema->system (translation), have->has (grammar), algoritm->algorithm (spelling), "
    "use->uses (grammar), dados->data (translation), resultado->result (translation). "
    "Review one English sentence, applying STRICT category priority: "
    "FIRST translate ALL Portuguese words into English (category=translation). "
    "SECOND correct misspelled English words (category=spelling). "
    "THIRD correct clear English grammar errors (category=grammar). "
    "Never classify a Portuguese word as spelling or grammar, even when it resembles a typo. "
    "For example sistema->system and resultado->result are translations; "
    "algoritm->algorithm is spelling; She go->She goes is grammar (original=go). "
    "Return separate minimal edits for every issue. Each original MUST exactly match "
    "a substring of the supplied input. All replacements MUST be English. "
    "Explain each edit briefly in Brazilian Portuguese. "
    "Protect Markdown and code. "
    "Do not invent issues; if correct return empty edits. Return JSON only."
)

SPELLING_PROMPT = (
    "You are an American English spelling proofreader. All replacements MUST be English words, NEVER Portuguese. "
    "Check only spelling errors in English words. Do not translate Portuguese or correct grammar. "
    "Return minimal edits categorized spelling with short Brazilian Portuguese explanations. "
    "Do not create a spelling edit for a Portuguese word. Never output an empty replacement. "
    "Preserve proper nouns, identifiers and Markdown. If correct, return empty edits."
)

GRAMMAR_PROMPT = (
    "You are a conservative American English proofreader. Correct only demonstrable errors; "
    "do not invent mistakes in already grammatical sentences. Do not suggest has->have "
    "when a singular subject such as This neural network already agrees with has. "
    "Review every adjective and noun modifier. For example, in "
    "'The model have multiples layers', propose two separate edits: "
    "have->has and multiples->multiple. The adjective multiple never "
    "takes a plural -s before a noun. Do not change correct 'multiple layers'. "
    "Review ALL English verbs, spelling and grammar in the complete sentence. "
    "Check spelling of every English content word, not only verbs. "
    "Check subject-verb agreement separately for EVERY coordinated verb. "
    "When a singular subject has two present-tense verbs joined by 'and', both verbs must agree with that subject, even if a noun phrase appears between them. "
    "Do not mistake a verb after 'and' for a plural-subject verb unless there is actually a new plural subject. "
    "Produce one separate edit per demonstrable issue; do not stop after the first correction. "
    "Pay special attention to coordinated verbs "
    "that share the same subject, including verbs after objects or foreign words. "
    "For example: 'The algorithm use data and generate predictions' requires "
    "use->uses AND generate->generates, because BOTH verbs refer to algorithm. "
    "Portuguese words may occur in the sentence; leave them unchanged in this pass. "
    "Make only the smallest separate edits to English words (category=grammar or spelling). "
    "Each original must be an EXACT substring of input. Explain ONLY this individual "
    "original->replacement edit in Brazilian Portuguese. Never claim that other words "
    "were changed, or that an uncorrected phrase is grammatically perfect. "
    "Identify the grammatical subject accurately. Preserve technical "
    "terminology, code and Markdown; return JSON edits; no issues means []."
)


def validate_edits(text: str, edits: list[dict], compact: bool = False) -> list[Suggestion]:
    """Retain only uniquely locatable, exact, nonoverlapping suggestions."""
    valid: list[Suggestion] = []
    for edit in edits[:20]:
        if not isinstance(edit, dict):
            continue
        original, replacement = edit.get("original"), edit.get("replacement")
        category, explanation = edit.get("category"), edit.get("explanation_pt")
        if compact and explanation is None:
            explanation = ""
        if not all(isinstance(v, str) for v in (original, replacement, category, explanation)):
            continue
        if not original or not replacement.strip() or (not compact and not explanation.strip()) or original == replacement or category not in {"translation", "grammar", "spelling", "style"}:
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
    """Enforce pass roles before merging; never trust the model category alone."""
    safe_translations = [edit for edit in translations
                         if isinstance(edit, dict) and edit.get("category") == "translation"]
    safe_grammar = [edit for edit in grammar
                    if isinstance(edit, dict) and edit.get("category") in {"grammar", "spelling"}]
    # Translation has priority on overlaps; grammar and spelling follow.
    return validate_edits(text, safe_translations + safe_grammar)


async def _analyze_pass(client: httpx.AsyncClient, text: str, instructions: str, compact: bool = False) -> list[dict]:
    response = await client.post(OLLAMA_URL, json={
        "model": MODEL, "system": instructions, "prompt": text, "stream": False,
        "think": False, "format": COMPACT_SCHEMA if compact else SCHEMA, "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 1024},
    })
    response.raise_for_status()
    edits = json.loads(response.json()["response"])["edits"]
    if not isinstance(edits, list):
        raise ValueError("edits must be an array")
    return edits


@router.post("/v1/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest) -> ExplainResponse:
    if request.original not in request.context:
        raise HTTPException(status_code=422, detail="Original not in context")
    query = json.dumps(request.model_dump(), ensure_ascii=False)
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(OLLAMA_URL, json={
                "model": MODEL, "system": EXPLAIN_PROMPT, "prompt": query,
                "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 160},
            })
            response.raise_for_status()
            result = response.json()["response"].strip()
            if not result:
                raise ValueError("Empty explanation")
            return ExplainResponse(explanation_pt=result[:600])
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        logger.warning("AI explanation failed: %s", type(error).__name__)
        raise HTTPException(status_code=502, detail="Local AI explanation failed") from error


@router.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            # The word path uses one combined prompt to reduce local inference latency.
            # The legacy all mode still keeps independent translation and grammar passes.
            if request.mode == "batch":
                edits = await _analyze_pass(client, request.text, BATCH_PROMPT + " Do not generate explanations; output ONLY original, replacement, category for each edit.", compact=True)
                edits = [item for item in edits if isinstance(item, dict) and item.get("category") in {"translation", "spelling", "grammar"}]
                priority = {"translation": 0, "spelling": 1, "grammar": 2}
                edits.sort(key=lambda item: priority[item["category"]])
                return AnalyzeResponse(original=request.text, suggestions=validate_edits(request.text, edits, compact=True))
            if request.mode == "word":
                word_edits = await _analyze_pass(client, request.text, WORD_PROMPT)
                word_edits = [edit for edit in word_edits if isinstance(edit, dict) and edit.get("category") in {"translation", "spelling"}]
                return AnalyzeResponse(original=request.text, suggestions=validate_edits(request.text, word_edits))
            translations = await _analyze_pass(client, request.text, TRANSLATE_PROMPT) if request.mode != "sentence" else []
            grammar = await _analyze_pass(client, request.text, GRAMMAR_PROMPT)
            if request.mode == "sentence":
                grammar = [edit for edit in grammar if isinstance(edit, dict) and edit.get("category") == "grammar"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        # Log only the error class, never note text or model output.
        logger.warning("AI analysis failed: %s", type(error).__name__)
        raise HTTPException(status_code=502, detail="Local AI analysis failed") from error
    return AnalyzeResponse(
        original=request.text, suggestions=merge_edits(request.text, translations, grammar)
    )
