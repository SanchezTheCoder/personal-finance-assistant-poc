from __future__ import annotations

import json
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from openai import OpenAI
from pydantic import ValidationError

from .schemas import ContextBundle, LLMResponse

ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "model_bundle_v1"
PROMPTS_DIR = ARTIFACTS_DIR / "prompts"
PROMPT_NAME = "response_system.txt"

PRICE_TABLE = {
    "gpt-5.2": {"input_per_1m": 10.0, "output_per_1m": 30.0},
    "gpt-5": {"input_per_1m": 10.0, "output_per_1m": 30.0},
    "gpt-5-mini": {"input_per_1m": 3.0, "output_per_1m": 9.0},
    "gpt-5-nano": {"input_per_1m": 1.0, "output_per_1m": 3.0},
}


@dataclass
class ModelConfig:
    api_key: str
    model: str
    base_url: Optional[str]


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _build_user_prompt(question: str, context: ContextBundle, citations: Iterable[str]) -> str:
    template = _load_prompt("response_user.txt")
    return template.replace("{{question}}", question).replace(
        "{{context_json}}", json.dumps(context.model_dump(), indent=2)
    ).replace("{{valid_citations}}", ", ".join(citations))


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICE_TABLE.get(model)
    if not pricing:
        return 0.0
    return round(
        (prompt_tokens / 1_000_000) * pricing["input_per_1m"]
        + (completion_tokens / 1_000_000) * pricing["output_per_1m"],
        6,
    )


def _json_schema() -> dict[str, Any]:
    return {
        "name": "LLMResponse",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer_markdown": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "needs_clarification": {"type": "boolean"},
                "clarifying_question": {"type": ["string", "null"]},
            },
            "required": [
                "answer_markdown",
                "citations",
                "confidence",
                "needs_clarification",
                "clarifying_question",
            ],
        },
    }


class ResponseError(RuntimeError):
    pass


def clean_answer_markdown(text: str) -> str:
    cleaned = text.strip().replace("\r\n", "\n")
    lines = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-•]\s*", "", line)
        line = re.sub(r"\s+", " ", line)
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r":(\d)", r": \1", cleaned)
    cleaned = re.sub(r"\byou(hold|own|have)\b", r"you \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcostbasis\b", "cost basis", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"([A-Za-z])\$", r"\1 $", cleaned)
    cleaned = re.sub(r"\bYTDcontributions\b", "YTD contributions", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"([\.!\?])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([,])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"(\d)([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([A-Za-z])(\d)", r"\1 \2", cleaned)
    cleaned = re.sub(r"\b[Nn]etcontributions\b", "Net contributions", cleaned)
    cleaned = re.sub(r"\s+Reasoning:", "\n\nReasoning:", cleaned)
    cleaned = re.sub(r"\nReasoning:", "\n\nReasoning:", cleaned)
    return cleaned.strip()


def strip_markdown(text: str) -> str:
    cleaned = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)
    cleaned = re.sub(r"^#{1,6}\\s+", "", cleaned, flags=re.MULTILINE)
    return cleaned


def _parse_llm_output(raw_text: str) -> LLMResponse:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ResponseError("LLM output was not valid JSON") from exc

    try:
        return LLMResponse.model_validate(payload)
    except ValidationError as exc:
        raise ResponseError("LLM output did not match schema") from exc


def _call_openai(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> Tuple[LLMResponse, dict[str, Any]]:
    system_prompt = system_prompt + "\n\nRequired JSON schema:\n" + json.dumps(_json_schema()["schema"], indent=2)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = getattr(response, "output_text", "") or ""
    if not raw_text and getattr(response, "output", None):
        for item in response.output:
            for content in getattr(item, "content", []):
                text = getattr(content, "text", "")
                if text:
                    raw_text += text

    parsed = _parse_llm_output(raw_text)

    usage = getattr(response, "usage", None) or {}
    prompt_tokens = getattr(usage, "input_tokens", 0) if not isinstance(usage, dict) else usage.get("input_tokens", 0)
    completion_tokens = getattr(usage, "output_tokens", 0) if not isinstance(usage, dict) else usage.get("output_tokens", 0)

    meta = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": _estimate_cost(model, prompt_tokens, completion_tokens),
    }
    return parsed, meta


def validate_response(llm_response: LLMResponse, valid_sources: list[str]) -> bool:
    if llm_response.needs_clarification:
        return True
    if not llm_response.citations:
        return False
    return set(llm_response.citations) == set(valid_sources)


def generate_response(
    utterance: str,
    context: ContextBundle,
    model_config: ModelConfig,
    max_retries: int = 1,
) -> Tuple[LLMResponse, dict[str, Any]]:
    if not model_config.api_key:
        fallback = LLMResponse(
            answer_markdown=(
                "I do not have access to the model right now. "
                "Please set OPENAI_API_KEY and retry."
            ),
            citations=[],
            confidence=0.0,
            needs_clarification=True,
            clarifying_question="Can you provide a valid API key?",
        )
        return fallback, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
            "retry_count": 0,
            "prompt_name": PROMPT_NAME,
        }

    client = OpenAI(api_key=model_config.api_key, base_url=model_config.base_url)
    system_prompt = _load_prompt(PROMPT_NAME)
    user_prompt = _build_user_prompt(utterance, context, context.sources)

    attempt = 0
    meta: dict[str, Any] = {}
    last_error: Optional[str] = None

    while attempt <= max_retries:
        attempt += 1
        try:
            llm_response, meta = _call_openai(client, model_config.model, system_prompt, user_prompt)
            if validate_response(llm_response, context.sources):
                meta["retry_count"] = attempt - 1
                meta["prompt_name"] = PROMPT_NAME
                return llm_response, meta
            last_error = "Missing or invalid citations"
        except ResponseError as exc:
            last_error = str(exc)

        system_prompt = (
            system_prompt
            + "\n\nIMPORTANT: Your previous output failed validation."
            + "\nReturn ONLY valid JSON that matches the schema, and include citations that match the valid list."
        )
        time.sleep(0.05)

    meta["retry_count"] = attempt - 1
    meta["prompt_name"] = PROMPT_NAME
    raise ResponseError(f"Failed to get valid response: {last_error}")


def stream_chunks(text: str, chunk_size: int = 16) -> Iterable[str]:
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
