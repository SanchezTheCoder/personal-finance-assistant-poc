from __future__ import annotations

import json
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .schemas import Intent, IntentRoute


class LLMRoute(BaseModel):
    intent: Intent
    extracted: dict[str, Any] = {}


def llm_reroute(
    utterance: str,
    api_key: str,
    model: str,
    base_url: Optional[str],
    candidates: list[dict[str, float]],
) -> Optional[IntentRoute]:
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = (
        "You are an intent router. Choose the single best intent for the user query.\n"
        "Return ONLY JSON in this schema: {\"intent\": <string>, \"extracted\": {}}.\n"
        "Valid intents: activity, positions, positions_list, portfolio_ranking, symbol_performance, performance, quotes, facts, transfers, account_value, cash_balance.\n"
        "User query: "
        + utterance
        + "\nCandidates (intent:score): "
        + ", ".join([f"{c['intent']}:{c['score']}" for c in candidates])
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    raw_text = getattr(response, "output_text", "") or ""
    if not raw_text and getattr(response, "output", None):
        for item in response.output:
            for content in getattr(item, "content", []):
                text = getattr(content, "text", "")
                if text:
                    raw_text += text

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    try:
        llm_route = LLMRoute.model_validate(payload)
    except ValidationError:
        return None

    return IntentRoute(
        intent=llm_route.intent,
        confidence=0.7,
        missing_params=[],
        extracted=llm_route.extracted or {},
        candidates=candidates,
        routing_mode="llm_reroute",
    )
