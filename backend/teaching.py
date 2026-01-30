from __future__ import annotations

import json
from typing import Any, Optional

from openai import OpenAI


def _build_prompt(trace: dict[str, Any]) -> str:
    summary = {
        "utterance": trace.get("utterance"),
        "intent": trace.get("intent"),
        "routing_mode": trace.get("routing_mode"),
        "routing_confidence": trace.get("routing_confidence"),
        "routing_extracted": trace.get("routing_extracted"),
        "routing_missing_params": trace.get("routing_missing_params"),
        "tool_calls": trace.get("tool_calls"),
        "tool_params": trace.get("tool_params"),
        "context_summary": trace.get("context_summary"),
        "grounded_sources": trace.get("grounded_sources"),
        "grounding_valid": trace.get("grounding_valid"),
        "formatter_used": trace.get("formatter_used"),
        "latency_ms": trace.get("latency_ms"),
        "policy_gate": trace.get("policy_gate"),
        "prompt_name": trace.get("prompt_name"),
        "retry_count": trace.get("retry_count"),
    }
    return (
        "You are explaining how a backend assistant handled a user request. "
        "Use the provided trace fields only. Write 4–6 sentences. "
        "Explain routing, tool calls, context building, grounding, and formatting. "
        "Be clear and teaching-oriented, but concise.\n\n"
        f"Trace JSON:\n{json.dumps(summary, indent=2)}"
    )


def generate_teaching_explanation(
    trace: dict[str, Any],
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = _build_prompt(trace)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "Return a short teaching explanation."},
            {"role": "user", "content": prompt},
        ],
    )

    text = getattr(response, "output_text", "") or ""
    if not text and getattr(response, "output", None):
        for item in response.output:
            for content in getattr(item, "content", []):
                chunk = getattr(content, "text", "")
                if chunk:
                    text += chunk
    return text.strip()
