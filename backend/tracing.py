from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

TRACE_DIR = Path(__file__).parent / "traces"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceLogger:
    def __init__(self) -> None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        self.trace_id = str(uuid.uuid4())
        self.data: dict[str, Any] = {
            "trace_id": self.trace_id,
            "timestamp": _utc_now(),
            "utterance": "",
            "intent": None,
            "routing_mode": None,
            "routing_confidence": None,
            "routing_candidates": [],
            "routing_extracted": {},
            "routing_missing_params": [],
            "tool_calls": [],
            "tool_latency_ms": [],
            "tool_params": [],
            "context_summary": {},
            "clarification": None,
            "grounding_valid": None,
            "formatter_used": None,
            "session_state": {},
            "prompt_name": None,
            "retry_count": 0,
            "latency_ms": {},
            "policy_gate": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
            "model_version": None,
            "artifact_version": None,
            "prompt_version": None,
            "grounded_sources": [],
            "grounding_rate": 0.0,
            "router_diagnostics": {},
        }

    def start(self, utterance: str) -> str:
        self.data["utterance"] = utterance
        return self.trace_id

    def log_intent(self, intent: str) -> None:
        self.data["intent"] = intent

    def log_routing(self, mode: str, confidence: float, candidates: list[dict[str, float]]) -> None:
        self.data["routing_mode"] = mode
        self.data["routing_confidence"] = confidence
        self.data["routing_candidates"] = candidates

    def log_routing_detail(self, extracted: dict[str, Any], missing: list[str]) -> None:
        self.data["routing_extracted"] = extracted
        self.data["routing_missing_params"] = missing

    def log_tool(self, name: str, latency_ms: int, source_id: str) -> None:
        self.data["tool_calls"].append({"name": name, "source_id": source_id})
        self.data["tool_latency_ms"].append(latency_ms)

    def log_tool_params(self, name: str, params: dict[str, Any]) -> None:
        self.data["tool_params"].append({"name": name, "params": params})

    def log_tokens(self, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        self.data["prompt_tokens"] = prompt_tokens
        self.data["completion_tokens"] = completion_tokens
        self.data["estimated_cost_usd"] = cost

    def log_grounding(self, citations: list[str], sources: list[str]) -> None:
        self.data["grounded_sources"] = citations
        denom = max(1, len(sources))
        self.data["grounding_rate"] = round(len(citations) / denom, 2)

    def log_grounding_validation(self, valid: bool) -> None:
        self.data["grounding_valid"] = valid

    def log_context_summary(self, summary: dict[str, Any]) -> None:
        self.data["context_summary"] = summary

    def log_clarification(self, text: str) -> None:
        self.data["clarification"] = text

    def log_formatter_used(self, name: Optional[str]) -> None:
        self.data["formatter_used"] = name

    def log_session_state(self, state: dict[str, Any]) -> None:
        self.data["session_state"] = state

    def log_prompt_info(self, prompt_name: str) -> None:
        self.data["prompt_name"] = prompt_name

    def log_retry_count(self, count: int) -> None:
        self.data["retry_count"] = count

    def log_latency_breakdown(self, breakdown: dict[str, int]) -> None:
        self.data["latency_ms"] = breakdown

    def log_policy_gate(self, payload: dict[str, Any]) -> None:
        self.data["policy_gate"] = payload

    def log_router_diagnostics(self, payload: dict[str, Any]) -> None:
        self.data["router_diagnostics"] = payload

    def finalize(self, model_version: str, artifact_version: str, prompt_version: str) -> None:
        self.data["model_version"] = model_version
        self.data["artifact_version"] = artifact_version
        self.data["prompt_version"] = prompt_version
        (TRACE_DIR / f"{self.trace_id}.json").write_text(json.dumps(self.data, indent=2))


def load_trace(trace_id: str) -> Optional[dict[str, Any]]:
    path = TRACE_DIR / f"{trace_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
