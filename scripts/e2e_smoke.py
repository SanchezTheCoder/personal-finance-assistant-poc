#!/usr/bin/env python3
from __future__ import annotations

import json
import uuid
from typing import Any

import requests

BASE_URL = "http://127.0.0.1:8000"


def post_chat(utterance: str, session_id: str) -> dict[str, Any]:
    payload = {"utterance": utterance, "stream": False, "session_id": session_id}
    resp = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def assert_has_answer(resp: dict[str, Any]) -> None:
    answer = resp.get("answer_markdown", "").strip()
    if not answer:
        raise AssertionError("Empty answer")
    if resp.get("needs_clarification"):
        raise AssertionError("Unexpected clarification")
    if not resp.get("citations"):
        raise AssertionError("Missing citations")


def main() -> int:
    session_id = str(uuid.uuid4())
    queries = [
        "What are my positions?",
        "in stocks",
        "How many shares of AAPL do I own?",
        "What was my most recent trade?",
    ]

    for idx, q in enumerate(queries):
        resp = post_chat(q, session_id)
        if q == "in stocks":
            if resp.get("needs_clarification"):
                # fallback: repeat with explicit question
                resp = post_chat("What are my positions in stocks?", session_id)
        assert_has_answer(resp)
        print(json.dumps({"utterance": q, "answer": resp["answer_markdown"]}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
