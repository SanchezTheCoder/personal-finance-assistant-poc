from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .schemas import ToolResult

DATA_DIR = Path(__file__).parent / "data"
MASTER_PATH = DATA_DIR / "user_master.json"
_MASTER_CACHE: Optional[dict[str, Any]] = None
_MASTER_MTIME: Optional[float] = None


def _load_master() -> dict[str, Any]:
    global _MASTER_CACHE, _MASTER_MTIME
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master data file: {MASTER_PATH}")
    mtime = MASTER_PATH.stat().st_mtime
    if _MASTER_CACHE is None or _MASTER_MTIME != mtime:
        _MASTER_CACHE = json.loads(MASTER_PATH.read_text())
        _MASTER_MTIME = mtime
    return _MASTER_CACHE


def _today_iso() -> str:
    return datetime.now().date().isoformat()


def _get_section(section: str) -> dict[str, Any]:
    payload = _load_master()
    data = payload.get(section)
    if not data:
        raise KeyError(f"Missing section '{section}' in {MASTER_PATH.name}")
    as_of = _today_iso()
    if "as_of" not in data and payload.get("as_of"):
        data = {**data, "as_of": payload["as_of"]}
    data = {**data, "as_of": as_of}
    return data


def get_activity(account: Optional[str] = None) -> ToolResult:
    payload = _get_section("activity")
    return ToolResult(source_id="tool:activity:v1", data=payload, as_of=payload["as_of"])


def get_positions(symbol: str, account: Optional[str] = None) -> ToolResult:
    payload = _get_section("positions")
    return ToolResult(source_id="tool:positions:v1", data=payload, as_of=payload["as_of"])


def get_positions_list(asset_class: Optional[str] = None, account: Optional[str] = None) -> ToolResult:
    payload = _get_section("positions")
    if asset_class:
        filtered = [
            p for p in payload["positions"] if p.get("asset_class") == asset_class
        ]
        payload = {**payload, "positions": filtered, "asset_class_filter": asset_class}
    return ToolResult(source_id="tool:positions_list:v1", data=payload, as_of=payload["as_of"])

def get_performance(timeframe: str, account: Optional[str] = None) -> ToolResult:
    payload = _get_section("performance")
    return ToolResult(source_id="tool:performance:v1", data=payload, as_of=payload["as_of"])


def get_quotes(symbol: str) -> ToolResult:
    payload = _get_section("quotes")
    return ToolResult(source_id="tool:quotes:v1", data=payload, as_of=payload["as_of"])


def get_facts(topic: str) -> ToolResult:
    # naive mapping based on keywords
    topic_lower = topic.lower()
    if "roth" in topic_lower:
        file_name = "roth_ira.md"
    elif "etf" in topic_lower:
        file_name = "etf_basics.md"
    else:
        file_name = "rebalancing.md"

    content = (DATA_DIR / "facts" / file_name).read_text().strip()
    payload = {
        "topic": topic,
        "snippet": content.split("\n", 2)[-1].strip(),
        "source": f"facts/{file_name}",
        "as_of": _today_iso(),
    }
    return ToolResult(source_id="tool:facts:v1", data=payload, as_of=payload["as_of"])


def get_transfers(account: Optional[str] = None) -> ToolResult:
    payload = _get_section("transfers")
    return ToolResult(source_id="tool:transfers:v1", data=payload, as_of=payload["as_of"])


def get_account_summary(account: Optional[str] = None) -> ToolResult:
    payload = _get_section("account_summary")
    return ToolResult(source_id="tool:account_summary:v1", data=payload, as_of=payload["as_of"])
class ToolRegistry:
    def call_tool(self, intent: str, **params: Any) -> ToolResult:
        if intent == "activity":
            return get_activity(account=params.get("account"))
        if intent == "positions":
            return get_positions(symbol=params["symbol"], account=params.get("account"))
        if intent == "positions_list":
            return get_positions_list(
                asset_class=params.get("asset_class"), account=params.get("account")
            )
        if intent == "performance":
            return get_performance(timeframe=params["timeframe"], account=params.get("account"))
        if intent == "quotes":
            return get_quotes(symbol=params["symbol"])
        if intent == "facts":
            return get_facts(topic=params["topic"])
        if intent == "transfers":
            return get_transfers(account=params.get("account"))
        if intent in {"account_value", "cash_balance"}:
            return get_account_summary(account=params.get("account"))
        raise ValueError(f"Unknown intent: {intent}")
