from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .schemas import Intent, IntentRoute
from .router_llm import llm_reroute
from .torch_router import torch_attempt, torch_router_status

SYMBOL_RE = re.compile(r"\b[A-Z]{1,5}\b")
SYMBOL_TOKEN_RE = re.compile(r"\b([A-Za-z]{1,5})\b")
SYMBOL_STOPWORDS = {
    "A",
    "AN",
    "THE",
    "HOW",
    "IS",
    "ARE",
    "AM",
    "WAS",
    "WERE",
    "BE",
    "BEEN",
    "BEING",
    "MANY",
    "SHARES",
    "SHARE",
    "OF",
    "DO",
    "DOES",
    "DID",
    "DONT",
    "DON'T",
    "I",
    "ME",
    "WE",
    "US",
    "OUR",
    "YOUR",
    "YOU",
    "THEY",
    "THEIR",
    "IT",
    "ITS",
    "IT'S",
    "TO",
    "FOR",
    "WITH",
    "WITHOUT",
    "IN",
    "ON",
    "AT",
    "FROM",
    "BY",
    "AS",
    "ABOUT",
    "ALL",
    "ANY",
    "GOOD",
    "TIME",
    "SELL",
    "BUY",
    "OWN",
    "WHAT",
    "PRICE",
    "PRICES",
    "QUOTE",
    "QUOTES",
    "TODAY",
    "TODAYS",
    "CHANGE",
    "MY",
    "MOST",
    "RECENT",
    "TRADE",
    "ACTIVITY",
    "PORTFOLIO",
    "YTD",
    "YEAR",
    "DATE",
    "POSITION",
    "POSITIONS",
    "HOLDING",
    "HOLDINGS",
    "STOCK",
    "STOCKS",
    "ETF",
    "ETFS",
    "FUND",
    "FUNDS",
    "EQUITY",
    "EQUITIES",
    "IRA",
    "ROTH",
    "ACCOUNT",
    "ACCOUNTS",
    "SHOW",
    "LIST",
    "ALL",
    "BALANCE",
    "VALUE",
    "CASH",
    "TRANSFER",
    "DEPOSIT",
    "WITHDRAWAL",
    "ACH",
    "THIS",
    "THAT",
    "PERFORMANCE",
    "PERFORMING",
    "PERFORM",
    "RETURN",
    "GAIN",
    "LOSS",
    "PROFIT",
    "BEST",
    "WORST",
    "TOP",
    "BIGGEST",
    "HIGHEST",
    "LOWEST",
    "UP",
    "DOWN",
    "TODAYS",
    "TODAY'S",
}

ASSET_CLASS_MAP = {
    "stocks": ["stocks", "stock", "equities", "equity"],
    "etf": ["etf", "etfs"],
    "funds": ["funds", "mutual funds"],
}

SYMBOL_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "vanguard": "VOO",
}

DATA_DIR = Path(__file__).parent / "data"
MASTER_PATH = DATA_DIR / "user_master.json"


def _load_known_symbols() -> set[str]:
    symbols: set[str] = set()
    if MASTER_PATH.exists():
        try:
            payload = json.loads(MASTER_PATH.read_text())
        except json.JSONDecodeError:
            payload = {}
        positions = payload.get("positions", {}).get("positions", [])
        quotes = payload.get("quotes", {}).get("quotes", [])
        for item in positions + quotes:
            symbol = item.get("symbol")
            if symbol:
                symbols.add(symbol.upper())
    else:
        for name, key in [("positions.json", "positions"), ("quotes.json", "quotes")]:
            path = DATA_DIR / name
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            for item in payload.get(key, []):
                symbol = item.get("symbol")
                if symbol:
                    symbols.add(symbol.upper())
    for alias_symbol in SYMBOL_ALIASES.values():
        symbols.add(alias_symbol.upper())
    return symbols


KNOWN_SYMBOLS = _load_known_symbols()

TYPO_MAP = {
    "perfomance": "performance",
    "postions": "positions",
    "posiions": "positions",
    "qoute": "quote",
    "balnce": "balance",
    "tranfer": "transfer",
}

SYNONYMS = {
    "positions": ["positions", "holdings", "portfolio holdings", "what do i own", "owned shares", "equities"],
    "performance": ["performance", "performer", "performing", "return", "gain", "loss", "p/l", "profit", "up", "down"],
    "quotes": ["price", "quote", "trading at", "current price"],
    "transfers": ["transfer", "deposit", "withdrawal", "cash transfer", "bank transfer", "ach"],
    "account_value": ["account value", "total value", "portfolio value", "account balance"],
    "cash_balance": ["cash value", "cash amount", "cash balance", "settled cash", "total cash", "available cash"],
}

RANK_KEYWORDS = [
    "best",
    "worst",
    "top",
    "biggest",
    "highest",
    "lowest",
    "most",
    "least",
    "winner",
    "loser",
    "outperform",
    "underperform",
    "strongest",
    "weakest",
]

SYMBOL_HINT_KEYWORDS = [
    "price",
    "quote",
    "quotes",
    "perform",
    "performance",
    "performing",
    "change",
    "ticker",
    "symbol",
]


def _extract_symbol(text: str, allow_unknown: bool = False) -> str | None:
    # Prefer explicit uppercase tokens in the original input (e.g., AAPL).
    direct_matches = SYMBOL_RE.findall(text)
    for m in direct_matches:
        symbol = m.upper()
        if symbol in SYMBOL_STOPWORDS:
            continue
        if symbol in KNOWN_SYMBOLS or allow_unknown:
            return symbol

    # Check for $-prefixed tickers.
    for match in re.finditer(r"\$([A-Za-z]{1,5})\b", text):
        symbol = match.group(1).upper()
        if symbol not in SYMBOL_STOPWORDS:
            return symbol

    # If the user typed lowercase, accept known symbols and (optionally) a single candidate.
    candidates: list[str] = []
    for token in SYMBOL_TOKEN_RE.findall(text):
        symbol = token.upper()
        if symbol in SYMBOL_STOPWORDS:
            continue
        if symbol in KNOWN_SYMBOLS:
            return symbol
        if allow_unknown:
            candidates.append(symbol)

    if allow_unknown:
        unique = []
        for symbol in candidates:
            if symbol not in unique:
                unique.append(symbol)
        if len(unique) == 1:
            return unique[0]

        stripped = text.strip()
        if stripped.isalpha() and len(stripped.split()) == 1:
            return stripped.upper()

    return None


def _extract_symbol_with_alias(text: str, allow_unknown: bool = False) -> str | None:
    alias = SYMBOL_ALIASES.get(text.strip().lower())
    if alias:
        return alias
    for key, symbol in SYMBOL_ALIASES.items():
        if key in text.lower():
            return symbol
    return _extract_symbol(text, allow_unknown=allow_unknown)


def _normalize(text: str) -> str:
    cleaned = text.lower().strip()
    for typo, correct in TYPO_MAP.items():
        cleaned = cleaned.replace(typo, correct)
    cleaned = re.sub(r"[^\w\s\-\.\/]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _score_candidates(text: str, utterance: str) -> list[dict[str, float]]:
    scores: dict[str, float] = {
        "activity": 0.0,
        "positions": 0.0,
        "positions_list": 0.0,
        "portfolio_ranking": 0.0,
        "symbol_performance": 0.0,
        "performance": 0.0,
        "quotes": 0.0,
        "facts": 0.0,
        "transfers": 0.0,
        "account_value": 0.0,
        "cash_balance": 0.0,
    }

    def bump(intent: str, amount: float) -> None:
        scores[intent] = min(1.0, scores[intent] + amount)

    has_symbol_hint = any(k in text for k in SYMBOL_HINT_KEYWORDS) or "$" in utterance
    symbol_in_text = _extract_symbol_with_alias(utterance, allow_unknown=has_symbol_hint)

    if any(k in text for k in ["recent trade", "last trade", "most recent trade"]):
        bump("activity", 0.9)

    if any(k in text for k in ["how many shares", "do i own"]):
        bump("positions", 0.9)

    if any(k in text for k in SYNONYMS["positions"]):
        bump("positions_list", 0.8)

    ranking_trigger = any(k in text for k in RANK_KEYWORDS)
    portfolio_context = any(
        k in text
        for k in [
            "position",
            "positions",
            "holding",
            "holdings",
            "portfolio",
            "investments",
            "unrealized",
        ]
    )
    if ranking_trigger and (portfolio_context or any(k in text for k in SYNONYMS["performance"])):
        bump("portfolio_ranking", 0.92)

    if any(k in text for k in SYNONYMS["account_value"]):
        bump("account_value", 0.85)

    if any(k in text for k in SYNONYMS["cash_balance"]):
        bump("cash_balance", 0.85)

    if any(k in text for k in ["ytd", "year to date", "this year", "how did i do", "did i do"]) or any(
        k in text for k in SYNONYMS["performance"]
    ):
        bump("performance", 0.8)
        if symbol_in_text:
            bump("symbol_performance", 0.95)

    if any(k in text for k in SYNONYMS["transfers"]):
        bump("transfers", 0.8)

    if any(k in text for k in SYNONYMS["quotes"]) or "change" in text:
        bump("quotes", 0.8)

    if any(k in text for k in ["what is", "explain", "define", "help"]):
        bump("facts", 0.6)

    if "performance" in text or "performing" in text:
        symbol = _extract_symbol_with_alias(utterance, allow_unknown=True)
        if symbol:
            bump("symbol_performance", 0.95)

    if any(k in text for k in SYNONYMS["quotes"]):
        symbol = _extract_symbol_with_alias(utterance, allow_unknown=True)
        if symbol:
            bump("quotes", 0.9)

    if "performance" in text and any(k in text for k in SYNONYMS["quotes"]):
        symbol = _extract_symbol_with_alias(utterance, allow_unknown=True)
        if symbol:
            bump("symbol_performance", 1.0)
            scores["quotes"] = min(scores["quotes"], 0.6)

    ranked = sorted(
        [{"intent": k, "score": v} for k, v in scores.items() if v > 0.0],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked


def _apply_extraction(intent: Intent, text: str, utterance: str) -> tuple[Intent, dict[str, Any], list[str]]:
    extracted: dict[str, Any] = {}
    missing: list[str] = []

    for key, terms in ASSET_CLASS_MAP.items():
        if any(term in text for term in terms):
            extracted["asset_class"] = key
            break

    if "brokerage" in text:
        extracted["account"] = "Brokerage"

    if intent is Intent.facts:
        extracted["topic"] = utterance.strip()

    if intent in {Intent.positions, Intent.quotes}:
        symbol = _extract_symbol_with_alias(utterance, allow_unknown=True)
        if symbol:
            extracted["symbol"] = symbol
        else:
            if intent is Intent.positions:
                intent = Intent.positions_list
            else:
                missing.append("symbol")

    if intent is Intent.symbol_performance:
        symbol = _extract_symbol_with_alias(utterance, allow_unknown=True)
        if symbol:
            extracted["symbol"] = symbol
        else:
            # fall back to portfolio performance if no symbol
            intent = Intent.performance

    if intent is Intent.portfolio_ranking:
        symbol = _extract_symbol_with_alias(utterance, allow_unknown=True)
        has_portfolio_terms = any(
            k in text for k in ["position", "positions", "holding", "holdings", "portfolio", "investments", "list"]
        )
        if symbol and not has_portfolio_terms:
            intent = Intent.symbol_performance
            extracted["symbol"] = symbol
        else:
            if any(k in text for k in ["worst", "lowest", "least", "loser", "underperform", "biggest loss", "most loss"]):
                extracted["direction"] = "worst"
            elif "loss" in text and any(k in text for k in RANK_KEYWORDS):
                extracted["direction"] = "worst"
            else:
                extracted["direction"] = "best"
            if any(k in text for k in ["percent", "percentage", "%", "pct"]):
                extracted["basis"] = "unrealized_pl_pct"
            else:
                extracted["basis"] = "unrealized_pl"

    if intent is Intent.cash_balance:
        if "settled" in text:
            extracted["cash_type"] = "settled"
        elif "total" in text:
            extracted["cash_type"] = "total"
        else:
            extracted["cash_type"] = "both"

    if intent is Intent.performance:
        if (
            "ytd" in text
            or "year to date" in text
            or "this year" in text
            or "year" in text
        ):
            extracted["timeframe"] = "YTD"
        else:
            extracted["timeframe"] = "YTD"

    if intent in {Intent.account_value, Intent.cash_balance} and "account" not in extracted:
        extracted["account"] = "Brokerage"

    if intent is Intent.clarify:
        missing = ["intent"]

    return intent, extracted, missing


def rule_route(utterance: str) -> IntentRoute:
    text = _normalize(utterance)
    candidates = _score_candidates(text, utterance)
    if not candidates:
        return IntentRoute(
            intent=Intent.clarify,
            confidence=0.2,
            missing_params=["intent"],
            extracted={},
            candidates=[],
            routing_mode="rules",
        )

    top = candidates[0]
    intent = Intent(top["intent"])
    intent, extracted, missing = _apply_extraction(intent, text, utterance)

    if missing:
        extracted["candidate_intent"] = intent.value
        return IntentRoute(
            intent=Intent.clarify,
            confidence=top["score"],
            missing_params=missing,
            extracted=extracted,
            candidates=candidates,
            routing_mode="rules",
        )

    return IntentRoute(
        intent=intent,
        confidence=top["score"],
        missing_params=[],
        extracted=extracted,
        candidates=candidates,
        routing_mode="rules",
    )


def route_intent(utterance: str, use_llm: bool = True) -> IntentRoute:
    rule_result = rule_route(utterance)

    threshold = float(os.getenv("ROUTER_CONF_THRESHOLD", "0.75"))
    candidates = rule_result.candidates
    ambiguous = False
    if len(candidates) >= 2:
        ambiguous = abs(candidates[0]["score"] - candidates[1]["score"]) < 0.1

    if not use_llm:
        rule_result.routing_meta = {
            **torch_router_status(),
            "torch_attempted": False,
            "torch_used": False,
        }
        return rule_result

    if rule_result.confidence >= threshold and not ambiguous:
        rule_result.routing_meta = {
            **torch_router_status(),
            "torch_attempted": False,
            "torch_used": False,
        }
        return rule_result

    torch_route, torch_meta = torch_attempt(utterance, candidates)
    if torch_route:
        text = _normalize(utterance)
        intent, extracted, missing = _apply_extraction(torch_route.intent, text, utterance)
        torch_route.intent = intent
        torch_route.extracted.update(extracted)
        torch_route.missing_params = missing
        torch_route.routing_meta = torch_meta
        return torch_route

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        rule_result.routing_meta = torch_meta if "torch_meta" in locals() else {
            **torch_router_status(),
            "torch_attempted": False,
            "torch_used": False,
        }
        return rule_result

    model = os.getenv("ROUTER_MODEL", "gpt-5-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    reroute = llm_reroute(utterance, api_key, model, base_url, candidates)
    if not reroute:
        rule_result.routing_meta = torch_meta if "torch_meta" in locals() else {
            **torch_router_status(),
            "torch_attempted": False,
            "torch_used": False,
        }
        return rule_result

    text = _normalize(utterance)
    intent, extracted, missing = _apply_extraction(reroute.intent, text, utterance)
    reroute.intent = intent
    reroute.extracted.update(extracted)
    reroute.missing_params = missing
    reroute.routing_meta = torch_meta if "torch_meta" in locals() else {
        **torch_router_status(),
        "torch_attempted": False,
        "torch_used": False,
    }
    return reroute
