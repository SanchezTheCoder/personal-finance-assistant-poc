#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import re
from itertools import product
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "ml" / "intent_templates.json"
OUT_DIR = ROOT / "ml"
SEED = 42
TARGET_SIZE = int(os.getenv("INTENT_DATASET_SIZE", "25000"))
MAX_COMBOS = int(os.getenv("INTENT_MAX_COMBOS", "200"))

PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _load_templates() -> dict:
    return json.loads(TEMPLATE_PATH.read_text())


def _render_template(template: str, slots: dict[str, list[str]], max_combos: int) -> list[str]:
    keys = PLACEHOLDER_RE.findall(template)
    if not keys:
        return [template]

    values = [slots.get(k, []) for k in keys]
    if any(not v for v in values):
        return [template]

    combos = list(product(*values))
    if len(combos) > max_combos:
        combos = combos[:max_combos]

    rendered = []
    for combo in combos:
        text = template
        for key, value in zip(keys, combo):
            text = text.replace(f"{{{key}}}", value)
        rendered.append(text)
    return rendered


def _variants(text: str, typos: dict[str, list[str]]) -> Iterable[str]:
    variants = {text}
    variants.add(text.lower())
    variants.add(re.sub(r"[\?\.!,,]", "", text))

    for word, typo_list in typos.items():
        for typo in typo_list:
            if word in text.lower():
                variants.add(re.sub(word, typo, text, flags=re.IGNORECASE))

    m = re.search(r"\b([A-Z]{1,5})\b", text)
    if m and "performance" in text.lower():
        variants.add(f"{m.group(1)} perf")
        variants.add(f"{m.group(1)} performance")
    if m and "price" in text.lower():
        variants.add(f"{m.group(1)} price")
        variants.add(f"{m.group(1)} quote")

    return variants


def _augment_with_affixes(
    text: str,
    prefixes: list[str],
    suffixes: list[str],
    slots: dict[str, list[str]],
    rng: random.Random,
) -> str:
    prefix = rng.choice(prefixes) if prefixes else ""
    suffix = rng.choice(suffixes) if suffixes else ""
    if "{ACCOUNT}" in suffix:
        accounts = slots.get("ACCOUNT", [])
        if accounts:
            suffix = suffix.replace("{ACCOUNT}", rng.choice(accounts))
    parts = [prefix.strip(), text.strip(), suffix.strip()]
    return " ".join([p for p in parts if p]).strip()


def _split_dataset(items: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    random.shuffle(items)
    n = len(items)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def main() -> int:
    random.seed(SEED)
    rng = random.Random(SEED)
    config = _load_templates()
    intents = config["intents"]
    slots = config["slots"]
    typos = config.get("typos", {})
    prefixes = config.get("prefixes", [])
    suffixes = config.get("suffixes", [])
    account_phrases = config.get("account_phrases", [])

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_row(text: str, intent: str) -> None:
        key = (text.lower(), intent)
        if key in seen:
            return
        seen.add(key)
        rows.append({"text": text, "intent": intent})

    for intent, templates in intents.items():
        for template in templates:
            rendered = _render_template(template, slots, max_combos=MAX_COMBOS)
            for text in rendered:
                for variant in _variants(text, typos):
                    add_row(variant, intent)

    if len(rows) < TARGET_SIZE:
        base_rows = list(rows)
        while len(rows) < TARGET_SIZE:
            base = rng.choice(base_rows)
            augmented = _augment_with_affixes(base["text"], prefixes, suffixes, slots, rng)
            add_row(augmented, base["intent"])
            if len(rows) >= TARGET_SIZE:
                break
            if account_phrases:
                phrase = rng.choice(account_phrases)
                add_row(f"{base['text']} {phrase}", base["intent"])
            if len(rows) >= TARGET_SIZE:
                break
            add_row(base["text"].replace("?", ""), base["intent"])

    train, val, test = _split_dataset(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUT_DIR / "intent_train.jsonl", train)
    _write_jsonl(OUT_DIR / "intent_val.jsonl", val)
    _write_jsonl(OUT_DIR / "intent_test.jsonl", test)

    print(f"Generated {len(rows)} examples")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
