#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"
MASTER_PATH = DATA_DIR / "user_master.json"

SECTION_MAP = {
    "account_summary": "account.json",
    "activity": "activity.json",
    "positions": "positions.json",
    "quotes": "quotes.json",
    "performance": "performance.json",
    "transfers": "transfers.json",
}


def main() -> int:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master file: {MASTER_PATH}")

    payload = json.loads(MASTER_PATH.read_text())
    for section, filename in SECTION_MAP.items():
        data = payload.get(section)
        if data is None:
            raise KeyError(f"Missing section '{section}' in {MASTER_PATH.name}")
        if "as_of" not in data and payload.get("as_of"):
            data = {**data, "as_of": payload["as_of"]}
        path = DATA_DIR / filename
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
