"""Download the public OpenF1 data used by this analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

SESSION_KEY = 11369
BASE_URL = "https://api.openf1.org/v1"
ENDPOINTS = (
    "drivers",
    "laps",
    "stints",
    "pit",
    "race_control",
    "weather",
    "position",
    "intervals",
    "session_result",
)
OUTPUT_DIR = Path(__file__).parent / "data" / "raw"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for endpoint in ENDPOINTS:
        response = requests.get(
            f"{BASE_URL}/{endpoint}",
            params={"session_key": SESSION_KEY},
            timeout=60,
        )
        response.raise_for_status()
        records = response.json()
        destination = OUTPUT_DIR / f"{endpoint}.json"
        destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Saved {len(records):,} {endpoint} records")
        time.sleep(0.4)


if __name__ == "__main__":
    main()

