import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_race import vsc_window


def load(name):
    with (ROOT / "data" / "raw" / f"{name}.json").open(encoding="utf-8") as source:
        return json.load(source)


class MadridRaceDataTests(unittest.TestCase):
    def test_podium_matches_session_result(self):
        classified = [row for row in load("session_result") if row["position"] is not None]
        results = sorted(classified, key=lambda row: row["position"])
        self.assertEqual([row["driver_number"] for row in results[:3]], [12, 3, 1])

    def test_vsc_window_is_present_and_ordered(self):
        race_control = pd.DataFrame(load("race_control"))
        race_control["date"] = pd.to_datetime(
            race_control["date"], utc=True, format="mixed"
        )
        start, end = vsc_window(race_control)
        self.assertLess(start, end)
        self.assertEqual((end - start).total_seconds(), 123)

    def test_podium_pit_timing(self):
        pits = {row["driver_number"]: row for row in load("pit") if row["driver_number"] in (1, 3, 12)}
        self.assertEqual(pits[12]["lap_number"], 14)
        self.assertEqual(pits[3]["lap_number"], 14)
        self.assertEqual(pits[1]["lap_number"], 15)


if __name__ == "__main__":
    unittest.main()
