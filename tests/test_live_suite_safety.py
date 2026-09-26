"""Offline checks for the live suite's exact historical request gate."""

import copy
import unittest
from pathlib import Path

from spinq_live_suite import (
    HISTORICAL_PHYSICAL_BASELINE, STUDY_MAX_CUMULATIVE_REQUESTED_RF_US,
    check_case, load_config, make_cases,
)


class HistoricalBaselineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "live_suite.example.toml")
        self.cases = make_cases(self.config)
        self.baseline = next(case for case in self.cases if case["id"] == "physical_baseline")

    def test_only_one_exact_historical_physical_request_is_allowed(self):
        self.config["historical_baseline_only"] = True
        self.assertEqual(self.baseline["params"], HISTORICAL_PHYSICAL_BASELINE)
        self.assertEqual(check_case(self.baseline, self.config, 0), 40.0)
        for case in self.cases:
            if case is not self.baseline:
                with self.subTest(case=case["id"]), self.assertRaises(ValueError):
                    check_case(case, self.config, 0)
        with self.assertRaises(ValueError):
            check_case(self.baseline, self.config, 40)

    def test_editing_historical_request_does_not_bypass_missing_limits(self):
        self.config["historical_baseline_only"] = True
        for field, value in (("width", 41), ("am", 99), ("phase", 91)):
            changed = copy.deepcopy(self.baseline)
            changed["params"]["pulse"]["hPulse"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_case(changed, self.config, 0)
        changed = copy.deepcopy(self.baseline)
        changed["params"]["relaxation_time"] = 16
        with self.assertRaises(ValueError):
            check_case(changed, self.config, 0)

    def test_bounded_study_covers_real_series_and_rejects_larger_changes(self):
        self.assertIs(self.config["bounded_study_enabled"], True)
        total = 0.0
        for case in self.cases:
            if case.get("enabled") is False:
                continue
            with self.subTest(case=case["id"]):
                total += check_case(case, self.config, total)
        self.assertLessEqual(total + 44, STUDY_MAX_CUMULATIVE_REQUESTED_RF_US)
        changed = copy.deepcopy(self.baseline)
        changed["params"]["pulse"]["hPulse"][0]["width"] = 80
        with self.assertRaises(ValueError):
            check_case(changed, self.config, 0)
        changed = copy.deepcopy(self.baseline)
        changed["params"]["pulse"]["pPulse"] = [{"width": 40, "am": 100, "phase": 90, "freshift": 0}]
        with self.assertRaises(ValueError):
            check_case(changed, self.config, 0)
        with self.assertRaises(ValueError):
            check_case(self.baseline, self.config, STUDY_MAX_CUMULATIVE_REQUESTED_RF_US-10)


if __name__ == "__main__":
    unittest.main()
