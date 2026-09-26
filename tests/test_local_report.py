"""Offline checks for the measured-results narrative; no hardware or network."""

import tempfile
import unittest
from pathlib import Path

from spinq_local.report import FIELDS, Results


class ReportNarrativeTests(unittest.TestCase):
    def test_report_distinguishes_measured_diagnostics_from_validated_claims(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Results(Path(folder), {"blocks": 10}, {})
            data = results.data
            data["hardware_results_present"] = True
            data["budgets"]["acquisitions_used"] = 149
            data["pilot"] = {
                "rabi": {"period_us": 152, "t90_us": 38},
                "reference_frequency_hz": -1570.673828125,
                "multiplet_bands_hz": [[-1810.673828125, -1570.673828125]],
                "fid_acquisition_plan": {"status": "MODEL_MISMATCH"},
            }
            data["modules"]["F"]["status"] = "REFERENCE_INADEQUATE"
            for block, short_error, full_error in [(0, 120, 10), (1, 140, 20)]:
                data["rows"] += [
                    {"module": "B", "method": "diagnostic_8000", "block": block,
                     "error": short_error},
                    {"module": "B", "method": "fixed_16000", "block": block,
                     "error": full_error},
                ]
            for block, raw, tv, hankel in [("block-00", 10, 9, 7),
                                           ("block-01", 12, 10, 8),
                                           ("block-02", 14, 13, 9)]:
                data["rows"] += [
                    {"module": "F", "method": method, "block": block, "error": error}
                    for method, error in [("unchanged", raw),
                                          ("complex_TV", tv),
                                          ("randomized_Hankel", hankel)]
                ]
            for method in ("coarse", "sequential", "nelder_mead", "gp"):
                data["rows"].append({"module": "H", "method": method,
                                     "block": 0, "error": .2})
            for row in data["rows"]:
                for field in FIELDS:
                    row.setdefault(field, None)
            report = results.markdown()
            self.assertIn("149", report)
            self.assertIn("10 meracích blokov", report)
            self.assertNotIn("Tri pilotné bloky", report)
            self.assertIn("2 párov blokov", report)
            self.assertIn("130 Hz", report)
            self.assertIn("okraji fit pásma", report)
            self.assertIn("MODEL_MISMATCH", report)
            self.assertIn("3 odložené meracie bloky", report)
            self.assertIn("3/3 párov", report)
            self.assertIn("nadradenosť nie je preukázaná", report)
            self.assertIn("4 fyzických skúšobných pulzov", report)
            self.assertIn("nejde o vernosť kvantovej brány", report)

    def test_offline_reanalysis_points_to_original_raw_data(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Results(Path(folder), {"blocks": 3}, {})
            results.data["reanalysis"] = {"source_run": "first_run",
                                          "source_raw": "../raw",
                                          "offline_only": True}
            report = results.markdown()
            self.assertIn("offline prepočet", report)
            self.assertIn("`../raw/`", report)
            self.assertNotIn("lokálne v `raw/`", report)


if __name__ == "__main__":
    unittest.main()
