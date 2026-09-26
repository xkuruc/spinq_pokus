"""The Windows reanalysis path must reuse measured FIDs without a device."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reanalyze_saved import load_saved_records, reanalyze, reclassify_clipped_b_reference
from spinq_local.core import RawFIDRecord
from spinq_local.report import Results


class SavedReanalysisTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        source = root / "results" / "completed_run"
        results = Results(source, {"blocks": 3}, {"torch": {"ready": False}})
        results.data["state"] = "COMPLETED_WITH_EXPLICIT_LIMITATIONS"
        results.data["pilot"] = {"rabi": {"t90_us": 38.0}}
        results.data["hardware_results_present"] = True
        results.save()
        record = RawFIDRecord(
            key="pilot_40_r0", task_id="measured-task", group="physical",
            path="H", qubit="0", step="1",
            axis_original=np.arange(64, dtype=float),
            time_seconds=np.arange(64, dtype=float) / 10000,
            re=np.ones(64), im=np.zeros(64), parameters_sent={},
            metadata={"dataset_role": "pilot", "measurement_block": "pilot",
                      "setting_family": "repeat"},
        )
        record.save(source / "raw")
        data = source / "data"
        data.mkdir()
        (data / "hardware_journal.json").write_text(json.dumps({
            "pilot_40_r0": {"phase": "completed"}}), encoding="utf-8")
        return source

    def test_reanalysis_preserves_source_and_passes_saved_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self._source(Path(folder))
            watched = {name: (source / name).read_bytes() for name in (
                "results.json", "raw/pilot_40_r0.npz", "raw/pilot_40_r0.json",
                "data/hardware_journal.json")}

            def offline_analysis(roles, pilot, out, runtime, results):
                self.assertEqual([r.key for r in roles["pilot"]], ["pilot_40_r0"])
                self.assertEqual(sum(len(v) for v in roles.values()), 1)
                self.assertEqual(pilot["rabi"]["t90_us"], 38.0)
                results.module("D", "DEPENDENCY_FAILED", "measured model incomplete")
                results.module("E", "DEPENDENCY_FAILED", "no physical gate reference")
                results.module("F", "REFERENCE_INADEQUATE", "few blocks")
                return {"D": {"status": "DEPENDENCY_FAILED"}}

            with patch("reanalyze_saved.check_runtime", return_value={
                     "numeric": {"ready": True}, "torch": {"ready": False}}), \
                 patch("reanalyze_saved.analyze_de_fg", side_effect=offline_analysis), \
                 patch("reanalyze_saved.plots_from_results"), \
                 patch("reanalyze_saved.publish_summary") as publish:
                out = reanalyze(source)
            publish.assert_not_called()
            self.assertEqual(out.parent, source.resolve())
            derived = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(derived["reanalysis"]["saved_task_count"], 1)
            self.assertEqual(derived["state"],
                             "COMPLETED_OFFLINE_REANALYSIS_WITH_EXPLICIT_LIMITATIONS")
            for name, content in watched.items():
                self.assertEqual((source / name).read_bytes(), content)

    def test_unresolved_journal_is_rejected_before_reanalysis(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self._source(Path(folder))
            journal = source / "data" / "hardware_journal.json"
            journal.write_text(json.dumps({
                "pilot_40_r0": {"phase": "completed"},
                "other": {"phase": "sent_unconfirmed"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unresolved hardware journal"):
                load_saved_records(source)
            journal.write_text(json.dumps({
                "pilot_40_r0": {"phase": "completed"}, "other": None}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "INVALID_ENTRY"):
                load_saved_records(source)

    def test_clipped_legacy_b_reference_becomes_diagnostic_only(self):
        data = {"pilot": {"reference_frequency_hz": -1570.673828125,
                          "multiplet_bands_hz": [[-1810.673828125, -1570.673828125]]},
                "modules": {"B": {"status": "METHOD_FAILED", "reason": "frequency changed"}},
                "rows": [{"module": "B", "status": "METHOD_FAILED", "error": 138.2}],
                "reanalysis": {}}
        reclassify_clipped_b_reference(data)
        self.assertEqual(data["modules"]["B"]["status"], "REFERENCE_INADEQUATE")
        self.assertIn("clipped", data["modules"]["B"]["reason"])
        self.assertEqual(data["rows"][0]["status"], "REFERENCE_INADEQUATE")
        self.assertEqual(data["rows"][0]["error"], 138.2)
        self.assertTrue(data["reanalysis"]["legacy_b_reference_reclassified"])


if __name__ == "__main__":
    unittest.main()
