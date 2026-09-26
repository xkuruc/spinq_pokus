"""Offline checks for diagnostics shown to the Windows benchmark operator."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from local_benchmark_windows import _print_pilot, _run_guarded
from spinq_local.report import Results


class ConsoleDiagnosticsTests(unittest.TestCase):
    def test_partial_row_and_final_module_are_printed_and_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            capture = io.StringIO()
            with redirect_stdout(capture):
                results = Results(out, {}, {})
                results.row(module="B", method="adaptive", baseline="fixed_16000",
                            task="B_b00_adaptive", block=0, acquisitions=1,
                            error=2.5, tolerance=3.0, status="RUNNING")
                results.module("B", "REFERENCE_INADEQUATE",
                               "independent frequency reference unavailable")
            shown = capture.getvalue()
            self.assertIn("ROW B block=0 method=adaptive task=B_b00_adaptive", shown)
            self.assertIn("error=2.5 tolerance=3.0 status=RUNNING", shown)
            self.assertIn("MODULE B: REFERENCE_INADEQUATE: independent frequency reference unavailable", shown)
            saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["rows"][0]["status"], "REFERENCE_INADEQUATE")
            self.assertEqual(saved["modules"]["B"]["status"], "REFERENCE_INADEQUATE")
            with redirect_stdout(capture := io.StringIO()):
                Results(out, {}, {})
            self.assertEqual(capture.getvalue(), "")

    def test_guarded_failure_prints_stage_and_trace_and_persists_details(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            results = Results(out, {}, {})

            def bad_map():
                value = 1 + 2j
                return value["coefficient"]

            capture = io.StringIO()
            with redirect_stdout(capture):
                outcome = _run_guarded(results, "H", bad_map, stage="rf_map")
            self.assertIsNone(outcome)
            shown = capture.getvalue()
            self.assertIn("MODULE H: METHOD_FAILED: rf_map: TypeError:", shown)
            self.assertIn("TRACE H/rf_map:", shown)
            self.assertIn("bad_map", shown)
            self.assertIn("'complex' object is not subscriptable", shown)
            saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["modules"]["H"]["status"], "METHOD_FAILED")
            self.assertEqual(saved["error_details"][0]["stage"], "rf_map")
            self.assertIn("bad_map", saved["error_details"][0]["traceback"])

    def test_pilot_substep_failures_and_partial_plan_are_printed(self):
        pilot = {
            "rabi": {"status": "METHOD_FAILED", "reason": "fringe absent"},
            "multiplet_bands_hz": [[-100.0, 100.0]],
            "fid_acquisition_plan": {"status": "REFERENCE_INADEQUATE"},
            "failures": {"independent_noise": "ValueError: no repeats",
                         "rabi_t90": "ValueError: fringe absent"},
        }
        session = SimpleNamespace(t90_us=None, noise=None)
        capture = io.StringIO()
        with redirect_stdout(capture):
            _print_pilot(pilot, session)
        shown = capture.getvalue()
        self.assertIn("PILOT: Rabi status=METHOD_FAILED", shown)
        self.assertIn("noise=UNAVAILABLE", shown)
        self.assertIn("PILOT WARNING independent_noise: ValueError: no repeats", shown)
        self.assertIn("PILOT WARNING rabi_t90: ValueError: fringe absent", shown)
        self.assertIn("FID length plan={'status': 'REFERENCE_INADEQUATE'}", shown)


if __name__ == "__main__":
    unittest.main()
