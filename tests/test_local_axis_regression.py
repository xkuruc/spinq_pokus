"""Offline regressions for the exported SpinQ chart axis and failed-run archive."""

import gzip
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spinq_benchmark.hardware import LiveHardware
from spinq_local.core import (Capabilities, IncompleteFID, RawFIDRecord, Segment,
                              SequenceIR, assemble_fid, compile_sequence, run_raw)
from spinq_local.report import bundle_complete
from spinq_local.report import Results
from spinq_local.signal import validate_axis


def fid_events(axis):
    common = {"taskId": "EXP_LAYER_PHYSICAL-example", "group": "exp_layer_physical",
              "path": "0", "qubit": "0", "step": "NMRSIG"}
    values = np.exp(-np.arange(len(axis)) / 10000).astype(float)

    def chart(name, y):
        return {"kind": "s_post_exp_chart_updated", "payload": {"chart_data": {
            **common, "chart_name": name,
            "points": np.column_stack((axis, y)).tolist()}}}

    return [chart("fidRe", values), chart("fidIm", np.zeros(len(axis))),
            {"kind": "s_post_exp_chart_updated_finished",
             "payload": {"json_data": common}},
            {"kind": "s_post_exp_finished", "payload": {"json_data": common}}]


class LocalAxisRegressionTests(unittest.TestCase):
    def test_float32_16000_point_chart_axis_is_uniform_at_export_precision(self):
        # Protobuf chart coordinates are float32; subtracting neighboring large
        # coordinates does not preserve an exactly constant 0.1 ms step.
        axis = np.asarray(np.arange(16000, dtype=np.float64) * .1,
                          dtype=np.float32).astype(np.float64)
        record = assemble_fid(fid_events(axis), "EXP_LAYER_PHYSICAL-example",
                              key="pilot_40_r0",
                              parameters_sent={"sampleFre": 10000,
                                               "sampleCount": 16000})
        contract = validate_axis(record)
        self.assertEqual(contract.point_count, 16000)
        self.assertEqual(contract.sample_hz, 10000)
        self.assertAlmostEqual(contract.seconds_per_original_unit, .001, delta=1e-6)
        self.assertAlmostEqual(record.time_seconds[-1], 1.5999, places=7)
        self.assertFalse(contract.physical_clock_verified)

    def test_irregular_chart_axis_still_rejected(self):
        axis = np.asarray(np.arange(16000, dtype=np.float64) * .1,
                          dtype=np.float32).astype(np.float64)
        axis[8000] += .02  # Still increasing, but too large for float32 rounding.
        with self.assertRaises(IncompleteFID):
            assemble_fid(fid_events(axis), "EXP_LAYER_PHYSICAL-example",
                         parameters_sent={"sampleFre": 10000,
                                          "sampleCount": 16000})
        record = RawFIDRecord(
            key="irregular", task_id="example", group="exp_layer_physical",
            path="0", qubit="0", step="NMRSIG", axis_original=axis,
            time_seconds=np.arange(len(axis)) / 10000,
            re=np.ones(len(axis)), im=np.zeros(len(axis)),
            parameters_sent={"sampleFre": 10000, "sampleCount": 16000})
        with self.assertRaises(ValueError):
            validate_axis(record)

    def test_archive_preserves_chart_events_when_fid_assembly_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "data").mkdir()
            (out / "data" / "pilot_40_r0.json").write_text(
                '{"chart":"original exported FID"}', encoding="utf-8")
            (out / "REPORT.md").write_text("pilot failed", encoding="utf-8")
            archive = bundle_complete(out)
            with zipfile.ZipFile(archive) as z:
                self.assertIn("data/pilot_40_r0.json", z.namelist())
                self.assertEqual(z.read("data/pilot_40_r0.json"),
                                 b'{"chart":"original exported FID"}')

    def test_resume_replays_completed_task_from_gzip_only_events(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            data = out / "data"
            raw = out / "raw"
            data.mkdir()
            raw.mkdir()
            key = "pilot_40_r0"
            task_id = "EXP_LAYER_PHYSICAL-example"
            spec = compile_sequence(
                SequenceIR((Segment(0, 40),), sample_count=4000), Capabilities())
            (data / "hardware_journal.json").write_text(json.dumps({key: {
                "phase": "completed", "requested_rf_us": 40,
                "result_file": f"data/{key}.json"}}), encoding="utf-8")
            (data / f"{key}.json").write_text(json.dumps({
                "task_id": task_id, "params": spec.payload,
                "wall_seconds": 4.0, "finished_utc": "2026-09-26T13:34:17Z"}),
                encoding="utf-8")
            (raw / f"{key}.error.json").write_text(json.dumps({
                "task_id": task_id, "error": "previous axis check failed"}),
                encoding="utf-8")
            axis = np.asarray(np.arange(4000, dtype=np.float64) * .1,
                              dtype=np.float32).astype(np.float64)
            with gzip.open(data / "events.jsonl.gz", "wt", encoding="utf-8") as stream:
                for event in fid_events(axis):
                    stream.write(json.dumps(event) + "\n")
            self.assertFalse((data / "events.jsonl").exists())
            hardware = LiveHardware(out)  # Constructor and cached measure use no network.
            record = run_raw(spec, key=key, hardware=hardware, output=out)
            self.assertEqual(record.task_id, task_id)
            self.assertEqual(len(record.fid), 4000)
            self.assertEqual(json.loads((raw / f"{key}.error.json").read_text())
                             ["status"], "RECOVERED_FROM_SAVED_EVENTS")
            self.assertTrue((raw / f"{key}.npz").is_file())

    def test_failed_pilot_keeps_measured_data_but_never_publishes_empty_report(self):
        from local_benchmark_windows import _finalize
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            results = Results(out, {}, {})
            results.data["state"] = "FAILED_PILOT"
            (out / "data").mkdir()
            (out / "data" / "hardware_journal.json").write_text(json.dumps({
                "pilot_40_r0": {"phase": "completed", "requested_rf_us": 40}}),
                encoding="utf-8")
            (out / "data" / "pilot_40_r0.json").write_text(
                '{"chart":"measured FID"}', encoding="utf-8")
            with patch("local_benchmark_windows.publish_summary") as publish:
                with redirect_stdout(io.StringIO()):
                    _finalize(out, results, upload=True)
                publish.assert_not_called()
            self.assertTrue(results.data["hardware_results_present"])
            self.assertEqual(results.data["upload"]["status"],
                             "UPLOAD_SKIPPED_INCOMPLETE")
            self.assertFalse((out / "results.zip").exists())
            self.assertEqual((out / "data" / "pilot_40_r0.json").read_text(),
                             '{"chart":"measured FID"}')
            self.assertEqual(results.data["archive"]["status"],"NOT_CREATED_THIS_RUN")
            with redirect_stdout(io.StringIO()):
                _finalize(out,results,upload=False,full_archive=True)
            with zipfile.ZipFile(out / "results.zip") as archive:
                self.assertIn("data/pilot_40_r0.json",archive.namelist())

    def test_completed_run_publishes_summary_only(self):
        from local_benchmark_windows import _finalize
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            results = Results(out, {}, {})
            results.data["state"] = "COMPLETED_WITH_EXPLICIT_LIMITATIONS"
            results.data["hardware_results_present"] = True
            results.row(module="B", method="fixed_16000", task="B_b00_fixed_16000",
                        block=0, acquisitions=1, status="REFERENCE_INADEQUATE")
            with patch("local_benchmark_windows.publish_summary",
                       return_value={"status": "UPLOAD_SUCCEEDED"}) as publish:
                with redirect_stdout(io.StringIO()):
                    _finalize(out, results, upload=True)
            self.assertEqual(publish.call_count, 1)
            self.assertEqual(publish.call_args.args[1], out)
            self.assertEqual(results.data["upload"]["status"], "UPLOAD_SUCCEEDED")


if __name__ == "__main__":
    unittest.main()
