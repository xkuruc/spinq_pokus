"""Receive-only event slicing and delayed FID chart regression tests."""

import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from spinq_benchmark.hardware import (LiveHardware, PreSubmissionFailure,
    last_complete_event_offset, physical_request, wait_completed_fid_events)
from spinq_local.core import (ExperimentSpec, Segment, SequenceIR,
                              read_task_events, run_raw)


def chart_events(task_id, count=64):
    base = {"taskId": task_id, "group": "physical", "path": "0",
            "qubit": "0", "step": "NMRSIG"}
    points = [[i * .1, float(i)] for i in range(count)]
    def chart(name, values):
        return {"kind": "s_post_exp_chart_updated", "payload": {"chart_data": {
            **base, "chart_name": name, "points": values}}}
    return [
        {"kind": "s_post_exp_finished", "payload": {"json_data": base}},
        chart("fidRe", points),
        chart("fidIm", [[x, -y] for x, y in points]),
        {"kind": "s_post_exp_chart_updated_finished", "payload": {"json_data": base}},
    ]


def append_events(path, events):
    with path.open("ab") as target:
        for event in events:
            target.write(json.dumps(event).encode("utf-8") + b"\n")
            target.flush()


class QuietRecorder:
    def __init__(self):
        self.queue = SimpleNamespace(unfinished_tasks=0)

    def status(self):
        return {"complete": True}


class EventCaptureTests(unittest.TestCase):
    def test_start_offset_is_journaled_before_run_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            hw = LiveHardware.__new__(LiveHardware)
            hw.out = Path(folder)
            hw.data = hw.out / "data"
            hw.data.mkdir()
            old = b'{"kind":"old"}\n'
            (hw.data / "events.jsonl").write_bytes(old + b'{"partial"')
            hw.journal_path = hw.data / "hardware_journal.json"
            hw.journal = {}
            hw.task_count = 0
            hw.rf_us = 0.
            hw.max_tasks = 180
            hw.max_rf = 12000.
            hw.pause = 0.
            hw.last_finished = 0.
            hw.halted = False
            hw.adapter = SimpleNamespace(own_task_ids=set(), pending_own_ack=False,
                                         ack_mismatch=False)
            exp = SimpleNamespace(id="new-task", get_experiment_parameter=lambda: {
                "params": json.dumps(physical_request())})
            hw.link = Mock()
            hw.link.register_experiment.return_value = (exp, object())
            sdk = types.ModuleType("spinqlablink")
            sdk.ExperimentType = SimpleNamespace(PHYSICAL_LAYER_EXPERIMENT=object())
            captured = {}

            def fail_checkpoint(path, payload):
                if path == hw.journal_path:
                    captured.update(payload["test-key"])
                    raise PermissionError("local journal unavailable")

            with patch.object(hw, "_preflight", return_value={}), \
                 patch("spinq_benchmark.hardware._configure_physical"), \
                 patch("spinq_benchmark.hardware.atomic_json", side_effect=fail_checkpoint), \
                 patch("spinq_benchmark.hardware.time.sleep"), \
                 patch.dict(sys.modules, {"spinqlablink": sdk}):
                with self.assertRaises(PreSubmissionFailure):
                    hw.measure("test-key", physical_request())
            self.assertEqual(captured["event_start_byte_offset"], len(old))
            hw.link.run_experiment.assert_not_called()

    def test_binary_offsets_skip_prior_data_and_retry_partial_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            path.write_bytes(b"not valid json before task\n" + b'{"kind":"other"')
            start = last_complete_event_offset(path)
            self.assertEqual(start, len(b"not valid json before task\n"))
            events, cursor = read_task_events(path, "new", start)
            self.assertEqual(events, [])
            self.assertEqual(cursor, start)
            with path.open("ab") as target:
                target.write(b',"payload":{"json_data":{"taskId":"old"}}}\n')
            append_events(path, chart_events("new"))
            events, cursor = read_task_events(path, "new", start)
            self.assertEqual(len(events), 4)
            self.assertEqual(cursor, path.stat().st_size)

    def test_terminal_before_late_charts_waits_and_never_submits(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            first, *rest = chart_events("task-late")
            append_events(path, [first])
            recorder = QuietRecorder()

            def late_writer():
                time.sleep(.08)
                append_events(path, rest[:1])
                time.sleep(.08)
                append_events(path, rest[1:])

            worker = threading.Thread(target=late_writer)
            worker.start()
            try:
                capture = wait_completed_fid_events(recorder, path, "task-late",
                    {"sampleFre": 10000, "sampleCount": 64}, 0,
                    seconds=1., settle_seconds=.05)
            finally:
                worker.join()
            self.assertEqual(capture["fid_capture_status"], "COMPLETE")
            self.assertEqual(capture["captured_task_events"], 4)
            self.assertEqual(capture["event_end_byte_offset"], path.stat().st_size)

    def test_timeout_keeps_incomplete_reason_and_does_not_fabricate_fid(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            append_events(path, chart_events("only-terminal")[:1])
            capture = wait_completed_fid_events(QuietRecorder(), path,
                "only-terminal", {"sampleFre": 10000, "sampleCount": 64},
                0, seconds=.1, settle_seconds=.01)
            self.assertEqual(capture["fid_capture_status"], "INCOMPLETE")
            self.assertIn("fidRe", capture["fid_capture_reason"])

    def test_run_raw_resumes_saved_record_without_measuring_again(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            data = output / "data"
            data.mkdir()
            path = data / "events.jsonl"
            append_events(path, chart_events("task-once"))
            class FakeHardware:
                def __init__(self):
                    self.data = data
                    self.calls = 0

                def measure(self, key, payload, *, allow_idle_probe=False):
                    self.calls += 1
                    return {"task_id": "task-once", "event_start_byte_offset": 0,
                            "event_end_byte_offset": path.stat().st_size,
                            "fid_capture_status": "COMPLETE"}

            hardware = FakeHardware()
            payload = {"sampleFre": 10000, "sampleCount": 64}
            sequence = SequenceIR((Segment(0., 40.),), sample_count=64)
            spec = ExperimentSpec(payload, sequence, 40., 40.)
            first = run_raw(spec, key="one", hardware=hardware, output=output)
            second = run_raw(spec, key="one", hardware=hardware, output=output)
            self.assertEqual(hardware.calls, 1)
            self.assertEqual(first.task_id, second.task_id)
            self.assertEqual(first.metadata["event_end_byte_offset"], path.stat().st_size)


if __name__ == "__main__":
    unittest.main()
