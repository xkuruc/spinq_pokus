"""Offline regressions for the live queue gate and runtime owner assertion.

These tests create no SpinQLabLink connection and submit no experiment.
"""

from __future__ import annotations

import tempfile
import time
import unittest
import json
import types
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from bayes_01_windows import main as windows_main
from experiments.bayes_calibration import BayesRun
from spinq_benchmark.hardware import HardwareUncertain, LiveHardware, physical_request


def hardware_with_queue(root: Path, queue, *, exclusive: bool) -> LiveHardware:
    hardware = LiveHardware(root, exclusive_use_confirmed=exclusive)
    hardware.link = Mock()
    hardware.link.get_connection.return_value = True
    hardware.adapter = SimpleNamespace(
        fresh=lambda *_: {"connected": True, "lockState": True,
                          "temperature": 23.0},
        queue=queue,
        lock_lost_observed=False,
        decoder_failures=0,
        own_task_ids=set(),
        pending_own_ack=False,
        ack_mismatch=False,
    )
    hardware.recorder = SimpleNamespace(status=lambda: {"complete": True})
    return hardware


class QueueGateTests(unittest.TestCase):
    def test_missing_queue_without_runtime_assertion_stops_before_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            hardware = hardware_with_queue(Path(folder), None, exclusive=False)
            with self.assertRaisesRegex(HardwareUncertain, "Queue unavailable"):
                hardware.measure("pilot_40", physical_request())
            hardware.link.register_experiment.assert_not_called()
            hardware.link.run_experiment.assert_not_called()
            self.assertEqual(hardware.journal, {})
            self.assertEqual(hardware.task_count, 0)

    def test_runtime_assertion_allows_missing_or_stale_queue_only(self):
        with tempfile.TemporaryDirectory() as folder:
            hardware = hardware_with_queue(Path(folder), None, exclusive=True)
            preflight = hardware._preflight()
            self.assertFalse(preflight["queue_fresh"])
            self.assertFalse(preflight["queue_empty"])

            hardware.adapter.queue = (
                time.monotonic_ns() - 121_000_000_000, {"queue": []})
            stale = hardware._preflight()
            self.assertFalse(stale["queue_fresh"])
            self.assertFalse(stale["queue_empty"])

    def test_fresh_busy_queue_blocks_even_with_runtime_assertion(self):
        with tempfile.TemporaryDirectory() as folder:
            hardware = hardware_with_queue(
                Path(folder), (time.monotonic_ns(), {"queue": [{"id": "other"}]}),
                exclusive=True)
            with self.assertRaisesRegex(HardwareUncertain, "queue is occupied"):
                hardware.measure("pilot_40", physical_request())
            hardware.link.register_experiment.assert_not_called()
            hardware.link.run_experiment.assert_not_called()
            self.assertEqual(hardware.journal, {})

    def test_fresh_empty_queue_needs_no_runtime_assertion(self):
        with tempfile.TemporaryDirectory() as folder:
            hardware = hardware_with_queue(
                Path(folder), (time.monotonic_ns(), {"queue": []}), exclusive=False)
            preflight = hardware._preflight()
            self.assertTrue(preflight["queue_fresh"])
            self.assertTrue(preflight["queue_empty"])

    def test_queue_becoming_busy_during_pause_blocks_before_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            hardware = hardware_with_queue(
                Path(folder), (time.monotonic_ns(), {"queue": []}), exclusive=False)
            hardware.pause = 2.
            hardware.last_finished = time.monotonic()
            request = physical_request()
            experiment = SimpleNamespace(
                id="local-exp",
                get_experiment_parameter=lambda: {"params": json.dumps(request)})
            hardware.link.register_experiment.return_value = (experiment, object())
            sdk = types.ModuleType("spinqlablink")
            sdk.ExperimentType = SimpleNamespace(
                PHYSICAL_LAYER_EXPERIMENT=object())

            def occupy_queue(_seconds):
                hardware.adapter.queue = (
                    time.monotonic_ns(), {"queue": [{"id": "other"}]})

            with patch("spinq_benchmark.hardware.time.sleep",
                       side_effect=occupy_queue) as sleeper, \
                 patch("spinq_benchmark.hardware._configure_physical"), \
                 patch.dict("sys.modules", {"spinqlablink": sdk}):
                with self.assertRaisesRegex(HardwareUncertain, "queue is occupied"):
                    hardware.measure("pilot_40", request)
            sleeper.assert_called_once()
            hardware.link.register_experiment.assert_not_called()
            hardware.link.run_experiment.assert_not_called()
            self.assertEqual(hardware.journal, {})
            self.assertEqual(hardware.task_count, 0)

    def test_cli_confirmation_reaches_run_without_changing_saved_config(self):
        with patch("bayes_01_windows.offline_preflight", return_value={"offline": True}), \
             patch("bayes_01_windows.BayesRun") as run_class, \
             redirect_stdout(StringIO()):
            run_class.return_value.execute.return_value = 0
            status = windows_main(["--exclusive-use-confirmed"])
        self.assertEqual(status, 0)
        self.assertTrue(run_class.call_args.kwargs["exclusive_use_confirmed"])
        self.assertFalse(run_class.call_args.args[2]["exclusive_use_confirmed"])

    def test_queue_precheck_pause_is_retryable_and_does_not_publish_empty_run(self):
        # An absent push queue message must not be reported as a submitted,
        # uncertain hardware task. The same saved run can be resumed safely.
        from spinq_benchmark.hardware import QueuePreflightUnavailable

        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-01-bayes.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            run = BayesRun(repo, Path(folder) / "run", config, {"offline": True})
            fake_hardware = MagicMock()
            fake_hardware.__enter__.return_value = fake_hardware
            fake_hardware.__exit__.return_value = False
            fake_hardware.task_count = 0
            with patch("experiments.bayes_calibration.HardwareLock",
                       return_value=nullcontext()), \
                 patch("experiments.bayes_calibration.LiveHardware",
                       return_value=fake_hardware), \
                 patch.object(run, "run_pilot",
                              side_effect=QueuePreflightUnavailable("queue absent")), \
                 patch("experiments.bayes_calibration.plot_comparisons"), \
                 patch("experiments.bayes_calibration.publish_results") as publish, \
                 redirect_stdout(StringIO()):
                code = run.execute()
            self.assertEqual(code, 2)
            self.assertEqual(run.data["state"], "PAUSED_QUEUE")
            self.assertEqual(run.data["hardware_tasks_completed"], 0)
            publish.assert_not_called()
            resumed = BayesRun(repo, run.out, config, {"offline": True}, resume=True)
            self.assertTrue(resumed.resuming)


if __name__ == "__main__":
    unittest.main()
