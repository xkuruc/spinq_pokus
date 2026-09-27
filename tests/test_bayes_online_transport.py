"""Offline transport tests; none opens a socket or submits hardware work."""

import unittest
from dataclasses import replace

import numpy as np

from bayes_online_core.transport import (
    AcquisitionRequest, AcquisitionResult, IncompleteData, PhysicalTransport, PulseSpec,
    TransportError, UnsupportedTiming, _prepare_graph, _schedule,
)


class TransportValidationTests(unittest.TestCase):
    def test_negative_detuning_and_padding_are_serialized_without_clipping(self):
        req = AcquisitionRequest(
            "x", (
                PulseSpec(0, 0, 40, 80, 90, -27),
                PulseSpec(0, 100, 20, 80, 90, -27),
                PulseSpec(1, 0, 40, 50, 0, 13),
            ), 0)
        segments, evidence = _schedule(req)
        self.assertEqual([(s["path"], s["width"], s["am"]) for s in segments],
                         [(0, 40, 80.0), (0, 60, 0.0), (0, 20, 80.0), (1, 40, 50.0)])
        self.assertEqual(segments[0]["freshift"], -27)
        self.assertTrue(evidence["uses_zero_amplitude_padding"])

    def test_unverified_or_invalid_commands_fail_before_submission(self):
        with self.assertRaises(TransportError):
            _schedule(AcquisitionRequest("x", (PulseSpec(0, 0, 40, 101, 90),), 0))
        with self.assertRaises(UnsupportedTiming):
            _schedule(AcquisitionRequest("x", (PulseSpec(0, 0.5, 40, 80, 90),), 0))
        with self.assertRaises(UnsupportedTiming):
            _schedule(AcquisitionRequest("x", (), -1))
        with self.assertRaises(UnsupportedTiming):
            _schedule(AcquisitionRequest("x", (
                PulseSpec(0, 0, 40, 80, 90), PulseSpec(0, 20, 40, 80, 90)), 0))

    def test_fid_axes_and_vendor_products_are_separate(self):
        x = np.arange(4000) / 10.0  # milliseconds at 10 kHz
        graph = [{"fidRe": np.column_stack((x, np.ones(4000))).tolist(),
                  "fidIm": np.column_stack((x, -np.ones(4000))).tolist(),
                  "fftMod": [[0, 99]]}]
        time_s, fid, exported, vendor, evidence = _prepare_graph(graph, 4000, 10000)
        self.assertAlmostEqual(time_s[1], 1e-4)
        self.assertEqual(evidence["exported_axis_unit_inferred"], "ms")
        self.assertEqual(exported[1], 0.1)
        self.assertEqual(fid[0], 1-1j)
        self.assertEqual(vendor, {"fftMod": [[0, 99]]})
        with self.assertRaises(IncompleteData):
            _prepare_graph([{"fidRe": graph[0]["fidRe"]}], 4000, 10000)

    def test_two_tasks_capture_fids_and_restore_exact_sdk_handlers(self):
        from spinqlablink import SpinQLabLink
        from spinqlablink.utils.types import MachineType

        client = SpinQLabLink("127.0.0.1", 8181, "offline", "offline")
        client.is_connected = True
        client.is_logged_in = True
        transport = PhysicalTransport(exclusive_use_confirmed=True, timeout_s=3)
        transport._client = client
        transport._admit = lambda: {"status": {"connected": True, "lock_state": True}}
        seen = {}
        points_re = [[i / 10000, float(np.cos(i / 50))] for i in range(4000)]
        points_im = [[i / 10000, float(np.sin(i / 50))] for i in range(4000)]
        registered_handlers = []
        original_register = client.register_experiment

        def tracked_register(*args, **kwargs):
            value = original_register(*args, **kwargs)
            registered_handlers.append(dict(client.exp_handler_map))
            return value

        client.register_experiment = tracked_register

        def fake_server_send(message_type, payload):
            self.assertEqual(message_type, MachineType.MSG_REQ_ADD_EXP_TASK_REQ)
            seen.update(payload)
            handlers = client.exp_handler_map
            task_id = f"offline-task-{len(registered_handlers)}"
            handlers[MachineType.MSG_RES_ADD_EXP_TASK_RES]({"code": 0, "taskId": task_id})
            handlers[MachineType.MSG_POST_EXP_STARTED]({"taskId": task_id,
                "queue": [{"startTime": 1, "name": "offline"}]})
            handlers[MachineType.MSG_POST_EXP_CHART_UPDATED_STARTED]({
                "taskId": task_id, "group": "exp_layer_physical"})
            for name, points in (("fidRe", points_re), ("fidIm", points_im),
                                 ("fftMod", [[0, 5]])):
                handlers[MachineType.MSG_POST_EXP_CHART_UPDATED]({
                    "taskId": task_id, "group": "exp_layer_physical",
                    "chart_name": name, "points": points})
            handlers[MachineType.MSG_POST_EXP_CHART_UPDATED_FINISHED]({
                "taskId": task_id, "group": "exp_layer_physical"})
            handlers[MachineType.MSG_POST_EXP_FINISHED]({"taskId": task_id,
                "data": {"isTerminated": False, "parameters": {"result": ""}}})

        client._send_message = fake_server_send
        req = AcquisitionRequest("fid", (PulseSpec(0, 0, 40, 80, 90, -27),), 0,
                                 sample_count=4000, initialize_state=False)
        result = transport.acquire(req)
        self.assertEqual(result.task_id, "offline-task-1")
        self.assertEqual(len(result.fid_complex), 4000)
        self.assertEqual(result.sent_payload["compute_type"], 0)
        self.assertFalse(result.sent_payload["makePps"])
        self.assertEqual(result.sent_payload["pulse"]["hPulse"][0]["freshift"], -27)
        self.assertEqual(result.vendor_reference["graph_step_0"]["fftMod"], [[0, 5]])
        self.assertTrue(any(e["kind"] == "chart_updated" for e in result.events))
        self.assertEqual(seen["account"], "offline")
        self.assertNotIn("account", result.to_dict()["sent_payload"])
        for code, original in registered_handlers[0].items():
            self.assertIs(client.exp_handler_map[code], original)

        second = transport.acquire(replace(req, key="fid_repeat"))
        self.assertEqual(second.task_id, "offline-task-2")
        self.assertEqual(len(second.fid_complex), 4000)
        self.assertEqual(len(registered_handlers), 2)
        for code, original in registered_handlers[1].items():
            self.assertIs(client.exp_handler_map[code], original)

    def test_idle_gap_qualification_requires_shifted_fid_not_just_padding(self):
        transport = PhysicalTransport(exclusive_use_confirmed=True)
        sample_count = 4000
        t = np.arange(sample_count) / 10000

        def signal(time_axis):
            return np.exp(-time_axis / 0.006 + 2j * np.pi * 330 * time_axis)

        def synthetic_acquire(request, *, qualification):
            self.assertTrue(qualification)
            zero = [p for p in request.pulses if p.amplitude_pct == 0]
            delay_s = zero[0].width_us / 1e6 if zero else 0.0
            return AcquisitionResult(request.key, request.key, "COMPLETED", t,
                signal(t + delay_s), {}, {}, {},
                {"axis_status": "EXPORT_MATCHES_REQUESTED_RATE"}, 0.1, {}, [])

        transport._acquire = synthetic_acquire
        saved = []
        report = transport.verify_idle_gap(PulseSpec(0, 0, 40, 80, 90), gap_us=1000,
                                           key_prefix="run1_h_1000", on_acquisition=saved.append)
        self.assertEqual(report["status"], "SAME_CHANNEL_IDLE_VERIFIED")
        self.assertEqual(len(saved), 4)
        self.assertEqual([x.key for x in saved],
                         ["run1_h_1000_idle_h_reference_a", "run1_h_1000_idle_h_1000us",
                          "run1_h_1000_idle_h_2000us", "run1_h_1000_idle_h_reference_b"])
        self.assertTrue(transport.idle_gap_verified(0, 1000))
        self.assertFalse(transport.idle_gap_verified(1, 1000))

    def test_two_segment_callback_runs_before_next_submission_and_can_abort(self):
        transport = PhysicalTransport(exclusive_use_confirmed=True)
        t = np.arange(4000) / 10000
        signal = np.exp(-t / 0.006 + 2j * np.pi * 330 * t)
        physical_keys = []
        saved_keys = []

        def synthetic_acquire(request, *, qualification):
            self.assertTrue(qualification)
            self.assertEqual(len(physical_keys), len(saved_keys))
            physical_keys.append(request.key)
            fid = np.zeros_like(signal) if request.key.endswith("inverse") else signal
            return AcquisitionResult(request.key, request.key, "COMPLETED", t,
                                     fid, {}, {}, {}, {}, 0.1, {}, [])

        transport._acquire = synthetic_acquire
        report = transport.verify_two_segment_equivalence(PulseSpec(0, 0, 40, 80, 90),
                        key_prefix="block2", on_acquisition=lambda r: saved_keys.append(r.key))
        self.assertEqual(report["status"], "CONTIGUOUS_SEGMENTS_VERIFIED")
        self.assertEqual(saved_keys, physical_keys)
        self.assertEqual(len(saved_keys), 4)

        physical_keys.clear()
        saved_keys.clear()

        def failing_sink(result):
            saved_keys.append(result.key)
            raise RuntimeError("disk full")

        with self.assertRaisesRegex(RuntimeError, "disk full"):
            transport.verify_two_segment_equivalence(PulseSpec(0, 0, 40, 80, 90),
                              key_prefix="block3", on_acquisition=failing_sink)
        self.assertEqual(len(physical_keys), 1)
        self.assertEqual(len(saved_keys), 1)


if __name__ == "__main__":
    unittest.main()
