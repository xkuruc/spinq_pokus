"""All tests run without importing SpinQLabLink or opening a socket."""

import json
import math
import struct
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path

from spinq_audit.adapter import AuditAdapter, audit_existing_client
from spinq_audit.analysis import analyze_events, read_events
from spinq_audit.common import redact
from spinq_audit.probes import wait_terminal
from spinq_audit.recorder import EventRecorder
from spinq_audit.report import build_report
from spinq_audit.safety import build_plan, select_feedback_candidate, validate_payload, verify_approval


def approved_fixture():
    params = {"compute_type": 0, "relaxation_time": 15000000.0, "stepList": [],
              "samplePath": 0, "h_freShift": 0, "p_freShift": 0,
              "h_freDemo": 0, "p_freDemo": 0, "makePps": True,
              "sampleFre": 10000, "sampleCount": 16, "sampleDelay": 0,
              "pulse": {"hPulse": [{"width": 40.0, "am": 10.0, "phase": 90.0, "freshift": 0.0}],
                        "pPulse": []}, "gradient": []}
    limits = {"max_pulse_amplitude_pct": 20, "max_pulse_width_us": 50,
              "max_total_rf_us": 100, "max_sequence_us": 100,
              "max_cumulative_rf_us": 300, "min_relaxation_us": 10000000,
              "max_sample_count": 100, "max_sample_frequency": 20000,
              "max_sample_delay_us": 1000, "max_acquisitions_per_request": 1,
              "min_inter_experiment_seconds": 1,
              "min_temperature_c": 18, "max_temperature_c": 30}
    config = {"active_enabled": True, "limits": limits,
              "limits_approved_by": "test", "limits_approved_utc": "synthetic",
              "limits_evidence": "synthetic-only"}
    baseline = {"approved": True, "workstation_verified": True,
                "approved_by": "test", "approved_utc": "synthetic",
                "approval_evidence": "synthetic-only",
                "verified_units": {"relaxation_time": "us", "sampleFre": "Hz", "temperature": "C"},
                "internal_acquisitions_upper_bound": 1,
                "preparation_rf_upper_bound_us": 0, "repeat_count": 1, "params": params}
    return config, baseline


class SafetyTests(unittest.TestCase):
    def test_no_baseline_blocks_plan(self):
        plan = build_plan({}, {})
        self.assertEqual(plan["status"], "blocked_safety")
        self.assertTrue(plan["blockers"])

    def test_missing_active_permission(self):
        config, baseline = approved_fixture()
        plan = build_plan(config, baseline)
        approval = {"approved": False, "operator": "x", "approved_utc": "now",
                    "approved_plan_snapshot": deepcopy(plan)}
        with self.assertRaises(RuntimeError):
            verify_approval(approval, plan, config, baseline, 2)

    def test_payload_tamper_and_final_revalidation(self):
        config, baseline = approved_fixture()
        plan = build_plan(config, baseline)
        self.assertEqual(plan["blockers"], [])
        approval = {"approved": True, "operator": "operator", "approved_utc": "now",
                    "approved_plan_snapshot": deepcopy(plan)}
        approval = json.loads(json.dumps(approval))
        plan = json.loads(json.dumps(plan))
        verify_approval(approval, plan, config, baseline, 2)
        plan["tests"][0]["params"]["pulse"]["hPulse"][0]["am"] = 90
        with self.assertRaises(RuntimeError):
            verify_approval(approval, plan, config, baseline, 2)
        self.assertTrue(validate_payload(plan["tests"][0]["params"], baseline, config))
        plan = deepcopy(approval["approved_plan_snapshot"])
        config["limits"]["max_pulse_amplitude_pct"] = 21
        with self.assertRaises(RuntimeError):
            verify_approval(approval, plan, config, baseline, 2)

    def test_unknown_units_and_preparation_block(self):
        config, baseline = approved_fixture()
        baseline["verified_units"] = {}
        baseline["preparation_rf_upper_bound_us"] = None
        reasons = validate_payload(baseline["params"], baseline, config)
        self.assertTrue(any("jednotky" in x for x in reasons))
        self.assertTrue(any("prípravy" in x for x in reasons))

    def test_failed_and_timeout(self):
        class Experiment:
            def __init__(self, state): self.state = state
            def get_status(self): return self.state
        class Recorder:
            def status(self): return {"complete": True}
        self.assertEqual(wait_terminal(Experiment("FAILED"), lambda: True, Recorder(), .01), "FAILED")
        with self.assertRaises(TimeoutError):
            wait_terminal(Experiment("RUNNING"), lambda: True, Recorder(), .01)

    def test_feedback_selects_only_preapproved_fake_candidate(self):
        proposal = {"threshold": 5.0, "below_id": "A", "above_id": "B",
                    "candidates": [{"id": "A", "params": {"phase": 90}},
                                   {"id": "B", "params": {"phase": 100}}]}
        self.assertEqual(select_feedback_candidate(4.0, proposal)["id"], "A")
        self.assertEqual(select_feedback_candidate(6.0, proposal)["id"], "B")
        proposal["above_id"] = "C"
        with self.assertRaises(ValueError):
            select_feedback_candidate(6.0, proposal)


class FakeProtocol:
    def __init__(self): self.remaining_data = b""
    def serialize_message(self, msg_id, metadata, data): return msg_id.encode()
    def deserialize_message(self, data):
        self.remaining_data += data
        if len(self.remaining_data) < 8: return False, {}
        magic, length = struct.unpack(">II", self.remaining_data[:8])
        if magic != 0xCAFEBABE or len(self.remaining_data) < 8 + length: return False, {}
        body = json.loads(self.remaining_data[8:8+length])
        self.remaining_data = self.remaining_data[8+length:]
        return True, body


def frame(message):
    body = json.dumps(message).encode()
    return struct.pack(">II", 0xCAFEBABE, len(body)) + body


class FakeClient:
    def __init__(self):
        self.protocol = FakeProtocol()
        self.connection = type("Connection", (), {})()
        self.connection._message_callback = self._received
        self.expMgr = type("Manager", (), {"current_experiment": None})()
        self.handler_map = {"s_post_exp_queue_update": lambda data: (_ for _ in ()).throw(RuntimeError("SDK bug"))}
        self.disconnected = False
    def _received(self, data):
        ok, message = self.protocol.deserialize_message(data)
        if ok and message.get("msg_id") in self.handler_map:
            self.handler_map[message["msg_id"]](message.get("json_data", {}))
    def disconnect(self): self.disconnected = True


class AdapterTests(unittest.TestCase):
    def test_passive_allowlist_and_fragment_coalesced_drain(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            recorder = EventRecorder(Path(temp))
            adapter = AuditAdapter(client, recorder, mode="passive", owns_connection=True, fake_transport=True)
            original = client.connection._message_callback
            adapter.attach()
            with self.assertRaises(RuntimeError):
                client.protocol.serialize_message("c_add_exp_task_req", {}, {})
            self.assertEqual(client.protocol.serialize_message("heartbeat_req", {}, {}), b"heartbeat_req")
            one = frame({"msg_id": "s_post_device_info", "json_data": {"temperature": 21}})
            two = frame({"msg_id": "s_post_exp_queue_update", "json_data": {"queue": []}})
            client.connection._message_callback(one[:5])
            client.connection._message_callback(one[5:] + two)
            self.assertEqual(recorder.count, 2)
            self.assertEqual(client.protocol.remaining_data, b"")
            adapter.detach()
            self.assertEqual(client.connection._message_callback, original)
            recorder.close()

    def test_borrowed_client_not_disconnected_or_written(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            old_decoder = client.protocol.deserialize_message
            result = audit_existing_client(client, out=temp, duration=0,
                                           owns_connection=False, fake_transport=True)
            self.assertFalse(client.disconnected)
            self.assertEqual(result["outgoing_by_auditor"], [])
            self.assertEqual(client.protocol.deserialize_message, old_decoder)

    def test_foreign_task_filtered_and_late_event_preserved_only_when_owned(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            recorder = EventRecorder(Path(temp))
            adapter = AuditAdapter(client, recorder, mode="active", own_task_ids={"own"},
                                   owns_connection=True, fake_transport=True).attach()
            client.connection._message_callback(frame({"msg_id": "s_post_exp_chart_updated",
                                                      "chart_data": {"taskId": "foreign", "points": [[1,2]]}}))
            self.assertEqual(recorder.count, 0)
            client.connection._message_callback(frame({"msg_id": "s_post_exp_chart_updated",
                                                      "chart_data": {"taskId": "own", "points": [[1,2]]}}))
            self.assertEqual(recorder.count, 1)
            adapter.detach()
            recorder.close()

    def test_foreign_ack_cannot_claim_local_task(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            recorder = EventRecorder(Path(temp))
            adapter = AuditAdapter(client, recorder, mode="active", own_task_ids={"local"},
                                   owns_connection=True, fake_transport=True).attach()
            adapter.pending_own_ack = True
            adapter.pending_sequence_id = 123
            client.connection._message_callback(frame({"msg_id": "s_add_exp_task_res",
                                                      "metadata": {"sequence_id": 999},
                                                      "json_data": {"code": 0, "taskId": "foreign"}}))
            self.assertTrue(adapter.ack_mismatch)
            self.assertNotIn("foreign", adapter.own_task_ids)
            self.assertEqual(recorder.count, 0)
            adapter.detach()
            recorder.close()


def event(seq, task, channel, name, points, block=1):
    return {"seq": seq, "kind": "s_post_exp_chart_updated", "source": "synthetic_test",
            "payload": {"chart_data": {"taskId": task, "group": "g", "path": channel,
                                       "qubit": 0, "step": block, "chart_name": name,
                                       "points": points}}}


class DataTests(unittest.TestCase):
    def test_multichannel_pair_fft_and_numeric_roundtrip(self):
        xs = [[i, math.cos(2*math.pi*i/8)] for i in range(8)]
        ys = [[i, math.sin(2*math.pi*i/8)] for i in range(8)]
        events = [event(1, "A", 0, "fidRe", xs), event(2, "A", 0, "fidIm", ys),
                  event(3, "A", 1, "fidRe", xs), event(4, "A", 1, "fidIm", ys)]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = analyze_events(events, out)
            self.assertTrue(result["synthetic_or_mock_data"])
            self.assertEqual(len([p for p in result["fid_pairs"] if p["status"] == "paired"]), 2)
            self.assertTrue(all(p["fft"]["peak_signed_bin"] == 1 for p in result["fid_pairs"]))
            original = json.loads((out / "original_scientific.json").read_text())
            self.assertEqual(original["charts"][0]["points"], xs)
            self.assertEqual(original["raw_adc_confirmed"], False)
            self.assertFalse(result["transport_complete_confirmed"])

    def test_duplicate_and_mismatched_axis_do_not_pair(self):
        re = [[0, 1.0], [1, 2.0]]
        im = [[0, 0.0], [2, 1.0]]
        with tempfile.TemporaryDirectory() as temp:
            result = analyze_events([event(1,"A",0,"fidRe",re), event(2,"A",0,"fidIm",im)], Path(temp))
            self.assertEqual(result["fid_pairs"][0]["status"], "unpaired")
        with tempfile.TemporaryDirectory() as temp:
            result = analyze_events([event(1,"A",0,"fidRe",re), event(2,"A",0,"fidRe",re),
                                     event(3,"A",0,"fidIm",re)], Path(temp))
            self.assertEqual(result["fid_pairs"][0]["status"], "unpaired")

    def test_redaction_and_recoverable_journal(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = EventRecorder(Path(temp))
            recorder.record("telemetry", {"session_id": "private", "password": "private",
                                          "points": [[1.234567890123, 2.345678901234]]})
            recorder.close()
            raw = (Path(temp) / "events.jsonl").read_text()
            self.assertNotIn("private", raw)
            replayed = list(read_events(Path(temp)))
            self.assertEqual(replayed[0]["payload"]["points"][0][0], 1.234567890123)
            self.assertEqual(replayed[0]["payload"]["session_id"], "[REDACTED]")

    def test_nonfinite_chart_is_flagged_without_fabricated_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            result = analyze_events([event(1,"A",0,"fidRe",[[0,float("nan")]]),
                                     event(2,"A",0,"fidIm",[[0,1.0]])], Path(temp))
            self.assertEqual(result["fid_pairs"][0]["status"], "unpaired")
            self.assertIn('NaN', (Path(temp)/"original_scientific.json").read_text())

    def test_bounded_queue_overflow_marks_incomplete(self):
        class SlowRecorder(EventRecorder):
            gate = threading.Event()
            def _writer(self):
                self.gate.wait(timeout=2)
                super()._writer()
        with tempfile.TemporaryDirectory() as temp:
            recorder = SlowRecorder(Path(temp), max_events=1)
            self.assertTrue(recorder.record("one", {}))
            self.assertFalse(recorder.record("two", {}))
            recorder.gate.set()
            self.assertFalse(recorder.close()["complete"])

    def test_report_escapes_untrusted_device_text(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            summary = {"mode": "offline", "sdk_version": None, "sdk_origin": "reference", "created_utc": "now",
                       "blockers": ["<script>alert(1)</script>"], "charts_observed": 0,
                       "fid_pairs": 0, "hardware_experiments_executed": 0,
                       "synthetic_or_mock_data": False}
            env = {"spinqlablink": {"origin": "reference"}}
            build_report(out, summary, env, [], [], {}, None)
            html = (out / "REPORT.html").read_text()
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
