"""Regression for a completed SpinQ chart with one fewer point than requested."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from spinq_benchmark.hardware import (HardwareUncertain, LiveHardware, PreSubmissionFailure,
                                      paired_fid_graphs, physical_request,
                                      same_physical_payload, stored_graph_fields)


class FidPairTests(unittest.TestCase):
    def test_local_compact_task_metadata_does_not_duplicate_chart_arrays(self):
        graph=[{"fidRe":[[0.0,1.0],[0.1,2.0]],
                "fidIm":[[0.0,0.0],[0.1,0.0]]}]
        compact=stored_graph_fields(graph,2,compact=True)
        self.assertEqual(compact["sdk_graph_blocks"],1)
        self.assertNotIn("fid_pairs",compact)
        self.assertNotIn("all_decoded_graphs",compact)
        full=stored_graph_fields(graph,2)
        self.assertEqual(full["all_decoded_graphs"],graph)
        self.assertEqual(len(full["fid_pairs"]),1)

    def test_physical_payload_numeric_equivalence_and_real_difference(self):
        expected={"sampleFre":10000,"pulse":{"hPulse":[{"width":40,"am":100.0}]}}
        serialized={"pulse":{"hPulse":[{"am":100,"width":40.00000001}]},
                    "sampleFre":10000.0}
        self.assertTrue(same_physical_payload(expected,serialized))
        serialized["pulse"]["hPulse"][0]["width"]=41.0
        self.assertFalse(same_physical_payload(expected,serialized))

    def test_one_missing_final_point_is_preserved_but_axis_mismatch_is_rejected(self):
        real=[[i/10000, i] for i in range(15999)]
        imag=[[i/10000, -i] for i in range(15999)]
        good=paired_fid_graphs([{"fidRe":real,"fidIm":imag}],16000)
        self.assertEqual(len(good),1)
        self.assertEqual(good[0]["actual_sample_count"],15999)
        self.assertEqual(good[0]["requested_sample_count"],16000)
        shifted=[[t+0.0001,v] for t,v in imag]
        self.assertEqual(paired_fid_graphs([{"fidRe":real,"fidIm":shifted}],16000),[])


class JournalCheckpointTests(unittest.TestCase):
    def test_failed_pre_submit_journal_write_cannot_send_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            hw=LiveHardware.__new__(LiveHardware)
            hw.out=Path(folder)
            hw.data=hw.out / "data"
            hw.data.mkdir()
            hw.journal_path=hw.data / "hardware_journal.json"
            hw.journal={}
            hw.task_count=0
            hw.rf_us=0.
            hw.max_tasks=180
            hw.max_rf=12000.
            hw.pause=0.
            hw.last_finished=0.
            hw.halted=False
            hw.adapter=SimpleNamespace(own_task_ids=set(),pending_own_ack=False,
                                       ack_mismatch=False)
            exp=SimpleNamespace(id="local-exp-1",
                                get_experiment_parameter=lambda: {
                                    "params":json.dumps(physical_request())})
            hw.link=Mock()
            hw.link.register_experiment.return_value=(exp,object())
            sdk=types.ModuleType("spinqlablink")
            sdk.ExperimentType=SimpleNamespace(PHYSICAL_LAYER_EXPERIMENT=object())
            with patch.object(hw,"_preflight",return_value={}), \
                 patch("spinq_benchmark.hardware._configure_physical"), \
                 patch("spinq_benchmark.hardware.atomic_json",
                       side_effect=PermissionError("Access is denied")), \
                 patch("spinq_benchmark.hardware.time.sleep"), \
                 patch.dict("sys.modules",{"spinqlablink":sdk}):
                with self.assertRaisesRegex(PreSubmissionFailure,
                                            "no experiment was sent"):
                    hw.measure("pilot_count4000_r1",physical_request())
            hw.link.run_experiment.assert_not_called()
            hw.link.deregister_experiment.assert_called_once_with()
            self.assertTrue(hw.halted)
            self.assertEqual(hw.task_count,0)
            self.assertEqual(hw.rf_us,0.)
            self.assertEqual(hw.journal,{})
            self.assertFalse(hw.journal_path.exists())
            self.assertFalse(hw.adapter.pending_own_ack)
            self.assertEqual(hw.adapter.own_task_ids,set())


if __name__=="__main__":unittest.main()
