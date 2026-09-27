"""Offline checks for the bounded H-only B/D comparison. No device is opened."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from bayes_online_core.runner import (HiddenCommandError, OnlineRun,
                                      quick_task_plan, validate_config)


REPO = Path(__file__).resolve().parents[1]


def quick_config() -> dict:
    config = json.loads((REPO / "config-bayes-online.json").read_text(encoding="utf-8"))
    config.update(channels=["H"], blocks=3,
                  calibration_acquisitions_per_method=10,
                  reference_acquisitions_per_block=5,
                  max_tasks=120, max_requested_rf_us=30000,
                  pause_seconds=1)
    return validate_config(config)


class QuickProfileTests(unittest.TestCase):
    def test_plan_counts_every_fid_and_rejects_insufficient_budget(self):
        config = quick_config()
        plan = quick_task_plan(config)
        self.assertEqual(plan["channel"], "H")
        self.assertEqual(plan["methods"], ["B_classical", "D_adaptive_bayes"])
        self.assertEqual(plan["scenarios"], ["neutral", "positive_command_error",
                                             "negative_command_error"])
        self.assertEqual(plan["counts"], {
            "shared_anchor": 18,
            "fresh_nominal_references": 6,
            "B_and_D_training": 60,
            "heldout_controls": 12,
            "hidden_perturbation_probes": 2,
            "pre_between_post_drift_checks": 9,
        })
        self.assertEqual(plan["planned_physical_tasks"], 107)
        self.assertEqual(sum(plan["counts"].values()), 107)
        self.assertLessEqual(plan["planned_physical_tasks"], plan["max_physical_tasks"])
        too_small = dict(config, max_tasks=106)
        with self.assertRaisesRegex(ValueError, "exceeds configured physical-task budget"):
            quick_task_plan(too_small)

    def test_orchestrator_reuses_paid_anchor_and_runs_only_paired_B_D(self):
        config = quick_config()
        with tempfile.TemporaryDirectory() as temporary:
            run = OnlineRun(REPO, Path(temporary) / "quick", config, "rabi", quick=True)
            anchor = object()
            pilot_calls: list[tuple[int, str]] = []
            reference_calls: list[tuple[int, object]] = []
            arm_calls: list[tuple[int, str, str, int, object]] = []
            drift_calls: list[tuple[int, str]] = []

            def pilot(block, channel):
                pilot_calls.append((block, channel))
                return anchor

            def references(block, anchors):
                reference_calls.append((block, anchors["H"]))
                return {"H": {"phase45": object(), "phase135": object()}}

            def scenario(block, anchors):
                run.state["scenarios"][str(block)] = {
                    "label": ("neutral", "positive_command_error",
                              "negative_command_error")[block]}
                return {"H": HiddenCommandError(0., 1., 0.)}

            def measure_arm(block, channel, method, measured_anchor,
                            controller, fresh_refs, probe, position):
                arm_calls.append((block, channel, method, position, measured_anchor))
                row = {"block": block, "channel": channel, "method": method,
                       "status": "TARGET_REACHED"}
                run.state["comparison"].append(row)
                return row

            def drift(block, channel, measured_anchor, stage="after"):
                self.assertIs(measured_anchor, anchor)
                drift_calls.append((block, stage))
                return {"status": "STABLE_WITHIN_REPEAT_NOISE"}

            run._pilot = pilot
            run._nominal_references = references
            run._scenario = scenario
            run._probe = Mock(return_value={"status": "NEUTRAL"})
            run._measure_arm = measure_arm
            run._drift_check = drift
            run.artifacts.report = Mock()
            run.log = Mock()

            latest = run.run_calibration()
            self.assertIs(latest["H"], anchor)
            self.assertEqual(pilot_calls, [(0, "H")])
            self.assertEqual(reference_calls, [(0, anchor), (1, anchor), (2, anchor)])
            self.assertEqual(len(arm_calls), 6)
            self.assertTrue(all(channel == "H" and measured_anchor is anchor
                                for _, channel, _, _, measured_anchor in arm_calls))
            orders = [[method for block, _, method, _, _ in arm_calls if block == i]
                      for i in range(3)]
            self.assertTrue(all(set(order) == {"B_classical", "D_adaptive_bayes"}
                                for order in orders))
            self.assertEqual(orders[1], list(reversed(orders[0])))
            self.assertEqual(orders[2], orders[0])
            self.assertEqual(drift_calls,
                             [(block, stage) for block in range(3)
                              for stage in ("pre", "between", "after")])
            self.assertEqual(len(run.state["comparison"]), 6)

    def test_quick_rejects_two_spin_task_before_connection(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "only for frequency/Rabi"):
                OnlineRun(REPO, Path(temporary) / "quick", quick_config(),
                          "bell", quick=True)

    def test_invalid_drift_pair_cannot_improve_claimed_mean(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = OnlineRun(REPO, Path(temporary) / "quick", quick_config(),
                            "rabi", quick=True)
            for block in range(3):
                for method, error, status in (
                    ("B_classical", .2, "TARGET_REACHED"),
                    ("D_adaptive_bayes", (.1, .001, .15)[block],
                     "PILOT_INCONCLUSIVE" if block == 1 else "TARGET_REACHED"),
                ):
                    run.state["comparison"].append({
                        "block": block, "channel": "H", "task": "frequency_rabi",
                        "method": method,
                        "heldout_complex_error": error, "status": status,
                        "total_acquisitions": 12, "end_to_end_seconds": 1.,
                    })
            result = run._paired_summary()["D_adaptive_bayes"]
            self.assertEqual(result["valid_target_pairs"], 2)
            self.assertAlmostEqual(result["paired_valid_heldout_error_difference_mean"],
                                   -.075)
            self.assertLess(result["paired_heldout_error_difference_mean"], -.075)
            self.assertEqual(result["conclusion"],
                             "INSUFFICIENT_VALID_PAIRED_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
