"""Offline orchestration checks; no socket or device is opened."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bayes_online_core.inference import Candidate, ReadoutAnchor, predict_feature
from bayes_online_core.runner import OnlineRun, PerturbationController
from bayes_online_core.transport import AcquisitionResult


class _SyntheticTransport:
    def __init__(self) -> None:
        self.count = 0
        self.anchor = ReadoutAnchor(1.0 + .2j, 90., 300., .008,
                                    np.eye(2) * 1e-6)

    def acquire(self, request):
        self.count += 1
        pulse = next(p for p in request.pulses if p.amplitude_pct > 0)
        candidate = Candidate("H" if pulse.path == 0 else "P", pulse.width_us,
                              pulse.amplitude_pct, pulse.phase_deg,
                              pulse.detuning_hz, 1.)
        t = np.arange(request.sample_count) / request.sample_frequency_hz
        magnitude = complex(predict_feature(candidate, np.array([0., 1., 0.]),
                                            self.anchor))
        rng = np.random.default_rng(self.count)
        noise = .001 * (rng.normal(size=len(t)) + 1j*rng.normal(size=len(t)))
        fid = magnitude * np.exp(-t/self.anchor.decay_s +
                                  2j*np.pi*self.anchor.fid_frequency_hz*t) + noise
        return AcquisitionResult(request.key, f"offline-{self.count}", "COMPLETED",
            t, fid, {}, {"request": request.key}, {"request": request.key},
            {"axis_status": "EXPORT_MATCHES_REQUESTED_RATE"}, 1., {}, [],
            exported_axis=t*1000)


class _BiasedStopTransport(_SyntheticTransport):
    def acquire(self, request):
        result = super().acquire(request)
        if "_stop_" in request.key:
            result.fid_complex *= 1.8
        return result


class _NoiseOnlyTransport(_SyntheticTransport):
    def acquire(self, request):
        result = super().acquire(request)
        rng = np.random.default_rng(1000 + self.count)
        result.fid_complex = 26 * (rng.normal(size=len(result.time_s)) +
                                   1j * rng.normal(size=len(result.time_s)))
        return result


class RunnerOfflineTests(unittest.TestCase):
    def test_noise_only_pilot_stops_after_six_paid_fids(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-bayes-online.json").read_text())
        config.update(channels=["H"], pause_seconds=0)
        with tempfile.TemporaryDirectory() as directory:
            run = OnlineRun(repo, Path(directory) / "run", config, "rabi", True)
            fake = _NoiseOnlyTransport()
            with patch("bayes_online_core.runner.PhysicalTransport") as transport:
                transport.return_value.__enter__.return_value = fake
                state = run.execute()
            self.assertEqual(state["status"], "PILOT_INCONCLUSIVE")
            self.assertEqual(fake.count, 6)
            self.assertEqual(run.task_count, 6)
            self.assertEqual(len(run.state["acquisitions"]), 6)
            self.assertEqual(run.state["capabilities"]["b0_H_pilot_triage"]["status"],
                             "NO_DETECTABLE_FID")
            self.assertTrue((run.artifacts.path / "profiles" /
                             "b0_H_pilot_triage.json").is_file())
            self.assertFalse(run.state["anchors"])

    def test_real_path_with_synthetic_transport_keeps_independent_controls(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-bayes-online.json").read_text())
        config["channels"] = ["H"]
        config["pause_seconds"] = 0
        with tempfile.TemporaryDirectory() as directory:
            run = OnlineRun(repo, Path(directory) / "run", config, "rabi", True)
            fake = _SyntheticTransport()
            run.transport = fake
            anchor = run._pilot(0, "H")
            references = run._nominal_references(0, {"H": anchor})["H"]
            controller = PerturbationController(run._scenario(0, {"H": anchor}), 100.)
            row = run._measure_arm(0, "H", "A_prior_only", anchor, controller,
                                   references, {"status": "NEUTRAL"}, 0)
            self.assertEqual(fake.count, 22)
            self.assertEqual(row["design_acquisitions"], 0)
            self.assertEqual(row["control_acquisitions"], 2)
            self.assertNotIn(row["heldout_keys"][0], anchor.pilot_keys)
            with np.load(run.artifacts.path / "data" /
                         f"{row['heldout_keys'][0]}.npz") as stored:
                self.assertEqual(stored["re"].size, 16000)
                self.assertEqual(stored["exported_x"].size, 16000)
            self.assertIn(row["status"], {"TARGET_REACHED", "NONIDENTIFIABLE",
                                               "BUDGET_EXHAUSTED"})
            arm_rows = [row]
            for index, method in enumerate(("B_classical", "C_fixed_bayes",
                                            "D_adaptive_bayes"), start=1):
                compared = run._measure_arm(0, "H", method, anchor, controller,
                                            references, {"status": "NEUTRAL"}, index)
                self.assertIn(compared["design_acquisitions"], (6, 12))
                self.assertIn(compared["sequential_stop_check_acquisitions"], (0, 2))
                self.assertEqual(compared["control_acquisitions"], 2)
                self.assertEqual(compared["total_acquisitions"],
                    compared["design_acquisitions"] +
                    compared["sequential_stop_check_acquisitions"] + 2)
                self.assertTrue(set(compared["stopping_checkpoint"]["measured_keys"])
                                .isdisjoint(compared["heldout_keys"]))
                self.assertTrue((run.artifacts.path / compared["estimate_file"]).exists())
                arm_rows.append(compared)
            self.assertEqual(fake.count, 20 + sum(item["total_acquisitions"]
                                                   for item in arm_rows))

    def test_sequential_stop_uses_two_disjoint_fids_and_final_scorer_is_fresh(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-bayes-online.json").read_text())
        config.update(channels=["H"], pause_seconds=0)
        with tempfile.TemporaryDirectory() as directory:
            run = OnlineRun(repo, Path(directory) / "run", config, "rabi", True)
            fake = _SyntheticTransport()
            run.transport = fake
            anchor = run._pilot(0, "H")
            references = run._nominal_references(0, {"H": anchor})["H"]
            controller = PerturbationController(run._scenario(0, {"H": anchor}), 100.)
            row = run._measure_arm(0, "H", "B_classical", anchor, controller,
                                   references, {"status": "NEUTRAL"}, 0)
            self.assertEqual(row["stopping_decision"], "EARLY_STOP")
            self.assertEqual(row["design_acquisitions"], 6)
            self.assertEqual(row["sequential_stop_check_acquisitions"], 2)
            self.assertEqual(row["control_acquisitions"], 2)
            self.assertEqual(row["total_acquisitions"], 10)
            self.assertEqual(fake.count, 30)
            checks = row["stopping_checkpoint"]["measured_keys"]
            self.assertEqual(len(checks), 2)
            self.assertTrue(set(checks).isdisjoint(row["heldout_keys"]))
            self.assertTrue(all(run.state["acquisitions"][key]["role"] ==
                                "sequential_stop_check" for key in checks))
            self.assertEqual(len(run.state["methods"]["b0_H_B_classical"]
                                 ["estimate"]["training_keys"]), 6)

    def test_failed_stop_checks_are_paid_and_training_continues(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-bayes-online.json").read_text())
        config.update(channels=["H"], pause_seconds=0)
        with tempfile.TemporaryDirectory() as directory:
            run = OnlineRun(repo, Path(directory) / "run", config, "rabi", True)
            fake = _BiasedStopTransport()
            run.transport = fake
            anchor = run._pilot(0, "H")
            references = run._nominal_references(0, {"H": anchor})["H"]
            controller = PerturbationController(run._scenario(0, {"H": anchor}), 100.)
            row = run._measure_arm(0, "H", "B_classical", anchor, controller,
                                   references, {"status": "NEUTRAL"}, 0)
            self.assertEqual(row["stopping_decision"], "MAX_TRAINING_BUDGET")
            self.assertEqual(row["stopping_checkpoint"]["status"], "FAIL")
            self.assertEqual(row["design_acquisitions"], 12)
            self.assertEqual(row["sequential_stop_check_acquisitions"], 2)
            self.assertEqual(row["control_acquisitions"], 2)
            self.assertEqual(row["total_acquisitions"], 16)
            self.assertEqual(fake.count, 36)


if __name__ == "__main__":
    unittest.main()
