"""Offline orchestration checks; no socket or device is opened."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


class RunnerOfflineTests(unittest.TestCase):
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
            for index, method in enumerate(("B_classical", "C_fixed_bayes",
                                            "D_adaptive_bayes"), start=1):
                compared = run._measure_arm(0, "H", method, anchor, controller,
                                            references, {"status": "NEUTRAL"}, index)
                self.assertEqual(compared["design_acquisitions"], 12)
                self.assertEqual(compared["control_acquisitions"], 2)
                self.assertTrue((run.artifacts.path / compared["estimate_file"]).exists())
            self.assertEqual(fake.count, 64)


if __name__ == "__main__":
    unittest.main()
