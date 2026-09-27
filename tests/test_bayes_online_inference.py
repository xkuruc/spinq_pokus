"""Numerical checks for the new, hardware-independent inference module."""

import unittest

import numpy as np

from bayes_online_core.inference import (
    Candidate, ComplexFeature, OnlineLearner, ReadoutAnchor,
    detect_coherent_window, estimate_anchor, numeric_selfcheck,
    pulse_design_identifiability,
)


class OnlineInferenceTests(unittest.TestCase):
    def test_synthetic_physical_inference(self):
        self.assertEqual(numeric_selfcheck()["status"], "PASS")

    def test_one_fid_cannot_be_counted_twice(self):
        anchor = ReadoutAnchor(1 + 0j, 1000, 300, 0.01, np.eye(2) * 0.01)
        learner = OnlineLearner(np.array([[-300, 300], [0.8, 1.2], [-0.5, 0.5]]),
                                anchor, n_particles=128, seed=1)
        candidate = Candidate("H", 40, 5, 0, 0, 10)
        feature = ComplexFeature(-0.8j, np.eye(2) * 0.01, 0.1, 64, 50,
                                 source_id="physical-task-1")
        learner.update(candidate, feature)
        with self.assertRaisesRegex(ValueError, "same FID"):
            learner.update(candidate, feature)
        self.assertEqual(learner.summary()["acquisitions"], 1)

    def test_fixed_receiver_frequency_is_not_transmit_calibration(self):
        anchor = ReadoutAnchor(1 + 0j, 1000, 300, 0.01, np.eye(2) * 0.01)
        bounds = np.array([[-300, 300], [0.8, 1.2], [-0.5, 0.5]])
        single = Candidate("H", 40, 5, 0, 0, 10)
        diagnostics = pulse_design_identifiability([single], anchor, bounds,
                                                    np.array([0.0, 1.0, 0.0]))
        self.assertEqual(diagnostics["status"], "NONIDENTIFIABLE")
        learner = OnlineLearner(bounds, anchor, n_particles=128, seed=1)
        self.assertEqual(learner.summary()["status"], "PRIOR_ONLY")

    def test_late_artifact_does_not_extend_coherent_window(self):
        rng = np.random.default_rng(99)
        t = np.arange(16000) / 10000.0
        signal = 100 * np.exp(-t / 0.0025 + 2j * np.pi * 330 * t)
        noise = 1.0 * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))
        fid = signal + noise
        fid[11000:11008] += 1000
        window = detect_coherent_window(fid, t)
        self.assertLess(window["seconds"], 0.03)
        self.assertGreater(window["samples"], 32)
        self.assertTrue(window["late_artifact_warning"])

    def test_imperfect_but_bounded_rabi_anchor_remains_explicitly_flagged(self):
        """A multiplet-like extra response must not be called a clean fit."""
        rng = np.random.default_rng(12)
        t = np.arange(2048) / 100000.0
        widths = np.array([10, 20, 30, 40, 50, 60, 80, 100, 20, 50], float)
        area = widths * 1e-6 * 5.0
        primary = -1j * np.sin(2 * np.pi * 1000 * area)
        nuisance = 0.6 * np.exp(0.5j) * np.sin(2 * np.pi * 1800 * area)
        basis = np.exp(-t / 0.018 + 2j * np.pi * 390.625 * t)
        fids = np.array([
            amp * basis + 0.002 * (rng.normal(size=len(t)) +
                                   1j * rng.normal(size=len(t)))
            for amp in primary + nuisance])
        anchor = estimate_anchor(fids, t, widths, 5.0)
        self.assertEqual(anchor.diagnostics["rabi_model_status"],
                         "RABI_MODEL_CHECK_REQUIRED")
        self.assertGreater(anchor.diagnostics["rabi_relative_residual"], 0.35)
        self.assertLess(anchor.diagnostics["rabi_relative_residual"], 0.7)
        self.assertTrue(np.isfinite(anchor.rabi_hz_per_pct))
        learner = OnlineLearner(np.array([[-300, 300], [0.8, 1.2], [-0.5, 0.5]]),
                                anchor, n_particles=128, seed=1)
        feature = ComplexFeature(-0.8j, np.eye(2) * 0.01, 0.1, 64, 50,
                                 source_id="flagged-pilot-next-acquisition")
        learner.update(Candidate("H", 40, 5, 0, 0, 10), feature)
        self.assertEqual(learner.summary()["status"], "MODEL_CHECK_REQUIRED")


if __name__ == "__main__":
    unittest.main()
