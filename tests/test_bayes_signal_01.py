"""Offline numeric contracts for the exported-FID calibration observable."""

import math
import unittest

import numpy as np

from experiments.signal_01 import (SignalIdentificationError, _feature_windows,
    demodulated_features, estimate_pilot_rabi, estimate_signal,
    fixed_pilot_projection, identify_pilot_multiplet)
from spinq_local.core import RawFIDRecord


FS = 10_000
N = 4_000
T = np.arange(N) / FS


def record(key, signal, *, sample_count=N, pulse_width=40.):
    signal = np.asarray(signal, complex)[:sample_count]
    return RawFIDRecord(
        key=key, task_id=f"task-{key}", group="g", path="0", qubit="0",
        step="NMRSIG", axis_original=(np.arange(len(signal)) * .1).astype(float),
        time_seconds=np.arange(len(signal)) / FS, re=signal.real, im=signal.imag,
        parameters_sent={"sampleFre": FS, "sampleCount": sample_count,
                         "pulse": {"hPulse": [{"width": pulse_width}]}}, metadata={})


def fid(primary=3 + 1j, secondary=.8 - 1.1j, frequency=-1700.,
        baseline=.2 - .1j):
    return baseline + primary * np.exp((-4 + 2j * np.pi * frequency) * T) + \
        secondary * np.exp((-7 + 2j * np.pi * -1300.) * T)


class BayesSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(18)
        cls.pilot_records = []
        for i in range(3):
            noise = .015 * (rng.normal(size=N) + 1j * rng.normal(size=N))
            cls.pilot_records.append(record(f"pilot-{i}", fid() + noise))
        cls.pilot = identify_pilot_multiplet(cls.pilot_records)

    def test_pilot_discovers_two_real_modes_and_positive_feature_covariance(self):
        pilot = self.pilot
        self.assertEqual(pilot.status, "IDENTIFIED")
        self.assertEqual(len(pilot.bands_hz), 2)
        self.assertAlmostEqual(pilot.component_frequencies_hz[0], -1700., delta=1.)
        self.assertAlmostEqual(pilot.component_frequencies_hz[1], -1300., delta=1.)
        self.assertEqual(len(pilot.feature_windows_s), 6)
        self.assertEqual(pilot.feature_covariance_re_im.shape, (12, 12))
        self.assertGreater(np.linalg.eigvalsh(pilot.feature_covariance_re_im).min(), 0)
        self.assertFalse(pilot.to_dict()["vendor_result_used"])

    def test_shorter_fid_keeps_component_identity_and_exact_feature_operator(self):
        shorter = record("short", fid(), sample_count=3000)
        estimate = estimate_signal(shorter, self.pilot)
        self.assertEqual(estimate.status, "FIT_OK")
        self.assertAlmostEqual(estimate.frequency_hz, -1700., delta=1.)
        self.assertEqual(estimate.features.shape, (6,))
        manual = demodulated_features(shorter.fid, shorter.time_seconds, self.pilot)
        np.testing.assert_allclose(manual, estimate.features, atol=1e-12)
        # A much shorter FID cannot silently zero-fill the missing window.
        with self.assertRaises(SignalIdentificationError):
            demodulated_features(shorter.fid[:100], shorter.time_seconds[:100], self.pilot)

    def test_phase_crosses_pi_and_fixed_projection_survives_rabi_null(self):
        pilot = self.pilot
        angles = [math.radians(179), math.radians(-179)]
        estimates = []
        for i, angle in enumerate(angles):
            z = record(f"phase-{i}", fid(primary=(3 + 1j) * np.exp(1j * angle)))
            estimates.append(estimate_signal(z, pilot))
        for estimate in estimates:
            self.assertEqual(estimate.status, "FIT_OK")
        wrapped_difference = np.angle(np.exp(1j * (
            estimates[0].relative_phase_rad - estimates[1].relative_phase_rad)))
        self.assertAlmostEqual(abs(math.degrees(wrapped_difference)), 2., delta=1.)
        zero = record("zero", fid(primary=0j))
        coefficient, residual = fixed_pilot_projection(zero, pilot)
        self.assertLess(abs(coefficient), .02)
        self.assertLess(residual, .03)

    def test_band_edge_diagnostic_fails_closed(self):
        outer = self.pilot.bands_hz[0][1] + 15
        shifted = record("outside", fid(frequency=outer))
        estimate = estimate_signal(shifted, self.pilot)
        self.assertNotEqual(estimate.status, "FIT_OK")
        with self.assertRaises(SignalIdentificationError):
            from experiments.signal_01 import estimate_frequency
            estimate_frequency(shifted, self.pilot)

    def test_rabi_period_is_data_driven_and_no_null_fit_is_required(self):
        widths = list(range(20, 201, 20))
        pilot_records = []
        for width in widths:
            amplitude = 1.1 * np.sin(2 * np.pi * width / 160)
            primary = (3 + 1j) * amplitude
            pilot_records.append(record(f"rabi-{width}", fid(primary=primary),
                                        pulse_width=width))
        result = estimate_pilot_rabi(widths, pilot_records, self.pilot)
        self.assertEqual(result.status, "IDENTIFIED")
        self.assertAlmostEqual(result.period_us, 160., delta=3.)
        self.assertAlmostEqual(result.t90_us, 40., delta=1.)
        self.assertGreater(result.signed_complex_r2, .99)
        self.assertEqual(len(result.to_dict()["diagnostics"]["frozen_projection_residual_rms"]),
                         len(widths))

    def test_repeated_pilot_requires_distinct_task_ids(self):
        same = [self.pilot_records[0]] * 3
        with self.assertRaisesRegex(SignalIdentificationError, "task ID"):
            identify_pilot_multiplet(same)

    def test_fast_decay_with_receiver_offset_keeps_early_signal(self):
        rng = np.random.default_rng(12)
        t = np.arange(16_000) / FS
        signal = (.2 - .1j + (3 + 1j) * np.exp((-60 - 2j * np.pi * 1700) * t)
                  + (.8 - 1.1j) * np.exp((-80 - 2j * np.pi * 1300) * t))
        records = []
        for i in range(3):
            noisy = signal + .015 * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))
            records.append(record(f"fast-{i}", noisy, sample_count=len(t)))
        pilot = identify_pilot_multiplet(records)
        self.assertEqual(pilot.status, "IDENTIFIED")
        self.assertEqual(len(pilot.component_frequencies_hz), 2)
        self.assertLess(abs(pilot.component_frequencies_hz[0] + 1700), 2.)
        self.assertLess(pilot.diagnostics["active_points"], 2000)

    def test_six_features_follow_measured_short_coherence_and_fail_if_undersampled(self):
        t = np.arange(16_000) / FS
        short_signal = 400 * np.exp(-270 * t)
        covariance = np.eye(2) * 15**2
        windows = _feature_windows(short_signal, covariance, FS, (-125., 330.), 1)
        self.assertEqual(len(windows), 6)
        self.assertLess(windows[-1][1], .012)
        self.assertTrue(all(np.count_nonzero((t >= a) & (t < b)) >= 4
                            for a, b in windows))
        with self.assertRaisesRegex(SignalIdentificationError, "fewer than four samples"):
            _feature_windows(300 * np.exp(-270 * t), np.eye(2) * 20**2,
                             FS, (-125., 330.), 1)


if __name__ == "__main__":
    unittest.main()
