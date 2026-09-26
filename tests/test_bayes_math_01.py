"""Offline checks for the standalone Bayesian NMR numerical engine."""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.stats import multivariate_normal

from experiments.design import (Tolerances, choose_candidate,
                                expected_posterior_score,
                                meets_uncertainty_and_validation)
from experiments.likelihood import (Candidate, GaussianObservation, NMRModel,
                                    covariance_from_repeats, interleaved_real_imag,
                                    predict_complex)
from experiments.signal_01 import demodulated_features
from experiments.smc import (ParticleFilter, PriorBounds,
                             effective_sample_size, normalized_log_weights,
                             particle_count_convergence, phase_difference,
                             systematic_resample_indices,
                             weighted_posterior_moments)


WINDOWS = ((0.0, 0.0007), (0.0031, 0.0038), (0.0122, 0.0129),
           (0.0413, 0.0420))


def candidate(width=40., phase=90., *, seconds=10.) -> Candidate:
    return Candidate("rabi", width, 100., phase, WINDOWS,
                     sample_hz=10_000, sample_count=500,
                     estimated_wall_seconds=seconds)


class PhysicalLikelihoodTests(unittest.TestCase):
    def test_short_pulse_matches_unitary_and_exact_discrete_features(self):
        """Known-fs synthetic multiplet checked against matrix propagation."""
        model = NMRModel(
            reference_frequency_hz=-1723.5, reference_coefficient=2.3-0.4j,
            baseline_complex=0.17+0.09j,
            component_offsets_hz=(0., 311.7),
            component_weights=(1+0j, 0.31-0.16j),
            component_decay_per_s=(19., 29.),
            pulse_detuning_offset_hz=67., amplitude_reference_pct=100.)
        setting = candidate(width=37.4, phase=117.)
        theta = np.array([[14.5, 39.2, math.radians(178.)]])
        actual = predict_complex(theta, setting, model)[0]
        sigma_x = np.array([[0, 1], [1, 0]], complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], complex)
        sigma_z = np.diag([1, -1]).astype(complex)
        rho = np.diag([1., 0.]).astype(complex)
        phi = math.radians(setting.phase_deg) + theta[0, 2]
        omega = 1 / (4 * theta[0, 1] * 1e-6)
        transition = []
        for offset in model.component_offsets_hz:
            detuning = model.pulse_detuning_offset_hz + theta[0, 0] + offset
            K = (detuning * sigma_z + omega *
                 (math.cos(phi) * sigma_x + math.sin(phi) * sigma_y)) / 2
            U = expm(-2j * np.pi * K * setting.width_us * 1e-6)
            evolved = U @ rho @ U.conj().T
            transition.append(np.trace(evolved @ sigma_x) +
                              1j * np.trace(evolved @ sigma_y))
        clock = np.arange(setting.sample_count) / setting.sample_hz
        raw = np.full_like(clock, model.baseline_complex, dtype=complex)
        for amplitude, offset, rate, weight in zip(
                transition, model.component_offsets_hz,
                model.component_decay_per_s, model.component_weights):
            raw += (model.reference_coefficient * weight * amplitude *
                    np.exp((-rate + 2j * np.pi *
                            (model.reference_frequency_hz + theta[0, 0] + offset)) * clock))
        fake_pilot = SimpleNamespace(
            baseline_complex=model.baseline_complex,
            reference_frequency_hz=model.reference_frequency_hz,
            reference_coefficient=model.reference_coefficient,
            feature_windows_s=setting.feature_windows_s)
        measured = demodulated_features(raw, clock, fake_pilot)
        np.testing.assert_allclose(actual, measured, rtol=1e-11, atol=1e-11)

    def test_drive_and_receiver_frequency_controls_are_distinct(self):
        theta = np.array([[20., 38., .2]])
        model = NMRModel(-1720., 1+0j)
        base = predict_complex(theta, candidate(), model)
        driven = predict_complex(theta, Candidate(**{
            **candidate().__dict__, "drive_frequency_shift_hz": 15.}), model)
        received = predict_complex(theta, Candidate(**{
            **candidate().__dict__, "demodulation_shift_hz": 15.}), model)
        self.assertGreater(float(np.max(np.abs(base - driven))), 1e-4)
        self.assertGreater(float(np.max(np.abs(base - received))), 1e-4)
        self.assertFalse(np.allclose(driven, received))

    def test_unverified_coherent_delay_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "verified coherent delay"):
            Candidate("ramsey", 40., 100., 90., WINDOWS,
                      sample_count=500, delay_us=60., readout_width_us=40.)

    def test_correlated_real_imag_gaussian_matches_reference_density(self):
        observation = np.array([1+2j, -1+0.5j])
        predictions = np.array([[.9+1.9j, -1.1+.4j], [0j, 0j]])
        basis = np.array([[.20, .04, .01, 0.], [.04, .25, .05, 0.],
                          [.01, .05, .30, -.02], [0., 0., -.02, .40]])
        ours = GaussianObservation(basis).logpdf(observation, predictions)
        expected = np.array([multivariate_normal.logpdf(
            interleaved_real_imag(observation - row), mean=np.zeros(4), cov=basis)
            for row in predictions])
        np.testing.assert_allclose(ours, expected, rtol=1e-12)

    def test_repeat_covariance_has_correlation_and_drift_allowance(self):
        rng = np.random.default_rng(5)
        shared = rng.normal(0, .2, size=(15, 1)) + 1j * rng.normal(0, .15, size=(15, 1))
        repeats = shared + .01 * (rng.standard_normal((15, 4)) +
                                  1j * rng.standard_normal((15, 4)))
        drift = np.tile(3 * shared, (1, 4))
        base = covariance_from_repeats(repeats)
        with_drift = covariance_from_repeats(repeats, drift_differences=drift)
        self.assertEqual(base.shape, (8, 8))
        self.assertGreater(abs(base[0, 2]), 0)
        self.assertGreater(np.trace(with_drift), np.trace(base))
        np.linalg.cholesky(with_drift)


class ParticleTests(unittest.TestCase):
    def test_log_weights_are_stable_under_extreme_likelihood(self):
        normalized, evidence = normalized_log_weights(np.array([-1e9, -1e9-1, -1e9-100]))
        self.assertAlmostEqual(float(np.exp(normalized).sum()), 1.)
        self.assertTrue(math.isfinite(evidence))

    def test_phase_wrap_and_multimodality_are_not_arithmetically_averaged(self):
        theta = np.array([[0., 40., math.radians(179.)],
                          [0., 40., math.radians(-179.)]])
        moments = weighted_posterior_moments(theta, np.array([.5, .5]))
        self.assertLess(abs(abs(moments["mean_phase_rad"]) - math.pi), .03)
        self.assertLess(moments["var_phase_rad2"], .002)
        self.assertAlmostEqual(float(phase_difference(math.radians(179),
                                                     math.radians(-179))),
                               math.radians(-2), places=9)
        bimodal = np.array([[0., 40., 0.], [0., 40., math.pi]])
        self.assertFalse(weighted_posterior_moments(
            bimodal, np.array([.5, .5]))["phase_identified"])

    def test_systematic_resampling_preserves_large_weight_and_ess(self):
        weights = np.array([.8, .1, .1])
        indices = systematic_resample_indices(weights, np.random.default_rng(4))
        self.assertEqual(len(indices), 3)
        self.assertGreaterEqual(sum(indices == 0), 2)
        self.assertLess(effective_sample_size(weights), 2.)

    def test_bayes_update_keeps_t90_when_phase_is_unidentified(self):
        rng = np.random.default_rng(4)
        prior = PriorBounds((-35., 35.), (34., 46.))
        pf = ParticleFilter.from_prior(prior, 128, rng)
        reference = NMRModel(-1700., 1+0j)
        observation = predict_complex(np.array([[8., 39., 0.2]]), candidate(), reference)[0]
        stats = pf.update(observation, candidate(), reference,
                          np.eye(2 * len(WINDOWS)) * .01)
        self.assertTrue(math.isfinite(stats.log_evidence))
        summary = pf.summary()
        self.assertGreater(summary["mean_t90_us"], 0)
        self.assertGreater(stats.ess_after_update, 0)

    def test_local_rejuvenation_respects_phase_boundary_and_bounds(self):
        rng = np.random.default_rng(21)
        bounds = PriorBounds((-10., 10.), (20., 60.),
                             phase_center_rad=math.pi, phase_half_width_rad=.15)
        pf = ParticleFilter.from_prior(bounds, 64, rng)
        pf.log_weights = np.log(np.linspace(1, 64, 64) / sum(range(1, 65)))
        pf.resample(bandwidth=.3)
        self.assertAlmostEqual(pf.ess, 64., places=8)
        self.assertTrue(np.all(pf.particles[:, 0] >= -10.))
        self.assertTrue(np.all(pf.particles[:, 1] > 0.))
        self.assertTrue(np.all(np.abs(phase_difference(
            pf.particles[:, 2], math.pi)) <= .15 + 1e-12))

    def test_tempered_replay_recovers_known_parameters_after_sharp_fid(self):
        """A narrow first likelihood must not lock onto one wrong ancestor."""
        rng = np.random.default_rng(22)
        prior = PriorBounds((-35., 35.), (30., 50.), phase_half_width_rad=.7)
        pf = ParticleFilter.from_prior(prior, 512, rng)
        model = NMRModel(-1700., 1+0j,
                         component_offsets_hz=(0., 300.),
                         component_weights=(1+0j, .3+.1j),
                         component_decay_per_s=(20., 30.))
        truth = np.array([[8., 39., .2]])
        windows = ((0., .0006), (.001, .0016), (.0023, .0029),
                   (.006, .0066), (.012, .0126), (.025, .0256))
        stages = []
        for width in (20., 40., 80., 120.):
            setting = Candidate("rabi", width, 100., 90., windows, sample_count=300)
            observed = predict_complex(truth, setting, model)[0] + .03 * (
                rng.standard_normal(6) + 1j * rng.standard_normal(6))
            stats = pf.update(observed, setting, model, np.eye(12) * .03**2)
            stages.append(stats.tempering_stages)
        summary = pf.summary()
        self.assertGreater(stages[0], 1)
        self.assertLess(abs(summary["mean_delta_hz"] - 8.), 2.)
        self.assertLess(abs(summary["mean_t90_us"] - 39.), 2.)
        self.assertLess(abs(float(phase_difference(summary["mean_phase_rad"], .2))), .15)
        self.assertEqual(summary["updates"], 4)
        with self.assertRaisesRegex(RuntimeError, "drift covariance"):
            pf.propagate_drift(delta_std_hz=1.)

    def test_particle_count_convergence_reports_failure_and_success(self):
        common = PriorBounds((-1., 1.), (39., 41.), phase_half_width_rad=.1)
        left = ParticleFilter.from_prior(common, 16, np.random.default_rng(1))
        right = ParticleFilter.from_prior(common, 32, np.random.default_rng(2))
        report = particle_count_convergence({16: left, 32: right},
            delta_tolerance_hz=10., t90_tolerance_us=10., phase_tolerance_deg=90.)
        self.assertEqual(report["selected_particles"], 16)
        shifted = ParticleFilter.from_prior(
            PriorBounds((100., 102.), (39., 41.), phase_half_width_rad=.1),
            32, np.random.default_rng(2))
        failed = particle_count_convergence({16: left, 32: shifted},
            delta_tolerance_hz=1., t90_tolerance_us=10., phase_tolerance_deg=90.)
        self.assertIsNone(failed["selected_particles"])


class DesignTests(unittest.TestCase):
    def test_predictive_mc_and_time_variant(self):
        rng = np.random.default_rng(8)
        pf = ParticleFilter.from_prior(PriorBounds((-20., 20.), (35., 45.),
                                                    phase_half_width_rad=.4), 64, rng)
        model = NMRModel(-1700., 1+0j)
        cov = np.eye(2 * len(WINDOWS)) * .02
        tolerances = Tolerances(3., 2., 10.)
        slow = candidate(seconds=20.)
        fast = candidate(seconds=10.)
        score = expected_posterior_score(pf, slow, model, cov, tolerances,
                                         mc_samples=8, rng=rng)
        self.assertTrue(math.isfinite(score) and score >= 0)
        choice = choose_candidate(pf, (slow, fast), model, cov, tolerances,
                                  rng=np.random.default_rng(41), mc_samples=8,
                                  time_weighted=True)
        self.assertEqual(choice.candidate, fast)
        self.assertEqual(len(choice.evaluations), 2)
        self.assertAlmostEqual(choice.evaluations[0].expected_score_after,
                               choice.evaluations[1].expected_score_after)

    def test_tolerance_requires_independent_validation(self):
        rng = np.random.default_rng(3)
        pf = ParticleFilter.from_prior(PriorBounds((-.01, .01), (39.99, 40.01),
                                                    phase_half_width_rad=.001), 32, rng)
        tol = Tolerances(1., 1., 5.)
        self.assertFalse(meets_uncertainty_and_validation(
            pf, tol, model_validated=True,
            independent_reference_validated=False,
            control_rotation_validated=True))
        self.assertTrue(meets_uncertainty_and_validation(
            pf, tol, model_validated=True,
            independent_reference_validated=True,
            control_rotation_validated=True))


if __name__ == "__main__":
    unittest.main()
