"""Deterministic numerical checks; generated observations are not hardware data."""

import math
import unittest

import numpy as np

from spinq_local.abh import (
    CalibrationParticleFilter, CalibrationSetting, EchoGridPosterior,
    RFGPOptimizer, RabiRateMap, TimingNotVerified, calibration_particle_prior,
    choose_fid_acquisition_plan, coarse_fine_calibration_schedule,
    conventional_log_echo_design,
    echo_fisher_information, fit_calibration_joint, hahn_echo_timing,
    fixed_calibration_schedule, iq_to_sdk, notebook_repetition_cost,
    predict_calibration, repetition_resource_count, rf_coarse_grid,
    rf_nelder_mead, rf_sequential_scan,
    static_echo_design,
)


class CalibrationTests(unittest.TestCase):
    def test_rabi_phase_and_ramsey_have_physical_units(self):
        theta = np.array([0., 1., 0.])
        x90 = CalibrationSetting("rabi", 40e-6, 100.)
        x180 = CalibrationSetting("rabi", 80e-6, 100.)
        y90 = CalibrationSetting("phase", 40e-6, 100., 90.)
        self.assertAlmostEqual(abs(predict_calibration(x90, theta, rf_hz_at_100pct=6250)), 1., places=10)
        self.assertLess(abs(predict_calibration(x180, theta, rf_hz_at_100pct=6250)), 1e-10)
        self.assertAlmostEqual(predict_calibration(y90, theta, rf_hz_at_100pct=6250).real, 1., places=10)
        ramsey = CalibrationSetting("ramsey", 40e-6, 100., 90., 120e-6)
        self.assertGreater(abs(predict_calibration(ramsey, [400., 1., 0.], rf_hz_at_100pct=6250)-
                               predict_calibration(ramsey, [0., 1., 0.], rf_hz_at_100pct=6250)), .1)

    def test_multistart_complex_fit_and_identifiability(self):
        truth = np.array([80., 1.08, .18])
        settings = ([CalibrationSetting("rabi", width, 100.) for width in (20e-6, 40e-6, 60e-6, 80e-6)] +
                    [CalibrationSetting("phase", 40e-6, 100., phase) for phase in (0., 90., 180.)] +
                    [CalibrationSetting("ramsey", 40e-6, 100., phase, wait)
                     for phase, wait in ((0., 50e-6), (90., 100e-6), (0., 150e-6))])
        covariance = np.diag([.02**2, .03**2])
        observations = [(s, predict_calibration(s, truth, rf_hz_at_100pct=6250), covariance)
                        for s in settings]
        fitted = fit_calibration_joint(observations, rf_hz_at_100pct=6250,
                                       delta_bounds_hz=(-500., 500.))
        self.assertTrue(fitted["identifiable"])
        self.assertAlmostEqual(fitted["delta_hz"], 80., places=3)
        self.assertAlmostEqual(fitted["rf_scale"], 1.08, places=5)
        self.assertAlmostEqual(fitted["phase_deg"], math.degrees(.18), places=3)
        self.assertEqual(fitted["n_observations"], len(settings))

    def test_particle_update_thresholding_and_absolute_parameters(self):
        particles = calibration_particle_prior((-200., 200.), (.7, 1.3), (-50., 50.), count=1200, seed=5)
        pf = CalibrationParticleFilter(particles, rf_hz_at_100pct=6250, seed=6)
        initial = pf.summary()
        truth = [40., 1.1, .12]
        covariance = np.eye(2)*.05**2
        settings = [CalibrationSetting("rabi", 40e-6, 100.),
                    CalibrationSetting("phase", 40e-6, 100., 90.),
                    CalibrationSetting("ramsey", 40e-6, 100., 0., 100e-6)]
        for setting in settings:
            datum = predict_calibration(setting, truth, rf_hz_at_100pct=6250)
            update = pf.update(setting, datum, covariance)
            self.assertGreaterEqual(update["tempering_stages"], 1)
        summary = pf.summary()
        self.assertLess(summary["rf_scale_sd"], initial["rf_scale_sd"])
        self.assertGreater(summary["rf_scale"], .8)  # no posterior reset to zero
        self.assertGreater(summary["ess"], 0.)
        selected, score = pf.select(settings, covariance, (50., .05, .1),
                                    policy="thresholded_per_second", n_outcomes=8)
        self.assertIn(selected, settings)
        self.assertTrue(np.isfinite(score["expected_normalized_variance"]))
        with self.assertRaises(ValueError):
            pf.select([CalibrationSetting("ramsey", 40e-6, 100., 0., 1.)],
                      covariance, (50., .05, .1), policy="thresholded_per_second", n_outcomes=4)

    def test_static_and_coarse_fine_use_same_candidate_pool(self):
        settings = [CalibrationSetting("rabi", width, 100.) for width in (20e-6, 40e-6, 80e-6)]
        settings += [CalibrationSetting("phase", 40e-6, 100., 90.)]
        fixed = fixed_calibration_schedule(settings, 6)
        coarse = coarse_fine_calibration_schedule(settings, 6,
                                                  preliminary_rf_scale=1.,
                                                  rf_hz_at_100pct=6250.)
        self.assertEqual(len(fixed), 6)
        self.assertEqual(len(coarse), 6)
        self.assertEqual(coarse[:3], fixed[:3])
        self.assertTrue(all(item in settings for item in coarse))

    def test_local_liu_west_keeps_separate_modes(self):
        rng = np.random.default_rng(8)
        particles = np.column_stack((rng.normal(0., 10., 400),
                                     np.r_[rng.normal(.8, .005, 200),
                                           rng.normal(1.2, .005, 200)],
                                     rng.normal(0., .02, 400)))
        pf = CalibrationParticleFilter(particles, rf_hz_at_100pct=6250, seed=9)
        pf._liu_west(.98)
        self.assertLess(np.mean((pf.particles[:, 1] > .9) &
                                (pf.particles[:, 1] < 1.1)), .01)
        self.assertGreater(np.mean(pf.particles[:, 1] < .9), .35)
        self.assertGreater(np.mean(pf.particles[:, 1] > 1.1), .35)


class AcquisitionTests(unittest.TestCase):
    def test_echo_requires_verified_timing_and_in_window(self):
        values = dict(width_90_s=40e-6, width_180_s=80e-6,
                      inter_pulse_gap_s=100e-6, acquisition_start_s=200e-6,
                      acquisition_end_s=500e-6)
        with self.assertRaises(TimingNotVerified):
            hahn_echo_timing(**values, timing_verified=False)
        timing = hahn_echo_timing(**values, timing_verified=True)
        self.assertAlmostEqual(timing["excitation_center_s"], 20e-6)
        self.assertAlmostEqual(timing["refocus_center_s"], 180e-6)
        self.assertAlmostEqual(timing["TE_s"], 320e-6)
        self.assertAlmostEqual(timing["echo_center_s"], 340e-6)
        with self.assertRaises(ValueError):
            hahn_echo_timing(**(values | {"acquisition_end_s": 300e-6}), timing_verified=True)

    def test_echo_grid_posterior_and_crlb_include_nuisance_amplitude(self):
        amps = np.linspace(.5, 1.5, 12)
        t2 = np.linspace(.03, .2, 18)
        posterior = EchoGridPosterior(amps, t2)
        prior_sd = posterior.summary()["T2_sd_s"]
        for te in (.02, .05, .09, .14):
            posterior.update(te, 1.1*np.exp(-te/.08), .02)
        self.assertLess(posterior.summary()["T2_sd_s"], prior_sd)
        self.assertAlmostEqual(posterior.summary()["T2_s"], .08, delta=.02)
        info = echo_fisher_information([.02, .12], 1.1, .08, .02)
        self.assertGreater(np.linalg.det(info), 0.)
        choice = posterior.choose_te(np.geomspace(.01, .2, 16), noise_sd=.02,
                                     preparation_s=.3, recovery_s=1.)
        self.assertGreater(choice["TE_s"], 0.)
        self.assertLess(choice["expected_relative_crlb"], posterior.expected_relative_crlb())
        frozen = static_echo_design(amps, t2, candidates_s=np.geomspace(.01, .2, 16),
                                    count=5, noise_sd=.02, preparation_s=.3, recovery_s=1.)
        self.assertEqual(len(frozen), 5)
        self.assertEqual(len(conventional_log_echo_design(.01, .2, 5)), 5)

    def test_fid_budget_uses_repeated_physical_acquisitions(self):
        pilot = [{"sample_count": 4000, "estimate_hz": f, "wall_seconds": 2.}
                 for f in (100., 104., 96.)] + [
                 {"sample_count": 16000, "estimate_hz": f, "wall_seconds": 4.}
                 for f in (100., 101., 99.)]
        plan = choose_fid_acquisition_plan(pilot, target_se_hz=.6, max_repeats=12)
        self.assertEqual(plan["status"], "PLAN_ONLY")
        self.assertLessEqual(plan["predicted_frequency_se_hz"], .6)
        self.assertTrue(plan["requires_independent_validation"])
        impossible = choose_fid_acquisition_plan(pilot, target_se_hz=.01, max_repeats=12)
        self.assertEqual(impossible["status"], "BUDGET_EXHAUSTED")
        shifted = [row.copy() for row in pilot]
        for row in shifted:
            if row["sample_count"] == 4000:
                row["estimate_hz"] += 1000.
        self.assertEqual(choose_fid_acquisition_plan(shifted, target_se_hz=2.)["status"],
                         "MODEL_MISMATCH")


class RFCalibrationTests(unittest.TestCase):
    def test_rabi_map_iq_conversion_headroom_and_cost(self):
        rate = RabiRateMap([10., 50., 100.], [500., 2500., 5000.],
                           [400., 2000., 4000.])
        self.assertAlmostEqual(rate.rate_hz(50., 0.), 2500.)
        self.assertAlmostEqual(rate.rate_hz(50., 90.), 2000.)
        self.assertLess(rate.non_linearity["x"], 1e-12)
        with self.assertRaises(ValueError):
            rate.rate_hz(110., 0.)
        segments = iq_to_sdk([30., 0.], [0., 40.], sample_scale=1.1,
                             relative_scale=.9, duration_s=10e-6,
                             verified_headroom_pct=50.)
        self.assertAlmostEqual(segments[0]["amplitude_pct"], 29.7)
        self.assertAlmostEqual(segments[1]["amplitude_pct"], 44.)
        self.assertAlmostEqual(segments[1]["phase_deg"], 90.)
        with self.assertRaises(ValueError):
            iq_to_sdk([30.], [40.], sample_scale=2., relative_scale=1.,
                      duration_s=10e-6, verified_headroom_pct=70.)
        ideal = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.assertAlmostEqual(notebook_repetition_cost(ideal, ideal), 0.)
        self.assertAlmostEqual(notebook_repetition_cost(-ideal, ideal), 1.)
        resources = repetition_resource_count([1, 5, 9], ["x", "y", "z"], independent_repeats=2)
        self.assertEqual(resources["physical_acquisitions"], 18)
        self.assertEqual(resources["rf_gate_uses"], 90)

    def test_gp_and_classical_rf_candidate_baselines(self):
        bounds = ((.8, 1.2), (.8, 1.2))
        self.assertEqual(len(rf_coarse_grid(bounds, levels=3)), 9)
        self.assertEqual(len(rf_sequential_scan(bounds, levels=5)), 10)
        gp = RFGPOptimizer(bounds, seed=11)
        center, first = gp.propose()
        self.assertEqual(center, [1., 1.])
        self.assertEqual(first["method"], "initial_center")
        for samp, srel in ((1., 1.), (.8, .8), (1.2, .8), (.8, 1.2), (1.2, 1.2)):
            gp.observe(samp, srel, cost=(samp-1.1)**2+(srel-.9)**2,
                       cost_variance=.0004)
        proposed, info = gp.propose(candidate_count=64)
        self.assertEqual(info["method"], "matern_gp_expected_improvement")
        self.assertTrue(.8 <= proposed[0] <= 1.2)
        self.assertTrue(.8 <= proposed[1] <= 1.2)
        with self.assertRaises(ValueError):
            gp.observe(1.3, 1., cost=1., cost_variance=.01)
        nm = rf_nelder_mead(lambda a, b: (a-1.08)**2+(b-.92)**2,
                            bounds, max_evaluations=30)
        self.assertLessEqual(nm["evaluations"], 30)
        self.assertLess(nm["best"]["cost"], .01)


if __name__ == "__main__":
    unittest.main()
