"""Offline contracts for the real experiment orchestrator and common fit."""

import json
import math
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.bayes_calibration import (
    BayesRun, _pilot_response_model, candidate, candidate_dict, candidate_from_dict,
    fit_shared, prior_and_tolerances, validate_config,
)
from experiments.likelihood import NMRModel, predict_complex
from experiments.signal_01 import PilotSignalModel, demodulated_features
from experiments.smc import PriorBounds
from experiments.smc import ParticleFilter
from spinq_benchmark.hardware import HardwareUncertain
from spinq_local.core import RawFIDRecord
from bayes_01_windows import main as windows_main, saved_config_for_reanalysis


def synthetic_pilot():
    windows = ((0., .002), (.01, .012), (.025, .027),
               (.04, .042), (.06, .062), (.09, .092))
    return PilotSignalModel(
        sample_hz=10000., bands_hz=((-100., 100.),),
        primary_component_index=0, component_frequencies_hz=(0.,),
        component_decay_per_s=(10.,), component_weights=(1 + 0j,),
        reference_frequency_hz=0., reference_coefficient=1 + 0j,
        baseline_complex=0j, receiver_phase_rad=0.,
        noise_covariance_re_im=np.eye(2) * .0004,
        lag_one_correlation=0.,
        between_acquisition_covariance_re_im=np.eye(2) * .0004,
        feature_windows_s=windows,
        feature_covariance_re_im=np.eye(12) * .0004,
        pilot_task_ids=("a", "b", "c"), pilot_residual_rms=.01,
        status="IDENTIFIED",
        diagnostics={"spectral_resolution_hz": 10.,
                     "pilot_fit_relative_residual_rms": .01,
                     "pilot_fit_coherent_relative_residual_rms": .01})


class DriverNumericalTests(unittest.TestCase):
    def test_candidate_serialization_is_resume_stable(self):
        pilot = synthetic_pilot()
        original = candidate(40., 90., pilot)
        serialized = json.loads(json.dumps(candidate_dict(original)))
        self.assertEqual(serialized, candidate_dict(candidate_from_dict(serialized)))
        self.assertEqual(serialized["feature_windows_s"][0], [0., .002])

    def test_common_weighted_fit_recovers_known_parameters(self):
        pilot = synthetic_pilot()
        model = NMRModel(0., 1 + 0j, receiver_gain=1 + 0j,
                         component_decay_per_s=(10.,))
        truth = np.asarray([[8., 39., math.radians(179.)]])
        rng = np.random.default_rng(44)
        observations = []
        for width in (20., 30., 40., 60., 80., 100., 120., 160.):
            for phase in (0., 90., 180.):
                setting = candidate(width, phase, pilot)
                y = predict_complex(truth, setting, model)[0]
                y += rng.normal(0., .02, 6) + 1j * rng.normal(0., .02, 6)
                observations.append((setting, y))
        result = fit_shared(observations, model, pilot.feature_covariance_re_im,
                            PriorBounds((-30., 30.), (25., 55.)))
        self.assertEqual(result["status"], "FIT_OK")
        self.assertAlmostEqual(result["delta_hz"], 8., delta=1.)
        self.assertAlmostEqual(result["t90_us"], 39., delta=1.)
        self.assertLess(abs(np.angle(np.exp(1j * (result["relative_phase_rad"] -
                          truth[0, 2])))), .1)

    def test_holdout_phase_response_must_match_before_live_comparison(self):
        pilot = synthetic_pilot()
        truth = np.asarray([[0., 40., 0.]])
        model = NMRModel(0., 1 + 0j, receiver_gain=2 - .3j,
                         component_decay_per_s=(10.,))
        train = [(candidate(width, 90., pilot),
                  predict_complex(truth, candidate(width, 90., pilot), model)[0])
                 for width in (20., 40., 60., 80., 120., 160.)]
        heldout = [(candidate(40., phase, pilot, family="phase"),
                    predict_complex(truth, candidate(40., phase, pilot,
                                                    family="phase"), model)[0])
                   for phase in (0., 180., 270.)]
        fitted, diagnostic = _pilot_response_model(pilot, {"t90_us": 40.},
                                                     train, heldout)
        self.assertEqual(diagnostic["status"], "PILOT_HOLDOUT_VALIDATED")
        self.assertLess(abs(fitted.receiver_gain - model.receiver_gain), .01)
        wrong = [(setting, -signal) for setting, signal in heldout]
        with self.assertRaisesRegex(ValueError, "independent pilot"):
            _pilot_response_model(pilot, {"t90_us": 40.}, train, wrong)
        # A noise-only tail can make the full-FID residual ratio large. It
        # must not turn a reversed physical phase response into a pass.
        noisy_tail = replace(pilot, diagnostics={**pilot.diagnostics,
            "pilot_fit_relative_residual_rms": .9})
        with self.assertRaisesRegex(ValueError, "independent pilot"):
            _pilot_response_model(noisy_tail, {"t90_us": 40.}, train, wrong)

    def test_data_driven_prior_has_separate_tolerances(self):
        pilot = synthetic_pilot()
        rabi = {"t90_us": 40., "t90_interval_us": [37., 43.]}
        bounds, tolerances, basis = prior_and_tolerances(
            pilot, rabi, [1., 2., 3.])
        self.assertGreater(bounds.delta_hz[1], 0.)
        self.assertLess(bounds.delta_hz[0], 0.)
        self.assertGreater(tolerances.delta_hz, 0.)
        self.assertIn("pilot repeat", basis["tolerance_rule"])

    def test_unverified_ramsey_config_is_rejected(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-01-bayes.json").read_text())
        validate_config(config)
        with self.assertRaisesRegex(ValueError, "Unverified coherent"):
            validate_config({**config, "allow_unverified_ramsey": True})
        with self.assertRaisesRegex(ValueError, "Task budget"):
            validate_config({**config, "max_tasks": 10})
        with self.assertRaisesRegex(ValueError, "Requested-RF budget"):
            validate_config({**config, "max_requested_rf_us": 100.})

    def test_stopped_uncertain_run_allows_only_disk_reanalysis(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-01-bayes.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            first = BayesRun(repo, out, config, {})
            first.data["state"] = "STOPPED_UNCERTAIN"
            first.save()
            with self.assertRaises(HardwareUncertain):
                BayesRun(repo, out, config, {}, resume=True)
            offline = BayesRun(repo, out, config, {}, resume=True, read_only=True)
            with self.assertRaisesRegex(RuntimeError, "Read-only"):
                offline.execute()

    def test_resume_gauge_rejects_changed_complex_fid_before_next_task(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-01-bayes.json").read_text())
        pilot = synthetic_pilot()
        count = 1000
        t = np.arange(count) / 10000.
        def record(key):
            return RawFIDRecord(key, key, "g", "0", "0", "NMRSIG",
                np.arange(count) * .1, t, np.ones(count), np.zeros(count),
                {"sampleFre": 10000, "sampleCount": count}, {})
        with tempfile.TemporaryDirectory() as folder:
            run = BayesRun(repo, Path(folder) / "run", config, {})
            run.pilot = pilot
            run.model = NMRModel(0., 1 + 0j)
            run.bounds = PriorBounds((-30., 30.), (20., 60.))
            from experiments.design import Tolerances
            run.tolerances = Tolerances(10., 4., 10.)
            run._record = lambda key: record(key)
            observed = record("resume_new")
            measured = demodulated_features(observed.fid, observed.time_seconds, pilot)
            run.acquire = lambda *args, **kwargs: (observed, measured + 100.)
            estimate = types.SimpleNamespace(status="FIT_OK", frequency_hz=0.)
            with patch("experiments.bayes_calibration.estimate_signal", return_value=estimate):
                with self.assertRaisesRegex(HardwareUncertain, "gauge changed"):
                    run.validate_resume_gauge()
            self.assertFalse(run.data["resume_checks"][0]["accepted"])

    def test_adaptive_qc_uses_measured_complex_features(self):
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-01-bayes.json").read_text())
        pilot = synthetic_pilot()
        with tempfile.TemporaryDirectory() as folder:
            run = BayesRun(repo, Path(folder) / "run", config, {})
            run.pilot = pilot
            run.model = NMRModel(0., 1 + 0j, receiver_gain=1 + 0j,
                                 component_decay_per_s=(10.,))
            run.bounds = PriorBounds((-30., 30.), (25., 55.))
            from experiments.design import Tolerances
            run.tolerances = Tolerances(10., 4., 10.)
            setting = candidate(40., 90., pilot)
            posterior = ParticleFilter.from_prior(run.bounds, 128,
                                                   np.random.default_rng(10))
            measured = predict_complex(posterior.particles[:1], setting, run.model)[0]
            fitted = types.SimpleNamespace(status="FIT_OK", frequency_hz=0.)
            with patch("experiments.bayes_calibration.estimate_signal", return_value=fitted):
                good = run._adaptive_observation_qc(None, measured, setting, posterior)
                bad = run._adaptive_observation_qc(None, measured + 100., setting,
                                                   posterior)
            self.assertEqual(good["status"], "OK")
            self.assertEqual(bad["status"], "CHECK_REQUIRED")

    def test_reanalysis_uses_saved_config_after_current_config_changes(self):
        repo = Path(__file__).resolve().parents[1]
        original = json.loads((repo / "config-01-bayes.json").read_text())
        changed = {**original, "seed": original["seed"] + 1}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "saved_run"
            out.mkdir()
            (out / "results.json").write_text(json.dumps({
                "experiment": "01_bayes_kalibracia", "config": original,
                "state": "COMPLETED_WITH_LIMITATIONS"}), encoding="utf-8")
            current_config = Path(folder) / "changed-config.json"
            current_config.write_text(json.dumps(changed), encoding="utf-8")
            with patch("bayes_01_windows.offline_preflight", return_value={"offline": True}), \
                 patch("bayes_01_windows.reanalyze_saved", return_value=0) as analyze, \
                 redirect_stdout(StringIO()):
                code = windows_main(["--config", str(current_config),
                                     "--reanalyze", str(out)])
            self.assertEqual(code, 0)
            self.assertEqual(analyze.call_args.args[2], original)

    def test_resume_rejects_changed_current_config_before_hardware(self):
        repo = Path(__file__).resolve().parents[1]
        original = json.loads((repo / "config-01-bayes.json").read_text())
        changed = {**original, "seed": original["seed"] + 1}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "saved_run"
            out.mkdir()
            (out / "results.json").write_text(json.dumps({
                "experiment": "01_bayes_kalibracia", "config": original,
                "state": "INTERRUPTED"}), encoding="utf-8")
            current_config = Path(folder) / "changed-config.json"
            current_config.write_text(json.dumps(changed), encoding="utf-8")
            with patch("bayes_01_windows.offline_preflight", return_value={"offline": True}), \
                 patch.object(BayesRun, "execute", side_effect=AssertionError(
                     "Hardware execution must not begin")) as execute, \
                 redirect_stdout(StringIO()):
                code = windows_main(["--config", str(current_config),
                                     "--resume", str(out)])
            self.assertEqual(code, 2)
            execute.assert_not_called()

    def test_reanalysis_requires_valid_saved_run_config(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "results.json").write_text(json.dumps({
                "experiment": "01_bayes_kalibracia", "config": {"seed": 1}}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Config fields changed"):
                saved_config_for_reanalysis(out)

    def test_full_pilot_freezes_plan_from_independent_complex_fids(self):
        """The orchestration must survive a sharp, near-noiseless Rabi profile."""
        repo = Path(__file__).resolve().parents[1]
        config = json.loads((repo / "config-01-bayes.json").read_text())
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as folder:
            session = BayesRun(repo, Path(folder) / "run", config, {"synthetic": True})

            def offline_acquire(self, key, command, *, role, block=None, method=None):
                count, fs = command.sample_count, command.sample_hz
                t = np.arange(count) / fs
                angle = (math.pi / 2) * command.width_us / 38 * \
                    (command.amplitude_pct / 100)
                phase = math.radians(command.phase_deg)
                transverse = (math.sin(phase) - 1j * math.cos(phase)) * math.sin(angle)
                lines = (np.exp((-40 + 2j * math.pi * -1920) * t) +
                         (.3 + .1j) * np.exp((-60 + 2j * math.pi * -1500) * t))
                fid = (.2 - .1j + (3 + 1j) * transverse * lines +
                       .005 * (rng.normal(size=count) + 1j * rng.normal(size=count)))
                record = RawFIDRecord(key, key, "g", "0", "0", "NMRSIG",
                    np.arange(count) * .1, t, fid.real, fid.imag,
                    {"sampleFre": fs, "sampleCount": count,
                     "pulse": {"hPulse": [{"width": command.width_us,
                                           "am": command.amplitude_pct,
                                           "phase": command.phase_deg,
                                           "freshift": 0.}]}},
                    {"wall_seconds": 10., "finished_utc": "synthetic"})
                record.save(self.out / "raw")
                self.data["acquisitions"][key] = {
                    "key": key, "role": role, "block": block, "method": method,
                    "task_id": key, "wall_seconds": 10., "full_cycle_seconds": 11.,
                    "requested_rf_us": command.width_us,
                    "raw_file": f"raw/{key}.npz"}
                self.save()
                features = (demodulated_features(record.fid, record.time_seconds,
                            self.pilot) if self.pilot is not None else np.empty(0, complex))
                return record, features

            session.acquire = types.MethodType(offline_acquire, session)
            session.run_pilot()
            self.assertEqual(session.plan["status"], "FROZEN")
            self.assertTrue((session.out / "plan.json").exists())
            self.assertGreaterEqual(session.plan["particle_count"], 128)
            self.assertEqual(session.plan["coherent_delay_status"],
                             "UNVERIFIED; no Ramsey tasks submitted")


if __name__ == "__main__":
    unittest.main()
