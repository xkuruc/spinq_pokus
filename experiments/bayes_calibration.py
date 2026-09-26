"""Real Windows acquisition and local analysis for research experiment 01.

All device commands pass through the established LiveHardware/run_raw path.
The numerical routines never read vendor FFT or server fit fields.  The Mac
can import and test this module, but only the Windows launcher performs RF
tasks.  The one-spin response is an explicitly validated short-H-pulse model;
unverified coherent delays are never submitted.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from spinq_audit.adapter import verify_installed_sdk
from spinq_audit.common import atomic_json, redact, utc_now
from spinq_audit.safety import HardwareLock
from spinq_benchmark.hardware import HardwareUncertain, LiveHardware
from spinq_local.core import (Capabilities, RawFIDRecord, Segment, SequenceIR,
                              compile_sequence, run_raw)
from spinq_local.signal import validate_axis

from .design import (Tolerances, choose_candidate, meets_uncertainty_and_validation,
                     normalized_variance_score)
from .likelihood import Candidate, NMRModel, interleaved_real_imag, predict_complex
from .output_01 import (archive_results, plot_comparisons, prepare_output,
                        save_results, snapshot_sources)
from .publish_01 import publish_results
from .signal_01 import (PilotSignalModel, demodulated_features,
                        estimate_pilot_rabi, estimate_signal,
                        identify_pilot_multiplet)
from .smc import ParticleFilter, PriorBounds, phase_difference, wrap_phase


EXPERIMENT = "01_bayes_kalibracia"
METHODS = ("A", "B", "C", "D", "E")
METHOD_NAMES = {
    "A": "fixed_identifiable_scan_multistart",
    "B": "coarse_fine_scan_multistart",
    "C": "fixed_plan_particle_filter",
    "D": "adaptive_min_normalized_variance",
    "E": "adaptive_variance_reduction_per_whole_task_second",
}
TERMINAL_STATES = {"COMPLETED", "COMPLETED_WITH_LIMITATIONS", "PILOT_FAILED",
                   "STOPPED_UNCERTAIN", "INTERRUPTED", "FAILED"}


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _from_pair(values: Sequence[float]) -> complex:
    return complex(float(values[0]), float(values[1]))


def candidate_dict(candidate: Candidate) -> dict[str, Any]:
    value = asdict(candidate)
    value["feature_windows_s"] = [[float(a), float(b)] for a, b in
                                  candidate.feature_windows_s]
    return value


def candidate_from_dict(data: dict[str, Any]) -> Candidate:
    fields = dict(data)
    fields["feature_windows_s"] = tuple(tuple(float(v) for v in w)
                                         for w in data["feature_windows_s"])
    return Candidate(**fields)


def model_dict(model: NMRModel) -> dict[str, Any]:
    return {
        "reference_frequency_hz": model.reference_frequency_hz,
        "reference_coefficient_re_im": _complex_pair(model.reference_coefficient),
        "baseline_re_im": _complex_pair(model.baseline_complex),
        "receiver_gain_re_im": _complex_pair(model.receiver_gain or model.reference_coefficient),
        "component_offsets_hz": list(model.component_offsets_hz),
        "component_weights_re_im": [_complex_pair(z) for z in model.component_weights],
        "component_decay_per_s": list(model.component_decay_per_s),
        "pulse_detuning_offset_hz": model.pulse_detuning_offset_hz,
        "amplitude_reference_pct": model.amplitude_reference_pct,
        "detection_sign": model.detection_sign,
        "physical_scope": "validated short H pulse effective one-spin response with frozen measured multiplet",
    }


def model_from_dict(data: dict[str, Any]) -> NMRModel:
    return NMRModel(
        reference_frequency_hz=float(data["reference_frequency_hz"]),
        reference_coefficient=_from_pair(data["reference_coefficient_re_im"]),
        baseline_complex=_from_pair(data["baseline_re_im"]),
        receiver_gain=_from_pair(data["receiver_gain_re_im"]),
        component_offsets_hz=tuple(float(v) for v in data["component_offsets_hz"]),
        component_weights=tuple(_from_pair(v) for v in data["component_weights_re_im"]),
        component_decay_per_s=tuple(float(v) for v in data["component_decay_per_s"]),
        pulse_detuning_offset_hz=float(data["pulse_detuning_offset_hz"]),
        amplitude_reference_pct=float(data["amplitude_reference_pct"]),
        detection_sign=int(data["detection_sign"]),
    )


def pilot_from_dict(data: dict[str, Any]) -> PilotSignalModel:
    """Reconstruct the frozen pilot exactly on resume; do not refit it."""
    return PilotSignalModel(
        sample_hz=float(data["sample_hz"]),
        bands_hz=tuple(tuple(float(v) for v in x) for x in data["bands_hz"]),
        primary_component_index=int(data["primary_component_index"]),
        component_frequencies_hz=tuple(float(v) for v in data["component_frequencies_hz"]),
        component_decay_per_s=tuple(float(v) for v in data["component_decay_per_s"]),
        component_weights=tuple(_from_pair(v) for v in data["component_weights_re_im"]),
        reference_frequency_hz=float(data["reference_frequency_hz"]),
        reference_coefficient=_from_pair(data["reference_coefficient_re_im"]),
        baseline_complex=_from_pair(data["baseline_re_im"]),
        receiver_phase_rad=float(data["receiver_phase_rad"]),
        noise_covariance_re_im=np.asarray(data["noise_covariance_re_im"], float),
        lag_one_correlation=float(data["lag_one_correlation"]),
        between_acquisition_covariance_re_im=np.asarray(
            data["between_acquisition_covariance_re_im"], float),
        feature_windows_s=tuple(tuple(float(v) for v in x) for x in data["feature_windows_s"]),
        feature_covariance_re_im=np.asarray(data["feature_covariance_re_im"], float),
        pilot_task_ids=tuple(str(v) for v in data["pilot_task_ids"]),
        pilot_residual_rms=float(data["pilot_residual_rms"]),
        status=str(data["status"]), diagnostics=dict(data["diagnostics"]),
    )


def _covariance_whitener(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Feature covariance is not square")
    return np.linalg.inv(np.linalg.cholesky(covariance))


def fit_shared(observations: Sequence[tuple[Candidate, np.ndarray]], model: NMRModel,
               covariance: np.ndarray, bounds: PriorBounds) -> dict[str, Any]:
    """The same weighted multistart estimator for A--E and references.

    The covariance is for one *independent acquisition* of a short complex
    feature vector. It is never divided by 16,000 FID samples. The fitted
    phase is relative to the independent receiver gauge; a singular phase
    direction does not invalidate an otherwise identifiable t90 estimate.
    """
    if len(observations) < 4:
        return {"status": "REFERENCE_INADEQUATE", "reason": "fewer than four independent acquisitions",
                "n_acquisitions": len(observations)}
    whiten = _covariance_whitener(covariance)
    items = [(candidate, np.asarray(features, complex)) for candidate, features in observations]
    if any(features.shape != (len(candidate.feature_windows_s),) for candidate, features in items):
        raise ValueError("Observation feature dimension differs from candidate")
    lower = np.asarray([bounds.delta_hz[0], bounds.t90_us[0], -math.pi], float)
    upper = np.asarray([bounds.delta_hz[1], bounds.t90_us[1], math.pi], float)

    def residual(theta: np.ndarray) -> np.ndarray:
        particles = theta[None, :]
        parts = []
        for candidate, observed in items:
            predicted = predict_complex(particles, candidate, model)[0]
            parts.append(whiten @ interleaved_real_imag(predicted - observed))
        return np.concatenate(parts)

    center_delta = sum(bounds.delta_hz) / 2
    center_t90 = sum(bounds.t90_us) / 2
    starts = [(center_delta, center_t90, phase) for phase in (0., -math.pi / 2,
                                                            math.pi / 2, math.pi - 1e-5)]
    starts += [(center_delta, bounds.t90_us[0] * .75 + bounds.t90_us[1] * .25, 0.),
               (center_delta, bounds.t90_us[0] * .25 + bounds.t90_us[1] * .75, 0.)]
    starts += [(bounds.delta_hz[0] * .75 + bounds.delta_hz[1] * .25, center_t90, 0.),
               (bounds.delta_hz[0] * .25 + bounds.delta_hz[1] * .75, center_t90, 0.)]
    best = None
    for start in starts:
        result = least_squares(residual, np.clip(start, lower + 1e-9, upper - 1e-9),
                               bounds=(lower, upper), max_nfev=100,
                               xtol=1e-6, ftol=1e-6, gtol=1e-6)
        rss = float(np.dot(result.fun, result.fun))
        if best is None or rss < best[0]:
            best = rss, result
    assert best is not None
    rss, result = best
    singular = np.linalg.svd(result.jac, compute_uv=False)
    rank = int(np.sum(singular > max(float(singular[0]) if len(singular) else 0., 1.) * 1e-7))
    dof = max(1, len(result.fun) - rank)
    scale = max(1., rss / dof)
    covariance_fit = np.linalg.pinv(result.jac.T @ result.jac) * scale
    stderr = np.sqrt(np.maximum(0., np.diag(covariance_fit)))
    edge = [name for index, name in enumerate(("delta_hz", "t90_us"))
            if min(result.x[index] - lower[index], upper[index] - result.x[index]) <=
            .015 * (upper[index] - lower[index])]
    phase_identified = rank == 3 and math.isfinite(float(stderr[2])) and stderr[2] < math.pi / 2
    center = [float(result.x[0]), float(result.x[1]), float(wrap_phase(result.x[2]))]
    ci95 = {"delta_hz": [center[0] - 1.96 * stderr[0], center[0] + 1.96 * stderr[0]],
            "t90_us": [center[1] - 1.96 * stderr[1], center[1] + 1.96 * stderr[1]],
            "phase_rad": ([float(wrap_phase(center[2] - 1.96 * stderr[2])),
                           float(wrap_phase(center[2] + 1.96 * stderr[2]))]
                          if phase_identified else None)}
    status = "FIT_OK" if result.success and not edge else "MODEL_CHECK_REQUIRED"
    return {"status": status, "reason": "" if status == "FIT_OK" else
            ("fit at pilot-derived bound: " + ", ".join(edge) if edge else "optimizer did not converge"),
            "n_acquisitions": len(items), "delta_hz": center[0],
            "frequency_hz": model.reference_frequency_hz + center[0],
            "t90_us": center[1], "relative_phase_rad": center[2],
            "relative_phase_deg": math.degrees(center[2]) if phase_identified else None,
            "phase_identified": phase_identified, "stderr": {
                "delta_hz": float(stderr[0]), "t90_us": float(stderr[1]),
                "phase_deg": math.degrees(float(stderr[2])) if phase_identified else None},
            "ci95": ci95, "weighted_rss": rss, "degrees_of_freedom": dof,
            "jacobian_rank": rank, "overdispersion_scale": scale,
            "boundary_parameters": edge, "multistart_count": len(starts),
            "uncertainty_scope": "conditional local linear CI; reference drift and model error separate"}


def prior_and_tolerances(pilot: PilotSignalModel, rabi: dict[str, Any],
                         frequency_repeats_hz: Sequence[float]) -> tuple[PriorBounds, Tolerances, dict]:
    """Freeze data-driven prior and tolerances before comparing any method."""
    t90 = float(rabi["t90_us"])
    interval = rabi.get("t90_interval_us")
    if not interval or interval[0] is None or interval[1] is None:
        raise ValueError("Rabi profile interval is unavailable")
    t90_lo, t90_hi = map(float, interval)
    if not 0 < t90_lo <= t90 <= t90_hi:
        raise ValueError("Rabi interval inconsistent with fitted t90")
    freq = np.asarray(frequency_repeats_hz, float)
    if len(freq) < 3 or not np.all(np.isfinite(freq)):
        raise ValueError("Independent pilot FID frequencies unavailable")
    freq_scatter = float(np.std(freq, ddof=1))
    resolution = float(pilot.diagnostics["spectral_resolution_hz"])
    # Prior is relative to the measured reference component, with a floor
    # derived from usable FID duration, not an old +/-20 or +/-200 Hz bound.
    delta_half = max(5 * freq_scatter, 3 * resolution)
    profile_half = max(t90 - t90_lo, t90_hi - t90)
    t90_half = max(3 * profile_half, .2 * t90)
    bounds = PriorBounds(delta_hz=(-delta_half, delta_half),
                         t90_us=(max(5., t90 - t90_half), min(200., t90 + t90_half)))
    if t90 <= bounds.t90_us[0] or t90 >= bounds.t90_us[1]:
        raise ValueError("Data-driven t90 prior exceeds completed H-pulse envelope")
    phase_cov = np.asarray(pilot.between_acquisition_covariance_re_im, float)
    phase_sigma_deg = math.degrees(math.sqrt(max(float(np.trace(phase_cov)), 0.)) /
                                    max(abs(pilot.reference_coefficient), 1e-12))
    tol = Tolerances(delta_hz=max(3 * freq_scatter, 1.5 * resolution),
                     t90_us=max(2 * profile_half, .08 * t90),
                     phase_deg=max(3 * phase_sigma_deg, 5.))
    details = {"frequency_repeat_scatter_hz": freq_scatter,
               "fid_resolution_hz": resolution,
               "rabi_profile_half_width_us": profile_half,
               "phase_repeat_scatter_deg_approx": phase_sigma_deg,
               "tolerance_rule": "pilot repeat scatter and Rabi profile; frozen before A-E; reference must independently resolve it",
               "prior_scope": "fid frequency shift, t90 at 100 percent RF, circular phase relative to pilot gauge"}
    return bounds, tol, details


def candidate(width_us: float, phase_deg: float, pilot: PilotSignalModel,
              *, family: str = "rabi", amplitude_pct: float = 100.,
              sample_count: int = 16000,
              estimated_wall_seconds: float | None = None) -> Candidate:
    width = float(width_us)
    if not (5 <= width <= 200):
        raise ValueError("Candidate width outside prior completed H-pulse range")
    return Candidate(family=family, width_us=width, amplitude_pct=float(amplitude_pct),
                     phase_deg=float(phase_deg) % 360,
                     feature_windows_s=pilot.feature_windows_s,
                     sample_hz=int(pilot.sample_hz), sample_count=sample_count,
                     estimated_wall_seconds=estimated_wall_seconds)


def candidate_pool(pilot: PilotSignalModel, t90_us: float,
                   full_task_seconds: float,
                   short_task_seconds: float | None = None) -> list[Candidate]:
    """Short H-pulse, phase-sensitive and FID-frequency families only."""
    widths = sorted({round(max(5., min(200., ratio * t90_us)), 3)
                     for ratio in (.5, 1., 1.5, 2., 3., 4.)})
    choices = []
    for width in widths:
        for phase in (0., 90., 180., 270.):
            family = "fid" if width == widths[0] else ("phase" if phase != 90 else "rabi")
            choices.append(candidate(width, phase, pilot, family=family,
                                     estimated_wall_seconds=full_task_seconds))
    if short_task_seconds is not None:
        for width in widths:
            for phase in (0., 90., 180., 270.):
                family = "fid" if width == widths[0] else ("phase" if phase != 90 else "rabi")
                choices.append(candidate(width, phase, pilot, family=family,
                                         sample_count=8000,
                                         estimated_wall_seconds=short_task_seconds))
    return choices


def fixed_schedule(pool: Sequence[Candidate], count: int) -> list[Candidate]:
    """Coverage of widths and phases, fixed before any method observation."""
    by_width = sorted({c.width_us for c in pool})
    grid = [next(c for c in pool if c.width_us == width and c.phase_deg == phase
                 and c.sample_count == 16000)
            for phase in (0., 90., 180., 270.) for width in by_width]
    return [grid[i % len(grid)] for i in range(count)]


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {"host", "port", "blocks", "acquisitions_per_method", "seed",
                "sample_hz", "sample_count", "pilot_widths_us", "pilot_repeat_width_us",
                "pilot_repeats", "pilot_phase_deg", "reference_acquisitions_per_block",
                "heldout_controls_per_method", "particles", "design_mc_samples",
                "max_tasks", "max_requested_rf_us", "timeout_seconds",
                "pause_seconds", "exclusive_use_confirmed", "allow_unverified_ramsey"}
    if set(config) != required:
        raise ValueError("Config fields changed; review before a physical run")
    if config["host"] != "172.19.20.100" or config["port"] != 8181:
        raise ValueError("Connection differs from the previously successful tablet endpoint")
    if config["sample_hz"] != 10000 or config["sample_count"] != 16000:
        raise ValueError("Only the completed 10 kHz / 16k sample mode is enabled")
    if config["blocks"] < 3 or config["acquisitions_per_method"] < 24:
        raise ValueError("At least three blocks and 24 new observations per arm are required")
    if config["pilot_repeats"] < 3 or len(set(config["pilot_widths_us"])) < 5:
        raise ValueError("Pilot needs three repeats and at least five Rabi widths")
    if config["reference_acquisitions_per_block"] < 6 or config["heldout_controls_per_method"] < 2:
        raise ValueError("Independent references and physical heldout controls are required")
    if config["heldout_controls_per_method"] != 2:
        raise ValueError("This controller implements exactly two heldout rotations per method")
    if config["particles"] < 128 or config["design_mc_samples"] < 4:
        raise ValueError("Numerical design budget is too small")
    if config["allow_unverified_ramsey"]:
        raise ValueError("Unverified coherent delays are not enabled")
    if config["exclusive_use_confirmed"]:
        raise ValueError("A saved config cannot prove exclusive use; fresh empty queue is required")
    if config["max_tasks"] < 1 or config["max_requested_rf_us"] <= 0:
        raise ValueError("Finite task and requested-RF budgets required")
    pilot_tasks = (len(config["pilot_widths_us"]) +
                   config["pilot_repeats"] - 1 +
                   len(set(config["pilot_phase_deg"]) - {90}) + 2)
    per_block_tasks = (len(METHODS) * (config["acquisitions_per_method"] + 2) +
                       config["reference_acquisitions_per_block"] + 2)
    if config["max_tasks"] < pilot_tasks + config["blocks"] * per_block_tasks:
        raise ValueError("Task budget cannot cover the frozen worst-case study plan")
    pilot_rf = (sum(float(x) for x in config["pilot_widths_us"]) +
                (config["pilot_repeats"] - 1 +
                 len(set(config["pilot_phase_deg"]) - {90}) + 2) *
                float(config["pilot_repeat_width_us"]))
    worst_rf = (pilot_rf + config["blocks"] *
                (200. * (len(METHODS) *
                 (config["acquisitions_per_method"] + 2) +
                 config["reference_acquisitions_per_block"]) +
                 2 * float(config["pilot_repeat_width_us"])))
    if config["max_requested_rf_us"] < worst_rf:
        raise ValueError("Requested-RF budget cannot cover the worst-case study plan")
    if config["timeout_seconds"] < 30 or config["pause_seconds"] < 0:
        raise ValueError("Invalid task timeout or relaxation pause")
    return config


def offline_preflight() -> dict[str, Any]:
    """Read-only SDK/API and numerical check before any live connection."""
    verify_installed_sdk()
    versions = {name: importlib.metadata.version(name) for name in
                ("spinqlablink", "numpy", "scipy", "scikit-learn", "matplotlib")}
    if versions["spinqlablink"] != "1.0.2":
        raise RuntimeError("Only the installed, previously functional SpinQLabLink 1.0.2 is supported")
    from spinq_benchmark.hardware import physical_request, same_physical_payload
    from spinq_audit.probes import _configure_physical
    from spinqlablink import ExperimentType, SpinQLabLink
    # Check the actual installed serializer without connecting or submitting.
    client = SpinQLabLink("127.0.0.1", 1, "offline", "offline")
    exp, parameters = client.register_experiment(ExperimentType.PHYSICAL_LAYER_EXPERIMENT)
    payload = physical_request(width_us=40., sample_count=16000, sample_hz=10000)
    _configure_physical(parameters, payload, check_serialization=False)
    actual = json.loads(exp.get_experiment_parameter()["params"])
    if not same_physical_payload(payload, actual):
        raise RuntimeError("Installed SDK physical-layer serialization differs from the established request")
    client.deregister_experiment()
    sample = np.exp(2j * np.pi * 37 * np.arange(256) / 256)
    if int(np.argmax(np.abs(np.fft.fft(sample)))) != 37:
        raise RuntimeError("Local complex numerical smoke test failed")
    return {"checked_utc": utc_now(), "versions": versions,
            "python": sys.version.split()[0], "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "gpu": "not used",
            "thread_limits": {name: os.environ.get(name) for name in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "sdk_physical_serializer": "MATCHES_ESTABLISHED_REQUEST",
            "raw_adc_verified": False, "server_fft_offload_disabled": False}


def _pilot_response_model(pilot: PilotSignalModel, rabi: dict[str, Any],
                          train: Sequence[tuple[Candidate, np.ndarray]],
                          heldout: Sequence[tuple[Candidate, np.ndarray]]) -> tuple[NMRModel, dict]:
    """Set the receiver gain from pilot widths, then test untouched phases.

    The RF detuning origin cannot be established by a demodulated FID alone.
    We leave it at the documented model convention of zero and require the
    independent phase/width holdouts to agree before using this approximation.
    """
    t90 = float(rabi["t90_us"])
    base_model = NMRModel(
        reference_frequency_hz=pilot.reference_frequency_hz,
        reference_coefficient=pilot.reference_coefficient,
        baseline_complex=pilot.baseline_complex,
        receiver_gain=pilot.reference_coefficient,
        component_offsets_hz=pilot.component_offsets_hz,
        component_weights=pilot.component_weights,
        component_decay_per_s=pilot.component_decay_per_s,
        pulse_detuning_offset_hz=0.,
        amplitude_reference_pct=100., detection_sign=1,
    )
    theta = np.asarray([[0., t90, 0.]], float)
    cov = pilot.feature_covariance_re_im
    noise_scale = math.sqrt(float(np.trace(cov)) / 2)
    alternatives = []
    for detection_sign in (1, -1):
        trial = replace(base_model, detection_sign=detection_sign)
        predicted = np.concatenate([predict_complex(theta, c, trial)[0] for c, _ in train])
        observed = np.concatenate([y for _, y in train])
        denominator = float(np.vdot(predicted, predicted).real)
        if denominator <= 1e-10:
            raise ValueError("Pilot predicted response is not identifiable")
        gain_factor = np.vdot(predicted, observed) / denominator
        if not np.isfinite(gain_factor) or abs(gain_factor) <= .01:
            raise ValueError("Pilot receiver gain collapsed")
        trial = replace(trial, receiver_gain=pilot.reference_coefficient * gain_factor)
        checks = []
        for c, y in heldout:
            estimate = predict_complex(theta, c, trial)[0]
            rms = float(np.sqrt(np.mean(np.abs(y - estimate)**2)))
            magnitude = float(np.sqrt(np.mean(np.abs(y)**2)))
            threshold = max(5 * noise_scale, .3 * magnitude,
                            2.5 * pilot.diagnostics["pilot_fit_relative_residual_rms"] * magnitude)
            checks.append({"candidate": candidate_dict(c), "complex_feature_rmse": rms,
                           "observed_feature_rms": magnitude,
                           "threshold": threshold, "passes": rms <= threshold})
        alternatives.append((sum(x["complex_feature_rmse"]**2 for x in checks),
                             trial, gain_factor, checks))
    _, model, gain_factor, checks = min(alternatives, key=lambda x: x[0])
    if not heldout or not all(item["passes"] for item in checks):
        raise ValueError("Frozen short-pulse NMR model failed independent pilot phase/return controls")
    return model, {"status": "PILOT_HOLDOUT_VALIDATED", "receiver_gain_factor_re_im":
                   _complex_pair(gain_factor), "pilot_train_acquisitions": len(train),
                   "pilot_heldout_acquisitions": len(heldout), "heldout_checks": checks,
                   "detection_sign_from_heldout_phase": model.detection_sign,
                   "pulse_detuning_origin_hz": 0.,
                   "pulse_detuning_origin_scope": "model convention, not independently measured RF carrier offset",
                   "fid_frequency_source": "complex FID; no unverified Ramsey delay submitted",
                   "receiver_gauge": "fixed independent pilot coefficient; only relative phase inferable"}


def _particle_convergence(pilot_observations: Sequence[tuple[Candidate, np.ndarray]],
                          model: NMRModel, covariance: np.ndarray,
                          bounds: PriorBounds, tolerances: Tolerances,
                          seed: int, requested_count: int) -> dict:
    """Replay heldout pilot data at increasing N before freezing N for A--E."""
    if len(pilot_observations) < 2:
        raise ValueError("Particle convergence needs two independent pilot observations")
    counts = sorted({max(128, requested_count // 4), max(256, requested_count // 2),
                     requested_count, min(4096, requested_count * 2)})
    results = []
    for number in counts:
        pf = ParticleFilter.from_prior(bounds, number, np.random.default_rng(seed + number))
        for c, observation in pilot_observations:
            pf.update(observation, c, model, covariance)
        summary = pf.summary()
        results.append({"particles": number, "posterior": summary,
                        "resampling_count": pf.resampling_count})
    def distance(a: dict, b: dict) -> dict:
        pa, pb = a["posterior"], b["posterior"]
        return {"delta_in_tolerances": abs(pa["mean_delta_hz"] - pb["mean_delta_hz"]) /
                tolerances.delta_hz,
                "t90_in_tolerances": abs(pa["mean_t90_us"] - pb["mean_t90_us"]) /
                tolerances.t90_us,
                "phase_in_tolerances": (abs(float(phase_difference(pa["mean_phase_rad"],
                    pb["mean_phase_rad"]))) / tolerances.phase_rad
                    if pa["mean_phase_rad"] is not None and pb["mean_phase_rad"] is not None
                    else None)}
    comparisons = []
    for first, second in zip(results[:-1], results[1:]):
        comparisons.append({"low": first["particles"], "high": second["particles"],
                            **distance(first, second)})
    accepted = None
    for row in comparisons:
        if row["low"] < requested_count:
            continue
        quantities = [row["delta_in_tolerances"], row["t90_in_tolerances"]]
        if row["phase_in_tolerances"] is not None:
            quantities.append(row["phase_in_tolerances"])
        if max(quantities) <= .35:
            accepted = int(row["low"])
            break
    return {"status": "CONVERGED_PILOT_REPLAY" if accepted else "NOT_CONVERGED",
            "selected_particles": accepted, "replay_acquisitions": len(pilot_observations),
            "criterion": "adjacent N posterior means within 0.35 frozen tolerance in identifiable coordinates",
            "runs": results, "comparisons": comparisons}


class BayesRun:
    def __init__(self, repo: Path, out: Path, config: dict[str, Any], preflight: dict,
                 *, resume: bool = False, read_only: bool = False):
        self.repo = Path(repo).resolve()
        self.out = prepare_output(Path(out).resolve())
        self.config = validate_config(config)
        self.preflight = preflight
        self.pilot: PilotSignalModel | None = None
        self.model: NMRModel | None = None
        self.bounds: PriorBounds | None = None
        self.tolerances: Tolerances | None = None
        self.plan: dict[str, Any] | None = None
        self.hw: LiveHardware | None = None
        self.read_only = read_only
        self.resuming = resume
        target = self.out / "results.json"
        if target.is_file():
            if not resume:
                raise ValueError("Output directory already contains a run; use --resume")
            self.data = json.loads(target.read_text(encoding="utf-8"))
            if self.data.get("config") != config:
                raise ValueError("Resume configuration differs from frozen original")
            if self.data.get("state") == "STOPPED_UNCERTAIN" and not read_only:
                raise HardwareUncertain("A task was uncertain; manual reconciliation is required")
        else:
            self.data = {"experiment": EXPERIMENT, "run_id": self.out.name,
                         "started_utc": utc_now(), "state": "RUNNING",
                         "config": config, "environment": preflight,
                         "pilot": {}, "plan": {}, "acquisitions": {}, "methods": {},
                         "references": {}, "rows": [], "errors": [],
                         "hardware_tasks_completed": 0,
                         "hardware_results_present": False,
                         "upload": {"status": "NOT_ATTEMPTED"},
                         "raw_data_type": "exported complex FID, not verified raw ADC"}
        save_results(self.out, self.data)

    def save(self) -> None:
        if self.hw is not None:
            self.data["hardware_tasks_completed"] = self.hw.task_count
            self.data["hardware_results_present"] = any((self.out / "raw").glob("*.npz"))
        save_results(self.out, self.data)

    def event(self, text: str, *, kind: str = "INFO") -> None:
        print(f"01 {kind}: {text}", flush=True)

    def _record(self, key: str) -> RawFIDRecord:
        return RawFIDRecord.load(self.out / "raw", key)

    def acquire(self, key: str, command: Candidate, *, role: str,
                block: int | None = None, method: str | None = None
                ) -> tuple[RawFIDRecord, np.ndarray]:
        if self.read_only:
            raise RuntimeError("Read-only reanalysis cannot submit hardware tasks")
        if self.hw is None:
            raise RuntimeError("No live hardware owner")
        if command.family == "ramsey" or command.delay_us or command.readout_width_us:
            raise ValueError("Coherent delay has not been verified on this connection")
        if command.demodulation_shift_hz != 0:
            raise ValueError("Receiver demodulation changes are not enabled")
        seq = SequenceIR(segments=(Segment(0., command.width_us,
                            amplitude_pct=command.amplitude_pct,
                            phase_deg=command.phase_deg,
                            detuning_hz=command.drive_frequency_shift_hz),),
                         sample_count=command.sample_count,
                         sample_hz=command.sample_hz, label=key)
        spec = compile_sequence(seq, Capabilities())
        previous = self.data["acquisitions"].get(key)
        if previous and previous["candidate"] != candidate_dict(command):
            raise ValueError(f"Resume candidate changed for {key}")
        saved_raw_before_call = (self.out / "raw" / f"{key}.json").exists()
        started = time.perf_counter()
        record = run_raw(spec, key=key, hardware=self.hw, output=self.out)
        cycle_s = time.perf_counter() - started
        if previous:
            full_cycle_s = float(previous["full_cycle_seconds"])
            cycle_scope = previous.get("full_cycle_scope", "measured whole local task cycle")
        elif saved_raw_before_call:
            saved_cycle = record.metadata.get("full_cycle_seconds")
            if isinstance(saved_cycle, (int, float)) and math.isfinite(saved_cycle):
                full_cycle_s = float(saved_cycle)
                cycle_scope = "restored from saved FID metadata"
            else:
                full_cycle_s = max(float(record.metadata.get("wall_seconds") or 0.), cycle_s)
                cycle_scope = "lower bound recovered after interrupted journal save; preflight/pause unavailable"
        else:
            full_cycle_s = cycle_s
            cycle_scope = "measured whole local task cycle"
        # Preserve exact payload and role in the original record. A resumed
        # completed task is reused from disk and is never submitted again.
        record.metadata.update({"role": role, "block": block, "method": method,
                                "candidate": candidate_dict(command),
                                "full_cycle_seconds": full_cycle_s,
                                "full_cycle_scope": cycle_scope})
        record.save(self.out / "raw")
        if self.pilot is not None:
            features = demodulated_features(record.fid, record.time_seconds, self.pilot)
        else:
            features = np.empty(0, complex)
        row = {"key": key, "role": role, "block": block, "method": method,
               "candidate": candidate_dict(command), "task_id": record.task_id,
               "wall_seconds": record.metadata.get("wall_seconds"),
               "full_cycle_seconds": full_cycle_s, "full_cycle_scope": cycle_scope,
               "requested_rf_us": spec.rf_duration_us,
               "raw_file": f"raw/{key}.npz", "completed_utc": utc_now()}
        self.data["acquisitions"][key] = row
        self.save()
        return record, features

    def _pilot_record(self, key: str, width: float, phase: float = 90.) -> RawFIDRecord:
        # The feature windows are not known until the three repeat FIDs have
        # independently established a pilot; a temporary physical command is
        # compiled with a two-sample valid window and never used as a model.
        windows = self.pilot.feature_windows_s if self.pilot else ((0., .002),)
        command = Candidate("rabi", width, 100., phase,
                            feature_windows_s=windows,
                            sample_hz=self.config["sample_hz"],
                            sample_count=self.config["sample_count"])
        record, _ = self.acquire(key, command, role="pilot")
        return record

    def run_pilot(self) -> None:
        if (self.out / "plan.json").is_file():
            self.plan = json.loads((self.out / "plan.json").read_text(encoding="utf-8"))
            self.pilot = pilot_from_dict(self.plan["pilot_signal"])
            self.model = model_from_dict(self.plan["model"])
            prior = self.plan["prior"]
            self.bounds = PriorBounds(tuple(prior["delta_hz"]), tuple(prior["t90_us"]),
                                      prior["phase_center_rad"], prior["phase_half_width_rad"])
            self.tolerances = Tolerances(**self.plan["tolerances"])
            self.event("Frozen pilot and plan restored from disk")
            return
        pilot_started = time.perf_counter()
        original_pilot_keys = {key for key, row in self.data["acquisitions"].items()
                               if row["role"] == "pilot"}
        repeat_width = float(self.config["pilot_repeat_width_us"])
        repeats = [self._pilot_record(f"pilot_repeat_{i}", repeat_width)
                   for i in range(self.config["pilot_repeats"])]
        self.pilot = identify_pilot_multiplet(repeats)
        self.data["pilot"]["signal"] = self.pilot.to_dict()
        self.save()
        if self.pilot.status != "IDENTIFIED":
            raise ValueError(f"Pilot multiplet unresolved: {self.pilot.status}; "
                             f"{self.pilot.diagnostics}")
        widths: list[float] = []
        rabi_records: list[RawFIDRecord] = []
        for width in self.config["pilot_widths_us"]:
            width = float(width)
            if width == repeat_width:
                records = repeats
            else:
                records = [self._pilot_record(f"pilot_width_{int(width)}", width)]
            for record in records:
                widths.append(width)
                rabi_records.append(record)
        rabi = estimate_pilot_rabi(widths, rabi_records, self.pilot)
        self.data["pilot"]["rabi"] = rabi.to_dict()
        self.save()
        if rabi.status != "IDENTIFIED" or rabi.t90_us is None:
            raise ValueError(f"Rabi t90 not identified: {rabi.status}; "
                             f"R2={rabi.signed_complex_r2}; {rabi.diagnostics}")
        # Different programmed phases, never used to choose the receiver gain.
        phases = [float(p) for p in self.config["pilot_phase_deg"] if float(p) != 90.]
        phase_records = [self._pilot_record(f"pilot_phase_{int(p)}", repeat_width, p)
                         for p in phases]
        frequency_rows = [estimate_signal(record, self.pilot) for record in repeats]
        if any(row.status != "FIT_OK" for row in frequency_rows):
            raise ValueError("Independent pilot frequency repeats fail frozen component check")
        frequency_repeats = [row.frequency_hz for row in frequency_rows]
        bounds, tolerances, basis = prior_and_tolerances(self.pilot, rabi.to_dict(),
                                                         frequency_repeats)
        train = []
        for width, record in zip(widths, rabi_records):
            if record.task_id in {repeats[1].task_id, repeats[2].task_id}:
                continue
            command = candidate(width, 90., self.pilot)
            train.append((command, demodulated_features(record.fid,
                                                        record.time_seconds, self.pilot)))
        heldout = [(candidate(repeat_width, 90., self.pilot),
                    demodulated_features(record.fid, record.time_seconds, self.pilot))
                   for record in repeats[1:]]
        heldout += [(candidate(repeat_width, phase, self.pilot, family="phase"),
                     demodulated_features(record.fid, record.time_seconds, self.pilot))
                    for phase, record in zip(phases, phase_records)]
        model, validation = _pilot_response_model(self.pilot, rabi.to_dict(), train,
                                                   heldout)
        # The short-spin model is approximate. Independent phase/return
        # residuals become a documented likelihood floor so a tiny repeat
        # noise covariance cannot create false posterior precision.
        model_rms = max(float(row["complex_feature_rmse"])
                        for row in validation["heldout_checks"])
        augmented = (self.pilot.feature_covariance_re_im +
                     np.eye(len(self.pilot.feature_covariance_re_im)) * model_rms**2 / 2)
        self.pilot = replace(self.pilot, feature_covariance_re_im=augmented,
            diagnostics={**self.pilot.diagnostics,
                "short_spin_model_discrepancy_floor_rms": model_rms,
                "likelihood_covariance_scope":
                    "independent repeats plus heldout physical-model discrepancy; systematic bias remains separately checked"})
        self.data["pilot"]["signal"] = self.pilot.to_dict()
        validation["likelihood_model_discrepancy_floor_rms"] = model_rms
        self.save()
        short_mode = {"status": "UNAVAILABLE", "reason": "frozen feature windows exceed short FID"}
        short_task_seconds = None
        if max(end for _, end in self.pilot.feature_windows_s) < 8000 / self.pilot.sample_hz:
            short_records = []
            try:
                for index in range(2):
                    short_command = candidate(repeat_width, 90., self.pilot,
                        sample_count=8000)
                    short_record, _ = self.acquire(f"pilot_short8000_{index}", short_command,
                                                   role="pilot")
                    short_records.append(short_record)
                short_estimates = [estimate_signal(record, self.pilot)
                                   for record in short_records]
                full_features = np.stack([demodulated_features(r.fid, r.time_seconds,
                                                               self.pilot) for r in repeats])
                short_features = np.stack([demodulated_features(r.fid, r.time_seconds,
                                                                self.pilot) for r in short_records])
                difference = short_features.mean(axis=0) - full_features.mean(axis=0)
                whitener = _covariance_whitener(self.pilot.feature_covariance_re_im)
                normalized = float(np.linalg.norm(whitener @ interleaved_real_imag(difference)))
                scatter = float(np.std(frequency_repeats, ddof=1))
                freq_tolerance = max(5 * scatter, 2 *
                    float(self.pilot.diagnostics["spectral_resolution_hz"]))
                freq_shift = max(abs(row.frequency_hz - self.pilot.reference_frequency_hz)
                                 for row in short_estimates)
                short_cycles = [float(self.data["acquisitions"][r.key]["full_cycle_seconds"])
                                for r in short_records]
                short_task_seconds = float(np.median(short_cycles))
                full_cycles = [float(self.data["acquisitions"][r.key]["full_cycle_seconds"])
                               for r in repeats]
                full_task_median = float(np.median(full_cycles))
                accepted = (all(row.status == "FIT_OK" for row in short_estimates) and
                            normalized <= 6 and freq_shift <= freq_tolerance and
                            short_task_seconds < .95 * full_task_median)
                short_mode = {"status": "VALIDATED_FASTER" if accepted else "NOT_SELECTED",
                              "short_count": 8000, "short_records": [r.key for r in short_records],
                              "feature_difference_whitened_norm": normalized,
                              "frequency_shift_max_hz": freq_shift,
                              "frequency_tolerance_hz": freq_tolerance,
                              "short_full_cycle_median_s": short_task_seconds,
                              "full_full_cycle_median_s": full_task_median,
                              "fit_statuses": [row.status for row in short_estimates],
                              "criterion": "same frozen complex FID feature/frequency; at least 5 percent measured whole-cycle saving"}
                if not accepted:
                    short_task_seconds = None
            except HardwareUncertain:
                raise
            except Exception as exc:
                short_mode = {"status": "NOT_SELECTED", "reason":
                              f"{type(exc).__name__}: {redact(str(exc))}"}
                short_task_seconds = None
        convergence = _particle_convergence(heldout, model,
            self.pilot.feature_covariance_re_im, bounds, tolerances,
            int(self.config["seed"]), int(self.config["particles"]))
        if convergence["status"] != "CONVERGED_PILOT_REPLAY":
            raise ValueError("Particle count did not converge on independent pilot replay")
        observed_cycles = [float(row["full_cycle_seconds"]) for key, row in
                           self.data["acquisitions"].items() if row["role"] == "pilot"]
        full_task_seconds = float(np.median(observed_cycles))
        pool = candidate_pool(self.pilot, rabi.t90_us, full_task_seconds,
                              short_task_seconds)
        count = int(self.config["acquisitions_per_method"])
        rng = random.Random(int(self.config["seed"]))
        block_order = []
        for _ in range(int(self.config["blocks"])):
            order = list(METHODS)
            rng.shuffle(order)
            block_order.append(order)
        self.plan = {"status": "FROZEN", "created_utc": utc_now(),
                     "pilot_signal": self.pilot.to_dict(), "rabi": rabi.to_dict(),
                     "model": model_dict(model), "model_validation": validation,
                     "short_fid_validation": short_mode,
                     "particle_convergence": convergence,
                     "prior": asdict(bounds), "tolerances": asdict(tolerances),
                     "tolerance_basis": basis,
                     "particle_count": convergence["selected_particles"],
                     "design_mc_samples": self.config["design_mc_samples"],
                     "candidate_pool": [candidate_dict(c) for c in pool],
                     "fixed_schedule": [candidate_dict(c) for c in fixed_schedule(pool, count)],
                     "block_method_order": block_order,
                     "acquisitions_per_method": count,
                     "reference_acquisitions_per_block":
                         self.config["reference_acquisitions_per_block"],
                     "heldout_controls_per_method":
                         self.config["heldout_controls_per_method"],
                     "coherent_delay_status": "UNVERIFIED; no Ramsey tasks submitted",
                     "frequency_mechanism": "complex FID fit and frozen feature likelihood",
                     "comparison_scope": "effective single-H-spin short-pulse response; not multi-qubit gate fidelity",
                     "server_fft_as_primary_input": False,
                     "internal_device_repetitions": "UNKNOWN"}
        self.model, self.bounds, self.tolerances = model, bounds, tolerances
        atomic_json(self.out / "plan.json", self.plan)
        self.data["pilot"]["status"] = "FROZEN_VALIDATED"
        self.data["pilot"]["frequency_repeats_hz"] = frequency_repeats
        self.data["pilot"]["model_validation"] = validation
        pilot_cycle_seconds = sum(float(row["full_cycle_seconds"]) for row in
            self.data["acquisitions"].values() if row["role"] == "pilot")
        new_pilot_cycle_seconds = sum(float(row["full_cycle_seconds"]) for key, row in
            self.data["acquisitions"].items() if row["role"] == "pilot" and
            key not in original_pilot_keys)
        self.data["pilot"]["numerical_preparation_seconds"] = (
            float(self.data["pilot"].get("numerical_preparation_seconds", 0.)) +
            max(0., time.perf_counter() - pilot_started - new_pilot_cycle_seconds))
        self.data["pilot"]["physical_cycle_seconds"] = pilot_cycle_seconds
        self.data["plan"] = {key: self.plan[key] for key in
                             ("status", "created_utc", "tolerances", "prior",
                              "particle_count", "acquisitions_per_method",
                              "coherent_delay_status")}
        self.save()
        self.event(f"PILOT: f={self.pilot.reference_frequency_hz:.3f} Hz, "
                   f"t90={rabi.t90_us:.2f} us, model=validated, "
                   f"particles={convergence['selected_particles']}")

    def validate_resume_gauge(self) -> None:
        """Check the frozen frequency/receiver frame before more study tasks.

        This is a new physical acquisition, never an inference from old pilot
        data. A changed sample or receiver phase invalidates continuation.
        """
        pilot, _, _, _ = self._requirements()
        checks = self.data.setdefault("resume_checks", [])
        key = f"resume_check_{len(checks):02d}"
        width = float(self.config["pilot_repeat_width_us"])
        command = candidate(width, 90., pilot, family="return_control")
        record, features = self.acquire(key, command, role="resume_check")
        repeat_records = [self._record(f"pilot_repeat_{index}")
                          for index in range(int(self.config["pilot_repeats"]))]
        center = np.mean([demodulated_features(item.fid, item.time_seconds, pilot)
                          for item in repeat_records], axis=0)
        residual = interleaved_real_imag(features - center)
        normalized = float(np.linalg.norm(
            _covariance_whitener(pilot.feature_covariance_re_im) @ residual))
        fitted = estimate_signal(record, pilot)
        repeat_frequencies = [estimate_signal(item, pilot).frequency_hz
                              for item in repeat_records]
        scatter = float(np.std(repeat_frequencies, ddof=1))
        allowed_frequency = max(5 * scatter,
            2 * float(pilot.diagnostics["spectral_resolution_hz"]))
        frequency_drift = (abs(float(fitted.frequency_hz) -
                           float(np.mean(repeat_frequencies)))
                           if fitted.frequency_hz is not None else None)
        accepted = (fitted.status == "FIT_OK" and
                    normalized <= 6. and frequency_drift is not None and
                    frequency_drift <= allowed_frequency)
        checks.append({"key": key, "task_id": record.task_id,
                       "accepted": accepted, "signal_fit_status": fitted.status,
                       "feature_whitened_norm": normalized,
                       "frequency_drift_hz": frequency_drift,
                       "frequency_drift_limit_hz": allowed_frequency,
                       "criterion": "frozen pilot receiver gauge and component frequency"})
        self.save()
        self.event(f"Resume gauge: {'OK' if accepted else 'MISMATCH'}; "
                   f"FID norm={normalized:.2f}, drift={frequency_drift} Hz")
        if not accepted:
            raise HardwareUncertain("Frozen pilot receiver/frequency gauge changed; "
                                    "start a new run after inspecting the device")

    def _requirements(self) -> tuple[PilotSignalModel, NMRModel, PriorBounds, Tolerances]:
        if self.pilot is None or self.model is None or self.bounds is None or self.tolerances is None:
            raise RuntimeError("Frozen pilot model, prior or tolerances unavailable")
        return self.pilot, self.model, self.bounds, self.tolerances

    def _step_observations(self, method_data: dict) -> list[tuple[Candidate, np.ndarray]]:
        pilot, _, _, _ = self._requirements()
        result = []
        for step in method_data["steps"]:
            record = self._record(step["key"])
            result.append((candidate_from_dict(step["candidate"]),
                           demodulated_features(record.fid, record.time_seconds, pilot)))
        return result

    def _fixed_method_candidate(self, method: str, step: int,
                                observations: list[tuple[Candidate, np.ndarray]],
                                method_data: dict) -> Candidate:
        pilot, model, bounds, _ = self._requirements()
        assert self.plan is not None
        fixed = [candidate_from_dict(value) for value in self.plan["fixed_schedule"]]
        if method in ("A", "C") or step < 8:
            return fixed[step]
        if method != "B":
            raise ValueError("Unsupported fixed method")
        if "coarse_fit" not in method_data:
            started = time.perf_counter()
            method_data["coarse_fit"] = fit_shared(observations[:8], model,
                                                   pilot.feature_covariance_re_im, bounds)
            method_data["costs"]["fitting_seconds"] += time.perf_counter() - started
            self.save()
        coarse = method_data["coarse_fit"]
        center = float(coarse["t90_us"]) if coarse.get("status") == "FIT_OK" else \
            float(self.plan["rabi"]["t90_us"])
        widths = [max(5., min(200., center * ratio))
                  for ratio in (.75, .875, 1., 1.125, 1.25)]
        phase = (0., 90., 180., 270.)[(step - 8) % 4]
        return candidate(widths[((step - 8) // 4) % len(widths)], phase, pilot,
                         family="phase" if phase != 90 else "rabi",
                         estimated_wall_seconds=float(fixed[0].estimated_wall_seconds))

    def _provisional_uncertainty_met(self, method: str, observations,
                                     particle_filter: ParticleFilter | None,
                                     method_data: dict) -> bool:
        pilot, model, bounds, tolerance = self._requirements()
        if len(observations) < 8 or len(observations) % 4:
            return False
        if method in ("C", "D", "E"):
            assert particle_filter is not None
            posterior = particle_filter.summary()
            return bool(posterior["phase_identified"] and
                math.sqrt(posterior["var_delta_hz2"]) <= tolerance.delta_hz and
                math.sqrt(posterior["var_t90_us2"]) <= tolerance.t90_us and
                math.degrees(math.sqrt(posterior["var_phase_rad2"])) <= tolerance.phase_deg)
        started = time.perf_counter()
        interim = fit_shared(observations, model,
                             pilot.feature_covariance_re_im, bounds)
        method_data["costs"]["fitting_seconds"] += time.perf_counter() - started
        method_data["interim_fits"].append({"after_acquisitions": len(observations),
                                             "fit": interim})
        if interim["status"] != "FIT_OK" or not interim["phase_identified"]:
            return False
        stderr = interim["stderr"]
        return bool(stderr["delta_hz"] <= tolerance.delta_hz and
                    stderr["t90_us"] <= tolerance.t90_us and
                    stderr["phase_deg"] <= tolerance.phase_deg)

    def _adaptive_observation_qc(self, record: RawFIDRecord,
                                 measured: np.ndarray, chosen: Candidate,
                                 particle_filter: ParticleFilter) -> dict:
        """Check each live adaptive FID against the frozen measured model."""
        pilot, model, bounds, tolerance = self._requirements()
        reasons = []
        try:
            signal = estimate_signal(record, pilot)
            status = signal.status
            frequency = float(signal.frequency_hz)
            frequency_limit = max(abs(value) for value in bounds.delta_hz) + \
                2 * float(pilot.diagnostics["spectral_resolution_hz"])
            if status != "FIT_OK":
                reasons.append(f"FID component fit {status}")
            if abs(frequency - pilot.reference_frequency_hz) > frequency_limit:
                reasons.append("FID component left frozen prior envelope")
        except Exception as exc:
            status = f"{type(exc).__name__}: {redact(str(exc))}"
            frequency = None
            frequency_limit = None
            reasons.append("FID component fit unavailable")
        predictions = predict_complex(particle_filter.particles, chosen, model)
        residual = interleaved_real_imag(measured[None, :] - predictions)
        whitener = _covariance_whitener(pilot.feature_covariance_re_im)
        minimum_norm = float(np.min(np.linalg.norm(residual @ whitener.T, axis=1)))
        if not math.isfinite(minimum_norm) or minimum_norm > 8.:
            reasons.append("complex FID features outside all predictive particles")
        before = normalized_variance_score(particle_filter.particles,
                                           particle_filter.weights, tolerance)
        return {"status": "OK" if not reasons else "CHECK_REQUIRED",
                "reasons": reasons, "signal_fit_status": status,
                "selected_frequency_hz": frequency,
                "frequency_envelope_hz": frequency_limit,
                "minimum_predictive_whitened_norm": minimum_norm,
                "normalized_variance_before": before,
                "criterion": "frozen component fit; frequency inside prior; predictive complex FID norm <= 8"}

    def _run_method(self, block: int, method: str) -> dict:
        pilot, model, bounds, tolerance = self._requirements()
        assert self.plan is not None
        identity = f"block_{block}_{method}"
        state = self.data["methods"].setdefault(identity, {
            "block": block, "method": method, "name": METHOD_NAMES[method],
            "status": "RUNNING", "steps": [], "interim_fits": [],
            "costs": {"acquisition_seconds": 0., "full_cycle_seconds": 0.,
                      "fitting_seconds": 0., "inference_seconds": 0.,
                      "design_seconds": 0.}, "control": [],
            "online_posterior": None})
        if state["status"] in ("MEASURED", "METHOD_FAILED"):
            return state
        observations = []
        particle_filter = None
        if method in ("C", "D", "E"):
            seed = int(self.config["seed"]) + 100000 * block + 1000 * METHODS.index(method)
            particle_filter = ParticleFilter.from_prior(bounds,
                int(self.plan["particle_count"]), np.random.default_rng(seed))
        count = int(self.plan["acquisitions_per_method"])
        choices = [candidate_from_dict(item) for item in self.plan["candidate_pool"]]
        fallback = min((item for item in choices if item.sample_count == 16000 and
                        item.phase_deg == 90.),
                       key=lambda item: abs(item.width_us - self.plan["rabi"]["t90_us"]))
        try:
            for index in range(count):
                key = f"b{block}_{method}_step{index:02d}"
                if index < len(state["steps"]):
                    chosen = candidate_from_dict(state["steps"][index]["candidate"])
                    if state["steps"][index]["key"] != key:
                        raise ValueError("Saved method step order differs from frozen plan")
                elif method in ("A", "B", "C"):
                    chosen = self._fixed_method_candidate(method, index,
                                                          observations, state)
                elif state.get("recovery_required"):
                    chosen = fallback
                    state.setdefault("design_history", []).append({
                        "step": index, "candidate": candidate_dict(chosen),
                        "fallback_used": True,
                        "reason": "previous measured FID failed frozen predictive check"})
                else:
                    design_start = time.perf_counter()
                    design_seed = int(self.config["seed"]) + 1000000 * block + \
                        10000 * METHODS.index(method) + index
                    decision = choose_candidate(particle_filter, choices, model,
                        pilot.feature_covariance_re_im, tolerance,
                        rng=np.random.default_rng(design_seed),
                        mc_samples=int(self.plan["design_mc_samples"]),
                        time_weighted=(method == "E"), fallback_candidate=fallback)
                    state["costs"]["design_seconds"] += time.perf_counter() - design_start
                    chosen = decision.candidate
                    state.setdefault("design_history", []).append({
                        "step": index, "candidate": candidate_dict(chosen),
                        "fallback_used": decision.fallback_used,
                        "time_weighted": decision.time_weighted,
                        "evaluations": [{"candidate": candidate_dict(value.candidate),
                                         "expected_score_after": value.expected_score_after,
                                         "expected_reduction": value.expected_reduction,
                                         "utility": value.utility,
                                         "estimated_wall_seconds": value.estimated_wall_seconds}
                                        for value in decision.evaluations]})
                record, measured = self.acquire(key, chosen, role="method",
                                                block=block, method=method)
                if not len(measured):
                    measured = demodulated_features(record.fid, record.time_seconds, pilot)
                observations.append((chosen, measured))
                new_step = index >= len(state["steps"])
                qc = (self._adaptive_observation_qc(record, measured, chosen,
                       particle_filter) if method in ("D", "E") and new_step else None)
                if particle_filter is not None:
                    inference_start = time.perf_counter()
                    update = particle_filter.update(measured, chosen, model,
                                                     pilot.feature_covariance_re_im)
                    state["costs"]["inference_seconds"] += time.perf_counter() - inference_start
                    online = {**particle_filter.summary(),
                        "update": asdict(update)}
                    if qc is not None:
                        after = normalized_variance_score(particle_filter.particles,
                                                          particle_filter.weights, tolerance)
                        qc["normalized_variance_after"] = after
                        if index >= 4 and after > 1.5 * max(
                                qc["normalized_variance_before"], 1e-12):
                            qc["status"] = "CHECK_REQUIRED"
                            qc["reasons"].append("posterior uncertainty rose after update")
                else:
                    online = None
                if new_step:
                    acquisition = self.data["acquisitions"][key]
                    state["costs"]["acquisition_seconds"] += float(acquisition["wall_seconds"])
                    state["costs"]["full_cycle_seconds"] += float(acquisition["full_cycle_seconds"])
                    state["steps"].append({"key": key, "candidate": candidate_dict(chosen),
                                           "online_posterior": online,
                                           "measured_fid_qc": qc,
                                           "task_id": record.task_id})
                    if qc is not None:
                        prior_streak = int(state.get("qc_failure_streak", 0))
                        failed = qc["status"] != "OK"
                        state["qc_failure_streak"] = prior_streak + 1 if failed else 0
                        state["recovery_required"] = failed
                        if failed:
                            self.event(f"Block {block + 1} {method}: measured FID check "
                                       f"requires recovery at {key}: {qc['reasons']}",
                                       kind="WARNING")
                    atomic_json(self.out / "pulses" / f"{key}.json", {
                        "command": candidate_dict(chosen),
                        "payload_sent": record.parameters_sent,
                        "task_id": record.task_id, "role": "method",
                        "model_scope": "short H pulse with explicit complex FID acquisition"})
                    self.save()
                    if qc is not None and state["qc_failure_streak"] >= 2:
                        raise ValueError("MODEL_MISMATCH: two consecutive physical FIDs "
                                         "failed frozen component/predictive checks")
                if (index + 1) % 4 == 0:
                    self.event(f"Block {block + 1}/{self.config['blocks']} {method}: "
                               f"{index + 1}/{count} acquisitions; "
                               f"ESS={particle_filter.ess:.0f}" if particle_filter is not None
                               else f"Block {block + 1}/{self.config['blocks']} {method}: "
                                    f"{index + 1}/{count} acquisitions")
                minimum = 16 if method == "B" else 8
                if (index + 1 >= minimum and not state.get("recovery_required") and
                    self._provisional_uncertainty_met(
                        method, observations, particle_filter, state)):
                    state["stop_reason"] = "PROVISIONAL_UNCERTAINTY_MET_AWAITING_INDEPENDENT_REFERENCE"
                    self.save()
                    break
            else:
                state["stop_reason"] = "BUDGET_EXHAUSTED"
            fit_start = time.perf_counter()
            state["common_final_fit"] = fit_shared(observations, model,
                                                    pilot.feature_covariance_re_im, bounds)
            state["costs"]["fitting_seconds"] += time.perf_counter() - fit_start
            if particle_filter is not None:
                state["online_posterior"] = particle_filter.summary()
                state["particle_resamplings"] = particle_filter.resampling_count
            state["status"] = "MEASURED"
            self.save()
            self.event(f"Block {block + 1} {method}: {len(state['steps'])} acquisitions, "
                       f"fit={state['common_final_fit']['status']}, "
                       f"stop={state['stop_reason']}")
        except HardwareUncertain:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            state["status"] = "METHOD_FAILED"
            state["reason"] = f"{type(exc).__name__}: {redact(str(exc))}"
            state["traceback"] = redact(traceback.format_exc())
            self.data["errors"].append({"block": block, "method": method,
                                        "reason": state["reason"],
                                        "traceback": state["traceback"]})
            self.save()
            self.event(f"Block {block + 1} {method}: {state['reason']}", kind="ERROR")
            print(state["traceback"], flush=True)
        return state

    def _run_controls(self, block: int, method: str, state: dict) -> None:
        pilot, model, _, _ = self._requirements()
        fit = state.get("common_final_fit", {})
        if fit.get("status") != "FIT_OK" or not fit.get("phase_identified"):
            state["control_status"] = "NOT_SUBMITTED_UNIDENTIFIED_CALIBRATION"
            self.save()
            return
        correction = -float(fit["relative_phase_deg"])
        width = float(fit["t90_us"])
        if not 5 <= width <= 200:
            state["control_status"] = "NOT_SUBMITTED_WIDTH_OUTSIDE_COMPLETED_ENVELOPE"
            self.save()
            return
        for index, nominal_phase in enumerate((0., 90.)):
            key = f"b{block}_{method}_control{index}"
            command = candidate(width, nominal_phase + correction, pilot,
                                family="return_control")
            record, features = self.acquire(key, command, role="control",
                                            block=block, method=method)
            if not len(features):
                features = demodulated_features(record.fid, record.time_seconds, pilot)
            if not any(item["key"] == key for item in state["control"]):
                acquisition = self.data["acquisitions"][key]
                state["costs"]["acquisition_seconds"] += float(acquisition["wall_seconds"])
                state["costs"]["full_cycle_seconds"] += float(acquisition["full_cycle_seconds"])
                state["control"].append({"key": key, "candidate": candidate_dict(command),
                                         "task_id": record.task_id,
                                         "nominal_phase_deg": nominal_phase,
                                         "calibration_applied": {
                                             "width_us": width, "phase_correction_deg": correction,
                                             "source_fit": f"block_{block}_{method}"}})
                atomic_json(self.out / "pulses" / f"{key}.json", {
                    "command": candidate_dict(command), "payload_sent": record.parameters_sent,
                    "calibration_applied": state["control"][-1]["calibration_applied"],
                    "task_id": record.task_id, "role": "heldout_control"})
                self.save()
        state["control_status"] = "MEASURED_HELDOUT"
        self.save()

    def _run_block_reference(self, block: int) -> dict:
        pilot, model, bounds, _ = self._requirements()
        assert self.plan is not None
        identity = f"block_{block}"
        reference = self.data["references"].setdefault(identity, {
            "block": block, "status": "RUNNING", "keys": [],
            "fit": None, "return_keys": []})
        if reference["status"] == "MEASURED":
            return reference
        pool = [candidate_from_dict(row) for row in self.plan["candidate_pool"]
                if int(row["sample_count"]) == 16000]
        widths = sorted({choice.width_us for choice in pool})
        count = int(self.plan["reference_acquisitions_per_block"])
        observations = []
        for index in range(count):
            fraction = index / max(count - 1, 1)
            width = widths[round(fraction * (len(widths) - 1))]
            phase = (0., 90., 180., 270.)[index % 4]
            command = next(x for x in pool if x.width_us == width and x.phase_deg == phase)
            key = f"b{block}_reference_{index:02d}"
            record, features = self.acquire(key, command, role="independent_reference",
                                            block=block)
            if not len(features):
                features = demodulated_features(record.fid, record.time_seconds, pilot)
            observations.append((command, features))
            if key not in reference["keys"]:
                reference["keys"].append(key)
                self.save()
        start = time.perf_counter()
        reference["fit"] = fit_shared(observations, model,
                                      pilot.feature_covariance_re_im, bounds)
        reference["fitting_seconds"] = time.perf_counter() - start
        reference["status"] = "MEASURED"
        self.save()
        self.event(f"Block {block + 1}: independent reference "
                   f"{reference['fit']['status']} from {len(observations)} new acquisitions")
        return reference

    def run_blocks(self) -> None:
        pilot, _, _, _ = self._requirements()
        assert self.plan is not None
        for block, order in enumerate(self.plan["block_method_order"]):
            self.event(f"Block {block + 1}/{self.config['blocks']}: order {' '.join(order)}")
            before_key = f"b{block}_return_before"
            before_command = candidate(float(self.config["pilot_repeat_width_us"]),
                                       90., pilot, family="return_control")
            self.acquire(before_key, before_command, role="return_control", block=block)
            for method in order:
                state = self._run_method(block, method)
                if state["status"] == "MEASURED":
                    self._run_controls(block, method, state)
            after_key = f"b{block}_return_after"
            self.acquire(after_key, before_command, role="return_control", block=block)
            self._run_block_reference(block)
            self.evaluate_block(block)

    def _reference_quality(self, block: int) -> tuple[bool, dict]:
        pilot, _, _, tolerances = self._requirements()
        reference = self.data["references"].get(f"block_{block}", {})
        fit = reference.get("fit") or {}
        before = self.data["acquisitions"].get(f"b{block}_return_before")
        after = self.data["acquisitions"].get(f"b{block}_return_after")
        drift = None
        if before and after:
            a = self._record(before["key"])
            b = self._record(after["key"])
            difference = demodulated_features(b.fid, b.time_seconds, pilot) - \
                demodulated_features(a.fid, a.time_seconds, pilot)
            whiten = _covariance_whitener(pilot.feature_covariance_re_im)
            drift = float(np.linalg.norm(whiten @ interleaved_real_imag(difference)))
        stderr = fit.get("stderr") or {}
        resolved = bool(fit.get("status") == "FIT_OK" and fit.get("phase_identified") and
            stderr.get("delta_hz") is not None and stderr["delta_hz"] <= tolerances.delta_hz / 2 and
            stderr.get("t90_us") is not None and stderr["t90_us"] <= tolerances.t90_us / 2 and
            stderr.get("phase_deg") is not None and stderr["phase_deg"] <= tolerances.phase_deg / 2)
        drift_ok = bool(drift is not None and drift <= 12.)
        return resolved and drift_ok, {
            "status": "VALIDATED" if resolved and drift_ok else "REFERENCE_INADEQUATE",
            "reference_parameter_precision_resolved": resolved,
            "return_control_whitened_difference": drift,
            "return_control_drift_limit": 12.,
            "drift_control_valid": drift_ok,
            "reference_fit_status": fit.get("status"),
            "reference_uncertainty": stderr,
            "reason": "" if resolved and drift_ok else
                      "independent reference uncertainty or block return drift does not resolve frozen tolerances"}

    def _control_quality(self, block: int, method: str,
                         reference_fit: dict[str, Any]) -> tuple[bool, dict]:
        pilot, model, _, _ = self._requirements()
        state = self.data["methods"][f"block_{block}_{method}"]
        if state.get("control_status") != "MEASURED_HELDOUT" or len(state["control"]) < 2:
            return False, {"status": state.get("control_status", "NOT_MEASURED"),
                           "reason": "two physical controls with applied calibration are required"}
        theta = np.asarray([[reference_fit["delta_hz"], reference_fit["t90_us"],
                             reference_fit["relative_phase_rad"]]], float)
        rows = []
        for item in state["control"]:
            record = self._record(item["key"])
            command = candidate_from_dict(item["candidate"])
            measured = demodulated_features(record.fid, record.time_seconds, pilot)
            # Compare against a *fixed* target defined by the independent
            # reference, not the response predicted for this method's own
            # command. Otherwise a wrong correction would move both sides of
            # the comparison and appear deceptively good.
            nominal_phase = float(item["nominal_phase_deg"])
            ideal_command = candidate(float(reference_fit["t90_us"]),
                nominal_phase - float(reference_fit["relative_phase_deg"]), pilot,
                family="return_control")
            expected = predict_complex(theta, ideal_command, model)[0]
            same_command_prediction = predict_complex(theta, command, model)[0]
            difference = measured - expected
            absolute = float(np.sqrt(np.mean(np.abs(difference)**2)))
            response = float(np.sqrt(np.mean(np.abs(expected)**2)))
            relative = absolute / max(response, 1e-10)
            noise = math.sqrt(float(np.trace(pilot.feature_covariance_re_im)) /
                              len(pilot.feature_covariance_re_im))
            limit = max(.3, 5 * noise / max(response, 1e-10))
            rows.append({"key": item["key"], "complex_feature_rmse": absolute,
                         "relative_complex_error": relative,
                         "same_command_model_error": float(np.sqrt(np.mean(
                             np.abs(measured - same_command_prediction)**2))),
                         "ideal_target": candidate_dict(ideal_command),
                         "threshold": limit, "passes": relative <= limit})
        success = all(item["passes"] for item in rows)
        return success, {"status": "VALIDATED" if success else "CONTROL_MISMATCH",
                         "relative_complex_error_mean": float(np.mean(
                             [item["relative_complex_error"] for item in rows])),
                         "controls": rows,
                         "scope": "heldout exported complex FID response to reference-defined 90-degree target; not quantum gate fidelity",
                         "frequency_rf_correction_applied": False,
                         "frequency_control_limit": "per-pulse RF carrier mapping outside established H envelope unverified; frequency validated against independent FID reference"}

    @staticmethod
    def _coverage(estimate: float | None, reference: float | None,
                  estimate_se: float | None, reference_se: float | None,
                  *, circular_degrees: bool = False) -> bool | None:
        if any(value is None or not math.isfinite(float(value)) for value in
               (estimate, reference, estimate_se, reference_se)):
            return None
        difference = (math.degrees(float(phase_difference(math.radians(estimate),
                 math.radians(reference)))) if circular_degrees else float(estimate) - float(reference))
        combined = math.hypot(float(estimate_se), float(reference_se))
        return abs(difference) <= 1.96 * combined

    def _equal_budget_fit(self, block: int, method: str, count: int) -> dict:
        pilot, model, bounds, _ = self._requirements()
        state = self.data["methods"][f"block_{block}_{method}"]
        if len(state.get("steps", [])) < count or count < 4:
            return {"status": "INSUFFICIENT_ACQUISITIONS"}
        observations = self._step_observations({"steps": state["steps"][:count]})
        return fit_shared(observations, model,
                          pilot.feature_covariance_re_im, bounds)

    def evaluate_block(self, block: int) -> None:
        """Independent reference is read only after all candidate choices stop."""
        pilot, model, _, tolerance = self._requirements()
        reference = self.data["references"].get(f"block_{block}", {})
        reference_fit = reference.get("fit") or {}
        reference_ok, reference_quality = self._reference_quality(block)
        reference["quality"] = reference_quality
        common_budget = min((len(self.data["methods"].get(f"block_{block}_{m}",
                       {}).get("steps", [])) for m in METHODS), default=0)
        rows = []
        for method in METHODS:
            state = self.data["methods"].get(f"block_{block}_{method}", {})
            fit = state.get("common_final_fit") or {}
            fit_ok = fit.get("status") == "FIT_OK"
            control_ok = False
            control = {"status": "NOT_EVALUABLE"}
            if fit_ok and reference_fit.get("status") == "FIT_OK":
                control_ok, control = self._control_quality(block, method, reference_fit)
            state["control_quality"] = control
            delta_f = (float(fit["frequency_hz"] - reference_fit["frequency_hz"])
                       if fit_ok and reference_fit.get("status") == "FIT_OK" else None)
            delta_t90 = (float(fit["t90_us"] - reference_fit["t90_us"])
                         if fit_ok and reference_fit.get("status") == "FIT_OK" else None)
            delta_phase = (math.degrees(float(phase_difference(
                fit["relative_phase_rad"], reference_fit["relative_phase_rad"])))
                if fit_ok and fit.get("phase_identified") and
                reference_fit.get("status") == "FIT_OK" and reference_fit.get("phase_identified")
                else None)
            estimate_se = fit.get("stderr") or {}
            reference_se = reference_fit.get("stderr") or {}
            tolerances_met = bool(delta_f is not None and delta_t90 is not None and
                delta_phase is not None and abs(delta_f) <= tolerance.delta_hz and
                abs(delta_t90) <= tolerance.t90_us and abs(delta_phase) <= tolerance.phase_deg)
            uncertainty_met = bool(fit_ok and fit.get("phase_identified") and
                estimate_se.get("delta_hz") is not None and
                estimate_se["delta_hz"] <= tolerance.delta_hz and
                estimate_se.get("t90_us") is not None and
                estimate_se["t90_us"] <= tolerance.t90_us and
                estimate_se.get("phase_deg") is not None and
                estimate_se["phase_deg"] <= tolerance.phase_deg)
            online = state.get("online_posterior") or {}
            if method in ("C", "D", "E"):
                online_qualified = bool(online.get("phase_identified") and
                    math.sqrt(online.get("var_delta_hz2", math.inf)) <= tolerance.delta_hz and
                    math.sqrt(online.get("var_t90_us2", math.inf)) <= tolerance.t90_us and
                    math.degrees(math.sqrt(online.get("var_phase_rad2", math.inf))) <= tolerance.phase_deg)
            else:
                online_qualified = True
            success = bool(reference_ok and fit_ok and control_ok and
                           tolerances_met and uncertainty_met and online_qualified)
            if state.get("status") == "METHOD_FAILED":
                status, reason = "METHOD_FAILED", state.get("reason", "unknown method failure")
            elif not fit_ok:
                status, reason = "MODEL_MISMATCH", fit.get("reason", "shared final fit unavailable")
            elif not reference_ok:
                status, reason = "REFERENCE_INADEQUATE", reference_quality["reason"]
            elif not fit.get("phase_identified") or not reference_fit.get("phase_identified"):
                status, reason = "PHASE_UNIDENTIFIED", (
                    "relative phase was not identifiable; heldout rotation control was not submitted")
            elif not control_ok:
                status, reason = "CONTROL_MISMATCH", control.get("status", "control unavailable")
            elif success:
                status, reason = "SUCCESS_VALIDATED", "independent reference and heldout controls resolve frozen tolerances"
            elif state.get("stop_reason") == "BUDGET_EXHAUSTED":
                status, reason = "BUDGET_EXHAUSTED", "maximal physical acquisition budget reached without validated tolerance"
            else:
                status, reason = "VALID_NEGATIVE_RESULT", "provisional stop did not satisfy independent validation"
            common_fit = self._equal_budget_fit(block, method, common_budget)
            equal_errors = None
            if common_fit.get("status") == "FIT_OK" and reference_fit.get("status") == "FIT_OK":
                equal_errors = {
                    "frequency_error_hz": abs(common_fit["frequency_hz"] - reference_fit["frequency_hz"]),
                    "t90_error_us": abs(common_fit["t90_us"] - reference_fit["t90_us"]),
                    "phase_error_deg": (abs(math.degrees(float(phase_difference(
                        common_fit["relative_phase_rad"], reference_fit["relative_phase_rad"]))))
                        if common_fit.get("phase_identified") and reference_fit.get("phase_identified")
                        else None)}
            acquisition_rows = [self.data["acquisitions"][step["key"]]
                                for step in state.get("steps", [])]
            prefix_seconds = sum(float(item["wall_seconds"])
                                 for item in acquisition_rows[:common_budget])
            costs = state.get("costs", {})
            pilot_share = sum(float(row["full_cycle_seconds"]) for row in
                self.data["acquisitions"].values() if row["role"] == "pilot") / \
                (len(METHODS) * int(self.config["blocks"]))
            reference_share = sum(float(self.data["acquisitions"][key]["full_cycle_seconds"])
                for key in reference.get("keys", [])) / len(METHODS)
            return_share = sum(float(self.data["acquisitions"][key]["full_cycle_seconds"])
                for key in (f"b{block}_return_before", f"b{block}_return_after")
                if key in self.data["acquisitions"]) / len(METHODS)
            pilot_preparation_share = float(self.data["pilot"].get(
                "numerical_preparation_seconds", 0.)) / (len(METHODS) * int(self.config["blocks"]))
            reference_fitting_share = float(reference.get("fitting_seconds", 0.)) / len(METHODS)
            connection_setup_share = sum(float(value) for value in self.data.get(
                "connection_setup_seconds", [])) / (len(METHODS) * int(self.config["blocks"]))
            repeated_use = (sum(float(costs.get(name, 0.)) for name in
                             ("full_cycle_seconds", "fitting_seconds", "inference_seconds",
                              "design_seconds")) + reference_share + return_share +
                            reference_fitting_share)
            end_to_end = (sum(float(costs.get(name, 0.)) for name in
                             ("full_cycle_seconds", "fitting_seconds", "inference_seconds",
                              "design_seconds")) + pilot_share + reference_share + return_share +
                            pilot_preparation_share + reference_fitting_share + connection_setup_share)
            row = {"method": method, "method_name": METHOD_NAMES[method],
                   "baseline": "C and B" if method in ("D", "E") else
                               ("A" if method == "B" else "none"),
                   "block": block + 1, "status": status, "reason": reason,
                   "acquisitions": len(state.get("steps", [])) + len(state.get("control", [])),
                   "design_acquisitions": len(state.get("steps", [])),
                   "control_acquisitions": len(state.get("control", [])),
                   "reference_acquisitions_shared": len(reference.get("keys", [])),
                   "pilot_acquisitions_shared": sum(row["role"] == "pilot"
                       for row in self.data["acquisitions"].values()),
                   "acquisition_seconds": costs.get("acquisition_seconds", 0.),
                   "full_cycle_seconds": costs.get("full_cycle_seconds", 0.),
                   "fitting_seconds": costs.get("fitting_seconds", 0.),
                   "inference_seconds": costs.get("inference_seconds", 0.),
                   "design_seconds": costs.get("design_seconds", 0.),
                   "shared_pilot_seconds_charged": pilot_share,
                   "shared_reference_seconds_charged": reference_share,
                   "shared_return_seconds_charged": return_share,
                   "shared_pilot_numerical_seconds_charged": pilot_preparation_share,
                   "shared_reference_fitting_seconds_charged": reference_fitting_share,
                   "shared_connection_setup_seconds_charged": connection_setup_share,
                   "shared_return_acquisitions": sum(key in self.data["acquisitions"]
                       for key in (f"b{block}_return_before", f"b{block}_return_after")),
                   "end_to_end_seconds": end_to_end,
                   "end_to_end_seconds_repeated_use_excluding_common_pilot": repeated_use,
                   "frequency_error_hz": abs(delta_f) if delta_f is not None else None,
                   "t90_error_us": abs(delta_t90) if delta_t90 is not None else None,
                   "phase_error_deg": abs(delta_phase) if delta_phase is not None else None,
                   "coverage_frequency": self._coverage(fit.get("frequency_hz"),
                       reference_fit.get("frequency_hz"), estimate_se.get("delta_hz"),
                       reference_se.get("delta_hz")),
                   "coverage_t90": self._coverage(fit.get("t90_us"),
                       reference_fit.get("t90_us"), estimate_se.get("t90_us"),
                       reference_se.get("t90_us")),
                   "coverage_phase": self._coverage(fit.get("relative_phase_deg"),
                       reference_fit.get("relative_phase_deg"), estimate_se.get("phase_deg"),
                       reference_se.get("phase_deg"), circular_degrees=True),
                   "reference_uncertainty": reference_se,
                   "estimate_uncertainty": estimate_se,
                   "calibration_bias": {"frequency_hz": delta_f, "t90_us": delta_t90,
                                        "relative_phase_deg": delta_phase},
                   "control_error": control.get("relative_complex_error_mean"),
                   "control_quality": control,
                   "online_posterior": online if method in ("C", "D", "E") else None,
                   "stop_reason": state.get("stop_reason"),
                   "equal_budget_acquisitions": common_budget,
                   "equal_budget_error": equal_errors,
                   "equal_budget_prefix_acquisition_seconds": prefix_seconds,
                   "equal_budget_mode": "offline refit of actually measured prefix; no physical time saving claimed",
                   "cost_to_tolerance_mode": "actual provisional online stop followed by independent physical validation",
                   "tolerances": asdict(tolerance),
                   "reference_status": reference_quality["status"],
                   "internal_device_repetitions": "UNKNOWN"}
            rows.append(row)
            atomic_json(self.out / "models" / f"block_{block}_{method}.json", {
                "method": method, "block": block + 1, "common_final_fit": fit,
                "online_posterior": online, "interim_fits": state.get("interim_fits", []),
                "design_history": state.get("design_history", []),
                "step_history": state.get("steps", []),
                "costs": costs, "control_quality": control,
                "equal_budget_fit": common_fit})
            self.event(f"Block {block + 1} {method}: {status}; "
                       f"acq={row['acquisitions']}, "
                       f"|df|={row['frequency_error_hz']}, "
                       f"|dt90|={row['t90_error_us']}, "
                       f"control={row['control_error']}")
        self.data["rows"] = [row for row in self.data["rows"] if row.get("block") != block + 1] + rows
        self.data["references"][f"block_{block}"] = reference
        calibration = {"experiment": EXPERIMENT, "units": {
            "frequency": "Hz", "t90": "microseconds", "relative_phase": "degrees"},
            "origin": "new independent pilot and per-method real Windows FID acquisitions",
            "validity": "short H pulses on this session/device under frozen measured multiplet and receiver gauge",
            "persistent_device_calibration_modified": False,
            "absolute_transmitter_phase_identified": False,
            "reference_measurements_not_used_for_candidate_selection": True,
            "blocks": {key: {"fit": value.get("fit"), "quality": value.get("quality")}
                       for key, value in self.data["references"].items()},
            "methods": {key: {"fit": value.get("common_final_fit"),
                               "online_posterior": value.get("online_posterior"),
                               "control_quality": value.get("control_quality")}
                        for key, value in self.data["methods"].items()}}
        atomic_json(self.out / "calibration.json", calibration)
        self.save()

    def _comparative_summary(self) -> None:
        rows = self.data["rows"]
        pairs = []
        for block in range(1, int(self.config["blocks"]) + 1):
            by_method = {row["method"]: row for row in rows if row["block"] == block}
            for baseline, proposed in (("C", "D"), ("C", "E"), ("B", "D"), ("B", "E")):
                a, b = by_method.get(baseline), by_method.get(proposed)
                if not a or not b:
                    continue
                comparable = all(row["status"] in ("SUCCESS_VALIDATED", "VALID_NEGATIVE_RESULT",
                                                     "BUDGET_EXHAUSTED") for row in (a, b))
                same_quality = bool(a["status"] == "SUCCESS_VALIDATED" and
                                    b["status"] == "SUCCESS_VALIDATED")
                time_saved = (a["end_to_end_seconds"] - b["end_to_end_seconds"]
                              if comparable else None)
                acquisition_saved = (a["acquisitions"] - b["acquisitions"]
                                     if comparable else None)
                pairs.append({"block": block, "baseline": baseline,
                              "proposed": proposed,
                              "status": ("SAME_VALIDATED_QUALITY" if same_quality else
                                         "QUALITY_NOT_JOINTLY_VALIDATED" if comparable else
                                         "INCOMPARABLE"),
                              "time_saved_seconds": time_saved,
                              "acquisitions_saved": acquisition_saved,
                              "benefit_at_same_validated_quality": bool(same_quality and
                                  (time_saved > 0 or acquisition_saved > 0)),
                              "absolute_frequency_error_difference_hz":
                                  (b["frequency_error_hz"] - a["frequency_error_hz"]
                                   if comparable and b["frequency_error_hz"] is not None and
                                   a["frequency_error_hz"] is not None else None)})
        self.data["comparative_pairs"] = pairs
        self.save()

    def finish(self, *, upload: bool = True) -> None:
        """No hardware commands; keeps partial data and a complete local ZIP."""
        if self.data["rows"]:
            self._comparative_summary()
        if not (self.out / "calibration.json").exists():
            atomic_json(self.out / "calibration.json", {
                "experiment": EXPERIMENT,
                "status": "PILOT_ONLY" if (self.out / "plan.json").exists() else
                          "UNAVAILABLE_PILOT_INCOMPLETE",
                "units": {"frequency": "Hz", "t90": "microseconds",
                          "relative_phase": "degrees"},
                "origin": "this run's measured FID pilot; no persistent device settings changed",
                "persistent_device_calibration_modified": False,
                "pilot": self.data.get("pilot", {}),
                "validity": "No method calibration certified without independent block reference and controls"})
        if not (self.out / "source_snapshot" / "manifest.json").exists():
            snapshot_sources(self.out, self.repo)
        plot_comparisons(self.out, self.data["rows"])
        self.data["finished_utc"] = utc_now()
        self.save()
        archive = archive_results(self.out)
        self.event(f"Report: {self.out / 'REPORT.md'}")
        self.event(f"Archive: {archive}")
        if upload:
            branch = f"benchmark/{EXPERIMENT}/{self.out.name}"
            published = publish_results(self.repo, self.out, branch)
            self.data["upload"] = published
            self.save()
            self.event(f"Upload: {published['status']}" +
                       (f" ({published.get('reason')})" if published["status"] != "UPLOAD_SUCCEEDED" else ""))

    def execute(self) -> int:
        if self.read_only:
            raise RuntimeError("Read-only reanalysis cannot open a device connection")
        if self.resuming and self.data["state"] in ("COMPLETED", "COMPLETED_WITH_LIMITATIONS"):
            self.event("Run is already complete; use --reanalyze to rebuild its report")
            return 0
        try:
            connection_started = time.perf_counter()
            with HardwareLock(Path("~/.spinq_live_gemini.lock")), LiveHardware(
                 self.out, host=self.config["host"], port=self.config["port"],
                 timeout_seconds=self.config["timeout_seconds"],
                 pause_seconds=self.config["pause_seconds"],
                 max_tasks=self.config["max_tasks"],
                 max_requested_rf_us=self.config["max_requested_rf_us"],
                 exclusive_use_confirmed=self.config["exclusive_use_confirmed"],
                 compact_result=True) as hardware:
                self.hw = hardware
                self.data.setdefault("connection_setup_seconds", []).append(
                    time.perf_counter() - connection_started)
                self.save()
                had_frozen_plan = (self.out / "plan.json").is_file()
                self.run_pilot()
                if self.resuming and had_frozen_plan:
                    self.validate_resume_gauge()
                self.run_blocks()
                self.data["state"] = ("COMPLETED" if self.data["rows"] and
                    all(row["status"] == "SUCCESS_VALIDATED" for row in self.data["rows"])
                    else "COMPLETED_WITH_LIMITATIONS")
                self.save()
        except KeyboardInterrupt:
            self.data["state"] = "INTERRUPTED"
            self.data["errors"].append("Operator interrupted; no next task was submitted")
            self.event("Interrupted; saved tasks can be resumed after checking the device", kind="ERROR")
        except HardwareUncertain as exc:
            self.data["state"] = "STOPPED_UNCERTAIN"
            self.data["errors"].append(f"HardwareUncertain: {redact(str(exc))}")
            self.event(f"Device state uncertain: {redact(str(exc))}; no further tasks", kind="ERROR")
        except Exception as exc:
            self.data["state"] = "PILOT_FAILED" if not (self.out / "plan.json").exists() else "FAILED"
            self.data["errors"].append({"reason": f"{type(exc).__name__}: {redact(str(exc))}",
                                        "traceback": redact(traceback.format_exc())})
            self.event(f"{type(exc).__name__}: {redact(str(exc))}", kind="ERROR")
            print(redact(traceback.format_exc()), flush=True)
        finally:
            if self.hw is not None:
                self.data["hardware_tasks_completed"] = self.hw.task_count
                self.data["hardware_results_present"] = any((self.out / "raw").glob("*.npz"))
                self.hw = None
            self.save()
            try:
                self.finish(upload=True)
            except Exception as exc:
                self.data["errors"].append(f"OutputFailure: {type(exc).__name__}: {redact(str(exc))}")
                self.save()
                self.event(f"Output error: {redact(str(exc))}", kind="ERROR")
        return 0 if self.data["state"] in ("COMPLETED", "COMPLETED_WITH_LIMITATIONS") else 2


def reanalyze_saved(repo: Path, out: Path, config: dict[str, Any],
                    preflight: dict) -> int:
    """Rebuild fits/comparisons from saved raw FIDs; no SpinQ connection."""
    session = BayesRun(repo, out, config, preflight, resume=True, read_only=True)
    if not (session.out / "plan.json").is_file():
        raise ValueError("No frozen plan; pilot cannot be reinterpreted as a comparison")
    session.run_pilot()  # disk-only branch above
    for block in range(int(config["blocks"])):
        ref = session.data["references"].get(f"block_{block}")
        if not ref or not ref.get("keys"):
            continue
        pilot, model, bounds, _ = session._requirements()
        reference_observations = []
        for key in ref["keys"]:
            row = session.data["acquisitions"][key]
            command = candidate_from_dict(row["candidate"])
            record = session._record(key)
            reference_observations.append((command,
                demodulated_features(record.fid, record.time_seconds, pilot)))
        ref["fit"] = fit_shared(reference_observations, model,
                                pilot.feature_covariance_re_im, bounds)
        for method in METHODS:
            state = session.data["methods"].get(f"block_{block}_{method}")
            if state and state.get("steps"):
                state["common_final_fit"] = fit_shared(
                    session._step_observations(state), model,
                    pilot.feature_covariance_re_im, bounds)
        session.evaluate_block(block)
    session.data["reanalysis"] = {"performed_utc": utc_now(),
                                  "source": "saved exported complex FIDs only",
                                  "hardware_commands_sent": 0}
    session.save()
    session.finish(upload=True)
    return 0
