"""Independent live protocol for Bayesian online SpinQ calibration.

This module owns its anchor, hidden command perturbation, learners and frozen
evaluator. It imports no earlier benchmark or its saved calibrations. The Mac
may run numerical preflight; only the Windows launcher opens the device.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifacts import RunArtifacts, atomic_json, jsonable, publish
from .inference import (Candidate, ComplexFeature, OnlineLearner, ReadoutAnchor,
                        diagnose_pilot_signal, estimate_anchor, extract_feature, fit_classical,
                        numeric_selfcheck, predict_feature)
from .physics import verify_gate_conventions
from .transport import (AcquisitionRequest, AcquisitionResult, PhysicalTransport, PulseSpec,
                        TaskUncertain, UnsupportedTiming)


EXPERIMENT = "01_bayes_online"
METHODS = ("A_prior_only", "B_classical", "C_fixed_bayes", "D_adaptive_bayes")
# The stopping rule is fixed before any DUT FID is acquired.  Its two physical
# checks are separate from training and from the final held-out scorer.
STOP_AFTER_TRAINING_FIDS = 6
STOP_CHECKS_PER_ARM = 2
STOP_MAX_RELATIVE_ERROR = 0.15
STOP_MAX_RELATIVE_NOISE = 0.05


class BudgetExhausted(RuntimeError):
    """The finite software task or requested RF budget has been reached."""


class PilotNoSignal(ValueError):
    """The first paid six-width round cannot support any learned calibration."""


SOURCE_FILES = ["run_01_bayes_online.cmd", "run_01_bayes_online_mac.sh",
                "bayes_online_windows.py",
                "config-bayes-online.json", "requirements-bayes-online.txt",
                "experiments/bayes_online.py", "bayes_online_core/__init__.py",
                "bayes_online_core/runner.py", "bayes_online_core/inference.py",
                "bayes_online_core/physics.py", "bayes_online_core/transport.py",
                "bayes_online_core/artifacts.py", "bayes_online_core/quantum_tasks.py",
                "bayes_online_core/SOURCES.md", "BAYES_ONLINE_README.md"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _phase_distance_deg(a: float, b: float) -> float:
    return float(np.rad2deg(np.angle(np.exp(1j * (a - b)))))


def _stop_control_quality(anchor: ReferenceAnchor, candidate: Candidate,
                          observed: ComplexFeature) -> dict[str, Any]:
    """Score one fresh stop-check FID against the paid nominal anchor model.

    The final held-out reference FIDs, injected truth, and Bell target never
    enter this decision.  An entire FID contributes one complex observation.
    """
    target = complex(predict_feature(candidate, np.array([0., 1., 0.]), anchor.readout))
    response = abs(target)
    if response < 0.2 * abs(anchor.readout.gain):
        return {"status": "WEAK_NOMINAL_RESPONSE", "passed": False,
                "relative_error": None, "relative_noise": None, "threshold": None}
    relative_error = float(abs(observed.value - target) / response)
    relative_noise = float(np.sqrt(np.trace(
        observed.covariance_ri + anchor.readout.feature_noise_ri)) / response)
    threshold = float(min(STOP_MAX_RELATIVE_ERROR,
                          max(3 * relative_noise,
                              2 * anchor.readout.model_floor_fraction)))
    passed = (observed.diagnostic == "OK" and
              relative_noise <= STOP_MAX_RELATIVE_NOISE and
              relative_error <= threshold)
    return {"status": "PASS" if passed else "FAIL", "passed": bool(passed),
            "relative_error": relative_error, "relative_noise": relative_noise,
            "threshold": threshold, "fid_diagnostic": observed.diagnostic,
            "target_source": "paid_nominal_anchor_forward_model"}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {"host", "port", "channels", "blocks", "seed",
                "sample_frequency_hz", "sample_count", "pilot_widths_us",
                "pilot_repeats", "nominal_amplitude_pct",
                "calibration_acquisitions_per_method",
                "heldout_controls_per_method", "reference_acquisitions_per_block",
                "particle_count", "design_mc_samples", "max_tasks",
                "max_requested_rf_us", "timeout_seconds", "pause_seconds",
                "software_pulse_width_envelope_us", "software_amplitude_ceiling_pct",
                "allow_unverified_multisegment"}
    if set(config) != required:
        raise ValueError("Unknown or missing configuration field; review before physical run")
    if not config["host"] or not 1 <= int(config["port"]) <= 65535:
        raise ValueError("Invalid tablet endpoint")
    if not set(config["channels"]).issubset({"H", "P"}) or not config["channels"]:
        raise ValueError("Channels must be H and/or P")
    if int(config["blocks"]) < 3:
        raise ValueError("At least three independent balanced scenarios are required")
    if config["sample_frequency_hz"] != 10000 or config["sample_count"] != 16000:
        raise ValueError("This controller supports the documented 10 kHz, 16000-point baseline")
    if config["pilot_repeats"] < 3 or len(set(config["pilot_widths_us"])) < 5:
        raise ValueError("Diverse pilot widths and three repeated anchors required")
    if config["heldout_controls_per_method"] != 2:
        raise ValueError("Exactly two independent controls per method implemented")
    if config["calibration_acquisitions_per_method"] < 10:
        raise ValueError("At least ten maximum training acquisitions per learned arm required")
    if not 512 <= config["particle_count"] <= 2048:
        raise ValueError("Particle count must be inside the specified CPU range")
    if config["max_tasks"] < 1 or config["max_requested_rf_us"] <= 0:
        raise ValueError("Finite software task/RF budget required")
    if config["allow_unverified_multisegment"]:
        raise ValueError("Unverified segment timing cannot be enabled")
    low, high = config["software_pulse_width_envelope_us"]
    if not 0 < low <= min(config["pilot_widths_us"]) or high < max(config["pilot_widths_us"]):
        raise ValueError("Pilot widths outside declared completed single-pulse envelope")
    if not 0 < config["nominal_amplitude_pct"] <= config["software_amplitude_ceiling_pct"] <= 100:
        raise ValueError("Invalid RF amplitude/headroom budget")
    if config["timeout_seconds"] < 30 or config["pause_seconds"] < 0:
        raise ValueError("Invalid timeout/pause")
    return config


def quick_task_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Count every planned physical FID in the bounded H-only B/D protocol."""
    config = validate_config(config)
    if config["channels"] != ["H"] or config["blocks"] != 3 or \
            config["calibration_acquisitions_per_method"] != 10:
        raise ValueError("Quick protocol requires H, three blocks and ten B/D acquisitions")
    counts = {"shared_anchor": len(config["pilot_widths_us"])*config["pilot_repeats"],
              "fresh_nominal_references": 2*config["blocks"],
              "B_and_D_training": 2*config["calibration_acquisitions_per_method"]*config["blocks"],
              "sequential_stop_checks": 2*STOP_CHECKS_PER_ARM*config["blocks"],
              "heldout_controls": 2*2*config["blocks"],
              "hidden_perturbation_probes": config["blocks"]-1,
              "pre_between_post_drift_checks": 3*config["blocks"]}
    total = sum(counts.values())
    if total > config["max_tasks"]:
        raise ValueError("Quick plan exceeds configured physical-task budget")
    return {"profile": "quick", "channel": "H", "methods": ["B_classical", "D_adaptive_bayes"],
            "scenarios": ["neutral", "positive_command_error", "negative_command_error"],
            "counts": counts, "planned_physical_tasks": total,
            "max_physical_tasks": config["max_tasks"],
            "comparison_scope": "three paired exploratory H-only blocks; no PPS/Bell or universal advantage claim"}


def numeric_preflight() -> dict:
    """No network, tablet, RF task or old benchmark is touched."""
    checks = {"spinqlablink": importlib.metadata.version("spinqlablink"),
              "numpy": importlib.metadata.version("numpy"),
              "scipy": importlib.metadata.version("scipy")}
    if checks["spinqlablink"] != "1.0.2":
        raise RuntimeError("Only the previously functional SpinQLabLink 1.0.2 is supported")
    checks["physics_max_error"] = max(verify_gate_conventions().values())
    checks["inference"] = numeric_selfcheck()
    checks["python"] = sys.version.split()[0]
    checks["platform"] = platform.platform()
    checks["cpu_count"] = os.cpu_count()
    checks["checked_utc"] = utc_now()
    checks["hardware_tested_here"] = False
    return checks


@dataclass(frozen=True)
class ReferenceAnchor:
    channel: str
    readout: ReadoutAnchor
    pilot_keys: tuple[str, ...]
    nominal_amplitude_pct: float
    t90_us: float
    full_cycle_s: float
    pilot_repeat_frequency_scatter_hz: float

    def public(self) -> dict:
        return {"channel": self.channel, "readout": jsonable(asdict(self.readout)),
                "pilot_keys": self.pilot_keys,
                "nominal_amplitude_pct": self.nominal_amplitude_pct,
                "t90_us": self.t90_us, "full_cycle_s": self.full_cycle_s,
                "pilot_repeat_frequency_scatter_hz": self.pilot_repeat_frequency_scatter_hz,
                "status": "PAID_NOMINAL_ANCHOR_NOT_PERTURBED_TRUTH"}


@dataclass(frozen=True)
class HiddenCommandError:
    df_hz: float
    rf_gain: float
    phase_deg: float


class PerturbationController:
    """Applies evaluator-owned reversible errors below the learner interface."""

    def __init__(self, errors: dict[str, HiddenCommandError],
                 amplitude_ceiling_pct: float):
        self._errors = errors.copy()
        self._ceiling = amplitude_ceiling_pct

    def apply(self, request: AcquisitionRequest) -> AcquisitionRequest:
        pulses = []
        for pulse in request.pulses:
            if pulse.role != "dut":
                pulses.append(pulse)
                continue
            channel = "H" if pulse.path == 0 else "P"
            error = self._errors.get(channel, HiddenCommandError(0., 1., 0.))
            amplitude = pulse.amplitude_pct * error.rf_gain
            if not 0 <= amplitude <= self._ceiling:
                raise ValueError("Hidden RF transformation exceeds validated command headroom")
            pulses.append(replace(pulse, amplitude_pct=amplitude,
                                  phase_deg=pulse.phase_deg + error.phase_deg,
                                  detuning_hz=pulse.detuning_hz + error.df_hz))
        return replace(request, pulses=tuple(pulses))

    def truth_for_evaluator(self, channel: str) -> HiddenCommandError:
        return self._errors.get(channel, HiddenCommandError(0., 1., 0.))

    def audit_only(self) -> dict:
        return {channel: asdict(error) for channel, error in self._errors.items()}


class Learner:
    """Fresh method state per scenario; never receives the hidden controller."""

    def __init__(self, method: str, anchor: ReferenceAnchor, bounds: np.ndarray,
                 particles: int, seed: int):
        if method not in METHODS:
            raise ValueError("Unknown method")
        self.method = method
        self.anchor = anchor
        self.bounds = np.asarray(bounds, float).copy()
        self.particles = particles
        self.seed = seed
        self.candidates: list[Candidate] = []
        self.features: list[ComplexFeature] = []
        self.smc = (OnlineLearner(self.bounds, anchor.readout, particles, seed=seed)
                    if method in ("C_fixed_bayes", "D_adaptive_bayes") else None)
        self.history: list[dict] = []

    def observe(self, candidate: Candidate, feature: ComplexFeature) -> dict:
        self.candidates.append(candidate)
        self.features.append(feature)
        if self.smc is not None:
            update = self.smc.update(candidate, feature)
            row = {"method": self.method, "acquisitions": len(self.features),
                   "update": jsonable(update), "posterior": self.smc.summary()}
        else:
            row = {"method": self.method, "acquisitions": len(self.features),
                   "feature": jsonable(asdict(feature))}
        self.history.append(row)
        return row

    def estimate(self) -> dict:
        if self.method == "A_prior_only":
            return {"status": "PRIOR_ONLY", "mean": [0., 1., 0.],
                    "interval_95": None, "training_acquisitions": 0}
        if self.method == "B_classical":
            fit = fit_classical(self.candidates, self.features,
                                self.anchor.readout, self.bounds,
                                starts=24, seed=self.seed)
            return {"status": fit.status, "mean": fit.parameters.tolist(),
                    "covariance": fit.covariance.tolist(),
                    "nll": fit.nll, "starts": fit.starts,
                    "condition_number": fit.condition_number,
                    "training_acquisitions": len(self.features)}
        assert self.smc is not None
        summary = self.smc.summary()
        summary["training_acquisitions"] = len(self.features)
        return summary


class FrozenEvaluator:
    """Scores only after a learner is frozen; owns targets and hidden truth."""

    def __init__(self, controller: PerturbationController,
                 references: dict[str, ComplexFeature], anchor: ReferenceAnchor,
                 probe: HiddenCommandError):
        self._controller = controller
        self._references = references.copy()
        self.anchor = anchor
        self.probe = probe

    def evaluate(self, channel: str, estimate: dict,
                 controls: dict[str, ComplexFeature]) -> dict:
        truth = self._controller.truth_for_evaluator(channel)
        if len(controls) != len(self._references):
            return {"status": "INCOMPLETE_DATA", "reason": "missing independent physical control"}
        residuals = []
        limits = []
        for name, observed in controls.items():
            reference = self._references[name]
            response = max(abs(reference.value), 1e-9)
            rms = abs(observed.value - reference.value) / response
            joint_noise = math.sqrt(float(np.trace(reference.covariance_ri +
                                                   observed.covariance_ri))) / response
            limit = max(3 * joint_noise, 2 * self.anchor.readout.model_floor_fraction)
            residuals.append(rms)
            limits.append(limit)
        mean = np.asarray(estimate.get("mean") or [0., 1., 0.], float)
        if mean.shape != (3,) or not np.all(np.isfinite(mean)):
            return {"status": "NONIDENTIFIABLE", "reason": "no finite calibration estimate"}
        error = [abs(mean[0] - truth.df_hz), abs(mean[1] - truth.rf_gain),
                 abs(_phase_distance_deg(mean[2], math.radians(truth.phase_deg)))]
        tolerance = [max(5., abs(self.probe.df_hz) / 2),
                     max(.01, abs(self.probe.rf_gain - 1.) / 2),
                     max(2., abs(self.probe.phase_deg) / 2)]
        response_ok = all(value <= bound for value, bound in zip(residuals, limits))
        parameter_ok = all(value <= bound for value, bound in zip(error, tolerance))
        source_status = estimate.get("status")
        supported = source_status in ("FIT_OK", "LEARNING", "PRIOR_ONLY")
        status = ("TARGET_REACHED" if supported and response_ok and parameter_ok else
                  "NONIDENTIFIABLE" if source_status in ("NONIDENTIFIABLE", "MODEL_CHECK_REQUIRED")
                  else "BUDGET_EXHAUSTED")
        return {"status": status, "parameter_error": {
                    "df_hz": error[0], "rf_scale": error[1], "phase_deg": error[2]},
                "parameter_tolerance": {"df_hz": tolerance[0],
                    "rf_scale": tolerance[1], "phase_deg": tolerance[2]},
                "heldout_relative_complex_errors": residuals,
                "heldout_thresholds": limits,
                "heldout_complex_error": float(np.mean(residuals)),
                "reference_scope": "separate nominal low-level FID controls; not Bell fidelity",
                "error_scope": "known injected command offset only; natural resonance unknown",
                "learner_status": source_status}


class OnlineRun:
    def __init__(self, repo: Path, out: Path, config: dict,
                 task: str, exclusive_use_confirmed: bool = False,
                 resume: bool = False, quick: bool = False):
        self.repo = Path(repo).resolve()
        self.artifacts = RunArtifacts(out)
        self.config = validate_config(config)
        self.task = task
        self.quick = bool(quick)
        if self.quick:
            quick_task_plan(self.config)
            if task not in ("rabi", "frequency"):
                raise ValueError("Quick protocol is only for frequency/Rabi calibration")
        self._started_monotonic = time.monotonic()
        self._quick_wall_limit_s = 3000 if self.quick else None
        self.resume = resume
        self.exclusive_use_confirmed = exclusive_use_confirmed
        self.transport: PhysicalTransport | None = None
        self.task_count = 0
        self.rf_us = 0.
        saved = self.artifacts.path / "results.json"
        if saved.exists():
            if not resume:
                raise ValueError("Run directory exists; use --resume")
            self.state = json.loads(saved.read_text(encoding="utf-8"))
            if (self.state.get("config") != config or
                    self.state.get("task_requested") != task or
                    self.state.get("profile", "full") != ("quick" if self.quick else "full")):
                raise ValueError("Resume config/task differs from original run")
            if self.state.get("status") == "STOPPED_UNCERTAIN":
                raise TaskUncertain("A prior task may still run; manual reconciliation required")
            orphans = [path.name for path in (self.artifacts.path / "data").glob("*.npz")
                       if path.stem not in self.state.get("acquisitions", {})]
            if orphans:
                raise TaskUncertain("Saved FID lacks a completed journal row; inspect "
                                    "task state and data before any resubmission: " + ", ".join(orphans[:4]))
            self.task_count = int(self.state.get("task_count", 0))
            self.rf_us = float(self.state.get("requested_rf_us", 0.))
        else:
            self.state = {"experiment": EXPERIMENT, "run_id": out.name,
                          "started_utc": utc_now(), "status": "RUNNING",
                          "profile": "quick" if self.quick else "full",
                          "task_requested": task, "config": config,
                          "acquisitions": {}, "anchors": {}, "scenarios": {},
                          "methods": {}, "capabilities": {}, "comparison": [],
                          "errors": [], "task_count": 0, "requested_rf_us": 0.,
                          "learner_input_provenance": "exported complex FID only; vendor FFT excluded"}
            self.save()
        atomic_json(self.artifacts.path / "profiles" / "effective_config.json",
                    {"profile": self.state.get("profile", "full"),
                     "task": task, "config": config,
                     "quick_plan": quick_task_plan(config) if self.quick else None})

    def log(self, message: str, *, kind: str = "INFO") -> None:
        print(f"01 ONLINE {kind}: {message}", flush=True)

    def save(self) -> None:
        self.state["task_count"] = self.task_count
        self.state["requested_rf_us"] = self.rf_us
        self.artifacts.write_state(self.state)

    def _request(self, key: str, candidate: Candidate,
                 *, initialize_state: bool = True) -> AcquisitionRequest:
        path = 0 if candidate.channel == "H" else 1
        return AcquisitionRequest(key=key, pulses=(PulseSpec(path=path, start_us=0,
                    width_us=float(round(candidate.width_us)),
                    amplitude_pct=float(candidate.amplitude_pct),
                    phase_deg=float(candidate.phase_deg),
                    detuning_hz=float(candidate.detuning_hz), role="dut"),),
                    sample_path=path,
                    sample_count=self.config["sample_count"],
                    sample_frequency_hz=self.config["sample_frequency_hz"],
                    initialize_state=initialize_state)

    def _check_budget(self, request: AcquisitionRequest) -> float:
        if self.quick and time.monotonic() - self._started_monotonic > self._quick_wall_limit_s:
            raise BudgetExhausted("Quick protocol reached 50-minute active runtime cap")
        low, high = self.config["software_pulse_width_envelope_us"]
        for pulse in request.pulses:
            if not 0 < pulse.width_us or (pulse.amplitude_pct > 0 and
                    not low <= pulse.width_us <= high):
                raise ValueError("Pulse outside observed software envelope")
            if not 0 <= pulse.amplitude_pct <= self.config["software_amplitude_ceiling_pct"]:
                raise ValueError("Pulse outside observed amplitude headroom")
        requested = sum(pulse.width_us for pulse in request.pulses
                        if pulse.amplitude_pct > 0)
        if self.task_count + 1 > self.config["max_tasks"] or \
                self.rf_us + requested > self.config["max_requested_rf_us"]:
            raise BudgetExhausted("Finite software task/RF budget exhausted")
        return requested

    def acquire(self, request: AcquisitionRequest, *, controller: PerturbationController | None,
                role: str, channel: str | None = None, block: int | None = None,
                method: str | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        key = request.key
        if key in self.state["acquisitions"]:
            old = self.state["acquisitions"][key]
            if old.get("logical_request") != jsonable(asdict(request)):
                raise ValueError("Resume logical request differs from original task")
            t, fid = self.artifacts.load_fid(key)
            return t, fid, float(old["elapsed_s"])
        if self.transport is None:
            raise RuntimeError("No physical transport open")
        actual = controller.apply(request) if controller else request
        rf = self._check_budget(actual)
        result = self.transport.acquire(actual)
        try:
            self.artifacts.save_acquisition(key, result, asdict(request), role=role,
                                            channel=channel, block=block, method=method)
        except Exception as exc:
            raise TaskUncertain(f"Completed task {result.task_id} could not be archived: {exc}",
                                task_id=result.task_id) from exc
        self.task_count += 1
        self.rf_us += rf
        row = {"key": key, "task_id": result.task_id,
               "role": role, "channel": channel, "block": block, "method": method,
               "elapsed_s": result.elapsed_s, "requested_rf_us": rf,
               "logical_request": jsonable(asdict(request)),
               "data_file": f"data/{key}.npz", "completed_utc": utc_now()}
        self.state["acquisitions"][key] = row
        self.save()
        self.log(f"FID {key}: {len(result.fid_complex)} points, {result.elapsed_s:.1f} s")
        if self.config["pause_seconds"]:
            time.sleep(self.config["pause_seconds"])
        return result.time_s, result.fid_complex, result.elapsed_s

    def _feature(self, key: str, anchor: ReferenceAnchor) -> ComplexFeature:
        axis, fid = self.artifacts.load_fid(key)
        return extract_feature(fid, axis, anchor.readout, source_id=key)

    def _pilot(self, block: int, channel: str) -> ReferenceAnchor:
        widths = self.config["pilot_widths_us"]
        repeats = self.config["pilot_repeats"]
        amplitude = float(self.config["nominal_amplitude_pct"])
        axes, fids, actual_widths, keys, durations = [], [], [], [], []
        self.log(f"Block {block}: measuring {channel} nominal anchor ({len(widths)*repeats} independent FIDs)")
        for repeat in range(repeats):
            for width in widths:
                key = f"b{block}_{channel}_anchor_w{width}_r{repeat}"
                candidate = Candidate(channel, float(width), amplitude, 0., 0., 12.)
                axis, fid, elapsed = self.acquire(self._request(key, candidate),
                    controller=None, role="nominal_anchor", channel=channel, block=block)
                axes.append(axis)
                fids.append(fid)
                actual_widths.append(width)
                keys.append(key)
                durations.append(elapsed)
            if repeat == 0:
                base = axes[0]
                if any(axis.shape != base.shape or
                       not np.allclose(axis, base, rtol=0, atol=1e-9)
                       for axis in axes[1:]):
                    raise ValueError("PILOT_INCONCLUSIVE: first-round exported FID axes differ")
                triage = diagnose_pilot_signal(np.asarray(fids), base)
                atomic_json(self.artifacts.path / "profiles" /
                            f"b{block}_{channel}_pilot_triage.json", triage)
                self.state["capabilities"][f"b{block}_{channel}_pilot_triage"] = triage
                self.save()
                self.log(f"Pilot {channel} first-round signal triage: "
                         f"{triage['status']}, detectable="
                         f"{triage['evidence']['detectable_fids']}/"
                         f"{triage['evidence']['pilot_fids']}")
                if triage["status"] != "PROCEED":
                    raise PilotNoSignal("PILOT_INCONCLUSIVE: no detectable coherent FID "
                                        "in the first six distinct pulse widths")
        base = axes[0]
        if any(axis.shape != base.shape or not np.allclose(axis, base, rtol=0, atol=1e-9)
               for axis in axes[1:]):
            raise ValueError("PILOT_INCONCLUSIVE: repeated exported FID axes differ")
        readout = estimate_anchor(np.asarray(fids), base, np.asarray(actual_widths), amplitude)
        axis_statuses = []
        for key in keys:
            metadata = json.loads((self.artifacts.path / "data" / f"{key}.json")
                                  .read_text(encoding="utf-8"))
            axis_statuses.append(metadata.get("timing_evidence", {}).get("axis_status"))
        readout.diagnostics["axis_statuses"] = sorted({str(status) for status in axis_statuses})
        if any(status != "EXPORT_MATCHES_REQUESTED_RATE" for status in axis_statuses):
            readout.diagnostics["spectral_model_status"] = "MODEL_CHECK_REQUIRED"
            readout.diagnostics["frequency_axis_status"] = "REQUESTED_RATE_ONLY_UNVERIFIED"
        else:
            readout.diagnostics["frequency_axis_status"] = "EXPORT_MATCHES_REQUESTED_RATE"
        t90 = 1e6 / (4 * readout.rabi_hz_per_pct * amplitude)
        if not np.isfinite(t90) or not 20 <= t90 <= 200:
            raise ValueError(f"PILOT_INCONCLUSIVE: t90={t90:.2f} us outside completed pulse envelope")
        # Between-acquisition frequency scatter is diagnostic only.  The
        # receiver frequency is not renamed as physical transmitter detuning.
        active = max(32, min(len(base), int(readout.coherent_window_s /
                                  np.median(np.diff(base)))))
        frequencies = []
        for fid in fids:
            phase = np.unwrap(np.angle(fid[:active]))
            frequencies.append(float(np.polyfit(base[:active], phase, 1)[0] / (2*math.pi)))
        scatter = float(statistics.pstdev(frequencies))
        anchor = ReferenceAnchor(channel, readout, tuple(keys), amplitude,
                                 float(t90), float(statistics.median(durations)), scatter)
        self.state["anchors"][f"b{block}_{channel}"] = anchor.public()
        self.state["anchors"][f"b{block}_{channel}"]["paid_acquisitions"] = len(keys)
        self.save()
        self.log(f"Pilot {channel}: t90={t90:.2f} us, coherent={readout.coherent_window_s*1e3:.2f} ms, "
                 f"Rabi residual={readout.diagnostics['rabi_relative_residual']:.3f}, "
                 f"spectral={readout.diagnostics['spectral_model_status']}, "
                 f"axis={readout.diagnostics['frequency_axis_status']}")
        return anchor

    def _heldout_candidates(self, anchor: ReferenceAnchor) -> dict[str, Candidate]:
        amplitude = anchor.nominal_amplitude_pct
        low, high = self.config["software_pulse_width_envelope_us"]
        widths = (round(np.clip(anchor.t90_us * 1.13, low, high)),
                  round(np.clip(anchor.t90_us * 2.27, low, high)))
        return {"phase45": Candidate(anchor.channel, widths[0], amplitude, 45., 0.,
                                     anchor.full_cycle_s),
                "phase135": Candidate(anchor.channel, widths[1], amplitude, 135., 0.,
                                      anchor.full_cycle_s)}

    def _nominal_references(self, block: int,
                            anchors: dict[str, ReferenceAnchor]) -> dict[str, dict[str, ComplexFeature]]:
        output = {}
        for channel, anchor in anchors.items():
            output[channel] = {}
            for name, candidate in self._heldout_candidates(anchor).items():
                key = f"b{block}_{channel}_reference_{name}"
                self.acquire(self._request(key, candidate), controller=None,
                             role="evaluator_reference", channel=channel, block=block)
                output[channel][name] = self._feature(key, anchor)
        return output

    def _scenario(self, block: int, anchors: dict[str, ReferenceAnchor]) -> dict[str, HiddenCommandError]:
        # Deterministic paired neutral/+/- command perturbations.  Sizes are
        # derived from the newly measured nutation scale and then bounded by
        # the SDK plus completed command envelope; they are not hardware limits.
        sign = (0, 1, -1)[block % 3]
        errors = {}
        for channel, anchor in anchors.items():
            df = min(600., max(100., .11 * anchor.readout.rabi_hz_per_pct *
                               anchor.nominal_amplitude_pct))
            errors[channel] = HiddenCommandError(float(sign * round(df)),
                                                  1. + sign * .07,
                                                  float(sign * 12))
        atomic_json(self.artifacts.path / "evaluator_only" / f"b{block}_scenario.json",
                    {"kind": "reversible_command_only", "errors":
                     {channel: asdict(error) for channel, error in errors.items()},
                     "applies_to_role": "dut", "demodulation_unchanged": True,
                     "naturally_occurring_resonance_truth": "UNKNOWN"})
        self.state["scenarios"][str(block)] = {"label":
            ("neutral", "positive_command_error", "negative_command_error")[block % 3],
            "truth_file": f"evaluator_only/b{block}_scenario.json"}
        self.save()
        return errors

    def _probe(self, block: int, channel: str, anchor: ReferenceAnchor,
               controller: PerturbationController, reference: dict[str, ComplexFeature]) -> dict:
        if block % 3 == 0:
            return {"status": "NEUTRAL", "relative_change": 0.}
        logical = self._heldout_candidates(anchor)["phase45"]
        key = f"b{block}_{channel}_hidden_probe"
        self.acquire(self._request(key, logical), controller=controller,
                     role="perturbation_probe_not_learner_input", channel=channel, block=block)
        feature = self._feature(key, anchor)
        baseline = reference["phase45"]
        delta = abs(feature.value - baseline.value)
        noise = math.sqrt(float(np.trace(feature.covariance_ri + baseline.covariance_ri)))
        ratio = float(delta / max(noise, 1e-12))
        report = {"status": "DISTINGUISHABLE" if ratio >= 2 else "WEAK_OR_UNRESOLVED",
                  "difference_over_joint_noise": ratio,
                  "relative_change": float(delta / max(abs(baseline.value), 1e-12)),
                  "fid": key}
        atomic_json(self.artifacts.path / "evaluator_only" /
                    f"b{block}_{channel}_probe.json", report)
        self.log(f"Probe {channel}: {report['status']}, difference/noise={ratio:.2f}")
        return report

    def _bounds(self, anchor: ReferenceAnchor) -> np.ndarray:
        df = min(1200., max(250., .22 * anchor.readout.rabi_hz_per_pct *
                              anchor.nominal_amplitude_pct))
        return np.asarray([[-df, df], [.8, 1.2],
                           [-math.radians(30), math.radians(30)]], float)

    def _candidate_pool(self, anchor: ReferenceAnchor) -> list[Candidate]:
        low, high = self.config["software_pulse_width_envelope_us"]
        widths = sorted({int(round(np.clip(anchor.t90_us * scale, low, high)))
                         for scale in (.55, .8, 1., 1.4, 1.9, 2.5, 3.2, 4.1)})
        detunings = (-600., 0., 600.)
        phases = (0., 90., 180., 270.)
        return [Candidate(anchor.channel, float(width), anchor.nominal_amplitude_pct,
                          phase, detuning, anchor.full_cycle_s)
                for width in widths for phase in phases for detuning in detunings]

    def _fixed_plan(self, pool: list[Candidate], total: int) -> list[Candidate]:
        # Distinct widths, all quadrature phases, and both detuning signs.
        widths = sorted({candidate.width_us for candidate in pool})
        phases = (0., 90., 180., 270.)
        detunings = (-600., 0., 600.)
        output = []
        for i in range(total):
            width = widths[(i * 3 + i // max(1, len(widths))) % len(widths)]
            phase = phases[i % len(phases)]
            detuning = detunings[(i // len(phases)) % len(detunings)]
            output.append(next(candidate for candidate in pool
                               if candidate.width_us == width and
                               candidate.phase_deg == phase and
                               candidate.detuning_hz == detuning))
        return output

    def _classical_next(self, learner: Learner, pool: list[Candidate],
                        fixed: list[Candidate], index: int) -> Candidate:
        if index < max(6, len(fixed)//2) or len(learner.features) < 6:
            return fixed[index]
        # A local D-optimal fine scan around the multistart complex fit;
        # the physical candidate remains in the same bounded command family.
        fit = learner.estimate()
        if fit.get("status") not in ("FIT_OK", "MODEL_CHECK_REQUIRED"):
            return fixed[index]
        theta = np.asarray(fit["mean"], float)
        bounds = learner.bounds
        steps = np.maximum((bounds[:, 1]-bounds[:, 0]) * 1e-3, 1e-6)
        seen = set(learner.candidates)
        scores = []
        for candidate in pool:
            deriv = np.asarray([(predict_feature(candidate, theta + np.eye(3)[j]*steps[j],
                                                 learner.anchor.readout) -
                                 predict_feature(candidate, theta - np.eye(3)[j]*steps[j],
                                                 learner.anchor.readout)) / (2*steps[j])
                                for j in range(3)])
            score = np.linalg.norm(deriv * (bounds[:, 1]-bounds[:, 0])) / candidate.full_cost_s
            scores.append(float(score / math.sqrt(1 + 4*(candidate in seen))))
        return pool[int(np.argmax(scores))]

    def _compensated(self, logical: Candidate, estimate: dict) -> Candidate:
        mean = np.asarray(estimate.get("mean", [0., 1., 0.]), float)
        if mean.shape != (3,) or not np.all(np.isfinite(mean)) or mean[1] <= 0:
            raise ValueError("No finite physical control correction")
        return replace(logical,
                       amplitude_pct=float(logical.amplitude_pct / mean[1]),
                       phase_deg=float(logical.phase_deg - math.degrees(mean[2])),
                       detuning_hz=float(round(logical.detuning_hz - mean[0])))

    def _stopping_candidates(self, anchor: ReferenceAnchor) -> tuple[tuple[str, Candidate], ...]:
        """Two prespecified checks disjoint from training and final controls."""
        low, high = self.config["software_pulse_width_envelope_us"]
        widths = (round(np.clip(1.11 * anchor.t90_us, low, high)),
                  round(np.clip(1.57 * anchor.t90_us, low, high)))
        return (("phase30", Candidate(anchor.channel, widths[0],
                    anchor.nominal_amplitude_pct, 30., -300., anchor.full_cycle_s)),
                ("phase150", Candidate(anchor.channel, widths[1],
                    anchor.nominal_amplitude_pct, 150., 300., anchor.full_cycle_s)))

    def _stopping_checkpoint(self, label: str, block: int, channel: str,
                             method: str, anchor: ReferenceAnchor,
                             controller: PerturbationController,
                             estimate: dict) -> dict[str, Any]:
        """Validate a provisional fit without updating it or using final controls."""
        allowed = "FIT_OK" if method == "B_classical" else "LEARNING"
        if estimate.get("status") != allowed:
            return {"status": "NOT_IDENTIFIABLE", "passed": False,
                    "estimate_status": estimate.get("status"), "measured_keys": [],
                    "checks": [], "training_fids_at_checkpoint": STOP_AFTER_TRAINING_FIDS}
        candidates = self._stopping_candidates(anchor)
        if len(candidates) != STOP_CHECKS_PER_ARM:
            raise AssertionError("Prespecified stop-check count changed")
        # All candidate targets must have usable anchor contrast before
        # sending either physical request.  No evaluator data is consulted.
        if any(abs(complex(predict_feature(candidate, np.array([0., 1., 0.]),
                                           anchor.readout))) < 0.2 * abs(anchor.readout.gain)
               for _, candidate in candidates):
            return {"status": "WEAK_NOMINAL_RESPONSE", "passed": False,
                    "estimate_status": estimate.get("status"), "measured_keys": [],
                    "checks": [], "training_fids_at_checkpoint": STOP_AFTER_TRAINING_FIDS}
        checks = []
        keys = []
        for name, logical in candidates:
            commanded = self._compensated(logical, estimate)
            key = f"{label}_stop_{name}"
            self.acquire(self._request(key, commanded), controller=controller,
                         role="sequential_stop_check", channel=channel,
                         block=block, method=method)
            check = _stop_control_quality(anchor, logical, self._feature(key, anchor))
            check["name"] = name
            check["key"] = key
            checks.append(check)
            keys.append(key)
        passed = all(check["passed"] for check in checks)
        return {"status": "PASS" if passed else "FAIL", "passed": passed,
                "estimate_status": estimate.get("status"), "measured_keys": keys,
                "checks": checks,
                "training_fids_at_checkpoint": STOP_AFTER_TRAINING_FIDS,
                "stopping_rule": "both disjoint physical controls pass identical "
                                 "relative response and noise limits"}

    def _measure_arm(self, block: int, channel: str, method: str,
                     anchor: ReferenceAnchor, controller: PerturbationController,
                     references: dict[str, ComplexFeature], probe: dict,
                     order_index: int) -> dict:
        label = f"b{block}_{channel}_{method}"
        if label in self.state["methods"]:
            previous = next((row for row in self.state["comparison"] if
                             row.get("block") == block and row.get("channel") == channel and
                             row.get("method") == method), None)
            if previous is not None:
                self.log(f"{label}: completed result reused on resume")
                return previous
        # A previous interrupted attempt may have left a status-only row.
        # Keep its error in the event log, but replace the row on a retry.
        self.state["comparison"] = [row for row in self.state["comparison"]
            if not (row.get("block") == block and row.get("channel") == channel
                    and row.get("method") == method and row.get("status") in
                    ("BUDGET_EXHAUSTED", "INCOMPLETE_DATA"))]
        started = time.monotonic()
        pool = self._candidate_pool(anchor)
        fixed = self._fixed_plan(pool, self.config["calibration_acquisitions_per_method"])
        bounds = self._bounds(anchor)
        learner = Learner(method, anchor, bounds, self.config["particle_count"],
                          self.config["seed"] + 1009*block + 31*(channel == "P") +
                          17*METHODS.index(method))
        self.log(f"Block {block} {channel} {method}: training budget "
                 f"{0 if method == 'A_prior_only' else len(fixed)} acquisitions")
        training_keys = []
        checkpoint = {"status": "NOT_APPLICABLE", "passed": False,
                      "measured_keys": [], "checks": []}
        estimated: dict | None = None
        for index in range(0 if method == "A_prior_only" else len(fixed)):
            if method == "D_adaptive_bayes":
                assert learner.smc is not None
                candidate = learner.smc.choose(pool, mode="adaptive")
            elif method == "C_fixed_bayes":
                candidate = fixed[index]
            else:
                candidate = self._classical_next(learner, pool, fixed, index)
            key = f"{label}_train_{index:02d}"
            self.acquire(self._request(key, candidate), controller=controller,
                         role="training", channel=channel, block=block, method=method)
            learner.observe(candidate, self._feature(key, anchor))
            training_keys.append(key)
            if learner.smc and (index+1) in (5, 10, 20, 40, 80):
                point = learner.smc.summary()
                self.log(f"{label} checkpoint {index+1}: status={point['status']}, "
                         f"ESS={point['ess']:.0f}, df={point['mean'][0]:.1f} Hz, "
                         f"RF={point['mean'][1]:.3f}, phase={math.degrees(point['mean'][2]):.1f} deg")
            if index + 1 == STOP_AFTER_TRAINING_FIDS:
                provisional = learner.estimate()
                provisional_file = f"models/{label}_after_{STOP_AFTER_TRAINING_FIDS}_training.json"
                atomic_json(self.artifacts.path / provisional_file, {
                    "estimate": provisional, "method": method,
                    "training_keys": training_keys,
                    "stopping_candidate_plan": [asdict(candidate) for _, candidate
                                                in self._stopping_candidates(anchor)],
                    "frozen_before_stopping_checks": True,
                })
                checkpoint = self._stopping_checkpoint(label, block, channel, method,
                                                       anchor, controller, provisional)
                checkpoint["provisional_estimate_file"] = provisional_file
                self.log(f"{label} sequential stop after {index+1} training FIDs: "
                         f"{checkpoint['status']}; separate check FIDs="
                         f"{len(checkpoint['measured_keys'])}")
                if checkpoint["passed"]:
                    estimated = provisional
                    break
        if estimated is None:
            estimated = learner.estimate()
        checkpoint_keys = list(checkpoint["measured_keys"])
        frozen = jsonable(estimated)
        frozen["method"] = method
        frozen["training_keys"] = training_keys
        frozen["sequential_stopping"] = checkpoint
        frozen["stopping_decision"] = ("EARLY_STOP" if checkpoint["passed"] else
                                       "MAX_TRAINING_BUDGET" if method != "A_prior_only" else
                                       "PRIOR_ONLY")
        frozen["prior_bounds"] = bounds.tolist()
        frozen["frozen_before_controls"] = True
        atomic_json(self.artifacts.path / "models" / f"{label}_frozen.json", frozen)
        # Heldout control phases are absent from all training plans.  A single
        # control FID is one observation, not 16000 Bernoulli trials.
        heldout = self._heldout_candidates(anchor)
        control_features = {}
        control_keys = []
        validation_started = time.monotonic()
        for name, logical in heldout.items():
            commanded = (logical if method == "A_prior_only" else
                         self._compensated(logical, estimated))
            key = f"{label}_control_{name}"
            _, _, elapsed = self.acquire(self._request(key, commanded),
                                         controller=controller,
                                         role="heldout_control", channel=channel,
                                         block=block, method=method)
            control_features[name] = self._feature(key, anchor)
            control_keys.append(key)
        evaluator = FrozenEvaluator(controller, references, anchor,
                                    controller.truth_for_evaluator(channel))
        scored = evaluator.evaluate(channel, estimated, control_features)
        if probe["status"] == "WEAK_OR_UNRESOLVED":
            scored["status"] = "PILOT_INCONCLUSIVE"
            scored["reason"] = "Command perturbation below independent probe resolution"
        if anchor.readout.diagnostics["spectral_model_status"] == "MODEL_CHECK_REQUIRED":
            scored["status"] = "NONIDENTIFIABLE"
            scored["reason"] = "Single-branch FID model misses observed coherent multiplet"
        total_s = time.monotonic() - started
        training_s = validation_started - started
        validation_s = total_s - training_s
        reference_keys = [f"b{block}_{channel}_reference_{name}" for name in references]
        shared_keys = list(anchor.pilot_keys) + reference_keys
        shared_seconds = sum(float(self.state["acquisitions"][key]["elapsed_s"])
                             for key in shared_keys)
        arm_acquisitions = len(training_keys) + len(checkpoint_keys) + len(control_keys)
        row = {"block": block, "scenario": self.state["scenarios"][str(block)]["label"],
               "task": "frequency_rabi", "channel": channel, "method": method,
               "method_order": order_index, "status": scored["status"],
               "design_acquisitions": len(training_keys),
               "sequential_stop_check_acquisitions": len(checkpoint_keys),
               "control_acquisitions": len(control_keys),
               "total_acquisitions": arm_acquisitions,
               "shared_anchor_acquisitions": len(anchor.pilot_keys),
               "shared_evaluator_acquisitions": len(reference_keys),
               "cold_start_acquisitions": arm_acquisitions + len(shared_keys),
               "cold_start_seconds": float(total_s + shared_seconds),
               "training_seconds": float(training_s),
               "validation_seconds": float(validation_s),
               "end_to_end_seconds": float(total_s),
               "frequency_error_hz": scored.get("parameter_error", {}).get("df_hz"),
               "rf_scale_error": scored.get("parameter_error", {}).get("rf_scale"),
               "phase_error_deg": scored.get("parameter_error", {}).get("phase_deg"),
               "heldout_complex_error": scored.get("heldout_complex_error"),
               "reason": scored.get("reason", ""),
               "scoring": scored, "estimate_file": f"models/{label}_frozen.json",
               "stopping_decision": frozen["stopping_decision"],
               "stopping_checkpoint": checkpoint,
               "heldout_keys": control_keys,
               "vendor_comparison": "NOT_MATCHED_PERTURBATION"}
        self.state["methods"][label] = {"estimate": frozen,
                                        "control_keys": control_keys,
                                        "score": scored,
                                        "duration_s": total_s,
                                        "order_index": order_index}
        self.state["comparison"].append(row)
        self.save()
        self.artifacts.comparison_csv(self.state["comparison"])
        self.log(f"{label}: {scored['status']}; heldout complex error="
                 f"{scored.get('heldout_complex_error', float('nan')):.3f}; "
                 f"physical acquisitions={row['total_acquisitions']}; total={total_s:.1f} s")
        return row

    def _drift_check(self, block: int, channel: str, anchor: ReferenceAnchor,
                     stage: str = "after") -> dict:
        if stage not in ("pre", "between", "after"):
            raise ValueError("Unknown nominal drift-check stage")
        width = self.config["pilot_widths_us"][1]
        suffix = f"_{stage}" if self.quick else ""
        key = f"b{block}_{channel}_return_to_nominal{suffix}"
        candidate = Candidate(channel, width, anchor.nominal_amplitude_pct,
                              0., 0., anchor.full_cycle_s)
        self.acquire(self._request(key, candidate), controller=None,
                     role="nominal_drift_check", channel=channel, block=block)
        original_key = next((pilot for pilot in anchor.pilot_keys
                             if pilot.endswith(f"_w{width}_r0")), None)
        if original_key is None:
            raise ValueError("Measured anchor lacks drift reference width")
        before = self._feature(original_key, anchor)
        after = self._feature(key, anchor)
        numerator = abs(before.value-after.value)
        denom = math.sqrt(float(np.trace(before.covariance_ri+after.covariance_ri)))
        ratio = float(numerator/max(denom, 1e-12))
        result = {"status": "STABLE_WITHIN_REPEAT_NOISE" if ratio <= 3 else "DRIFT_WARNING",
                  "difference_over_joint_noise": ratio,
                  "acquisition_key": key}
        self.state["capabilities"][f"b{block}_{channel}_drift{suffix}"] = result
        self.save()
        self.log(f"Block {block} {channel} {stage} return-to-nominal: "
                 f"{result['status']} ({ratio:.2f} sigma)")
        return result

    def run_calibration(self) -> dict[str, ReferenceAnchor]:
        latest: dict[str, ReferenceAnchor] = {}
        shared_anchor: dict[str, ReferenceAnchor] | None = None
        for block in range(self.config["blocks"]):
            anchors: dict[str, ReferenceAnchor] = {}
            if self.quick and block > 0:
                if shared_anchor is None:
                    break
                anchors = shared_anchor
                self.log(f"Block {block}: using the same paid H anchor; "
                         "fresh reference and drift FIDs follow")
            else:
                for channel in self.config["channels"]:
                    try:
                        anchors[channel] = self._pilot(block, channel)
                    except TaskUncertain:
                        raise
                    except BudgetExhausted:
                        raise
                    except PilotNoSignal:
                        raise
                    except Exception as exc:
                        self.log(f"Block {block} {channel} pilot: {type(exc).__name__}: {exc}", kind="ERROR")
                        self.state["errors"].append({"block": block, "channel": channel,
                                                      "stage": "pilot", "error": str(exc),
                                                      "trace": traceback.format_exc()})
                        self.save()
            if not anchors:
                continue
            if self.quick and block == 0:
                shared_anchor = anchors
            latest = anchors
            references = self._nominal_references(block, anchors)
            controller = PerturbationController(self._scenario(block, anchors),
                            self.config["software_amplitude_ceiling_pct"])
            probes = {channel: self._probe(block, channel, anchor, controller,
                                          references[channel]) for channel, anchor in anchors.items()}
            # Quick compares only B and D; their order alternates across
            # neutral/+/- blocks. Full protocol retains its four-arm rotations.
            base = list(("B_classical", "D_adaptive_bayes") if self.quick else METHODS)
            random.Random(self.config["seed"]).shuffle(base)
            self.state["scenarios"][str(block)]["method_order_by_channel"] = {}
            for channel_index, (channel, anchor) in enumerate(anchors.items()):
                shift = ((block + channel_index) % 2 if self.quick else
                         (2*block + channel_index) % 4)
                order = base[shift:] + base[:shift]
                self.state["scenarios"][str(block)]["method_order_by_channel"][channel] = order
                self.save()
                drift_checks = []
                if self.quick:
                    drift_checks.append(self._drift_check(block, channel, anchor, "pre"))
                for position, method in enumerate(order):
                    try:
                        self._measure_arm(block, channel, method, anchor, controller,
                                          references[channel], probes[channel], position)
                    except TaskUncertain:
                        raise
                    except BudgetExhausted as exc:
                        self.state["comparison"].append({"block": block,
                            "scenario": self.state["scenarios"][str(block)]["label"],
                            "task": "frequency_rabi", "channel": channel,
                            "method": method, "status": "BUDGET_EXHAUSTED",
                            "reason": str(exc)})
                        self.save()
                        self.artifacts.comparison_csv(self.state["comparison"])
                        raise
                    except Exception as exc:
                        label = f"b{block}_{channel}_{method}"
                        self.log(f"{label}: {type(exc).__name__}: {exc}", kind="ERROR")
                        self.state["errors"].append({"block": block, "channel": channel,
                            "method": method, "error": str(exc),
                            "trace": traceback.format_exc()})
                        self.state["comparison"].append({"block": block,
                            "scenario": self.state["scenarios"][str(block)]["label"],
                            "task": "frequency_rabi", "channel": channel,
                            "method": method, "status": "INCOMPLETE_DATA",
                            "reason": f"{type(exc).__name__}: {exc}"})
                        self.save()
                        self.artifacts.comparison_csv(self.state["comparison"])
                    if self.quick and position == 0:
                        drift_checks.append(self._drift_check(block, channel, anchor, "between"))
                drift_checks.append(self._drift_check(block, channel, anchor, "after"))
                if self.quick and any(item["status"] == "DRIFT_WARNING" for item in drift_checks):
                    self.log(f"Block {block} {channel}: paired comparison invalidated by "
                             "nominal drift warning", kind="WARNING")
                    for row in self.state["comparison"]:
                        if (row.get("block") == block and row.get("channel") == channel
                                and row.get("method") in order):
                            row["status"] = "PILOT_INCONCLUSIVE"
                            row["reason"] = "Nominal drift changed beyond repeat-derived uncertainty"
                            label = f"b{block}_{channel}_{row['method']}"
                            if label in self.state["methods"]:
                                self.state["methods"][label]["score"]["status"] = "PILOT_INCONCLUSIVE"
                                self.state["methods"][label]["score"]["reason"] = row["reason"]
                    self.state["capabilities"][f"b{block}_{channel}_paired_validity"] = {
                        "status": "PILOT_INCONCLUSIVE", "reason": "Nominal drift warning"}
                    self.save()
                    self.artifacts.comparison_csv(self.state["comparison"])
                if self.quick:
                    paired = {row.get("method"): row for row in self.state["comparison"]
                              if row.get("block") == block and row.get("channel") == channel
                              and row.get("method") in order}
                    classical = paired.get("B_classical", {})
                    adaptive = paired.get("D_adaptive_bayes", {})
                    b_error = classical.get("heldout_complex_error")
                    d_error = adaptive.get("heldout_complex_error")
                    difference = (f"{d_error-b_error:+.4f}" if
                                  isinstance(b_error, (int, float)) and
                                  isinstance(d_error, (int, float)) else "unavailable")
                    self.log(f"QUICK PAIR block={block} scenario="
                             f"{self.state['scenarios'][str(block)]['label']} "
                             f"B={classical.get('status', 'MISSING')} "
                             f"D={adaptive.get('status', 'MISSING')} "
                             f"heldout_D_minus_B={difference}")
            self.artifacts.report(self.state)
        return latest

    def run_required_pilots(self) -> dict[str, ReferenceAnchor]:
        """Only anchors required by coupling/PPS/Bell; no unrelated A/B/C/D arms."""
        anchors = {}
        for channel in self.config["channels"]:
            anchors[channel] = self._pilot(0, channel)
        return anchors

    def _paired_summary(self) -> dict:
        rows = [row for row in self.state["comparison"]
                if row.get("task") == "frequency_rabi" and
                row.get("method") in METHODS]
        by_pair = {(row["block"], row["channel"]): {} for row in rows}
        for row in rows:
            by_pair[(row["block"], row["channel"])][row["method"]] = row
        summary = {}
        compared_methods = ("D_adaptive_bayes",) if self.quick else \
            (METHODS[0], METHODS[2], METHODS[3])
        for method in compared_methods:
            delta, acquisition_delta, time_delta, valid_delta = [], [], [], []
            matched, method_target, classical_target = 0, 0, 0
            valid_pairs = 0
            per_block = []
            for (block, channel), pair in sorted(by_pair.items()):
                if method not in pair or "B_classical" not in pair:
                    continue
                test, classic = pair[method], pair["B_classical"]
                matched += 1
                method_target += int(test.get("status") == "TARGET_REACHED")
                classical_target += int(classic.get("status") == "TARGET_REACHED")
                valid = (test.get("status") == "TARGET_REACHED" and
                         classic.get("status") == "TARGET_REACHED")
                valid_pairs += int(valid)
                comparison = {"block": block, "channel": channel,
                              "method_status": test.get("status"),
                              "classical_status": classic.get("status"),
                              "valid_for_advantage_claim": valid,
                              "method_heldout_error": test.get("heldout_complex_error"),
                              "classical_heldout_error": classic.get("heldout_complex_error")}
                per_block.append(comparison)
                if not all(isinstance(row.get("heldout_complex_error"), (int, float))
                           for row in (test, classic)):
                    continue
                delta.append(test["heldout_complex_error"] - classic["heldout_complex_error"])
                comparison["paired_error_difference"] = delta[-1]
                if valid:
                    valid_delta.append(delta[-1])
                acquisition_delta.append(test["total_acquisitions"] - classic["total_acquisitions"])
                time_delta.append(test["end_to_end_seconds"] - classic["end_to_end_seconds"])
            summary[method] = {"matched_block_channel_pairs": matched,
                "independent_block_channel_pairs": len(delta),
                "valid_target_pairs": valid_pairs,
                "per_block": per_block,
                "method_target_reached": method_target,
                "classical_target_reached": classical_target,
                "paired_heldout_error_difference_mean": float(np.mean(delta)) if delta else None,
                "paired_heldout_error_difference_sd": float(np.std(delta, ddof=1)) if len(delta)>1 else None,
                "paired_valid_heldout_error_difference_mean":
                    float(np.mean(valid_delta)) if valid_delta else None,
                "paired_acquisition_difference_mean": float(np.mean(acquisition_delta)) if delta else None,
                "paired_total_time_difference_mean_s": float(np.mean(time_delta)) if delta else None,
                "conclusion": "INSUFFICIENT_VALID_PAIRED_EVIDENCE" if valid_pairs < 3 else
                    "PRELIMINARY_EQUAL_BUDGET_COMPARISON_NO_GENERAL_ADVANTAGE_CLAIM"}
        return summary

    def _quantum_anchors(self, latest: dict[str, ReferenceAnchor]) -> dict:
        h, p = latest["H"], latest["P"]
        coherent = min(h.readout.coherent_window_s, p.readout.coherent_window_s)
        return {"h90_width_us": round(h.t90_us), "p90_width_us": round(p.t90_us),
                "h90_amplitude_pct": h.nominal_amplitude_pct,
                "p90_amplitude_pct": p.nominal_amplitude_pct,
                "sample_count": self.config["sample_count"],
                "sample_frequency_hz": self.config["sample_frequency_hz"],
                "relaxation_delay_s": 15,
                "coherent_window_s": coherent,
                "coupling_delays_us": (0, 500, 1000, 1500, 2000, 3000),
                "coupling_center_bands_hz": {
                    "H": (h.readout.fid_frequency_hz-1000,
                          h.readout.fid_frequency_hz+1000),
                    "P": (p.readout.fid_frequency_hz-1000,
                          p.readout.fid_frequency_hz+1000)},
                "coupling_j_search_hz": (20., 1800.),
                "readout_gains": {"H": h.readout.gain, "P": p.readout.gain},
                "line_decay_per_s": {"H": float(1/h.readout.decay_s),
                                     "P": float(1/p.readout.decay_s)},
                "sequence_budget_status": "HARDWARE_SEQUENCE_LIMIT_UNMEASURED"}

    def _qualify_coherent_timing(self, latest: dict[str, ReferenceAnchor],
                                 anchors: dict) -> dict:
        if self.transport is None:
            raise RuntimeError("No physical transport")
        evidence: dict[str, Any] = {"idle_gap_verified": False,
              "idle_gap_verified_by_channel": {"H": False, "P": False},
              "verified_idle_us_by_channel": {"H": [], "P": []},
              "cross_channel_alignment_verified": False,
              "j_sign_verified": False, "cnot_verified": False,
              "line_resolved_readout_verified": False,
              "frozen_readout_verified": False,
              "absolute_scale_verified": False}
        qualification_round = int(self.state.get("qualification_round", -1)) + 1
        self.state["qualification_round"] = qualification_round
        self.save()
        delays = set(anchors["coupling_delays_us"]) - {0}
        for channel in ("H", "P"):
            anchor = latest[channel]
            path = 0 if channel == "H" else 1
            eligible_widths = [value for value in self.config["pilot_widths_us"]
                               if value >= 2*self.config["software_pulse_width_envelope_us"][0]]
            if not eligible_widths:
                raise ValueError("No completed pilot width permits bounded split timing test")
            width = min(eligible_widths,
                        key=lambda value: abs(value - anchor.t90_us))
            reference = PulseSpec(path, 0, width, anchor.nominal_amplitude_pct,
                                  0., 0., role="timing_reference")
            self.log(f"Qualifying measured {channel} contiguous pulse timing and coherent idle")

            def on_result(result: AcquisitionResult) -> None:
                if result.key in self.state["acquisitions"]:
                    # A resumed qualification must not re-issue duplicate
                    # physical tasks under an already archived key.
                    raise ValueError("Duplicate physical qualification key")
                try:
                    self.artifacts.save_acquisition(result.key, result,
                        {"qualification": True, "requested_payload": result.requested_payload},
                        role="timing_qualification", channel=channel)
                except Exception as exc:
                    raise TaskUncertain(f"Completed qualification could not be archived: {exc}",
                                        task_id=result.task_id) from exc
                self.task_count += 1
                self.rf_us += width
                self.state["acquisitions"][result.key] = {
                    "key": result.key, "task_id": result.task_id,
                    "role": "timing_qualification", "channel": channel,
                    "elapsed_s": result.elapsed_s, "requested_rf_us": width,
                    "data_file": f"data/{result.key}.npz", "completed_utc": utc_now()}
                self.save()
                self.log(f"FID {result.key}: {len(result.fid_complex)} points, "
                         f"{result.elapsed_s:.1f} s")
            if self.task_count + 16 > self.config["max_tasks"] or \
                    self.rf_us + 16*width > self.config["max_requested_rf_us"]:
                evidence["reason"] = "Finite qualification budget exhausted"
                break
            split = self.transport.verify_two_segment_equivalence(reference,
                sample_count=4000, sample_frequency_hz=self.config["sample_frequency_hz"],
                key_prefix=f"qual_r{qualification_round}_{channel}_split",
                on_acquisition=on_result)
            self.state["capabilities"][f"{channel}_contiguous_timing"] = jsonable(split)
            self.save()
            self.log(f"{channel} contiguous timing: {split['status']}")
            if split["status"] != "CONTIGUOUS_SEGMENTS_VERIFIED":
                continue
            for gap in (500, 1000, 1500):
                if self.task_count + 4 > self.config["max_tasks"]:
                    evidence["reason"] = "Finite idle qualification task budget exhausted"
                    break
                report = self.transport.verify_idle_gap(reference, gap_us=gap,
                    sample_count=4000, sample_frequency_hz=self.config["sample_frequency_hz"],
                    key_prefix=f"qual_r{qualification_round}_{channel}_{gap}",
                    on_acquisition=on_result)
                self.state["capabilities"][f"{channel}_idle_{gap}us"] = jsonable(report)
                self.save()
                self.log(f"{channel} idle {gap}/{gap*2} us: {report['status']}")
                if report["status"] != "SAME_CHANNEL_IDLE_VERIFIED":
                    break
            evidence["idle_gap_verified_by_channel"][channel] = all(
                self.transport.idle_gap_verified(path, value) for value in delays)
            evidence["verified_idle_us_by_channel"][channel] = sorted(
                value for value in delays if self.transport.idle_gap_verified(path, value))
        evidence["idle_gap_verified"] = all(evidence["idle_gap_verified_by_channel"].values())
        evidence["contiguous_verified_both_channels"] = self.transport.timing_verified
        self.state["capabilities"]["timing_evidence"] = evidence
        self.save()
        return evidence

    def _quantum_callback(self, request: AcquisitionRequest):
        if self.transport is None:
            raise RuntimeError("No live transport")
        if request.key in self.state["acquisitions"]:
            previous = self.state["acquisitions"][request.key]
            if previous.get("logical_request") != jsonable(asdict(request)):
                raise ValueError("Resume quantum logical request differs from original task")
            time_s, fid = self.artifacts.load_fid(request.key)
            raw = json.loads((self.artifacts.path / "data" /
                              f"{request.key}.json").read_text(encoding="utf-8"))
            applied = json.loads((self.artifacts.path / "evaluator_only" /
                                  f"{request.key}_applied.json").read_text(encoding="utf-8"))
            vendor = json.loads((self.artifacts.path / "vendor_reference" /
                                 f"{request.key}.json").read_text(encoding="utf-8"))
            return AcquisitionResult(request.key, previous["task_id"], "COMPLETED",
                    time_s, fid, vendor, applied["requested_payload"],
                    applied["sent_payload"], raw["timing_evidence"],
                    previous["elapsed_s"], raw["device_snapshot"])
        rf = self._check_budget(request)
        result = self.transport.acquire(request)
        try:
            self.artifacts.save_acquisition(request.key, result, asdict(request),
                                            role="quantum_task", channel=None)
        except Exception as exc:
            raise TaskUncertain(f"Completed quantum task could not be archived: {exc}",
                                task_id=result.task_id) from exc
        self.task_count += 1
        self.rf_us += rf
        self.state["acquisitions"][request.key] = {
            "key": request.key, "task_id": result.task_id,
            "role": "quantum_task", "elapsed_s": result.elapsed_s,
            "requested_rf_us": rf, "logical_request": jsonable(asdict(request)),
            "data_file": f"data/{request.key}.npz", "completed_utc": utc_now()}
        self.save()
        self.log(f"FID {request.key}: {len(result.fid_complex)} points, {result.elapsed_s:.1f} s")
        if self.config["pause_seconds"]:
            time.sleep(self.config["pause_seconds"])
        return result

    def _run_quantum_tasks(self, latest: dict[str, ReferenceAnchor]) -> None:
        from .quantum_tasks import run_bell, run_coupling, run_pps
        requested = self.task
        self.state["capabilities"]["full_chain_comparison"] = {
            "status": "UNSUPPORTED_REQUIRED_PRIMITIVE",
            "reason": "Paired A/B/C/D PPS/gate and R0/R1 ablations require physically "
                      "qualified H/P alignment, signed J, native CNOT and frozen readout; "
                      "this launcher does not assert those capabilities"}
        if not all(channel in latest for channel in ("H", "P")):
            for name in ("coupling", "pps", "bell"):
                if requested in (name, "all"):
                    self.state["capabilities"][name] = {"status": "PILOT_INCONCLUSIVE",
                        "reason": "H and P nominal anchors not both identified"}
            self.save()
            return
        anchors = self._quantum_anchors(latest)
        # These checks submit bounded real FIDs and archive each one before
        # proceeding.  Nothing infers H/P alignment or J sign from an API name.
        timing = self._qualify_coherent_timing(latest, anchors)
        def record_task(name: str, result: dict, start_tasks: int,
                        start_time: float) -> None:
            result = jsonable(result)
            self.state["capabilities"][name] = result
            row = {"block": 0, "scenario": "nominal_unpaired", "task": name,
                   "channel": "H/P", "method": "native_low_level",
                   "status": result.get("status", "INCOMPLETE_DATA"),
                   "design_acquisitions": self.task_count-start_tasks,
                   "control_acquisitions": 0,
                   "total_acquisitions": self.task_count-start_tasks,
                   "training_seconds": time.monotonic()-start_time,
                   "validation_seconds": None,
                   "end_to_end_seconds": time.monotonic()-start_time,
                   "reason": result.get("reason", ""),
                   "vendor_comparison": "NOT_MATCHED_PERTURBATION"}
            self.state["comparison"].append(row)
            self.artifacts.comparison_csv(self.state["comparison"])
            self.log(f"{name}: {row['status']} — {row['reason']}; "
                     f"physical acquisitions={row['total_acquisitions']}")
            self.save()
        if requested in ("coupling", "pps", "bell", "all"):
            task_count, clock = self.task_count, time.monotonic()
            coupling = run_coupling(self._quantum_callback, anchors, timing)
            record_task("coupling", coupling, task_count, clock)
        else:
            coupling = None
        if requested in ("pps", "bell", "all"):
            task_count, clock = self.task_count, time.monotonic()
            pps = run_pps(self._quantum_callback, anchors, timing, coupling=coupling)
            record_task("pps", pps, task_count, clock)
        if requested in ("bell", "all"):
            task_count, clock = self.task_count, time.monotonic()
            labels = ("Phi+", "Psi+") if requested == "bell" else \
                ("Phi+", "Phi-", "Psi+", "Psi-")
            bell = run_bell(self._quantum_callback, anchors, timing,
                            coupling=coupling, labels=labels)
            record_task("bell", bell, task_count, clock)

    def execute(self) -> dict:
        self.artifacts.snapshot_sources(self.repo, SOURCE_FILES)
        if self.state.get("status") == "COMPLETED_WITH_EXPLICIT_LIMITATIONS":
            return self.state
        try:
            with PhysicalTransport(host=self.config["host"], port=self.config["port"],
                    exclusive_use_confirmed=self.exclusive_use_confirmed,
                    timeout_s=self.config["timeout_seconds"]) as transport:
                self.transport = transport
                self.log(f"SpinQLabLink connected to {self.config['host']}:{self.config['port']}")
                latest = (self.run_calibration() if self.task in
                          ("frequency", "rabi", "all") else self.run_required_pilots())
                if self.task in ("coupling", "pps", "bell", "all"):
                    self._run_quantum_tasks(latest)
                self.state["paired_summary"] = self._paired_summary()
                if self.quick:
                    paired = self.state["paired_summary"].get("D_adaptive_bayes", {})
                    self.log(f"QUICK SUMMARY valid_pairs="
                             f"{paired.get('valid_target_pairs', 0)}/3 "
                             f"valid_mean_D_minus_B="
                             f"{paired.get('paired_valid_heldout_error_difference_mean')} "
                             f"conclusion={paired.get('conclusion')}")
                self.state["status"] = ("COMPLETED_WITH_EXPLICIT_LIMITATIONS"
                    if self.state["comparison"] or self.state["capabilities"]
                    else "PILOT_INCONCLUSIVE")
        except TaskUncertain as exc:
            self.state["status"] = "STOPPED_UNCERTAIN"
            self.state["errors"].append({"stage": "transport", "task_id": exc.task_id,
                "error": str(exc), "trace": traceback.format_exc()})
            self.log(f"Uncertain task {exc.task_id}: {exc}; no further task submitted", kind="ERROR")
        except KeyboardInterrupt:
            self.state["status"] = "STOPPED_UNCERTAIN"
            self.state["errors"].append({"stage": "keyboard",
                "error": "Ctrl+C received; physical task state must be checked before resume"})
            self.log("Ctrl+C received; inspect tablet task state before any resume", kind="WARNING")
        except BudgetExhausted as exc:
            self.state["status"] = "BUDGET_EXHAUSTED"
            self.state["errors"].append({"stage": "budget", "error": str(exc)})
            self.log(str(exc), kind="ERROR")
        except PilotNoSignal as exc:
            self.state["status"] = "PILOT_INCONCLUSIVE"
            self.state["errors"].append({"stage": "pilot_signal", "error": str(exc)})
            self.log(str(exc) + "; no further physical task submitted", kind="WARNING")
        except Exception as exc:
            self.state["status"] = "FAILED"
            self.state["errors"].append({"stage": "run", "error": f"{type(exc).__name__}: {exc}",
                                          "trace": traceback.format_exc()})
            self.log(f"{type(exc).__name__}: {exc}", kind="ERROR")
            self.log(traceback.format_exc(), kind="ERROR")
        finally:
            self.transport = None
            if self.state["comparison"]:
                # A timed stop still leaves the already measured paired arms
                # inspectable; recompute after every resume to avoid stale rows.
                self.state["paired_summary"] = self._paired_summary()
            self.state["ended_utc"] = utc_now()
            self.save()
            self.artifacts.comparison_csv(self.state["comparison"])
            self.artifacts.report(self.state)
            archive = self.artifacts.archive()
            self.log(f"Report: {self.artifacts.path / 'REPORT.md'}")
            self.log(f"Archive: {archive}")
            self.log(f"State={self.state['status']}; physical tasks={self.task_count}; "
                     f"comparison rows={len(self.state['comparison'])}; "
                     f"errors={len(self.state['errors'])}")
        return self.state
