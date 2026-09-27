"""Hardware-intended coupling, temporal PPS, and Bell tasks for bayes_online.

Only the caller's acquisition callback touches SpinQLabLink.  Every call is
one physical exported FID and must be persisted by the caller.  No result in
this module is a device result until the callback returns a completed task.

The matrices in :mod:`physics` are mathematical conventions.  The capability
gates here require separate experimental evidence before compiling entangling
sequences or interpreting a full density matrix on Gemini Lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from scipy.optimize import least_squares

from .physics import (
    PAULIS,
    bell_preparation_gates,
    bell_score,
    default_readout_unitaries,
    pps_target_deviation,
    reconstruct_deviation,
    reconstruct_full_density,
    temporal_pps_branches,
    tomography_design,
)
from .transport import AcquisitionRequest, AcquisitionResult, PulseSpec, TaskUncertain


class Acquire(Protocol):
    def __call__(self, request: AcquisitionRequest) -> AcquisitionResult: ...


def _unsupported(reason: str, **data: Any) -> dict[str, Any]:
    return {"status": "UNSUPPORTED_REQUIRED_PRIMITIVE", "reason": reason,
            "acquisitions": 0, **data}


def _complete(acquire: Acquire, request: AcquisitionRequest) -> AcquisitionResult:
    result = acquire(request)
    if result.status != "COMPLETED":
        raise ValueError(f"{request.key}: task status {result.status}; no completed FID")
    if result.key != request.key or result.fid_complex.size != request.sample_count:
        raise ValueError(f"{request.key}: completed task data/key mismatch")
    if result.time_s.size != request.sample_count or not np.all(np.isfinite(result.fid_complex)):
        raise ValueError(f"{request.key}: incomplete exported complex FID")
    return result


def _baseline(anchors: Mapping[str, Any]) -> dict[str, Any]:
    required = ("h90_width_us", "p90_width_us", "h90_amplitude_pct",
                "p90_amplitude_pct", "sample_count", "sample_frequency_hz",
                "relaxation_delay_s", "coherent_window_s")
    absent = [name for name in required if name not in anchors]
    if absent:
        raise ValueError("missing measured anchors: " + ", ".join(absent))
    data = {name: anchors[name] for name in required}
    for name in ("h90_width_us", "p90_width_us"):
        value = float(data[name])
        if not value.is_integer() or not 1 <= value <= 1_000_000:
            raise ValueError(f"{name} must be a measured positive integer µs")
        data[name] = int(value)
    for name in ("h90_amplitude_pct", "p90_amplitude_pct"):
        value = float(data[name])
        if not np.isfinite(value) or not 0 < value < 100:
            raise ValueError(f"{name} must be a measured unclipped RF percentage")
        data[name] = value
    data["sample_count"] = int(data["sample_count"])
    data["sample_frequency_hz"] = int(data["sample_frequency_hz"])
    data["relaxation_delay_s"] = float(data["relaxation_delay_s"])
    data["coherent_window_s"] = float(data["coherent_window_s"])
    if data["coherent_window_s"] <= 0 or not np.isfinite(data["coherent_window_s"]):
        raise ValueError("coherent_window_s must come from measured pilot FIDs")
    return data


def _request(
    key: str, pulses: Sequence[PulseSpec], sample_path: int,
    anchors: Mapping[str, Any],
) -> AcquisitionRequest:
    return AcquisitionRequest(
        key=key, pulses=tuple(pulses), sample_path=sample_path,
        sample_count=int(anchors["sample_count"]),
        sample_frequency_hz=int(anchors["sample_frequency_hz"]),
        relaxation_delay_s=float(anchors["relaxation_delay_s"]),
        h_frequency_shift_hz=float(anchors.get("h_frequency_shift_hz", 0)),
        p_frequency_shift_hz=float(anchors.get("p_frequency_shift_hz", 0)),
        h_demodulation_hz=float(anchors.get("h_demodulation_hz", 0)),
        p_demodulation_hz=float(anchors.get("p_demodulation_hz", 0)),
        initialize_state=False,
    )


def _coupling_pulses(path: int, delay_us: int, a: Mapping[str, Any]) -> tuple[PulseSpec, ...]:
    channel = "h" if path == 0 else "p"
    width = int(a[f"{channel}90_width_us"])
    amplitude = float(a[f"{channel}90_amplitude_pct"])
    # Only the observed channel is scheduled.  Independent H/P delay FIDs can
    # qualify |J| without assuming their simultaneous cross-channel alignment.
    # A later CNOT still requires separate alignment qualification.
    preparation = PulseSpec(path, 0, width, amplitude, 0,
                            role=f"{channel}_prepare_90")
    if delay_us == 0:
        return (preparation,)
    return (
        preparation,
        PulseSpec(path, width, delay_us, 0, 0,
                  role="verified_coherent_idle"),
    )


def _short_complex_fid(result: AcquisitionResult, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(result.time_s, dtype=float)
    y = np.asarray(result.fid_complex, dtype=complex)
    if len(t) < 32 or not np.all(np.diff(t) > 0):
        raise ValueError("exported FID axis too short or nonmonotone")
    n = min(int(np.searchsorted(t - t[0], window_s)), len(t))
    if n < 32:
        raise ValueError("measured coherent FID window contains fewer than 32 points")
    stride = max(1, n // 192)
    return t[:n:stride] - t[0], y[:n:stride]


@dataclass(frozen=True)
class _CouplingFit:
    center_hz: float
    j_abs_hz: float
    j_sd_hz: float
    decay_per_s: float
    train_relative_residual: float
    heldout_relative_residual: float
    component_balance: float
    alias_ambiguous: bool
    fit_params: np.ndarray
    component_amplitudes: np.ndarray


def _joint_delay_fit(
    records: Sequence[tuple[float, AcquisitionResult]],
    window_s: float,
    center_band_hz: tuple[float, float],
    j_search_hz: tuple[float, float],
) -> _CouplingFit:
    """Two linked spectral components fitted across actual coherent delays.

    The final delay is held out.  Components share frequencies and complex
    amplitudes across runs; no independent per-FID gain is allowed to absorb
    control errors.  The inferred J is an *effective commanded-time* rate.
    """

    if len(records) < 6:
        raise ValueError("at least six independently acquired delay FIDs required")
    delays = np.array([float(row[0]) for row in records], dtype=float)
    if not np.all(np.isfinite(delays)) or len(set(delays)) != len(delays):
        raise ValueError("coherent delays must be finite and distinct")
    if delays.max() - delays.min() <= 0:
        raise ValueError("no coherent delay span")
    local = [_short_complex_fid(result, window_s) for _, result in records]
    train_t = np.concatenate([t + tau for (tau, _), (t, _) in zip(records[:-1], local[:-1])])
    train_y = np.concatenate([y for _, y in local[:-1]])
    test_t = local[-1][0] + delays[-1]
    test_y = local[-1][1]
    f_lo, f_hi = map(float, center_band_hz)
    j_lo, j_hi = map(float, j_search_hz)
    if not (np.isfinite([f_lo, f_hi, j_lo, j_hi]).all()
            and f_lo < f_hi and 0 < j_lo < j_hi):
        raise ValueError("finite bounded center/J search intervals are required")

    def design(time_s: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        center, j_abs, decay = parameters
        decay_term = np.exp(-decay * time_s)
        lower = decay_term * np.exp(2j * np.pi * (center - j_abs / 2) * time_s)
        upper = decay_term * np.exp(2j * np.pi * (center + j_abs / 2) * time_s)
        return np.column_stack((lower, upper, np.ones_like(time_s)))

    def solve(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = design(train_t, parameters)
        amplitudes, _, _, _ = np.linalg.lstsq(matrix, train_y, rcond=None)
        return amplitudes, matrix @ amplitudes - train_y

    def residual(parameters: np.ndarray) -> np.ndarray:
        difference = solve(parameters)[1]
        return np.concatenate((difference.real, difference.imag))

    gamma_start = min(1 / window_s, 2_000.0)
    starts = []
    for center in np.linspace(f_lo, f_hi, 7):
        for j_abs in np.linspace(j_lo, j_hi, 9):
            p = np.array([center, j_abs, gamma_start])
            starts.append((float(np.linalg.norm(residual(p))), p))
    starts.sort(key=lambda item: item[0])
    bounds = ([f_lo, j_lo, 0.0], [f_hi, j_hi, 10_000.0])
    best = None
    minima = []
    for _, start in starts[:6]:
        trial = least_squares(residual, start, bounds=bounds, max_nfev=200,
                              x_scale=[max(f_hi - f_lo, 1), max(j_hi - j_lo, 1), 500])
        minima.append(trial)
        if best is None or np.linalg.norm(trial.fun) < np.linalg.norm(best.fun):
            best = trial
    assert best is not None
    p = best.x
    amplitudes, difference = solve(p)
    prediction = design(test_t, p) @ amplitudes
    signal_norm = max(np.linalg.norm(train_y - amplitudes[2]), 1e-12)
    heldout_norm = max(np.linalg.norm(test_y - amplitudes[2]), 1e-12)
    train_rel = float(np.linalg.norm(difference) / signal_norm)
    heldout_rel = float(np.linalg.norm(prediction - test_y) / heldout_norm)
    dof = max(len(best.fun) - 9, 1)
    variance = np.dot(best.fun, best.fun) / dof
    fisher = best.jac.T @ best.jac
    j_sd = float(np.sqrt(max(variance * np.linalg.pinv(fisher)[1, 1], 0)))
    balance = float(min(abs(amplitudes[0]), abs(amplitudes[1])) /
                    max(abs(amplitudes[0]), abs(amplitudes[1]), 1e-12))
    best_cost = float(np.linalg.norm(best.fun))
    alias = any(
        abs(trial.x[1] - p[1]) > max(2 * j_sd, 0.05 * p[1])
        and float(np.linalg.norm(trial.fun)) <= best_cost * 1.05 + 1e-8
        for trial in minima
    )
    return _CouplingFit(float(p[0]), float(p[1]), j_sd, float(p[2]),
                        train_rel, heldout_rel, balance, alias, p, amplitudes)


def run_coupling(
    acquire_callback: Acquire,
    anchors: Mapping[str, Any],
    timing_evidence: Mapping[str, Any],
    key_prefix: str = "coupling",
) -> dict[str, Any]:
    """Measure H/P coherent-delay multiplets, validate on withheld delay."""

    by_channel = timing_evidence.get("idle_gap_verified_by_channel")
    idle_ok = (bool(by_channel.get("H")) and bool(by_channel.get("P"))
               if isinstance(by_channel, Mapping) else
               bool(timing_evidence.get("idle_gap_verified")))
    if not idle_ok:
        return _unsupported("coherent zero-RF idle gap has not been measured")
    try:
        a = _baseline(anchors)
        requested_delays = tuple(float(x) for x in anchors["coupling_delays_us"])
        if any(not np.isfinite(x) or x != int(x) for x in requested_delays):
            raise ValueError("coupling_delays_us must use the verified integer-µs grid exactly")
        delays_us = tuple(int(x) for x in requested_delays)
        if (len(delays_us) < 6 or len(set(delays_us)) != len(delays_us)
                or min(delays_us) < 0 or 0 not in delays_us):
            raise ValueError("coupling_delays_us needs >=6 distinct nonnegative delays including zero")
        center_bands = anchors["coupling_center_bands_hz"]
        j_search = tuple(anchors["coupling_j_search_hz"])
        if len(j_search) != 2:
            raise ValueError("coupling_j_search_hz must have lower and upper bounds")
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "PILOT_INCONCLUSIVE", "reason": str(exc), "acquisitions": 0}
    verified_gaps = timing_evidence.get("verified_idle_us_by_channel")
    if not isinstance(verified_gaps, Mapping):
        return _unsupported("per-channel verified idle gap duration records are missing")
    for channel in ("H", "P"):
        try:
            raw_qualified = tuple(float(x) for x in verified_gaps[channel])
            if any(not np.isfinite(x) or x != int(x) for x in raw_qualified):
                raise ValueError("qualification list contains a noninteger gap")
            qualified = {int(x) for x in raw_qualified}
        except (KeyError, TypeError, ValueError):
            return _unsupported(f"{channel} verified idle gap duration record is missing")
        missing = sorted(set(delays_us) - {0} - qualified)
        if missing:
            return _unsupported(f"{channel} coherent idle durations not qualified: {missing}")

    evidence: dict[str, Any] = {"H": [], "P": []}
    measured_keys: list[str] = []
    fits: dict[str, _CouplingFit] = {}
    try:
        for path, channel in ((0, "H"), (1, "P")):
            for index, delay in enumerate(delays_us):
                key = f"{key_prefix}_{channel}_{index:02d}"
                request = _request(key, _coupling_pulses(path, delay, a), path, a)
                result = _complete(acquire_callback, request)
                evidence[channel].append((delay / 1_000_000.0, result))
                measured_keys.append(key)
            fits[channel] = _joint_delay_fit(evidence[channel], a["coherent_window_s"],
                                             tuple(center_bands[channel]), j_search)
    except TaskUncertain:
        raise
    except Exception as exc:
        return {"status": "INCOMPLETE_DATA", "reason": f"{type(exc).__name__}: {exc}",
                "acquisitions": len(measured_keys), "measured_keys": measured_keys}

    report = {channel: {
        "center_hz": fit.center_hz, "effective_j_abs_hz": fit.j_abs_hz,
        "j_conditional_sd_hz": fit.j_sd_hz,
        "decay_per_s": fit.decay_per_s,
        "train_relative_residual": fit.train_relative_residual,
        "heldout_relative_residual": fit.heldout_relative_residual,
        "component_balance": fit.component_balance,
        "alias_ambiguous": fit.alias_ambiguous,
    } for channel, fit in fits.items()}
    h, p = fits["H"], fits["P"]
    j_abs = (h.j_abs_hz + p.j_abs_hz) / 2
    disagreement = abs(h.j_abs_hz - p.j_abs_hz)
    agreed = disagreement <= max(2 * np.hypot(h.j_sd_hz, p.j_sd_hz), 0.2 * j_abs)
    quality = all(fit.heldout_relative_residual <= 0.3
                  and fit.train_relative_residual <= 0.3
                  and fit.component_balance >= 0.05
                  and not fit.alias_ambiguous
                  and fit.j_abs_hz > 2 * fit.j_sd_hz
                  for fit in fits.values())
    signed = (bool(timing_evidence.get("j_sign_verified"))
              and bool(timing_evidence.get("cross_channel_alignment_verified"))
              and anchors.get("j_sign") in (-1, 1))
    return {
        "status": "TARGET_REACHED" if agreed and quality else "NONIDENTIFIABLE",
        "reason": ("effective |J| fits agree; sign and alignment independently qualified"
                   if agreed and quality and signed else
                   "effective |J| fits agree; sign and/or H/P alignment unverified"
                   if agreed and quality else
                   "H/P delay fits, heldout prediction, or component separation unverified"),
        "acquisitions": len(measured_keys), "measured_keys": measured_keys,
        "channels": report, "effective_j_abs_hz": float(j_abs),
        "effective_j_hz": float(j_abs * anchors["j_sign"]) if agreed and quality and signed else None,
        "magnitude_validated": bool(agreed and quality),
        "sign_validated": bool(signed),
        "cross_channel_alignment_verified": bool(timing_evidence.get("cross_channel_alignment_verified")),
        "cross_channel_j_difference_hz": float(disagreement),
        "heldout_delay_us": delays_us[-1],
        "interpretation": "effective commanded-delay rate, not independently proven physical J",
    }


class _NativeCompiler:
    """Compile one complete preparation/circuit/readout into explicit mode-0 pulses.

    Its ideal decompositions are not claimed as device-qualified by themselves;
    run_pps/run_bell require independent CNOT and timing evidence first.
    """

    def __init__(self, anchors: Mapping[str, Any], effective_j_hz: float):
        self.anchors = anchors
        self.j = float(effective_j_hz)
        if not np.isfinite(self.j) or self.j == 0:
            raise ValueError("signed effective J is required for compiled CZ")
        self.cursor_us = 0
        self.pulses: list[PulseSpec] = []

    def _rotation(self, qubit: str, axis: str, quarter_turns: int, role: str) -> None:
        if quarter_turns == 0:
            return
        channel = "h" if qubit == "H" else "p"
        path = 0 if qubit == "H" else 1
        width = abs(quarter_turns) * int(self.anchors[f"{channel}90_width_us"])
        phase = (0 if axis == "X" else 90) + (180 if quarter_turns < 0 else 0)
        phase += float(self.anchors.get(f"{channel}_rf_phase_deg", 0))
        self.pulses.append(PulseSpec(
            path=path, start_us=self.cursor_us, width_us=width,
            amplitude_pct=float(self.anchors[f"{channel}90_amplitude_pct"]),
            phase_deg=phase,
            detuning_hz=float(self.anchors.get(f"{channel}_pulse_detuning_hz", 0)),
            role=role,
        ))
        self.cursor_us += width

    def _parallel_rotations(self, axis: str, quarter_turns: int, role: str) -> None:
        if quarter_turns == 0:
            return
        start = self.cursor_us
        widths = []
        for qubit, channel, path in (("H", "h", 0), ("P", "p", 1)):
            width = abs(quarter_turns) * int(self.anchors[f"{channel}90_width_us"])
            phase = (0 if axis == "X" else 90) + (180 if quarter_turns < 0 else 0)
            phase += float(self.anchors.get(f"{channel}_rf_phase_deg", 0))
            self.pulses.append(PulseSpec(
                path, start, width, float(self.anchors[f"{channel}90_amplitude_pct"]),
                phase, float(self.anchors.get(f"{channel}_pulse_detuning_hz", 0)),
                f"{role}_{qubit}",
            ))
            widths.append(width)
        self.cursor_us += max(widths)

    def _hadamard(self, qubit: str, role: str) -> None:
        # Ideal H = i Rx(pi) Ry(pi/2); physical finite pulses retain J.
        self._rotation(qubit, "Y", 1, role + "_Ry90")
        self._rotation(qubit, "X", 2, role + "_Rx180")

    def _z(self, qubit: str, sign: int, role: str) -> None:
        # Rz(sign*pi/2) = Rx(pi/2) Ry(sign*pi/2) Rx(-pi/2).
        self._rotation(qubit, "X", -1, role + "_RxMinus90")
        self._rotation(qubit, "Y", sign, role + "_Ry90")
        self._rotation(qubit, "X", 1, role + "_Rx90")

    def _cz(self, role: str) -> None:
        j_idle_us = round(1_000_000 / (2 * abs(self.j)))
        if j_idle_us < 1:
            raise ValueError("J evolution requires sub-µs timing not qualified")
        self.cursor_us += j_idle_us
        # Both local Z rotations are decomposed into explicit XY pulses.
        sign = -1 if self.j > 0 else 1
        self._parallel_rotations("X", -1, role + "_local_z_1")
        self._parallel_rotations("Y", sign, role + "_local_z_2")
        self._parallel_rotations("X", 1, role + "_local_z_3")

    def gate(self, gate: str, role: str) -> None:
        if gate == "H_H" or gate == "H_P":
            self._hadamard(gate[-1], role)
        elif gate in ("X_H", "X_P"):
            self._rotation(gate[-1], "X", 2, role)
        elif gate in ("Z_H", "Z_P"):
            # Z differs from Rz(pi) only by global phase.
            self._rotation(gate[-1], "X", -1, role + "_RxMinus90")
            self._rotation(gate[-1], "Y", 2, role + "_Ry180")
            self._rotation(gate[-1], "X", 1, role + "_Rx90")
        elif gate in ("CNOT_H_P", "CNOT_P_H"):
            target = gate[-1]
            self._hadamard(target, role + "_pre_H")
            self._cz(role + "_CZ")
            self._hadamard(target, role + "_post_H")
        elif gate == "CZ":
            self._cz(role)
        elif gate == "RX90_H" or gate == "RX90_P":
            self._rotation(gate[-1], "X", 1, role)
        elif gate == "RY90_H" or gate == "RY90_P":
            self._rotation(gate[-1], "Y", 1, role)
        else:
            raise ValueError(f"unsupported low-level gate {gate}")

    def finish(self) -> tuple[PulseSpec, ...]:
        # The readout design omits the all-identity setting, so every request
        # contains RF.  No unqualified 1 µs zero-RF marker is appended.
        if not self.pulses:
            raise ValueError("pulse-less physical-layer acquisition is unverified")
        max_sequence = int(self.anchors["max_sequence_us"])
        max_rf = int(self.anchors["max_rf_us_per_acquisition"])
        rf_sum = sum(p.width_us for p in self.pulses if p.amplitude_pct > 0)
        if self.cursor_us > max_sequence or rf_sum > max_rf:
            raise ValueError("compiled PPS/circuit/readout exceeds measured sequence or RF budget")
        return tuple(self.pulses)


_READOUT_OPTIONS = (("I", ()), ("Rx90", ("RX90",)), ("Ry90", ("RY90",)))


def _readout_settings() -> tuple[tuple[str, tuple[str, ...]], ...]:
    settings = []
    for name_h, h in _READOUT_OPTIONS:
        for name_p, p in _READOUT_OPTIONS:
            label_h = "I" if not h else f"{name_h}H"
            label_p = "I" if not p else f"{name_p}P"
            gates = tuple(x + "_H" for x in h) + tuple(x + "_P" for x in p)
            settings.append((f"{label_h}+{label_p}", gates))
    assert tuple(name for name, _ in settings) == tuple(name for name, _ in default_readout_unitaries())
    # The eight remaining settings still have rank 15 and condition ~1.41
    # in the ideal two-line model.  This avoids an unqualified pulse-less
    # acquisition for the identity temporal-PPS branch.
    return tuple(item for item in settings if item[0] != "I+I")


def _evaluator(
    anchors: Mapping[str, Any], timing_evidence: Mapping[str, Any],
    require_absolute: bool,
) -> tuple[Any, dict[str, tuple[float, float]], dict[str, float], dict[str, float]]:
    if not timing_evidence.get("line_resolved_readout_verified"):
        raise ValueError("two complex multiplet lines per H/P FID have not been qualified")
    if not timing_evidence.get("frozen_readout_verified"):
        raise ValueError("independent frozen readout unitaries/gains are unavailable")
    if require_absolute and not timing_evidence.get("absolute_scale_verified"):
        raise ValueError("full-state Bell fidelity lacks independently calibrated absolute scale")
    matrices = anchors["readout_unitaries"]
    names = [name for name, _ in _readout_settings()]
    if not set(names).issubset(matrices):
        raise ValueError("frozen readout matrices must cover the eight physical settings")
    settings = [(name, np.asarray(matrices[name], complex)) for name in names]
    gains = anchors["readout_gains"]
    design = tomography_design(settings, gains)
    if design.rank != 15 or design.condition > 1e8:
        raise ValueError(f"readout rank={design.rank}/15 condition={design.condition:.3g}")
    freqs = anchors["line_frequencies_hz"]
    decay = anchors["line_decay_per_s"]
    noise_ceiling = anchors["max_fid_model_residual_rms"]
    out_freqs = {}
    out_decay = {}
    out_noise = {}
    window_s = float(anchors["coherent_window_s"])
    for channel in ("H", "P"):
        pair = tuple(map(float, freqs[channel]))
        if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] == pair[1]:
            raise ValueError(f"two distinct frozen {channel} line frequencies required")
        if abs(pair[1] - pair[0]) * window_s < 0.5:
            raise ValueError(f"{channel} lines are not resolved by measured coherent FID window")
        out_freqs[channel] = pair
        out_decay[channel] = float(decay[channel])
        if not 0 <= out_decay[channel] <= 10_000:
            raise ValueError(f"{channel} frozen decay must be bounded and finite")
        out_noise[channel] = float(noise_ceiling[channel])
        if not np.isfinite(out_noise[channel]) or out_noise[channel] <= 0:
            raise ValueError(f"{channel} independent residual/noise ceiling required")
    if require_absolute:
        scale = float(anchors["polarization_scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("independent positive polarization scale required")
        tolerance = float(anchors["max_projection_frobenius"])
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("independent projection tolerance required")
    return design, out_freqs, out_decay, out_noise


def _fit_two_lines(
    result: AcquisitionResult, frequencies_hz: tuple[float, float],
    decay_per_s: float, window_s: float, max_residual_rms: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    t, y = _short_complex_fid(result, window_s)
    a = np.column_stack((
        np.exp((2j * np.pi * frequencies_hz[0] - decay_per_s) * t),
        np.exp((2j * np.pi * frequencies_hz[1] - decay_per_s) * t),
        np.ones_like(t),
    ))
    condition = float(np.linalg.cond(a))
    if condition > 100 or not np.isfinite(condition):
        raise ValueError(f"two-line FID fit ill-conditioned: {condition:.1f}")
    coefficients, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    residual = y - a @ coefficients
    residual_rms = float(np.sqrt(np.mean(abs(residual) ** 2)))
    relative = float(np.linalg.norm(residual) /
                     max(np.linalg.norm(y - coefficients[2]), 1e-12))
    if residual_rms > max_residual_rms:
        raise ValueError(f"frozen two-line FID model fails: residual RMS {residual_rms:.3g} "
                         f"> independent ceiling {max_residual_rms:.3g}")
    # Conditional covariance of linear coefficients under measured complex
    # residual scatter.  Coherent residuals are separately flagged, never
    # relabeled as independent noise to claim precision.
    centered = np.column_stack((residual.real, residual.imag))
    point_cov = np.cov(centered.T, ddof=1)
    floor = max(float(np.mean(abs(y) ** 2)) * 1e-12, 1e-18)
    point_cov += floor * np.eye(2)
    n = len(t)
    real_design = np.block([[a.real, -a.imag], [a.imag, a.real]])
    inverse = np.linalg.pinv(real_design)
    data_cov = np.block([
        [point_cov[0, 0] * np.eye(n), point_cov[0, 1] * np.eye(n)],
        [point_cov[1, 0] * np.eye(n), point_cov[1, 1] * np.eye(n)],
    ])
    parameter_cov = inverse @ data_cov @ inverse.T
    index = [0, 1, 3, 4]
    coefficient_cov = parameter_cov[np.ix_(index, index)]
    lag_one = float(abs(np.vdot(residual[:-1], residual[1:])) /
                    max(np.vdot(residual, residual).real, 1e-12))
    return coefficients[:2], coefficient_cov, {
        "relative_residual": relative, "residual_rms": residual_rms,
        "max_residual_rms": max_residual_rms, "condition": condition,
        "residual_lag_one": lag_one,
    }


def _native_ready(
    anchors: Mapping[str, Any], timing_evidence: Mapping[str, Any],
    coupling: Mapping[str, Any] | None, *, require_absolute: bool,
) -> tuple[dict[str, Any], Any, dict[str, tuple[float, float]], dict[str, float], dict[str, float]]:
    if not timing_evidence.get("idle_gap_verified"):
        raise ValueError("coherent idle primitive unverified")
    if not timing_evidence.get("cross_channel_alignment_verified"):
        raise ValueError("H/P pulse alignment unverified")
    if not timing_evidence.get("cnot_verified"):
        raise ValueError("own low-level CNOT/CZ not independently qualified")
    if coupling is None or coupling.get("status") != "TARGET_REACHED":
        raise ValueError("signed effective J coupling not independently validated")
    if coupling.get("effective_j_hz") is None:
        raise ValueError("J sign/timing unavailable")
    a = _baseline(anchors)
    for key in ("max_sequence_us", "max_rf_us_per_acquisition"):
        if key not in anchors or not 0 < float(anchors[key]) <= 20_000_000:
            raise ValueError(f"measured per-acquisition {key} required")
    design, freqs, decay, noise = _evaluator(anchors, timing_evidence, require_absolute)
    return a, design, freqs, decay, noise


def _tomography_acquisitions(
    acquire_callback: Acquire,
    anchors: Mapping[str, Any],
    coupling: Mapping[str, Any],
    design: Any,
    line_frequencies_hz: Mapping[str, tuple[float, float]],
    line_decay_per_s: Mapping[str, float],
    residual_ceiling: Mapping[str, float],
    key_prefix: str,
    circuit_gates: Sequence[str],
    measured_keys: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    branch_coefficients = []
    branch_covariances = []
    diagnostics = []
    settings = _readout_settings()
    for branch in temporal_pps_branches():
        coefficients = []
        covariance = np.zeros((2 * len(design.measurement_labels),) * 2, dtype=float)
        for setting_index, (setting_name, readout_gates) in enumerate(settings):
            for channel, path in (("H", 0), ("P", 1)):
                compiler = _NativeCompiler(anchors, float(coupling["effective_j_hz"]))
                for gate in branch.gates:
                    compiler.gate(gate, f"pps_{branch.name}")
                for gate in circuit_gates:
                    compiler.gate(gate, "heldout_circuit")
                for gate in readout_gates:
                    compiler.gate(gate, "frozen_readout")
                key = f"{key_prefix}_{branch.name}_{setting_index:02d}_{channel}"
                request = _request(key, compiler.finish(), path, anchors)
                # Every branch starts from a new thermal preparation.  The
                # automatic server PPS is explicitly disabled in _request.
                result = _complete(acquire_callback, request)
                measured_keys.append(key)
                pair, pair_cov, info = _fit_two_lines(
                    result, line_frequencies_hz[channel],
                    line_decay_per_s[channel], float(anchors["coherent_window_s"]),
                    residual_ceiling[channel],
                )
                diagnostics.append({"key": key, "setting": setting_name, **info})
                coefficients.extend(pair)
                first = setting_index * 4 + (0 if channel == "H" else 2)
                indices = [first, first + 1, len(design.measurement_labels) + first,
                           len(design.measurement_labels) + first + 1]
                covariance[np.ix_(indices, indices)] = pair_cov
        branch_coefficients.append(np.asarray(coefficients, dtype=complex))
        branch_covariances.append(covariance)
    return branch_coefficients, branch_covariances, diagnostics


def _complex_matrix_json(matrix: np.ndarray) -> dict[str, Any]:
    return {"re": np.asarray(matrix).real.tolist(),
            "im": np.asarray(matrix).imag.tolist()}


def run_pps(
    acquire_callback: Acquire,
    anchors: Mapping[str, Any],
    timing_evidence: Mapping[str, Any],
    key_prefix: str = "pps",
    coupling: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute three native temporal branches and reconstruct their deviation."""

    try:
        _, design, freqs, decay, noise = _native_ready(anchors, timing_evidence, coupling,
                                                       require_absolute=False)
    except (KeyError, TypeError, ValueError) as exc:
        return _unsupported(str(exc))
    measured_keys: list[str] = []
    try:
        branches, covariances, diagnostics = _tomography_acquisitions(
            acquire_callback, anchors, coupling, design, freqs, decay, noise,
            key_prefix, (), measured_keys,
        )
        combined = sum(branches) / 3
        combined_covariance = sum(covariances) / 9
        reconstruction = reconstruct_deviation(combined, design,
                                               covariance=combined_covariance)
    except TaskUncertain:
        raise
    except Exception as exc:
        return {"status": "INCOMPLETE_DATA", "reason": f"{type(exc).__name__}: {exc}",
                "acquisitions": len(measured_keys), "measured_keys": measured_keys}
    delta = reconstruction["delta_rho"]
    target = pps_target_deviation()
    amplitude = float(np.trace(delta.conj().T @ target).real /
                      np.trace(target.conj().T @ target).real)
    shape_error = float(np.linalg.norm(delta - amplitude * target) /
                        max(np.linalg.norm(delta), 1e-12))
    reference_norm = anchors.get("pps_reference_signal_norm")
    preserved = (float(np.linalg.norm(delta) / float(reference_norm))
                 if reference_norm is not None and float(reference_norm) > 0 else None)
    return {
        "status": "EVALUATED" if preserved is not None else "PILOT_INCONCLUSIVE",
        "reason": ("three native temporal branches reconstructed under one frozen readout"
                   if preserved is not None else
                   "deviation reconstructed, but independent signal-retention reference missing"),
        "preparation_kind": "TEMPORAL_AVERAGED_EFFECTIVE_PPS",
        "acquisitions": len(measured_keys), "branch_count": 3,
        "measured_keys": measured_keys, "readout_diagnostics": diagnostics,
        "delta_rho": _complex_matrix_json(delta),
        "pauli_coefficients": reconstruction["pauli_coefficients"],
        "shape_error": shape_error, "target_amplitude_common_scale": amplitude,
        "relative_signal_to_frozen_reference": preserved,
        "tomography_rank": design.rank, "tomography_condition": design.condition,
        "fid_points_are_not_shots": True,
    }


def run_bell(
    acquire_callback: Acquire,
    anchors: Mapping[str, Any],
    timing_evidence: Mapping[str, Any],
    key_prefix: str = "bell",
    coupling: Mapping[str, Any] | None = None,
    labels: Sequence[str] = ("Phi+", "Phi-", "Psi+", "Psi-"),
) -> dict[str, Any]:
    """Held-out native Bell circuits and independent rank-15 tomography.

    The reconstruction receives coefficients and frozen readout only.  The
    mathematical Bell target enters *after* reconstruction, in bell_score.
    """

    try:
        _, design, freqs, decay, noise = _native_ready(anchors, timing_evidence, coupling,
                                                       require_absolute=True)
        for label in labels:
            bell_preparation_gates(label)
    except (KeyError, TypeError, ValueError) as exc:
        return _unsupported(str(exc))
    all_keys: list[str] = []
    states: dict[str, Any] = {}
    for label in labels:
        try:
            keys: list[str] = []
            branch_coefficients, branch_covariances, diagnostics = _tomography_acquisitions(
                acquire_callback, anchors, coupling, design, freqs, decay, noise,
                f"{key_prefix}_{label.replace('+', 'plus').replace('-', 'minus')}",
                bell_preparation_gates(label),
                keys,
            )
            all_keys.extend(keys)
            # The three branches are combined in the same calibrated signal
            # scale.  No branch is normalized to its own Bell target.
            combined = sum(branch_coefficients) / 3
            covariance = sum(branch_covariances) / 9
            independent = reconstruct_deviation(combined, design, covariance)
            state = reconstruct_full_density(
                independent["delta_rho"], float(anchors["polarization_scale"]),
                project_physical=True,
            )
            correction = float(state["projection_frobenius"])
            if correction > float(anchors["max_projection_frobenius"]):
                states[label] = {
                    "status": "NONIDENTIFIABLE",
                    "reason": "unprojected state too nonphysical for prespecified projection tolerance",
                    "projection_frobenius": correction,
                    "minimum_eigenvalue_unprojected": state["minimum_eigenvalue_unprojected"],
                    "measured_keys": keys,
                }
                continue
            # Target is introduced only here, after tomography is frozen.
            score = bell_score(state["rho_physical"], label)
            states[label] = {
                "status": "EVALUATED", "acquisitions": len(keys),
                "measured_keys": keys, "state_fidelity": score["state_fidelity"],
                "frobenius_error": score["frobenius_error"],
                "pauli_difference": score["pauli_difference"],
                "pauli_coefficients": independent["pauli_coefficients"],
                "rho_unprojected": _complex_matrix_json(state["rho_unprojected"]),
                "rho_physical": _complex_matrix_json(state["rho_physical"]),
                "projection_frobenius": correction,
                "minimum_eigenvalue_unprojected": state["minimum_eigenvalue_unprojected"],
                "readout_diagnostics": diagnostics,
            }
        except TaskUncertain:
            raise
        except Exception as exc:
            all_keys.extend(keys)
            states[label] = {"status": "INCOMPLETE_DATA",
                             "reason": f"{type(exc).__name__}: {exc}",
                             "acquisitions": len(keys), "measured_keys": keys}
            break
    success = bool(states) and all(row["status"] == "EVALUATED" for row in states.values())
    return {
        "status": "EVALUATED" if success else "INCOMPLETE_DATA",
        "acquisitions": len(all_keys), "measured_keys": all_keys,
        "states": states, "tomography_rank": design.rank,
        "tomography_condition": design.condition,
        "preparation_kind": "TEMPORAL_AVERAGED_EFFECTIVE_PPS",
        "fidelity_scope": "calibrated effective/full density per specified polarization scale; "
                          "Bell state fidelity is not CNOT process fidelity",
    }
