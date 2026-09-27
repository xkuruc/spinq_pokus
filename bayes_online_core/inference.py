"""Independent CPU inference for the SpinQ Bayesian online experiment.

The measured object is an exported *complex* FID.  A FID and a coefficient
extracted from it must never be entered as two independent observations.
The control likelihood uses one shared, calibrated readout gain per channel;
it never fits a fresh gain to each pulse width or phase.

Conventions: frequency in Hz, time in seconds, phase in radians internally.
The effective rotating-frame Hamiltonian for one pulse is
    K = detuning*Z/2 + Rabi*(cos(phi)*X + sin(phi)*Y)/2,
with U(t) = exp(-i*2*pi*K*t).  A common receiver phase fixes the gauge.
This reduced one-spin model is checked against held-out FIDs and is not a
claim that a coupled H/P sample is exactly one spin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import logsumexp


@dataclass(frozen=True)
class Candidate:
    channel: str
    width_us: float
    amplitude_pct: float
    phase_deg: float
    detuning_hz: float
    expected_wall_s: float
    acquisition_count: int = 1

    def __post_init__(self) -> None:
        vals = (self.width_us, self.amplitude_pct, self.phase_deg,
                self.detuning_hz, self.expected_wall_s)
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("Candidate fields must be finite")
        if self.width_us <= 0 or not 0 <= self.amplitude_pct <= 100:
            raise ValueError("Pulse width/amplitude outside command domain")
        if self.expected_wall_s <= 0 or self.acquisition_count < 1:
            raise ValueError("Total candidate time and acquisition count must be positive")

    @property
    def full_cost_s(self) -> float:
        return self.expected_wall_s * self.acquisition_count


@dataclass(frozen=True)
class ReadoutAnchor:
    """One shared gain and fixed spectral branch from independent anchor data.

    ``rabi_hz_per_pct`` is effective nutation frequency at 1 percent command
    amplitude. ``feature_noise_ri`` is a 2x2 covariance of one *complex*
    feature's real/imaginary coordinates, estimated from repeat acquisitions.
    Spectator branches are fitted together but are not silently interpreted
    as independent qubits.
    """

    gain: complex
    rabi_hz_per_pct: float
    fid_frequency_hz: float
    decay_s: float
    feature_noise_ri: np.ndarray = field(default_factory=lambda: np.eye(2) * 0.01)
    spectator_frequencies_hz: tuple[float, ...] = ()
    model_floor_fraction: float = 0.03
    coherent_window_s: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.gain.real) or not np.isfinite(self.gain.imag) or abs(self.gain) <= 0:
            raise ValueError("Anchor gain must be finite and nonzero")
        if self.rabi_hz_per_pct <= 0 or self.decay_s <= 0:
            raise ValueError("Rabi slope and decay time must be positive")
        if not np.isfinite(self.fid_frequency_hz):
            raise ValueError("FID frequency must be finite")
        _check_covariance(self.feature_noise_ri)
        if self.coherent_window_s is not None and self.coherent_window_s <= 0:
            raise ValueError("coherent_window_s must be positive")


@dataclass(frozen=True)
class ComplexFeature:
    value: complex
    covariance_ri: np.ndarray
    residual_rms: float
    n_points: int
    effective_points: float
    diagnostic: str = "OK"
    source_id: str | None = None

    def __post_init__(self) -> None:
        _check_covariance(self.covariance_ri)
        if not np.isfinite(self.value.real) or not np.isfinite(self.value.imag):
            raise ValueError("Complex feature must be finite")
        if self.n_points < 16 or self.effective_points <= 0:
            raise ValueError("Insufficient FID samples")


def _check_covariance(cov: np.ndarray) -> np.ndarray:
    mat = np.asarray(cov, dtype=float)
    if mat.shape != (2, 2) or not np.all(np.isfinite(mat)):
        raise ValueError("Expected finite real/imag 2x2 covariance")
    if not np.allclose(mat, mat.T, rtol=1e-8, atol=1e-12):
        raise ValueError("Covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(mat)) <= 0:
        raise ValueError("Covariance must be positive definite")
    return mat


def _noise_correlation_factor(residual: np.ndarray) -> float:
    """Conservative small-lag effective-sample adjustment for one FID."""
    centered = residual - np.mean(residual)
    power = np.real(np.vdot(centered, centered))
    if power <= 0:
        return 1.0
    length = min(64, len(residual) // 8)
    factor = 1.0
    for lag in range(1, length + 1):
        acf = float(np.real(np.vdot(centered[:-lag], centered[lag:])) / power)
        if acf <= 0:
            break
        factor += 2.0 * acf
    return float(min(max(factor, 1.0), max(len(residual) / 16, 1.0)))


def diagnose_pilot_signal(fids: np.ndarray, time_s: np.ndarray) -> dict:
    """Triage the first six diverse pilot FIDs before paying for repeats.

    An early, contiguous excess in a trace's complex envelope is compared
    with *that same trace's* late variability.  The tail is only a noise
    candidate: coherent late signal or drift can inflate the threshold and
    cause a conservative false negative.  This is a stop/go signal check,
    not an estimate of resonance, Rabi rate, or hardware calibration.
    """
    data = np.asarray(fids, dtype=complex)
    axis = np.asarray(time_s, dtype=float)
    if data.ndim != 2 or data.shape[0] < 6 or data.shape[1] < 128:
        raise ValueError("INCOMPLETE_DATA: signal triage needs six FIDs with >=128 points")
    if axis.shape != (data.shape[1],):
        raise ValueError("INCOMPLETE_DATA: signal triage needs one common FID time axis")
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(axis)):
        raise ValueError("INCOMPLETE_DATA: nonfinite FID or time samples")
    steps = np.diff(axis)
    if np.any(steps <= 0):
        raise ValueError("INCOMPLETE_DATA: FID time axis does not increase")
    dt = float(np.median(steps))
    if np.max(np.abs(steps - dt)) > max(1e-9, 0.01 * dt):
        raise ValueError("INCOMPLETE_DATA: FID time axis is not uniform")

    count, points = data.shape
    window = int(np.clip(round(0.0016 / dt), 8, min(64, points // 32)))
    early_windows = min(8, points // (8 * window))
    if early_windows < 2:
        raise ValueError("INCOMPLETE_DATA: FID is too short for early signal triage")
    tail_points = max(64, points // 8)
    threshold_ratio = 2.5
    threshold_coherence = 0.35
    traces = []
    for row in data:
        tail = row[-tail_points:]
        # Component-wise median/MAD resists occasional late spikes without
        # assuming that the tail is a known noise-only region.
        center = complex(np.median(tail.real), np.median(tail.imag))
        sigma_re = float(np.median(np.abs(tail.real - center.real)) / 0.6744897501960817)
        sigma_im = float(np.median(np.abs(tail.imag - center.imag)) / 0.6744897501960817)
        tail_rms = float(np.hypot(sigma_re, sigma_im))
        floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(row))))
        denominator = max(tail_rms, floor)
        window_rms = np.array([
            np.sqrt(np.mean(np.abs(row[i * window:(i + 1) * window] - center) ** 2))
            for i in range(early_windows)
        ])
        ratios = window_rms / denominator
        above = ratios >= threshold_ratio
        pairs = np.flatnonzero(above[:-1] & above[1:])
        # A true FID should start near acquisition onset.  A late isolated
        # burst, even within the first eighth of the export, is insufficient.
        candidate_pairs = pairs[pairs <= 2]
        coherence = []
        for start in candidate_pairs:
            segment = row[start * window:(start + 2) * window] - center
            power = float(np.vdot(segment, segment).real)
            coherence.append(float(abs(np.vdot(segment[:-1], segment[1:])) /
                                   max(power, np.finfo(float).tiny)))
        connected = bool(coherence and max(coherence) >= threshold_coherence)
        traces.append({
            "detected": connected,
            "early_tail_rms_ratio": float(np.max(ratios)),
            "early_lag_one_coherence": float(max(coherence, default=0.0)),
            "consecutive_early_windows": int(max(
                (len(group) for group in np.split(np.flatnonzero(above),
                                                   np.flatnonzero(np.diff(np.flatnonzero(above)) > 1) + 1)
                 if len(group)), default=0)),
            "tail_candidate_noise_rms": tail_rms,
        })
    detected = sum(trace["detected"] for trace in traces)
    return {
        "status": "PROCEED" if detected >= 2 else "NO_DETECTABLE_FID",
        "evidence": {
            "pilot_fids": count,
            "detectable_fids": detected,
            "required_detectable_fids": 2,
            "early_window_samples": window,
            "threshold_early_to_tail_rms": threshold_ratio,
            "threshold_lag_one_coherence": threshold_coherence,
            "trace": traces,
            "tail_caveat": "The late FID is a noise candidate, not certified noise-only; "
                           "late coherence or drift can make this check conservative.",
        },
    }


def detect_coherent_window(fids: np.ndarray, time_s: np.ndarray) -> dict:
    """Find the early *connected* signal interval, ignoring late spikes.

    The tail is a noise candidate, not automatically a clean noise-only
    region: its mean/RMS are used only for a robust activity threshold and
    the returned diagnostics expose a late excess that needs model review.
    """
    y = np.asarray(fids, dtype=complex)
    if y.ndim == 1:
        y = y[None, :]
    t = np.asarray(time_s, dtype=float)
    if y.ndim != 2 or t.shape != (y.shape[1],) or y.shape[1] < 64:
        raise ValueError("Expected repeat FIDs and one common time axis")
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or np.max(np.abs(np.diff(t) - dt)) > max(1e-9, 0.01 * dt):
        raise ValueError("FID time axis is not uniform")
    n = y.shape[1]
    tail = y[:, -max(64, n // 8):]
    center = np.mean(tail, axis=1)
    tail_power = np.mean(np.abs(tail - center[:, None]) ** 2, axis=1)
    noise_rms = float(np.median(np.sqrt(tail_power)))
    if not np.isfinite(noise_rms) or noise_rms <= 0:
        raise ValueError("PILOT_INCONCLUSIVE: tail noise cannot be estimated")
    smooth = min(max(16, int(round(0.0016 / dt))), max(16, n // 32))
    kernel = np.ones(smooth) / smooth
    envelope = np.median(np.stack([
        np.sqrt(np.convolve(np.abs(row - baseline) ** 2, kernel, mode="same"))
        for row, baseline in zip(y, center)]), axis=0)
    threshold = 2.5 * noise_rms
    early_limit = max(32, n // 4)
    early = np.flatnonzero(envelope[:early_limit] > threshold)
    if len(early) == 0 or int(early[0]) > max(16, 2 * smooth):
        raise ValueError("PILOT_INCONCLUSIVE: no connected early coherent FID")
    end = int(early[0])
    for index in early[1:]:
        if int(index) - end - 1 > smooth:
            break
        end = int(index)
    end = min(n, end + 1)
    if end < 32:
        raise ValueError("PILOT_INCONCLUSIVE: coherent FID is shorter than 32 samples")
    late_excess = bool(np.max(envelope[end + smooth:]) > 6 * noise_rms) if end + smooth < n else False
    return {"samples": end, "seconds": float(end * dt),
            "tail_noise_rms": noise_rms, "threshold_rms": threshold,
            "late_artifact_warning": late_excess,
            "first_sample_snr": float(envelope[0] / noise_rms)}


def estimate_anchor(repeated_fids: np.ndarray, time_s: np.ndarray,
                    widths_us: Sequence[float], amplitude_pct: Sequence[float] | float,
                    phases_deg: Sequence[float] | float = 0.0,
                    rabi_bounds_hz_per_pct: tuple[float, float] | None = None) -> ReadoutAnchor:
    """Estimate a *nominal* common readout/Rabi anchor from a paid pilot.

    This fit uses one complex gain for all widths.  Its result is only an
    initial prior/receiver gauge, not a ground truth for hidden detuning.
    Diverse widths and phase conventions are required; ambiguous Rabi aliases
    raise NONIDENTIFIABLE instead of returning an arbitrary oscillation.
    `time_s` may be one shared axis or a 2D array of equal per-FID axes.
    """
    data = np.asarray(repeated_fids, dtype=complex)
    if data.ndim != 2 or data.shape[0] < 5 or data.shape[1] < 64:
        raise ValueError("Pilot needs >=5 FIDs at >=3 distinct widths and >=64 time samples")
    count, n_points = data.shape
    if not np.all(np.isfinite(data)):
        raise ValueError("Pilot FIDs contain nonfinite samples")
    axis = np.asarray(time_s, dtype=float)
    if axis.ndim == 2:
        if axis.shape != data.shape or not np.allclose(axis, axis[0], rtol=1e-6, atol=1e-9):
            raise ValueError("Pilot axes differ; align using measured time bases first")
        axis = axis[0]
    if axis.ndim != 1 or len(axis) != n_points or np.any(np.diff(axis) <= 0):
        raise ValueError("Invalid pilot time axis")
    step = float(np.median(np.diff(axis)))
    if np.max(np.abs(np.diff(axis) - step)) > max(1e-9, 0.01 * step):
        raise ValueError("Pilot time axis is not uniform")
    widths = np.asarray(widths_us, dtype=float)
    amplitudes = np.broadcast_to(np.asarray(amplitude_pct, dtype=float), (count,))
    phases = np.broadcast_to(np.asarray(phases_deg, dtype=float), (count,))
    if widths.shape != (count,) or len(np.unique(widths)) < 3:
        raise ValueError("NONIDENTIFIABLE: pilot needs >=3 distinct pulse widths")
    if np.any(widths <= 0) or np.any(amplitudes <= 0) or np.any(amplitudes > 100):
        raise ValueError("Invalid pilot pulse command")
    coherence = detect_coherent_window(data, axis)
    active_n = int(coherence["samples"])
    data = data[:, :active_n]
    axis = axis[:active_n]
    n_points = active_n
    # The baseband frequency belongs to receiver/FID evolution, not to the
    # hidden transmitter offset.  This FFT is used only for a frozen basis.
    fft_n = int(2 ** np.ceil(np.log2(max(8 * n_points, 2048))))
    window = 0.5 + 0.5 * np.hanning(n_points)
    spectra = np.abs(np.fft.fft(data * window, n=fft_n, axis=1))
    spectrum = np.median(spectra, axis=0)
    frequencies = np.fft.fftfreq(fft_n, d=step)
    frequency = float(frequencies[int(np.argmax(spectrum))])
    demod = data * np.exp(-2j * np.pi * frequency * (axis - axis[0]))
    # Estimate coherence envelope using the median over all shots to avoid a
    # zero-crossing in any one Rabi response being mistaken for relaxation.
    bins = np.array_split(np.arange(n_points), max(4, min(20, n_points // 16)))
    bt = np.array([np.mean(axis[b] - axis[0]) for b in bins])
    envelope = np.array([np.median(np.abs(np.mean(demod[:, b], axis=1))) for b in bins])
    valid = envelope > max(np.median(envelope[-3:]) * 1.15, np.max(envelope) * 0.04)
    if np.sum(valid) >= 5:
        slope = float(np.polyfit(bt[valid], np.log(envelope[valid]), 1)[0])
        decay = float(np.clip(-1 / slope if slope < 0 else 20 * bt[-1],
                              max(bt[-1] / 4, 1e-6), max(20 * bt[-1], 1e-5)))
    else:
        decay = float(max(5 * bt[-1], 1e-5))
    basis = np.exp(-(axis - axis[0]) / decay +
                   2j * np.pi * frequency * (axis - axis[0]))
    coefficients = (data @ np.conj(basis)) / np.vdot(basis, basis).real
    normalized_width = widths * 1e-6 * amplitudes
    unique_width = np.unique(normalized_width)
    if len(unique_width) < 3:
        raise ValueError("NONIDENTIFIABLE: pulse area lacks diversity")
    if rabi_bounds_hz_per_pct is None:
        span = float(np.ptp(unique_width))
        spacing = float(np.min(np.diff(unique_width)))
        lower = 0.08 / span
        upper = 0.49 / spacing
    else:
        lower, upper = map(float, rabi_bounds_hz_per_pct)
    if not 0 < lower < upper or not np.isfinite(upper):
        raise ValueError("Invalid Rabi search bounds")

    def profile(rate: float) -> tuple[float, complex]:
        shape = -1j * np.exp(1j * np.deg2rad(phases)) * np.sin(
            2 * np.pi * rate * normalized_width)
        denom = float(np.vdot(shape, shape).real)
        if denom < 1e-10:
            return np.inf, 0j
        common_gain = complex(np.vdot(shape, coefficients) / denom)
        error = float(np.vdot(coefficients - common_gain * shape,
                              coefficients - common_gain * shape).real)
        return error, common_gain

    grid = np.linspace(lower, upper, 320)
    losses = np.array([profile(rate)[0] for rate in grid])
    minima = [i for i in range(1, len(grid) - 1)
              if losses[i] <= losses[i - 1] and losses[i] <= losses[i + 1]]
    minima.extend([0, len(grid) - 1])
    modes = []
    for i in minima:
        lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
        if hi > lo:
            optimized = minimize_scalar(lambda rate: profile(rate)[0], bounds=(lo, hi),
                                        method="bounded")
            modes.append((float(optimized.fun), float(optimized.x)))
    modes.sort()
    best_loss, rate = modes[0]
    relevant_aliases = [(loss, candidate_rate) for loss, candidate_rate in modes[1:]
                        if abs(candidate_rate - rate) > 0.08 * rate]
    signal_power = float(np.vdot(coefficients, coefficients).real)
    relative_fit = float(np.sqrt(best_loss / max(signal_power, 1e-30)))
    if relative_fit > 0.7:
        raise ValueError(f"PILOT_INCONCLUSIVE: common-gain Rabi model residual {relative_fit:.3f}")
    if relevant_aliases and relevant_aliases[0][0] < best_loss + max(0.04 * signal_power, 1e-10):
        raise ValueError("NONIDENTIFIABLE: multiple Rabi aliases fit pilot widths")
    _, gain = profile(rate)
    residuals = data - coefficients[:, None] * basis[None, :]
    residual_rms = float(np.sqrt(np.mean(np.abs(residuals) ** 2)))
    spectral_residual_fraction = float(np.linalg.norm(residuals) /
                                       max(np.linalg.norm(data), 1e-30))
    complex_var = float(np.mean(np.abs(residuals) ** 2) /
                        max(np.vdot(basis, basis).real, 1))
    # Repeats of an identical pulse area measure between-acquisition drift.
    repeat_differences = []
    for area in unique_width:
        group = coefficients[np.isclose(normalized_width, area, rtol=1e-7, atol=1e-12)]
        if len(group) > 1:
            repeat_differences.extend(group - np.mean(group))
    if repeat_differences:
        complex_var = max(complex_var, float(np.mean(np.abs(repeat_differences) ** 2)))
    feature_noise = np.eye(2) * max(complex_var / 2, 1e-12)
    return ReadoutAnchor(gain=gain, rabi_hz_per_pct=rate,
                         fid_frequency_hz=frequency, decay_s=decay,
                         feature_noise_ri=feature_noise,
                         coherent_window_s=coherence["seconds"],
                         diagnostics={"pilot_fids": count, "distinct_pulse_areas": len(unique_width),
                                      "rabi_relative_residual": relative_fit,
                                      "rabi_model_status": "RABI_MODEL_CHECK_REQUIRED"
                                      if relative_fit > 0.35 else "RABI_FIT_OK",
                                      "rabi_alias_count": len(relevant_aliases),
                                      "fid_residual_rms": residual_rms,
                                      "single_branch_residual_fraction": spectral_residual_fraction,
                                      "spectral_model_status": "MODEL_CHECK_REQUIRED"
                                      if spectral_residual_fraction > 0.35 else "EFFECTIVE_BRANCH_OK",
                                      "receiver_frequency_hz": frequency,
                                      "coherent_window": coherence,
                                      "transmitter_detuning_inferred_from_FID_carrier": False})


def extract_feature(fid: np.ndarray, time_s: np.ndarray,
                    anchor: ReadoutAnchor,
                    source_id: str | None = None) -> ComplexFeature:
    """Project one exported FID onto a frozen damped spectral basis.

    The first basis column is the target branch.  Other known multiplet
    columns remove spectator leakage.  Frequencies and decay are fixed by
    separate anchor data; a new peak is not selected after every pulse.
    Structured residuals are flagged and their covariance is inflated,
    rather than being called independent Gaussian noise.
    """
    y = np.asarray(fid, dtype=complex)
    t = np.asarray(time_s, dtype=float)
    if y.ndim != 1 or t.ndim != 1 or y.shape != t.shape or y.size < 32:
        raise ValueError("FID and time axis must be equal 1D arrays with >=32 points")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(t)):
        raise ValueError("FID/time axis contains nonfinite values")
    steps = np.diff(t)
    if np.any(steps <= 0) or np.max(np.abs(steps - np.median(steps))) > max(1e-9, 0.01 * np.median(steps)):
        raise ValueError("Nonuniform or unordered FID time axis")
    if anchor.coherent_window_s is not None:
        active_n = min(len(t), max(32, int(np.rint(anchor.coherent_window_s / np.median(steps)))))
        y = y[:active_n]
        t = t[:active_n]
    t = t - t[0]
    freqs = (anchor.fid_frequency_hz,) + tuple(anchor.spectator_frequencies_hz)
    basis = np.column_stack([np.exp(-t / anchor.decay_s + 2j * np.pi * f * t)
                             for f in freqs] + [np.ones_like(t)])
    condition = np.linalg.cond(basis)
    if condition > 1e7:
        raise ValueError("Frozen spectral branches are not identifiable on this FID axis")
    coefficients, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
    residual = y - basis @ coefficients
    corr = _noise_correlation_factor(residual)
    sigma2 = max(float(np.mean(np.abs(residual) ** 2)), np.finfo(float).tiny)
    gram_inv = np.linalg.inv(basis.conj().T @ basis)
    target_var_complex = sigma2 * float(np.real(gram_inv[0, 0])) * corr
    # The real/imag covariance is an effective covariance, not a count of
    # independent NMR shots.  Repeat-derived anchor noise is a lower bound.
    covariance = np.asarray(anchor.feature_noise_ri, dtype=float) + np.eye(2) * target_var_complex / 2
    rms = float(np.sqrt(sigma2))
    expected_noise = float(np.sqrt(np.trace(anchor.feature_noise_ri)))
    diagnostic = "MODEL_CHECK_REQUIRED" if rms > max(6 * expected_noise, 0.35 * abs(coefficients[0])) else "OK"
    return ComplexFeature(complex(coefficients[0]), covariance, rms, y.size,
                          float(y.size / corr), diagnostic, source_id)


def _bloch_transverse(candidate: Candidate, theta: np.ndarray,
                      anchor: ReadoutAnchor) -> np.ndarray:
    """z-polarized input evolved through a finite detuned physical pulse."""
    x = np.asarray(theta, dtype=float)
    if x.shape[-1] != 3:
        raise ValueError("Parameter order is [hidden_df_hz, rf_scale, hidden_phase_rad]")
    detuning = candidate.detuning_hz + x[..., 0]
    rabi = anchor.rabi_hz_per_pct * candidate.amplitude_pct * x[..., 1]
    norm = np.hypot(detuning, rabi)
    safe_norm = np.maximum(norm, np.finfo(float).tiny)
    phase = np.deg2rad(candidate.phase_deg) + x[..., 2]
    nx = rabi * np.cos(phase) / safe_norm
    ny = rabi * np.sin(phase) / safe_norm
    nz = detuning / safe_norm
    angle = 2 * np.pi * norm * candidate.width_us * 1e-6
    mx = ny * np.sin(angle) + nx * nz * (1 - np.cos(angle))
    my = -nx * np.sin(angle) + ny * nz * (1 - np.cos(angle))
    return mx + 1j * my


def predict_feature(candidate: Candidate, theta: np.ndarray,
                    anchor: ReadoutAnchor) -> np.ndarray:
    return anchor.gain * _bloch_transverse(candidate, theta, anchor)


def _combined_covariance(feature: ComplexFeature, anchor: ReadoutAnchor) -> np.ndarray:
    floor = (anchor.model_floor_fraction * abs(anchor.gain)) ** 2 / 2
    return _check_covariance(np.asarray(feature.covariance_ri) + np.eye(2) * floor)


def _prediction_matrix(z: np.ndarray) -> np.ndarray:
    return np.stack((np.real(z), np.imag(z)), axis=-1)


def _weighted_mean_cov(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.sum(weights[:, None] * values, axis=0)
    d = values - mu
    cov = (weights[:, None] * d).T @ d
    return mu, cov


def _wrap_phase(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2 * np.pi) - np.pi


def pulse_design_identifiability(candidates: Sequence[Candidate],
                                 anchor: ReadoutAnchor, prior_bounds: np.ndarray,
                                 theta: np.ndarray) -> dict:
    """Local pulse-response rank in dimensionless prior-width coordinates.

    This separates physical transmitter detuning from the frozen baseband
    receiver frequency.  A weak/degenerate pulse plan cannot silently claim
    a frequency calibration merely because its FID has a sharp FFT peak.
    """
    bounds = np.asarray(prior_bounds, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if bounds.shape != (3, 2) or theta.shape != (3,):
        raise ValueError("Expected three physical parameters and (3,2) bounds")
    if not candidates:
        return {"rank": 0, "condition_number": float("inf"),
                "singular_values": [0.0, 0.0, 0.0], "status": "NONIDENTIFIABLE"}
    steps = np.maximum((bounds[:, 1] - bounds[:, 0]) * 1e-4, 1e-9)
    jac = np.empty((2 * len(candidates), 3))
    for j in range(3):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += steps[j]
        minus[j] -= steps[j]
        prediction = np.array([(predict_feature(c, plus, anchor) -
                                predict_feature(c, minus, anchor)) /
                               (2 * steps[j]) for c in candidates])
        jac[0::2, j] = np.real(prediction) * (bounds[j, 1] - bounds[j, 0])
        jac[1::2, j] = np.imag(prediction) * (bounds[j, 1] - bounds[j, 0])
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > max(1e-8, singular[0] * 1e-3)))
    condition = float(np.inf if singular[-1] <= 1e-12 else singular[0] / singular[-1])
    return {"rank": rank, "condition_number": condition,
            "singular_values": singular.tolist(),
            "status": "IDENTIFIABLE" if rank == 3 and condition < 1e3
            else "NONIDENTIFIABLE"}


class OnlineLearner:
    """SMC posterior over one H or P control channel, updated once per FID.

    Use separate instances for H and P.  `sample_joint_calibration` carries
    their uncertainty into coupling/gate design instead of plugging in means.
    The receiver gain/frequency basis is shared across all observations in an
    instance.  A model mismatch flag must stop automatic control deployment.
    """

    def __init__(self, prior_bounds: np.ndarray, anchor: ReadoutAnchor,
                 n_particles: int = 768, seed: int | None = None,
                 resample_fraction: float = 0.55) -> None:
        bounds = np.asarray(prior_bounds, dtype=float)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("Expected finite prior_bounds shape (3,2)")
        if bounds[1, 0] <= 0 or bounds[2, 0] < -np.pi or bounds[2, 1] > np.pi:
            raise ValueError("RF scale must be positive and phase within [-pi,pi]")
        if not 128 <= n_particles <= 16384 or not 0 < resample_fraction < 1:
            raise ValueError("Invalid SMC settings")
        self.bounds = bounds
        self.anchor = anchor
        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, 3))
        self.log_weights = np.full(n_particles, -np.log(n_particles), dtype=float)
        self.resample_fraction = resample_fraction
        self.history: list[dict] = []
        self.status = "PRIOR_ONLY"
        self._noise_floor_cov = np.asarray(anchor.feature_noise_ri, dtype=float) + \
            np.eye(2) * (anchor.model_floor_fraction * abs(anchor.gain)) ** 2 / 2
        self._last_noise_cov = self._noise_floor_cov.copy()
        self.channel: str | None = None
        self._seen_source_ids: set[str] = set()
        self.model_mismatch_count = int(
            anchor.diagnostics.get("spectral_model_status") == "MODEL_CHECK_REQUIRED" or
            anchor.diagnostics.get("rabi_model_status") == "RABI_MODEL_CHECK_REQUIRED")

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.log_weights)

    @property
    def ess(self) -> float:
        return float(1 / np.sum(np.square(self.weights)))

    def _resample(self) -> None:
        n = len(self.particles)
        w = self.weights
        positions = (self.rng.random() + np.arange(n)) / n
        index = np.searchsorted(np.cumsum(w), positions, side="right")
        index = np.minimum(index, n - 1)
        phase_mean = np.angle(np.sum(w * np.exp(1j * self.particles[:, 2])))
        local = self.particles.copy()
        local[:, 2] = phase_mean + _wrap_phase(local[:, 2] - phase_mean)
        mu, cov = _weighted_mean_cov(local, w)
        a = 0.98
        h = np.sqrt(1 - a * a)
        cov += np.diag(np.square(self.bounds[:, 1] - self.bounds[:, 0]) * 1e-12)
        noise = self.rng.multivariate_normal(np.zeros(3), h * h * cov, size=n)
        moved = a * local[index] + (1 - a) * mu + noise
        moved[:, 2] = _wrap_phase(moved[:, 2])
        # No invisible clipping of physical commands.  Particle support is
        # reflected into the prior domain; boundary pileup is a diagnostic.
        for j in range(3):
            lo, hi = self.bounds[j]
            width = hi - lo
            moved[:, j] = lo + np.abs((moved[:, j] - lo + width) % (2 * width) - width)
        self.particles = moved
        self.log_weights[:] = -np.log(n)

    def update(self, candidate: Candidate, feature: ComplexFeature) -> dict:
        if self.channel is not None and candidate.channel != self.channel:
            raise ValueError("Separate H/P learners are required")
        if feature.source_id is not None and feature.source_id in self._seen_source_ids:
            raise ValueError("The same FID cannot be used twice as independent evidence")
        predicted = _prediction_matrix(predict_feature(candidate, self.particles, self.anchor))
        observed = np.array([feature.value.real, feature.value.imag])
        individual_covariance = _combined_covariance(feature, self.anchor)
        covariance = _check_covariance(0.75 * individual_covariance +
                                       0.25 * self._last_noise_cov)
        inverse = np.linalg.inv(covariance)
        difference = predicted - observed
        chi2 = np.einsum("ni,ij,nj->n", difference, inverse, difference)
        loglike = -0.5 * chi2 - 0.5 * np.linalg.slogdet(2 * np.pi * covariance)[1]
        old = self.log_weights.copy()
        evidence = float(logsumexp(old + loglike))
        self.log_weights = old + loglike - evidence
        ess_before = self.ess
        post_mean = np.sum(self.weights[:, None] * predicted, axis=0)
        post_delta = observed - post_mean
        predictive_chi2 = float(post_delta @ inverse @ post_delta)
        best_chi2 = float(np.min(chi2))
        mismatch = feature.diagnostic != "OK" or best_chi2 > 25
        row = {"candidate": candidate, "value": feature.value,
               "log_evidence": evidence, "ess_before_resampling": ess_before,
               "predictive_chi2": predictive_chi2, "best_particle_chi2": best_chi2,
               "feature_diagnostic": feature.diagnostic,
               "status": "MODEL_CHECK_REQUIRED" if mismatch else "UPDATED"}
        if ess_before < self.resample_fraction * len(self.particles):
            self._resample()
            row["resampled"] = True
        else:
            row["resampled"] = False
        if not mismatch and predictive_chi2 < 9:
            proposed = 0.9 * self._last_noise_cov + 0.1 * individual_covariance
            # Only two shared noise parameters (real/imag scale) drift, within
            # a bounded factor of the paid anchor.  A bad control response is
            # never fit away as arbitrary observation noise.
            base_diag = np.diag(self._noise_floor_cov)
            proposed_diag = np.clip(np.diag(proposed), 0.5 * base_diag,
                                     4.0 * base_diag)
            self._last_noise_cov = np.diag(proposed_diag)
        row["learned_noise_covariance_ri"] = self._last_noise_cov.tolist()
        self.history.append(row)
        self.channel = candidate.channel
        if feature.source_id is not None:
            self._seen_source_ids.add(feature.source_id)
        self.model_mismatch_count += int(mismatch)
        self.status = "MODEL_CHECK_REQUIRED" if self.model_mismatch_count else "LEARNING"
        return row

    def design_scores(self, candidates: Sequence[Candidate]) -> np.ndarray:
        """Approximate Gaussian information gain divided by full wall time."""
        if not candidates:
            raise ValueError("Need at least one realizable candidate")
        weights = self.weights
        noise = self._last_noise_cov
        inverse_noise = np.linalg.inv(noise)
        scores = []
        for candidate in candidates:
            predictions = _prediction_matrix(predict_feature(candidate, self.particles, self.anchor))
            _, cov = _weighted_mean_cov(predictions, weights)
            # 0.5 log det(I + Sigma_noise^{-1} Sigma_prediction).
            information = max(0.0, 0.5 * np.linalg.slogdet(np.eye(2) + inverse_noise @ cov)[1])
            scores.append(float(information / candidate.full_cost_s))
        return np.asarray(scores)

    def choose(self, candidates: Sequence[Candidate], mode: str = "adaptive") -> Candidate:
        if mode not in {"adaptive", "fixed"}:
            raise ValueError("mode must be 'adaptive' or 'fixed'")
        if not candidates:
            raise ValueError("Candidate list is empty")
        if mode == "fixed":
            used = [entry["candidate"] for entry in self.history]
            for candidate in candidates:
                if used.count(candidate) == 0:
                    return candidate
            return candidates[len(used) % len(candidates)]
        scores = self.design_scores(candidates)
        # A soft reuse penalty prevents a high-gain point from being repeated
        # forever, while permitting a repeat when it remains informative.
        counts = np.array([sum(row["candidate"] == c for row in self.history)
                           for c in candidates])
        return candidates[int(np.argmax(scores / np.sqrt(1 + counts)))]

    def summary(self) -> dict:
        w = self.weights
        p = self.particles
        mean = np.sum(w[:, None] * p, axis=0)
        mean[2] = np.angle(np.sum(w * np.exp(1j * p[:, 2])))
        local = p.copy()
        local[:, 2] = mean[2] + _wrap_phase(p[:, 2] - mean[2])
        cov = (w[:, None] * (local - mean)).T @ (local - mean)
        intervals = []
        for j in range(3):
            order = np.argsort(local[:, j])
            cumul = np.cumsum(w[order])
            intervals.append([float(np.interp(q, cumul, local[order, j]))
                              for q in (0.025, 0.975)])
        width = np.array([hi - lo for lo, hi in intervals])
        prior_width = self.bounds[:, 1] - self.bounds[:, 0]
        design = pulse_design_identifiability([row["candidate"] for row in self.history],
                                              self.anchor, self.bounds, mean)
        nonidentifiable = bool(np.any(width > 0.85 * prior_width) or
                               design["status"] == "NONIDENTIFIABLE")
        edge_fraction = float(np.sum(w * np.any(
            (p - self.bounds[:, 0] < 0.025 * prior_width) |
            (self.bounds[:, 1] - p < 0.025 * prior_width), axis=1)))
        status = "PRIOR_ONLY" if not self.history else \
            "MODEL_CHECK_REQUIRED" if self.model_mismatch_count else \
            "NONIDENTIFIABLE" if nonidentifiable else self.status
        return {"mean": mean.tolist(), "covariance": cov.tolist(),
                "interval_95": intervals, "ess": self.ess,
                "acquisitions": int(sum(row["candidate"].acquisition_count for row in self.history)),
                "status": status,
                "prior_width_fraction_95": (width / prior_width).tolist(),
                "boundary_mass": edge_fraction,
                "pulse_design_identifiability": design,
                "model_mismatch_count": self.model_mismatch_count,
                "learned_noise_covariance_ri": self._last_noise_cov.tolist()}

    def sample(self, count: int) -> np.ndarray:
        if count < 1:
            raise ValueError("sample count must be positive")
        return self.particles[self.rng.choice(len(self.particles), size=count, p=self.weights)].copy()

    def posterior_predictive_fid(self, candidate: Candidate, time_s: np.ndarray) -> dict:
        """Denoised predictive FID with posterior uncertainty, not a new datum."""
        t = np.asarray(time_s, dtype=float)
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("Time axis must increase")
        amplitude = predict_feature(candidate, self.particles, self.anchor)
        basis = np.exp(-(t - t[0]) / self.anchor.decay_s +
                       2j * np.pi * self.anchor.fid_frequency_hz * (t - t[0]))
        mean_amplitude = np.sum(self.weights * amplitude)
        uncertainty = np.sum(self.weights * np.abs(amplitude - mean_amplitude) ** 2)
        return {"mean": mean_amplitude * basis,
                "model_std": np.sqrt(uncertainty) * np.abs(basis),
                "noise_std": float(np.sqrt(np.trace(self._last_noise_cov))),
                "source": "posterior_predictive_from_training_FID_only"}


def sample_joint_calibration(h: OnlineLearner, p: OnlineLearner,
                             count: int) -> np.ndarray:
    """Independent channel posterior draws, columns H[3] then P[3]."""
    if count < 1:
        raise ValueError("count must be positive")
    return np.column_stack((h.sample(count), p.sample(count)))


@dataclass(frozen=True)
class ClassicalFit:
    parameters: np.ndarray
    covariance: np.ndarray
    nll: float
    status: str
    starts: int
    condition_number: float


def fit_classical(candidates: Sequence[Candidate], features: Sequence[ComplexFeature],
                  anchor: ReadoutAnchor, prior_bounds: np.ndarray,
                  starts: int = 24, seed: int | None = None) -> ClassicalFit:
    """Multistart complex Gaussian NLLS using exactly the Bayesian forward model.

    No per-acquisition complex amplitude, phase or gain is fitted.  All FIDs
    contribute one extracted feature and its real/imag covariance once.
    """
    if len(candidates) != len(features) or len(features) < 3:
        raise ValueError("Need paired candidates/features and >=3 acquisitions")
    bounds = np.asarray(prior_bounds, dtype=float)
    if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("prior_bounds must have shape (3,2)")
    if starts < 1:
        raise ValueError("starts must be positive")
    root_inv = [np.linalg.inv(np.linalg.cholesky(_combined_covariance(f, anchor)))
                for f in features]
    observations = [np.array([f.value.real, f.value.imag]) for f in features]

    def residual(theta: np.ndarray) -> np.ndarray:
        pieces = []
        for c, y, whiten in zip(candidates, observations, root_inv):
            z = complex(predict_feature(c, theta, anchor))
            pieces.append(whiten @ (np.array([z.real, z.imag]) - y))
        return np.concatenate(pieces)

    rng = np.random.default_rng(seed)
    starts_array = np.vstack((bounds.mean(axis=1),
                              rng.uniform(bounds[:, 0], bounds[:, 1], size=(max(0, starts - 1), 3))))
    best = None
    for initial in starts_array:
        fit = least_squares(residual, initial, bounds=(bounds[:, 0], bounds[:, 1]),
                            max_nfev=1000, x_scale="jac")
        if best is None or np.dot(fit.fun, fit.fun) < np.dot(best.fun, best.fun):
            best = fit
    assert best is not None
    singular = np.linalg.svd(best.jac, compute_uv=False)
    condition = float(np.inf if singular[-1] <= 1e-12 else singular[0] / singular[-1])
    covariance = np.linalg.pinv(best.jac.T @ best.jac)
    proximity = np.minimum(best.x - bounds[:, 0], bounds[:, 1] - best.x)
    at_bound = bool(np.any(proximity < 0.005 * (bounds[:, 1] - bounds[:, 0])))
    design = pulse_design_identifiability(candidates, anchor, bounds, best.x)
    status = "FIT_OK" if best.success and condition < 1e6 and not at_bound and \
        design["status"] == "IDENTIFIABLE" else "NONIDENTIFIABLE"
    if (anchor.diagnostics.get("spectral_model_status") == "MODEL_CHECK_REQUIRED" or
            anchor.diagnostics.get("rabi_model_status") == "RABI_MODEL_CHECK_REQUIRED"):
        status = "MODEL_CHECK_REQUIRED"
    return ClassicalFit(best.x, covariance, float(0.5 * np.dot(best.fun, best.fun)),
                        status, starts, condition)


def numeric_selfcheck() -> dict:
    """Short deterministic *synthetic* algebra check; never hardware evidence."""
    rng = np.random.default_rng(47)
    t = np.arange(2048, dtype=float) / 100_000.0
    nominal = ReadoutAnchor(1.4 + 0.5j, 1000.0, 390.625, 0.018,
                            np.eye(2) * 1e-5)
    widths = np.array([10, 20, 30, 40, 50, 60, 80, 100, 20, 50], dtype=float)
    pilot = []
    for width in widths:
        c = Candidate("H", width, 5.0, 0.0, 0.0, 10.0)
        amp = complex(predict_feature(c, np.array([0.0, 1.0, 0.0]), nominal))
        noise = 0.002 * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))
        pilot.append(amp * np.exp(-t / nominal.decay_s +
                                  2j * np.pi * nominal.fid_frequency_hz * t) + noise)
    anchor = estimate_anchor(np.asarray(pilot), t, widths, 5.0)
    if abs(anchor.rabi_hz_per_pct / nominal.rabi_hz_per_pct - 1) > 0.1:
        raise AssertionError("Synthetic anchor Rabi slope was not recovered")
    bounds = np.array([[-300.0, 300.0], [0.8, 1.2], [-0.5, 0.5]])
    hidden = np.array([100.0, 1.06, 0.14])
    learner = OnlineLearner(bounds, anchor, n_particles=512, seed=13)
    candidates = [Candidate("H", float(w), 5.0, float(phase), float(det), 10.0)
                  for w, det, phase in zip(
                      [20, 40, 60, 80, 100, 120, 140, 160],
                      [-300, 0, 300, -200, 200, 0, 250, -250],
                      [0, 90, 0, 90, 0, 90, 0, 90])]
    features = []
    for index, c in enumerate(candidates):
        amp = complex(predict_feature(c, hidden, anchor))
        y = amp * np.exp(-t / anchor.decay_s +
                         2j * np.pi * anchor.fid_frequency_hz * t)
        y += 0.002 * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))
        feature = extract_feature(y, t, anchor, source_id=f"synthetic-{index}")
        features.append(feature)
        learner.update(c, feature)
    classical = fit_classical(candidates, features, anchor, bounds, starts=12, seed=19)
    if classical.status != "FIT_OK" or abs(classical.parameters[1] - hidden[1]) > 0.03:
        raise AssertionError("Synthetic multistart physical fit failed")
    posterior = learner.summary()
    if posterior["status"] == "MODEL_CHECK_REQUIRED" or abs(posterior["mean"][1] - hidden[1]) > 0.08:
        raise AssertionError("Synthetic SMC physical inference failed")
    return {"kind": "SYNTHETIC_NUMERIC_SELFCHECK_ONLY", "status": "PASS",
            "anchor_rabi_hz_per_pct": anchor.rabi_hz_per_pct,
            "classical_parameters": classical.parameters.tolist(),
            "smc_mean": posterior["mean"],
            "smc_ess": posterior["ess"]}


if __name__ == "__main__":
    import json
    print(json.dumps(numeric_selfcheck(), indent=2))
