"""Pilot-frozen, local complex-FID observables for experiment 01.

The input is the exported Re+iIm FID, not raw ADC.  The vendor's FFT, fit,
frequency estimate and matrix fields are intentionally absent from this API.
All frequency bands and the receiver gauge come from independent pilot FIDs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import find_peaks, peak_widths, windows

from spinq_local.core import RawFIDRecord
from spinq_local.signal import validate_axis


class SignalIdentificationError(ValueError):
    """The measured pilot does not support a frozen component identity."""


@dataclass(frozen=True)
class PilotSignalModel:
    """Independent-pilot choices frozen before any A--E method is compared."""

    sample_hz: float
    bands_hz: tuple[tuple[float, float], ...]
    primary_component_index: int
    component_frequencies_hz: tuple[float, ...]
    component_decay_per_s: tuple[float, ...]
    component_weights: tuple[complex, ...]
    reference_frequency_hz: float
    reference_coefficient: complex
    baseline_complex: complex
    receiver_phase_rad: float
    noise_covariance_re_im: np.ndarray
    lag_one_correlation: float
    between_acquisition_covariance_re_im: np.ndarray
    feature_windows_s: tuple[tuple[float, float], ...]
    feature_covariance_re_im: np.ndarray
    pilot_task_ids: tuple[str, ...]
    pilot_residual_rms: float
    status: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def component_offsets_hz(self) -> tuple[float, ...]:
        return tuple(f - self.reference_frequency_hz for f in self.component_frequencies_hz)

    def to_dict(self) -> dict:
        return {
            "sample_hz": self.sample_hz,
            "bands_hz": [list(b) for b in self.bands_hz],
            "primary_component_index": self.primary_component_index,
            "component_frequencies_hz": list(self.component_frequencies_hz),
            "component_offsets_hz": list(self.component_offsets_hz),
            "component_decay_per_s": list(self.component_decay_per_s),
            "component_weights_re_im": [[z.real, z.imag] for z in self.component_weights],
            "reference_frequency_hz": self.reference_frequency_hz,
            "reference_coefficient_re_im": [self.reference_coefficient.real,
                                            self.reference_coefficient.imag],
            "baseline_re_im": [self.baseline_complex.real, self.baseline_complex.imag],
            "receiver_phase_rad": self.receiver_phase_rad,
            "receiver_gauge": "reference primary coefficient of independent pilot; TX/RX global phase unresolved",
            "noise_covariance_re_im": self.noise_covariance_re_im.tolist(),
            "lag_one_correlation": self.lag_one_correlation,
            "between_acquisition_covariance_re_im":
                self.between_acquisition_covariance_re_im.tolist(),
            "feature_windows_s": [list(w) for w in self.feature_windows_s],
            "feature_covariance_re_im": self.feature_covariance_re_im.tolist(),
            "pilot_task_ids": list(self.pilot_task_ids),
            "pilot_residual_rms": self.pilot_residual_rms,
            "status": self.status,
            "diagnostics": self.diagnostics,
            "vendor_result_used": False,
            "physical_adc_clock_independently_verified": False,
        }


@dataclass(frozen=True)
class SignalEstimate:
    """Compact likelihood observation and separate multiplet-fit diagnostics."""

    features: np.ndarray
    feature_covariance_re_im: np.ndarray
    selected_coefficient: complex
    relative_coefficient: complex
    relative_phase_rad: float
    frequency_hz: float
    frequency_se_hz_conditional: float | None
    component_frequencies_hz: tuple[float, ...]
    component_decay_per_s: tuple[float, ...]
    residual_rms: float
    relative_residual_rms: float
    status: str
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "features_re_im": [[z.real, z.imag] for z in self.features],
            "feature_covariance_re_im": self.feature_covariance_re_im.tolist(),
            "selected_coefficient_re_im": [self.selected_coefficient.real,
                                            self.selected_coefficient.imag],
            "relative_coefficient_re_im": [self.relative_coefficient.real,
                                            self.relative_coefficient.imag],
            "relative_phase_rad": self.relative_phase_rad,
            "frequency_hz": self.frequency_hz,
            "frequency_se_hz_conditional": self.frequency_se_hz_conditional,
            "component_frequencies_hz": list(self.component_frequencies_hz),
            "component_decay_per_s": list(self.component_decay_per_s),
            "residual_rms": self.residual_rms,
            "relative_residual_rms": self.relative_residual_rms,
            "status": self.status,
            "diagnostics": self.diagnostics,
            "uncertainty_scope": "conditional fit SE; repeated-pilot covariance supplies likelihood noise",
            "vendor_result_used": False,
        }


@dataclass(frozen=True)
class RabiPilotEstimate:
    """Complex signed Rabi initialization, separate from the FID frequency fit."""

    period_us: float | None
    t90_us: float | None
    t90_interval_us: tuple[float, float] | None
    signed_complex_r2: float
    receiver_offset: complex
    receiver_sine_gain: complex
    receiver_cosine_gain: complex
    status: str
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "period_us": self.period_us, "t90_us": self.t90_us,
            "t90_interval_us": list(self.t90_interval_us) if self.t90_interval_us else None,
            "signed_complex_r2": self.signed_complex_r2,
            "receiver_offset_re_im": [self.receiver_offset.real, self.receiver_offset.imag],
            "receiver_sine_gain_re_im": [self.receiver_sine_gain.real,
                                         self.receiver_sine_gain.imag],
            "receiver_cosine_gain_re_im": [self.receiver_cosine_gain.real,
                                           self.receiver_cosine_gain.imag],
            "status": self.status,
            "diagnostics": self.diagnostics,
            "scope": "complex FID mode amplitude versus pulse width; t90=period/4 under sinusoidal calibration model",
        }


def _positive_covariance(covariance: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    covariance = np.asarray(covariance, float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.all(np.isfinite(covariance)):
        raise SignalIdentificationError("Noise covariance is nonfinite or not square")
    covariance = (covariance + covariance.T) / 2
    eigenvalues, basis = np.linalg.eigh(covariance)
    scale = max(float(np.trace(covariance) / len(covariance)), floor)
    eigenvalues = np.maximum(eigenvalues, scale * 1e-5)
    return (basis * eigenvalues) @ basis.T


def _repeat_noise(records: Sequence[RawFIDRecord]) -> tuple[np.ndarray, float]:
    if len(records) < 3:
        raise SignalIdentificationError("At least three independent same-setting pilot FIDs are required")
    n = min(len(r.re) for r in records)
    rows = np.stack([r.fid[:n] for r in records])
    centered = rows - rows.mean(axis=0)
    components = np.column_stack((centered.real.ravel(), centered.imag.ravel()))
    covariance = _positive_covariance(np.cov(components, rowvar=False))
    a = centered[:, :-1].ravel()
    b = centered[:, 1:].ravel()
    lag = float(np.real(np.vdot(a, b)) / max(np.vdot(a, a).real, 1e-30))
    return covariance, float(np.clip(lag, -0.98, 0.98))


def _validate_repeats(records: Sequence[RawFIDRecord]) -> tuple[float, int]:
    if len(records) < 3:
        raise SignalIdentificationError("Three independent pilot acquisitions are needed")
    task_ids = [r.task_id for r in records]
    if len(set(task_ids)) != len(task_ids):
        raise SignalIdentificationError("Pilot repetition reuses a task ID")
    axes = [validate_axis(r) for r in records]
    sample_hz = axes[0].sample_hz
    if any(a.sample_hz != sample_hz for a in axes):
        raise SignalIdentificationError("Pilot repeats have different sampling frequencies")
    minimum = min(len(r.re) for r in records)
    if any(len(r.re) - minimum > 1 for r in records):
        raise SignalIdentificationError("Pilot repeats have different FID lengths")
    # The SDK may omit its final chart point, but distinct pulse payloads are
    # never silently pooled as noise repetitions.
    settings = [r.parameters_sent.get("pulse") for r in records]
    if any(setting != settings[0] for setting in settings[1:]):
        raise SignalIdentificationError("Pilot noise records have different pulse settings")
    return sample_hz, minimum


def _discover_bands(mean_fid: np.ndarray, sample_hz: float, noise_cov: np.ndarray,
                    centered_repeats: np.ndarray, *, max_components: int) -> tuple[tuple[tuple[float, float], ...], dict]:
    n = len(mean_fid)
    noise_rms = math.sqrt(float(np.trace(noise_cov)))
    # Exclude a measured receiver offset *before* finding the active interval.
    # Otherwise an offset can keep the whole chart above the noise threshold,
    # and a full-length Hann window will suppress a rapidly decaying FID.
    baseline = complex(np.mean(mean_fid[-max(32, n // 12):]))
    centered_signal = mean_fid - baseline
    smooth_n = max(16, min(n // 20, int(sample_hz / 100)))
    envelope = np.sqrt(np.convolve(np.abs(centered_signal) ** 2,
                                   np.ones(smooth_n) / smooth_n, mode="same"))
    active = np.flatnonzero(envelope > 3 * noise_rms)
    n_active = min(n, max(256, int(active[-1] + 1) if len(active) else 0))
    if n_active < 256 or float(np.max(envelope)) < 6 * noise_rms:
        raise SignalIdentificationError("Pilot signal is below independent-repeat noise threshold")
    # A Hann taper suppresses the spectral sidelobes of a strong mode, which
    # otherwise masquerade as extra weak components in high-SNR FIDs.
    taper = windows.hann(n_active)
    # A complex constant is part of the subsequent variable-projection model.
    nfft = 1 << int(math.ceil(math.log2(max(1024, 4 * n_active))))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(centered_signal[:n_active] * taper, n=nfft)))
    frequency = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / sample_hz))
    residual_spectra = [np.abs(np.fft.fftshift(np.fft.fft(row[:n_active] * taper, n=nfft)))
                        for row in centered_repeats]
    noise_spectrum = float(np.median(np.stack(residual_spectra)))
    threshold = max(7 * noise_spectrum, 0.02 * float(np.max(spectrum)), 1e-12)
    peaks, properties = find_peaks(spectrum, prominence=threshold / 2, height=threshold)
    if not len(peaks):
        raise SignalIdentificationError("No pilot spectral component rises above repeat-derived noise")
    order = sorted(peaks, key=lambda p: -float(spectrum[p]))
    selected: list[tuple[int, float]] = []
    resolution = sample_hz / n_active
    for peak in order:
        width_bins = float(peak_widths(spectrum, [peak], rel_height=0.5)[0][0])
        width_hz = max(resolution, width_bins * sample_hz / nfft)
        exclusion_hz = max(6 * resolution, 2.5 * width_hz)
        if all(abs(frequency[peak] - frequency[prior]) > max(exclusion_hz, 2.5 * prior_width)
               for prior, prior_width in selected):
            selected.append((int(peak), width_hz))
        if len(selected) >= max_components:
            break
    if not selected:
        raise SignalIdentificationError("Pilot component identity unavailable")
    selected.sort(key=lambda pair: frequency[pair[0]])
    bands = []
    for i, (peak, width_hz) in enumerate(selected):
        center = float(frequency[peak])
        half = max(4 * resolution, 1.6 * width_hz)
        lo, hi = center - half, center + half
        if i:
            lo = max(lo, (center + float(frequency[selected[i - 1][0]])) / 2 +
                     0.01 * resolution)
        if i + 1 < len(selected):
            hi = min(hi, (center + float(frequency[selected[i + 1][0]])) / 2 -
                     0.01 * resolution)
        lo = max(lo, -sample_hz / 2 + resolution)
        hi = min(hi, sample_hz / 2 - resolution)
        if lo >= hi:
            raise SignalIdentificationError("Pilot components have unresolved frequency bands")
        bands.append((lo, hi))
    return tuple(bands), {
        "peak_centers_hz": [float(frequency[p]) for p, _ in selected],
        "peak_snr_to_repeat_spectrum": [float(spectrum[p] / max(noise_spectrum, 1e-30))
                                        for p, _ in selected],
        "active_points": n_active,
        "spectral_resolution_hz": resolution,
        "spectral_noise_source": "independent pilot repetition differences",
        "band_rule": "pilot spectral peak width and active-duration resolution, disjoint at midpoints",
    }


def _fit(record: RawFIDRecord, bands: tuple[tuple[float, float], ...],
         covariance: np.ndarray, *, frequency_seeds: Sequence[float] | None = None,
         max_points: int = 1280) -> dict:
    axis = validate_axis(record)
    n = len(record.re)
    if any(not (-axis.sample_hz / 2 < lo < hi < axis.sample_hz / 2) for lo, hi in bands):
        raise SignalIdentificationError("Frozen component band is outside this FID's Nyquist range")
    if len(bands) > 3 or not bands:
        raise SignalIdentificationError("Expected one to three disjoint pilot components")
    if any(bands[i][1] >= bands[i + 1][0] for i in range(len(bands) - 1)):
        raise SignalIdentificationError("Frozen component bands overlap")
    # Include early samples in a decaying FID while still checking later phase.
    uniform = np.linspace(0, n - 1, min(n, max_points - min(192, n // 5)), dtype=int)
    early = np.arange(min(192, n))
    ix = np.unique(np.concatenate((uniform, early)))
    t = record.time_seconds[ix]
    y = record.fid[ix]
    chol_inverse = np.linalg.inv(np.linalg.cholesky(_positive_covariance(covariance)))
    p = len(bands) + 1
    yw = (np.stack((y.real, y.imag), axis=1) @ chol_inverse.T).reshape(-1)
    rate_lo = 1e-3
    rate_hi = max(1000., 20. / max(float(t[-1]), 1e-3))
    lo = np.asarray([v for band in bands for v in (band[0], rate_lo)])
    hi = np.asarray([v for band in bands for v in (band[1], rate_hi)])

    def linear(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design = np.ones((len(t), p), complex)
        for component in range(p - 1):
            frequency, rate = parameters[2 * component:2 * component + 2]
            design[:, component + 1] = np.exp((-rate + 2j * np.pi * frequency) * t)
        real = np.concatenate((design.real, -design.imag), axis=1)
        imag = np.concatenate((design.imag, design.real), axis=1)
        real_design = np.stack((real, imag), axis=1)
        weighted = np.einsum("ab,nbc->nac", chol_inverse, real_design).reshape(2 * len(t), 2 * p)
        beta = np.linalg.lstsq(weighted, yw, rcond=None)[0]
        coefficients = beta[:p] + 1j * beta[p:]
        predicted = design @ coefficients
        return predicted, coefficients, weighted

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted, _, _ = linear(parameters)
        delta = np.stack(((predicted - y).real, (predicted - y).imag), axis=1)
        return (delta @ chol_inverse.T).reshape(-1)

    centers = np.asarray([(a + b) / 2 for a, b in bands])
    seeds = np.asarray(frequency_seeds if frequency_seeds is not None else centers, float)
    if seeds.shape != centers.shape:
        raise ValueError("Frequency seed count differs from pilot bands")
    seeds = np.clip(seeds, lo[::2] + 1e-6, hi[::2] - 1e-6)
    start_frequencies = [seeds, centers]
    for component in range(len(bands)):
        for fraction in (0.18, 0.82):
            trial = seeds.copy()
            a, b = bands[component]
            trial[component] = a + fraction * (b - a)
            start_frequencies.append(trial)
    starts = []
    for frequencies in start_frequencies:
        candidate = np.asarray([v for f in frequencies for v in (f, 1 / max(t[-1], .02))])
        candidate = np.maximum(lo + 1e-7, np.minimum(hi - 1e-7, candidate))
        if not any(np.allclose(candidate, prior, atol=1e-9, rtol=0) for prior in starts):
            starts.append(candidate)
    best = None
    for initial in starts:
        result = least_squares(residual, initial, bounds=(lo, hi),
                               max_nfev=90, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        rss = float(np.dot(result.fun, result.fun))
        if best is None or rss < best[0]:
            best = (rss, result)
    if best is None:
        raise SignalIdentificationError("No multiplet fit start completed")
    rss, result = best
    _, coefficients, _ = linear(result.x)
    full_t = record.time_seconds
    full_prediction = np.full(n, coefficients[0], complex)
    for component in range(len(bands)):
        f, rate = result.x[2 * component:2 * component + 2]
        full_prediction += coefficients[component + 1] * np.exp((-rate + 2j * np.pi * f) * full_t)
    error = record.fid - full_prediction
    residual_rms = float(np.sqrt(np.mean(np.abs(error) ** 2)))
    signal_rms = float(np.sqrt(np.mean(np.abs(record.fid - coefficients[0]) ** 2)))
    relative_residual_rms = residual_rms / max(signal_rms, 1e-30)
    frequencies = tuple(float(v) for v in result.x[::2])
    rates = tuple(float(v) for v in result.x[1::2])
    edge = tuple(i for i, f in enumerate(frequencies)
                 if min(f - bands[i][0], bands[i][1] - f) <= 0.02 * (bands[i][1] - bands[i][0]))
    rate_edge = tuple(i for i, rate in enumerate(rates)
                      if min(rate - rate_lo, rate_hi - rate) <= 0.002 * (rate_hi - rate_lo))
    dof = max(1, 2 * len(ix) - (2 + 4 * len(bands)))
    covariance_nonlinear = np.linalg.pinv(result.jac.T @ result.jac) * rss / dof
    lag_naive_note = "conditional independent-sample Hessian; do not use as calibrated coverage"
    frequency_se = tuple(float(math.sqrt(max(covariance_nonlinear[2 * i, 2 * i], 0.)))
                         for i in range(len(bands)))
    return {
        "coefficients": tuple(coefficients[1:]),
        "baseline": complex(coefficients[0]),
        "frequencies_hz": frequencies,
        "decay_per_s": rates,
        "frequency_se_hz_conditional": frequency_se,
        "frequency_se_scope": lag_naive_note,
        "residual_rms": residual_rms,
        "relative_residual_rms": relative_residual_rms,
        "band_edge_components": edge,
        "decay_edge_components": rate_edge,
        "fit_points": len(ix),
        "input_points": n,
        "multi_starts": len(starts),
        "fit_success": bool(result.success),
        "whitened_rss": rss,
    }


def fixed_pilot_projection(record: RawFIDRecord, pilot: PilotSignalModel,
                           *, max_points: int = 2048) -> tuple[complex, float]:
    """Fast signed selected-mode coefficient with frozen pilot line shapes.

    Unlike a nonlinear frequency refit, this remains defined when a Rabi
    experiment passes through a near-zero amplitude. It is not itself a
    physical population or a separately calibrated phase measurement.
    """
    axis = validate_axis(record)
    if axis.sample_hz != pilot.sample_hz:
        raise SignalIdentificationError("Projection clock differs from pilot")
    n = len(record.re)
    ix = np.unique(np.concatenate((
        np.linspace(0, n - 1, min(max_points - min(192, n), n), dtype=int),
        np.arange(min(192, n)))))
    t = record.time_seconds[ix]
    y = record.fid[ix]
    design = np.ones((len(ix), 1 + len(pilot.component_frequencies_hz)), complex)
    for k, (frequency, rate) in enumerate(zip(pilot.component_frequencies_hz,
                                               pilot.component_decay_per_s), start=1):
        design[:, k] = np.exp((-rate + 2j * np.pi * frequency) * t)
    p = design.shape[1]
    real = np.concatenate((design.real, -design.imag), axis=1)
    imag = np.concatenate((design.imag, design.real), axis=1)
    raw_design = np.stack((real, imag), axis=1)
    inv_chol = np.linalg.inv(np.linalg.cholesky(pilot.noise_covariance_re_im))
    weighted_design = np.einsum("ab,nbc->nac", inv_chol, raw_design).reshape(2 * len(ix), 2 * p)
    stacked_y = (np.stack((y.real, y.imag), axis=1) @ inv_chol.T).reshape(-1)
    beta = np.linalg.lstsq(weighted_design, stacked_y, rcond=None)[0]
    coefficients = beta[:p] + 1j * beta[p:]
    selected = complex(coefficients[pilot.primary_component_index + 1])
    residual_rms = float(np.sqrt(np.mean(np.abs(y - design @ coefficients) ** 2)))
    return selected, residual_rms


def _feature_windows(signal: np.ndarray, noise_covariance: np.ndarray,
                     sample_hz: float, frequencies: Sequence[float],
                     primary: int) -> tuple[tuple[float, float], ...]:
    noise_rms = math.sqrt(float(np.trace(noise_covariance)))
    smoothed = np.sqrt(np.convolve(np.abs(signal) ** 2,
                                  np.ones(max(16, int(sample_hz / 100))) /
                                  max(16, int(sample_hz / 100)), mode="same"))
    active = np.flatnonzero(smoothed > 3 * noise_rms)
    if not len(active):
        raise SignalIdentificationError("No pilot FID interval remains above repeat-derived noise")
    # All established acquisition modes contain at least 4000 points at
    # 10 kHz; keep the likelihood vector within their common first 0.25 s.
    # Longer tails remain available to the separate full-FID model check.
    horizon = min(float((active[-1] + 1) / sample_hz), 0.25, len(signal) / sample_hz)
    if horizon < 0.012:
        raise SignalIdentificationError("Pilot FID coherence window is too short for six phase features")
    separation = max((abs(f - frequencies[primary]) for i, f in enumerate(frequencies)
                      if i != primary), default=0.)
    width = min(horizon / 18, 0.004, 1 / (4 * separation) if separation else math.inf)
    width = max(2 / sample_hz, width)
    centers = np.linspace(width / 2, 0.85 * horizon, 6)
    windows_s = tuple((float(center - width / 2), float(center + width / 2)) for center in centers)
    grid = np.arange(len(signal)) / sample_hz
    if any(np.count_nonzero((grid >= start) & (grid < end)) < 2 for start, end in windows_s):
        raise SignalIdentificationError("Pilot feature window has fewer than two FID samples")
    return windows_s


def demodulated_features(fid: np.ndarray, time_seconds: np.ndarray,
                         pilot: PilotSignalModel) -> np.ndarray:
    """Exact discrete feature operator used by both data and forward model.

    The windows are `[start, end)` on the acquisition's integer sample clock.
    Prediction must use the same sample grid, subtraction, demodulation and
    arithmetic mean, rather than only evaluating each window at its center.
    """
    fid = np.asarray(fid, complex)
    time_seconds = np.asarray(time_seconds, float)
    if fid.ndim != 1 or fid.shape != time_seconds.shape or not np.all(np.isfinite(fid)):
        raise ValueError("FID and time axis must be finite equal-length vectors")
    if abs(pilot.reference_coefficient) < 1e-12:
        raise SignalIdentificationError("Pilot receiver reference coefficient is zero")
    demodulated = (fid - pilot.baseline_complex) * np.exp(
        -2j * np.pi * pilot.reference_frequency_hz * time_seconds) / pilot.reference_coefficient
    values = []
    for start, end in pilot.feature_windows_s:
        window = (time_seconds >= start) & (time_seconds < end)
        if np.count_nonzero(window) < 2:
            raise SignalIdentificationError("Acquisition cannot support a frozen feature window")
        values.append(complex(np.mean(demodulated[window])))
    return np.asarray(values, complex)


def _interleaved(features: np.ndarray) -> np.ndarray:
    return np.stack((features.real, features.imag), axis=-1).reshape(-1)


def identify_pilot_multiplet(records: Sequence[RawFIDRecord], *, max_components: int = 3) -> PilotSignalModel:
    """Find pilot multiplet/noise/gauge from >=3 independent same-setting FIDs.

    Component bands are discovered by measured spectral width and usable FID
    duration, then frozen across all candidate pulses and acquisition lengths.
    The strongest fitted mode fixes a *receiver* phase gauge; it does not prove
    absolute transmitter phase or physical qubit-state normalization.
    """
    if not 1 <= max_components <= 3:
        raise ValueError("max_components must be 1..3")
    sample_hz, n = _validate_repeats(records)
    covariance, lag = _repeat_noise(records)
    signals = np.stack([r.fid[:n] for r in records])
    mean_fid = signals.mean(axis=0)
    bands, diagnostics = _discover_bands(mean_fid, sample_hz, covariance,
                                         signals - mean_fid, max_components=max_components)
    mean_record = RawFIDRecord(**{**records[0].__dict__,
        "axis_original": records[0].axis_original[:n],
        "time_seconds": records[0].time_seconds[:n],
        "re": mean_fid.real, "im": mean_fid.imag,
        "parameters_sent": {**records[0].parameters_sent, "sampleCount": n}})
    fit = _fit(mean_record, bands, covariance,
               frequency_seeds=diagnostics["peak_centers_hz"])
    coefficients = fit["coefficients"]
    primary = int(np.argmax(np.abs(coefficients)))
    reference = complex(coefficients[primary])
    if abs(reference) < 8 * math.sqrt(float(np.trace(covariance)) / max(n, 1)):
        raise SignalIdentificationError("Primary pilot mode is not separated from noise")
    features_windows = _feature_windows(mean_fid - fit["baseline"], covariance,
                                        sample_hz, fit["frequencies_hz"], primary)
    preliminary = PilotSignalModel(
        sample_hz=sample_hz, bands_hz=bands, primary_component_index=primary,
        component_frequencies_hz=fit["frequencies_hz"],
        component_decay_per_s=fit["decay_per_s"],
        component_weights=tuple(z / reference for z in coefficients),
        reference_frequency_hz=fit["frequencies_hz"][primary],
        reference_coefficient=reference, baseline_complex=fit["baseline"],
        receiver_phase_rad=float(np.angle(reference)),
        noise_covariance_re_im=covariance, lag_one_correlation=lag,
        between_acquisition_covariance_re_im=np.eye(2),
        feature_windows_s=features_windows, feature_covariance_re_im=np.eye(2 * len(features_windows)),
        pilot_task_ids=tuple(r.task_id for r in records), pilot_residual_rms=fit["residual_rms"],
        status="PENDING_NOISE", diagnostics={})
    feature_rows = np.stack([_interleaved(demodulated_features(r.fid[:n],
                                   r.time_seconds[:n], preliminary)) for r in records])
    repeat_cov = np.cov(feature_rows, rowvar=False)
    # Three pilot repetitions provide at most rank two in feature space.
    # Keep their measured common-mode drift but add an independent-sample
    # positive floor. The floor is deliberately conservative, not 16000 shots.
    width_counts = [np.count_nonzero((records[0].time_seconds[:n] >= a) &
                                     (records[0].time_seconds[:n] < b)) for a, b in features_windows]
    rotation = np.array([[reference.real, reference.imag],
                         [-reference.imag, reference.real]], float) / abs(reference) ** 2
    normalized_sample_cov = rotation @ covariance @ rotation.T
    correlation_inflation = max(1., (1 + max(lag, 0.)) / max(1 - max(lag, 0.), 0.02))
    floor_cov = np.zeros_like(repeat_cov)
    for i, count in enumerate(width_counts):
        floor_cov[2 * i:2 * i + 2, 2 * i:2 * i + 2] = \
            normalized_sample_cov * correlation_inflation / max(count, 1)
    feature_cov = _positive_covariance(repeat_cov + floor_cov)
    # Single-component reference scatter is a systematic gauge floor shared
    # by all later arms, recorded separately from their acquisition noise.
    repeat_fits = [_fit(r, bands, covariance, frequency_seeds=fit["frequencies_hz"],
                        max_points=768) for r in records]
    reference_rows = np.asarray([[f["coefficients"][primary].real,
                                  f["coefficients"][primary].imag] for f in repeat_fits])
    between_cov = _positive_covariance(np.cov(reference_rows, rowvar=False))
    edge = fit["band_edge_components"]
    decay_edge = fit["decay_edge_components"]
    status = "MODEL_CHECK_REQUIRED" if edge or decay_edge or not fit["fit_success"] else "IDENTIFIED"
    diagnostics = {
        **diagnostics,
        "pilot_fit_residual_rms": fit["residual_rms"],
        "pilot_fit_relative_residual_rms": fit["relative_residual_rms"],
        "pilot_band_edge_components": list(edge),
        "pilot_decay_edge_components": list(decay_edge),
        "feature_window_sample_counts": width_counts,
        "feature_covariance_rule": "independent-repeat covariance plus AR(1)-inflated sample-noise floor; 3 repeats are rank deficient",
        "receiver_phase_identifiability": "relative to independent pilot only; absolute TX/RX split unavailable",
        "FID_sign_convention": "exported Re+iIm fitted as exp(+2pi*i*f*t); sign of physical Mxy precession not independently verified",
        "frequency_fit_uncertainty_scope": fit["frequency_se_scope"],
    }
    return PilotSignalModel(
        sample_hz=sample_hz, bands_hz=bands, primary_component_index=primary,
        component_frequencies_hz=fit["frequencies_hz"],
        component_decay_per_s=fit["decay_per_s"],
        component_weights=tuple(z / reference for z in coefficients),
        reference_frequency_hz=fit["frequencies_hz"][primary],
        reference_coefficient=reference, baseline_complex=fit["baseline"],
        receiver_phase_rad=float(np.angle(reference)),
        noise_covariance_re_im=covariance, lag_one_correlation=lag,
        between_acquisition_covariance_re_im=between_cov,
        feature_windows_s=features_windows, feature_covariance_re_im=feature_cov,
        pilot_task_ids=tuple(r.task_id for r in records), pilot_residual_rms=fit["residual_rms"],
        status=status, diagnostics=diagnostics)


def estimate_signal(record: RawFIDRecord, pilot: PilotSignalModel) -> SignalEstimate:
    """Extract a measured complex observation and check frozen multiplet fit."""
    axis = validate_axis(record)
    if axis.sample_hz != pilot.sample_hz:
        raise SignalIdentificationError("Acquisition clock differs from frozen pilot")
    features = demodulated_features(record.fid, record.time_seconds, pilot)
    fit = _fit(record, pilot.bands_hz, pilot.noise_covariance_re_im,
               frequency_seeds=pilot.component_frequencies_hz)
    selected = complex(fit["coefficients"][pilot.primary_component_index])
    relative = selected / pilot.reference_coefficient
    status = "MODEL_CHECK_REQUIRED" if fit["band_edge_components"] or \
        fit["decay_edge_components"] or not fit["fit_success"] else "FIT_OK"
    if fit["relative_residual_rms"] > max(0.35, 2.5 * pilot.diagnostics.get(
            "pilot_fit_relative_residual_rms", math.inf)):
        status = "MODEL_MISMATCH"
    return SignalEstimate(
        features=features,
        feature_covariance_re_im=pilot.feature_covariance_re_im.copy(),
        selected_coefficient=selected, relative_coefficient=relative,
        relative_phase_rad=float(np.angle(relative)),
        frequency_hz=fit["frequencies_hz"][pilot.primary_component_index],
        frequency_se_hz_conditional=fit["frequency_se_hz_conditional"][pilot.primary_component_index],
        component_frequencies_hz=fit["frequencies_hz"],
        component_decay_per_s=fit["decay_per_s"],
        residual_rms=fit["residual_rms"],
        relative_residual_rms=fit["relative_residual_rms"], status=status,
        diagnostics={
            "band_edge_components": list(fit["band_edge_components"]),
            "decay_edge_components": list(fit["decay_edge_components"]),
            "fit_points": fit["fit_points"], "input_points": fit["input_points"],
            "multi_starts": fit["multi_starts"],
            "fit_success": fit["fit_success"],
            "frequency_se_scope": fit["frequency_se_scope"],
            "frozen_component_identity": "independent-pilot disjoint measured frequency bands",
            "receiver_gauge": "reference pilot primary coefficient; only relative phase identifiable",
        })


def estimate_frequency(record: RawFIDRecord, pilot: PilotSignalModel) -> float:
    estimate = estimate_signal(record, pilot)
    if estimate.status != "FIT_OK":
        raise SignalIdentificationError(
            f"Selected component frequency invalid: {estimate.status}; {estimate.diagnostics}")
    return estimate.frequency_hz


def estimate_pilot_rabi(widths_us: Sequence[float], records: Sequence[RawFIDRecord],
                        pilot: PilotSignalModel) -> RabiPilotEstimate:
    """Initialize t90 and readout response from a measured complex Rabi scan.

    A broad period range is derived from scan spacing and width span, not a
    historical t90. The returned interval is a *profile diagnostic*, not a
    confidence interval with demonstrated coverage. Multimodal or boundary
    profiles fail closed so no narrow prior is silently inferred.
    """
    if len(widths_us) != len(records) or len(records) < 5:
        raise ValueError("Rabi pilot needs at least five width/record pairs")
    x = np.asarray(widths_us, float)
    if not np.all(np.isfinite(x)) or np.any(x <= 0) or len(np.unique(x)) < 5:
        raise ValueError("Rabi pilot needs five distinct positive widths")
    distinct = np.unique(x)
    min_step = float(np.min(np.diff(distinct)))
    span = float(distinct[-1] - distinct[0])
    period_min = max(2.05 * min_step, 1.1 * min_step)
    period_max = 2.5 * span
    if period_max <= period_min:
        raise ValueError("Rabi scan width range does not resolve a period")
    projections = [fixed_pilot_projection(record, pilot) for record in records]
    y = np.asarray([coefficient / pilot.reference_coefficient
                    for coefficient, _ in projections])

    def solve(period: float) -> tuple[float, np.ndarray]:
        angle = 2 * np.pi * x / period
        basis = np.column_stack((np.ones(len(x)), np.sin(angle), np.cos(angle)))
        coefficients = np.linalg.lstsq(basis, y, rcond=None)[0]
        error = float(np.sum(np.abs(y - basis @ coefficients) ** 2))
        return error, coefficients

    grid = np.linspace(period_min, period_max, 800)
    errors = np.asarray([solve(p)[0] for p in grid])
    index = int(np.argmin(errors))
    low = float(grid[max(0, index - 1)])
    high = float(grid[min(len(grid) - 1, index + 1)])
    refined = minimize_scalar(lambda p: solve(float(p))[0], bounds=(low, high),
                              method="bounded")
    period = float(refined.x)
    error, coefficients = solve(period)
    scatter = float(np.sum(np.abs(y - y.mean()) ** 2))
    r2 = 1 - error / max(scatter, 1e-30)
    # The variance floor comes from repeated control acquisitions when present.
    # A smooth five-point fit by itself is not evidence of narrow uncertainty.
    repeated_scatter = []
    for width in distinct:
        group = y[x == width]
        if len(group) >= 2:
            repeated_scatter.extend(np.abs(group - group.mean()) ** 2)
    repeat_variance = float(np.mean(repeated_scatter)) if repeated_scatter else 0.
    residual_variance = error / max(1, 2 * len(y) - 7)
    variance = max(repeat_variance, residual_variance, 1e-10)
    # The refined continuous minimum can be lower than every point of the
    # finite search grid when measurement noise is tiny. Include that minimum
    # so the data-driven profile interval never vanishes due to grid spacing.
    threshold = max(error, float(errors[index])) + 3.84 * variance
    accepted = np.r_[grid[errors <= threshold], period]
    interval = (float(accepted.min() / 4), float(accepted.max() / 4))
    at_boundary = index < 2 or index > len(grid) - 3
    # Check whether an equally plausible, separated period mode exists.
    near_modes, _ = find_peaks(-errors)
    alternatives = [float(grid[i]) for i in near_modes if abs(i - index) > 5 and
                    errors[i] <= threshold]
    status = "IDENTIFIED" if r2 >= .6 and not at_boundary and not alternatives else \
        "AMBIGUOUS_OR_WEAK_RABI"
    return RabiPilotEstimate(
        period_us=period if status == "IDENTIFIED" else None,
        t90_us=period / 4 if status == "IDENTIFIED" else None,
        t90_interval_us=interval,
        signed_complex_r2=r2,
        receiver_offset=complex(coefficients[0]),
        receiver_sine_gain=complex(coefficients[1]),
        receiver_cosine_gain=complex(coefficients[2]),
        status=status,
        diagnostics={
            "period_search_us": [period_min, period_max],
            "widths_us": x.tolist(), "profile_error": error,
            "profile_variance_rule": "max(complex fit residual, repeated same-width pilot scatter)",
            "profile_interval_is_validated_coverage": False,
            "alternative_period_modes_us": alternatives,
            "fit_hits_search_edge": at_boundary,
            "five_distinct_widths": len(distinct) >= 5,
            "frozen_projection_residual_rms": [r for _, r in projections],
            "projection_scope": "linear signed pilot-frozen mode, defined at Rabi nulls",
        })
