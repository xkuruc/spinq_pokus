"""Local complex NMR analysis; vendor FFT/fit never enters these functions."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .core import RawFIDRecord, uniform_axis_step


@dataclass(frozen=True)
class AxisContract:
    sample_hz: float
    point_count: int
    original_step: float
    seconds_per_original_unit: float
    physical_clock_verified: bool


@dataclass(frozen=True)
class NoiseModel:
    re_im_covariance: np.ndarray
    lag_one_correlation: float
    between_acquisition_amplitude_sd: float
    repetitions: int
    source: str


@dataclass(frozen=True)
class MultipletSpec:
    """Frozen component identities as frequency bands from an independent pilot."""
    bands_hz: tuple[tuple[float, float], ...]
    decay_bounds_s: tuple[float, float] = (0.002, 20.0)
    starts_per_band: int = 3
    max_fit_points: int = 2048


def validate_axis(record: RawFIDRecord) -> AxisContract:
    x, t = record.axis_original, record.time_seconds
    if len(x) != len(t) or len(x) != len(record.re) or len(x) != len(record.im) or len(x) < 64:
        raise ValueError("Axis/FID lengths inconsistent")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(t)) and
            np.all(np.diff(x) > 0) and np.all(np.diff(t) > 0)):
        raise ValueError("Axis nonfinite or nonmonotonic")
    fs = float(record.parameters_sent["sampleFre"])
    if not np.allclose(np.diff(t), 1 / fs, rtol=1e-8, atol=1e-12):
        raise ValueError("Derived time axis does not match this experiment's sampleFre")
    dx, _, _ = uniform_axis_step(x)
    return AxisContract(fs, len(x), dx, (1/fs)/dx, False)


def estimate_noise(repeated_records: list[RawFIDRecord]) -> NoiseModel:
    """Noise from independent acquisitions, with drift reported separately."""
    if len(repeated_records) < 3:
        raise ValueError("At least three independent records are needed")
    lengths = [len(r.re) for r in repeated_records]
    n = min(lengths)
    fs = {validate_axis(r).sample_hz for r in repeated_records}
    if len(fs) != 1:
        raise ValueError("Cannot pool distinct acquisition clocks")
    signals = np.stack([r.fid[:n] for r in repeated_records])
    centered = signals - signals.mean(axis=0, keepdims=True)
    # Per-time covariance, pooled only after centering each time point. This
    # excludes the deterministic FID envelope from the noise estimate.
    stacked = np.column_stack((centered.real.ravel(), centered.imag.ravel()))
    covariance = np.cov(stacked, rowvar=False, ddof=1)
    if not np.all(np.isfinite(covariance)) or np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Independent-repeat covariance not positive definite")
    a = centered[:, :-1].ravel()
    b = centered[:, 1:].ravel()
    lag = float(np.real(np.vdot(a, b)) / max(np.vdot(a, a).real, 1e-12))
    amplitudes = np.abs(signals[:, : min(256, n)].mean(axis=1))
    return NoiseModel(covariance, lag, float(np.std(amplitudes, ddof=1)),
                      len(repeated_records), "independent repeated FID differences")


def _whiten(error: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    inv_chol = np.linalg.inv(np.linalg.cholesky(covariance))
    stacked = np.stack((error.real, error.imag), axis=-1)
    return (stacked @ inv_chol.T).ravel()


def fit_complex_multiplet(record: RawFIDRecord, model_spec: MultipletSpec,
                          noise: NoiseModel | None = None) -> dict:
    """Multi-start variable projection with fixed component frequency identities.

    One complex coefficient and positive R2* is fitted per frozen band. Each
    nonlinear evaluation solves offset and coefficients by linear least squares.
    A small-FID run still fits the same bands, including a weak component.
    """
    axis = validate_axis(record)
    bands = tuple((float(a), float(b)) for a, b in model_spec.bands_hz)
    if not bands or len(bands) > 3 or any(a >= b or abs(a) >= axis.sample_hz/2 or
                                       abs(b) >= axis.sample_hz/2 for a, b in bands):
        raise ValueError("Bands must be distinct, ordered intervals inside this mode's Nyquist range")
    if any(bands[i][1] >= bands[i+1][0] for i in range(len(bands)-1)):
        raise ValueError("Component identity bands overlap")
    count = min(len(record.re), model_spec.max_fit_points)
    ix = np.unique(np.linspace(0, len(record.re)-1, count, dtype=int))
    t = record.time_seconds[ix]
    y = record.fid[ix]
    covariance = noise.re_im_covariance if noise is not None else np.eye(2)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2): raise ValueError("Expected Re/Im covariance 2x2")
    lower = np.array([v for band in bands for v in (band[0], 1/model_spec.decay_bounds_s[1])])
    upper = np.array([v for band in bands for v in (band[1], 1/model_spec.decay_bounds_s[0])])
    if np.any(lower >= upper): raise ValueError("Invalid frequency or decay bounds")

    def solve_linear(params):
        cols = [np.ones(len(t), dtype=np.complex128)]
        for index in range(len(bands)):
            f, rate = params[2*index:2*index+2]
            cols.append(np.exp((-rate + 2j*np.pi*f)*t))
        design = np.stack(cols, axis=1)
        # Complex least squares handles signed and phase-rotated amplitudes.
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        return design @ coefficients, coefficients

    def residual(params):
        predicted, _ = solve_linear(params)
        return _whiten(predicted-y, covariance)

    starts = []
    for band in bands:
        grid = np.linspace(*band, model_spec.starts_per_band+2)[1:-1]
        starts.append(grid)
    best = None
    for frequencies in itertools.product(*starts):
        initial = np.array([v for f in frequencies for v in (f, 1/max(t[-1], 0.02))])
        initial = np.clip(initial, lower+1e-9, upper-1e-9)
        result = least_squares(residual, initial, bounds=(lower, upper),
                               max_nfev=140, ftol=1e-7, xtol=1e-7)
        rss = float(np.dot(result.fun, result.fun))
        if best is None or rss < best[0]: best = (rss, result)
    assert best is not None
    rss, result = best
    prediction, coefficients = solve_linear(result.x)
    residual_complex = y-prediction
    modes = []
    for k, band in enumerate(bands):
        frequency, rate = result.x[2*k:2*k+2]
        amp = coefficients[k+1]
        modes.append({"component_id": k, "band_hz": list(band),
                      "frequency_hz": float(frequency), "r2star_per_s": float(rate),
                      "t2star_s": float(1/rate), "coefficient_re": float(amp.real),
                      "coefficient_im": float(amp.imag), "amplitude": float(abs(amp)),
                      "phase_deg": float(np.degrees(np.angle(amp))),
                      "frequency_at_band_edge": bool(min(frequency-band[0],band[1]-frequency) <
                                                     0.01*(band[1]-band[0]))})
    dof = max(1, 2*len(y) - (2+4*len(bands)))
    try:
        covariance_nonlinear = np.linalg.pinv(result.jac.T@result.jac) * rss/dof
        for k, mode in enumerate(modes):
            mode["frequency_se_hz_conditional"] = float(math.sqrt(max(0.,covariance_nonlinear[2*k,2*k])))
    except np.linalg.LinAlgError:
        for mode in modes: mode["frequency_se_hz_conditional"] = None
    return {"modes": modes, "offset_re": float(coefficients[0].real),
            "offset_im": float(coefficients[0].imag), "residual_rms": float(np.sqrt(np.mean(np.abs(residual_complex)**2))),
            "whitened_rss": rss, "fit_points": len(ix), "input_points": len(record.re),
            "noise_covariance_source": noise.source if noise else "UNCALIBRATED_IDENTITY",
            "component_identity": "frozen disjoint frequency bands",
            "clock_basis": "configured sampleFre checked against original chart axis",
            "physical_clock_verified": False,
            "status": "MODEL_CHECK_REQUIRED" if any(m["frequency_at_band_edge"] for m in modes) else "FIT_COMPLETED"}


def fft_local(record: RawFIDRecord, *, window: str = "hann", nfft: int | None = None,
              normalization: str = "none") -> dict:
    axis = validate_axis(record)
    y = record.fid
    nfft = nfft or len(y)
    if nfft < len(y): raise ValueError("nfft cannot truncate FID")
    windows = {"none": np.ones(len(y)), "hann": np.hanning(len(y)),
               "hamming": np.hamming(len(y)), "blackman": np.blackman(len(y))}
    if window not in windows: raise ValueError("Unknown apodization window")
    spectrum = np.fft.fftshift(np.fft.fft(y*windows[window], n=nfft))
    if normalization == "sum_window": spectrum /= max(np.sum(windows[window]),1e-12)
    elif normalization == "unitary": spectrum /= math.sqrt(nfft)
    elif normalization != "none": raise ValueError("Unknown FFT normalization")
    frequency = np.fft.fftshift(np.fft.fftfreq(nfft, 1/axis.sample_hz))
    return {"frequency_hz": frequency, "spectrum": spectrum,
            "convention": "FFT of Re+iIm, exp(-i2πft), fftshift",
            "window": window, "normalization": normalization, "sample_hz": axis.sample_hz}


def vendor_fft_replica(record: RawFIDRecord, frozen_parameters: dict) -> dict:
    """Testable hypothesis; never a claim about unknown server source code."""
    axis = validate_axis(record)
    c = complex(frozen_parameters["complex_scale_re"],frozen_parameters.get("complex_scale_im",0.0))
    x_scale = float(frozen_parameters["axis_scale_original_units"])
    nfft = int(frozen_parameters.get("nfft",16384))
    if x_scale <= 0 or nfft < len(record.re): raise ValueError("Invalid frozen replica parameters")
    apodized = record.fid/(1+(record.axis_original/x_scale)**2)
    spectrum = c*np.fft.fftshift(np.fft.fft(apodized,n=nfft))
    frequency = np.fft.fftshift(np.fft.fftfreq(nfft,1/axis.sample_hz))
    return {"frequency_hz":frequency,"spectrum":spectrum,
            "status":"WORKING_HYPOTHESIS_NOT_VENDOR_SOURCE","parameters":frozen_parameters}


def extract_observables(records: list[RawFIDRecord], readout_calibration: dict) -> dict:
    """Regularized analog inversion with an explicit fixed receiver gauge."""
    response = np.asarray(readout_calibration["complex_response"], dtype=np.complex128)
    baseline = np.asarray(readout_calibration.get("baseline", np.zeros(response.shape[0])),complex)
    features = np.asarray([r.fid[0] for r in records],complex)
    if response.shape[0] != len(features) or len(baseline) != len(features):
        raise ValueError("Readout calibration and records have different channels/steps")
    lam = float(readout_calibration.get("ridge",1e-6))
    if lam < 0: raise ValueError("Negative readout regularization")
    gram = response.conj().T@response + lam*np.eye(response.shape[1])
    observables = np.linalg.solve(gram,response.conj().T@(features-baseline))
    predicted = response@observables+baseline
    return {"observable_re":observables.real.tolist(),"observable_im":observables.imag.tolist(),
            "relative_residual":float(np.linalg.norm(features-predicted)/max(np.linalg.norm(features),1e-12)),
            "gauge":readout_calibration.get("gauge","relative calibrated receiver phase and scale"),
            "physical_population_normalization_verified":False}
