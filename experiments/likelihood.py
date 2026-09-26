"""Short H-pulse NMR observation model for the Bayesian calibration experiment.

This is a *local* model of pilot-frozen, demodulated complex FID window means.
It does not call SpinQLabLink, assume that exported FID is raw ADC, or regard
the individual FID samples as independent quantum shots.  The one-spin
Hamiltonian is K = delta*Iz + Omega*(cos(phi)*Ix + sin(phi)*Iy), where
I = sigma/2, K is in Hz, and U(dt) = exp(-2j*pi*K*dt).  Known multiplet lines
are represented as a weighted sum of transitions.  This approximation is only
valid for short H pulses after a separate fit check on the real pilot FIDs.

The fixed receiver gain/phase is a gauge chosen using independent pilot data.
The inferred phase is *relative* to that gauge, not an independently
identifiable absolute transmitter or receiver phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Candidate:
    """One physical command and the pilot-frozen FID feature operator.

    `drive_frequency_shift_hz` changes the RF Hamiltonian.  A receiver
    demodulation change is a separate input and does not move the transmitter.
    A nonzero coherent delay is only representable after the driver has
    positively established its physical timing contract.
    """

    family: str
    width_us: float
    amplitude_pct: float
    phase_deg: float
    feature_windows_s: tuple[tuple[float, float], ...]
    sample_hz: int = 10_000
    sample_count: int = 16_000
    drive_frequency_shift_hz: float = 0.0
    demodulation_shift_hz: float = 0.0
    delay_us: float = 0.0
    readout_width_us: float = 0.0
    readout_phase_deg: float = 0.0
    coherent_delay_verified: bool = False
    estimated_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_windows_s",
                           tuple((float(a), float(b)) for a, b in self.feature_windows_s))
        if self.family not in {"rabi", "phase", "fid", "ramsey", "return_control"}:
            raise ValueError(f"Unknown candidate family: {self.family}")
        scalars = (self.width_us, self.amplitude_pct, self.phase_deg,
                   self.drive_frequency_shift_hz, self.demodulation_shift_hz,
                   self.delay_us, self.readout_width_us, self.readout_phase_deg)
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError("Nonfinite candidate setting")
        if self.width_us <= 0 or not 0 < self.amplitude_pct <= 100:
            raise ValueError("Pulse width and amplitude must be positive")
        if self.sample_hz <= 0 or self.sample_count < 2:
            raise ValueError("Invalid FID sample clock/count")
        if not self.feature_windows_s:
            raise ValueError("At least one pilot-frozen FID window is required")
        clock = np.arange(self.sample_count, dtype=np.float64) / self.sample_hz
        previous_end = -math.inf
        for start, end in self.feature_windows_s:
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start < previous_end:
                raise ValueError("FID feature windows must be finite, sorted and disjoint")
            # Match exactly the feature mask used for exported FID samples:
            # start <= n/sample_hz < end.
            if np.count_nonzero((clock >= start) & (clock < end)) < 2:
                raise ValueError("Each FID feature window needs at least two samples")
            previous_end = end
        if self.family == "ramsey":
            if not self.coherent_delay_verified or self.delay_us <= 0 or self.readout_width_us <= 0:
                raise ValueError("Ramsey requires a separately verified coherent delay and readout pulse")
        elif self.delay_us != 0 or self.readout_width_us != 0:
            raise ValueError("A second pulse or delay requires the verified Ramsey family")
        if self.estimated_wall_seconds is not None and (
                not math.isfinite(self.estimated_wall_seconds) or self.estimated_wall_seconds <= 0):
            raise ValueError("Estimated whole-task time must be positive")


@dataclass(frozen=True)
class NMRModel:
    """Pilot-frozen nuisance parameters and empirically checked spin model.

    The particle frequency `delta_hz` is a *shift* relative to
    `reference_frequency_hz` in the exported FID. `component_offsets_hz`
    are known relative offsets from that selected component. Pulse detuning
    relative to the programmed RF frequency has its own measured origin,
    because a receiver demodulation setting cannot establish that origin.
    """

    reference_frequency_hz: float
    reference_coefficient: complex
    baseline_complex: complex = 0j
    receiver_gain: complex | None = None
    component_offsets_hz: tuple[float, ...] = (0.0,)
    component_weights: tuple[complex, ...] = (1.0 + 0.0j,)
    component_decay_per_s: tuple[float, ...] = (0.0,)
    pulse_detuning_offset_hz: float = 0.0
    amplitude_reference_pct: float = 100.0
    detection_sign: int = 1

    def __post_init__(self) -> None:
        n = len(self.component_offsets_hz)
        if not n or len(self.component_weights) != n or len(self.component_decay_per_s) != n:
            raise ValueError("Multiplet offsets, weights and decays must match")
        if self.detection_sign not in (-1, 1):
            raise ValueError("Detection sign must be +1 or -1 from pilot validation")
        if self.amplitude_reference_pct <= 0 or not math.isfinite(self.amplitude_reference_pct):
            raise ValueError("Invalid RF amplitude reference")
        if abs(self.reference_coefficient) <= 1e-12:
            raise ValueError("FID normalization coefficient is zero")
        real_values = (self.reference_frequency_hz, self.pulse_detuning_offset_hz,
                       *self.component_offsets_hz, *self.component_decay_per_s)
        complex_values = (self.reference_coefficient, self.baseline_complex,
                          self.receiver_gain if self.receiver_gain is not None else 1 + 0j,
                          *self.component_weights)
        if not all(math.isfinite(float(value)) for value in real_values):
            raise ValueError("Nonfinite model parameter")
        if not all(np.isfinite(value) for value in complex_values):
            raise ValueError("Nonfinite complex model parameter")
        if any(rate < 0 for rate in self.component_decay_per_s):
            raise ValueError("Negative FID decay rate")


def _rotate_bloch(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  detuning_hz: np.ndarray, rabi_hz: np.ndarray,
                  phase_rad: np.ndarray, duration_us: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Rodrigues rotation equivalent to exp(-2j*pi*K*dt)."""
    total_hz = np.hypot(detuning_hz, rabi_hz)
    nx = np.divide(rabi_hz * np.cos(phase_rad), total_hz,
                   out=np.zeros_like(total_hz), where=total_hz > 0)
    ny = np.divide(rabi_hz * np.sin(phase_rad), total_hz,
                   out=np.zeros_like(total_hz), where=total_hz > 0)
    nz = np.divide(detuning_hz, total_hz,
                   out=np.zeros_like(total_hz), where=total_hz > 0)
    angle = 2 * np.pi * total_hz * duration_us * 1e-6
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dot = nx * x + ny * y + nz * z
    return (x * cos_a + (ny * z - nz * y) * sin_a + nx * dot * (1 - cos_a),
            y * cos_a + (nz * x - nx * z) * sin_a + ny * dot * (1 - cos_a),
            z * cos_a + (nx * y - ny * x) * sin_a + nz * dot * (1 - cos_a))


@lru_cache(maxsize=512)
def _window_sample_ranges(candidate: Candidate) -> tuple[tuple[int, int], ...]:
    clock = np.arange(candidate.sample_count, dtype=np.float64) / candidate.sample_hz
    return tuple((int(np.searchsorted(clock, start, side="left")),
                  int(np.searchsorted(clock, end, side="left")))
                 for start, end in candidate.feature_windows_s)


def _discrete_window_exponential_mean(exponent_per_s: np.ndarray,
                                      first_sample: int, stop_sample: int,
                                      sample_hz: int) -> np.ndarray:
    """Exact arithmetic mean of exp(lambda*n/fs) over integer sample indices."""
    count = stop_sample - first_sample
    z = exponent_per_s / sample_hz
    small = np.abs(z) < 1e-8
    # The direct mean is used only for the removable singularity near zero.
    # Else the geometric series is exact and avoids a 3D particle×line×time
    # array at every candidate evaluation.
    result = np.empty_like(z, dtype=np.complex128)
    if np.any(~small):
        active = ~small
        result[active] = (np.exp(z[active] * first_sample) *
                          np.expm1(z[active] * count) /
                          (count * np.expm1(z[active])))
    if np.any(small):
        zs = z[small]
        average_n = first_sample + (count - 1) / 2
        average_n2 = (first_sample**2 + first_sample * (count - 1) +
                      (count - 1) * (2 * count - 1) / 6)
        result[small] = 1 + zs * average_n + 0.5 * zs**2 * average_n2
    return result


def predict_complex(particles: np.ndarray, candidate: Candidate,
                    model: NMRModel) -> np.ndarray:
    """Predict exactly the complex, demodulated window means used by the data path.

    `particles` has columns `[delta_hz, t90_us, relative_phase_rad]`; the
    returned shape is `(n_particles, n_windows)`.  The only moving physical
    parameters are these three; fixed multiplet/receiver quantities come
    from independent pilot observations. The FID timepoint covariance is
    handled by `log_likelihood`, never by multiplying single-sample shots.
    """
    theta = np.asarray(particles, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 3 or not len(theta):
        raise ValueError("Particles must have shape (n, 3)")
    if not np.all(np.isfinite(theta)) or np.any(theta[:, 1] <= 0):
        raise ValueError("Particles must be finite with positive t90")
    offsets = np.asarray(model.component_offsets_hz, dtype=np.float64)
    weights = np.asarray(model.component_weights, dtype=np.complex128)
    decays = np.asarray(model.component_decay_per_s, dtype=np.float64)
    delta = theta[:, 0, None]
    t90_s = theta[:, 1, None] * 1e-6
    phase = theta[:, 2, None] + np.deg2rad(candidate.phase_deg)
    # Omega is cycles/second in K=Omega*I; a resonant t90 pulse satisfies
    # 2*pi*Omega*t90=pi/2, hence Omega=1/(4*t90).
    rabi_hz = (candidate.amplitude_pct / model.amplitude_reference_pct) / (4 * t90_s)
    pulse_delta = (model.pulse_detuning_offset_hz + delta + offsets[None, :]
                   - candidate.drive_frequency_shift_hz)
    zeros = np.zeros_like(pulse_delta)
    x, y, z = _rotate_bloch(zeros, zeros, np.ones_like(pulse_delta),
                            pulse_delta, rabi_hz, phase, candidate.width_us)
    if candidate.family == "ramsey":
        free_angle = 2 * np.pi * pulse_delta * candidate.delay_us * 1e-6
        cos_f, sin_f = np.cos(free_angle), np.sin(free_angle)
        x, y = x * cos_f - y * sin_f, x * sin_f + y * cos_f
        readout_phase = theta[:, 2, None] + np.deg2rad(candidate.readout_phase_deg)
        x, y, z = _rotate_bloch(x, y, z, pulse_delta, rabi_hz,
                                readout_phase, candidate.readout_width_us)
    transverse = x + 1j * model.detection_sign * y
    # The signed empirical FID reference is fixed by the pilot.  A receiver
    # demodulation change moves this observed frequency; it does not change
    # the pulse Hamiltonian above.
    relative_f = delta + offsets[None, :] - candidate.demodulation_shift_hz
    exponent = -decays[None, :] + 1j * 2 * np.pi * relative_f
    gain = model.receiver_gain if model.receiver_gain is not None else model.reference_coefficient
    scale = gain / model.reference_coefficient
    out = np.empty((len(theta), len(candidate.feature_windows_s)), dtype=np.complex128)
    for column, (first, stop) in enumerate(_window_sample_ranges(candidate)):
        # Exact arithmetic mean over the same exported-sample mask as
        # demodulated_features in the data path. No center-time shortcut.
        modes = _discrete_window_exponential_mean(exponent, first, stop,
                                                   candidate.sample_hz)
        out[:, column] = scale * np.sum(weights[None, :] * transverse * modes, axis=1)
    if not np.all(np.isfinite(out)):
        raise ValueError("Nonfinite physical-model prediction")
    return out


def interleaved_real_imag(values: np.ndarray) -> np.ndarray:
    """Return [..., Re0, Im0, Re1, Im1, ...] for complex feature vectors."""
    complex_values = np.asarray(values, dtype=np.complex128)
    if complex_values.ndim < 1:
        raise ValueError("Complex feature vector is required")
    out = np.empty(complex_values.shape[:-1] + (2 * complex_values.shape[-1],), dtype=np.float64)
    out[..., 0::2] = complex_values.real
    out[..., 1::2] = complex_values.imag
    return out


@dataclass(frozen=True)
class GaussianObservation:
    """A full real/imag Gaussian observation with a covariance per acquisition.

    Adjacent FID windows may be correlated.  The covariance must be
    estimated from independent repeated acquisitions or an explicit physical
    noise model, with drift allowance; it is not divided by the number of
    exported FID samples.
    """

    covariance: np.ndarray

    def __post_init__(self) -> None:
        cov = np.asarray(self.covariance, dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] % 2:
            raise ValueError("Covariance must be square with paired Re/Im dimensions")
        if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("Covariance must be finite and symmetric")
        try:
            cholesky = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Covariance is not positive definite; estimate a justified floor") from exc
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "_cholesky", cholesky)
        object.__setattr__(self, "_log_normalizer", -0.5 * (
            len(cov) * np.log(2 * np.pi) + 2 * np.log(np.diag(cholesky)).sum()))

    def logpdf(self, observed: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        observed = np.asarray(observed, dtype=np.complex128)
        predicted = np.asarray(predictions, dtype=np.complex128)
        if observed.ndim != 1 or predicted.ndim != 2 or predicted.shape[1:] != observed.shape:
            raise ValueError("Observation/prediction feature shapes differ")
        if 2 * len(observed) != self.covariance.shape[0]:
            raise ValueError("Feature and covariance dimensions differ")
        residual = interleaved_real_imag(observed[None, :] - predicted)
        solved = np.linalg.solve(self._cholesky, residual.T)
        return self._log_normalizer - 0.5 * np.sum(solved**2, axis=0)

    def sample(self, means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        means = np.asarray(means, dtype=np.complex128)
        if means.ndim != 2 or means.shape[1] * 2 != self.covariance.shape[0]:
            raise ValueError("Predictive means do not match covariance")
        normals = rng.standard_normal((len(means), self.covariance.shape[0]))
        perturbation = normals @ self._cholesky.T
        return means + perturbation[:, 0::2] + 1j * perturbation[:, 1::2]


def log_likelihood(observed: np.ndarray, predictions: np.ndarray,
                   covariance: np.ndarray) -> np.ndarray:
    """One multivariate Gaussian likelihood per *acquisition*, not per shot."""
    return GaussianObservation(covariance).logpdf(observed, predictions)


def covariance_from_repeats(repeated_features: Sequence[Sequence[complex]], *,
                            shrinkage: float = 0.2, relative_floor: float = 1e-4,
                            drift_differences: Sequence[Sequence[complex]] | None = None
                            ) -> np.ndarray:
    """Estimate full Re/Im covariance from independent pilot acquisitions.

    The `drift_differences` are separate return-control differences spanning a
    similar wall-time interval to the experiment. Their empirical covariance
    is added conservatively. Shrinkage/floor are explicit because a handful
    of repeated acquisitions cannot estimate a full 12x12 covariance at full
    rank. They must be frozen before the comparison, and reported.
    """
    repeats = np.asarray(repeated_features, dtype=np.complex128)
    if repeats.ndim != 2 or repeats.shape[0] < 2 or repeats.shape[1] < 1:
        raise ValueError("At least two independent complex feature vectors are required")
    if not np.all(np.isfinite(repeats)) or not 0 <= shrinkage <= 1 or relative_floor <= 0:
        raise ValueError("Invalid repeated features or covariance regularization")
    real = interleaved_real_imag(repeats)
    raw = np.cov(real, rowvar=False, ddof=1)
    raw = np.atleast_2d(raw)
    diagonal = np.diag(raw).copy()
    positive = diagonal[diagonal > 0]
    scale = float(np.median(positive)) if positive.size else 0.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Repeated pilot has no measurable scatter")
    covariance = (1 - shrinkage) * raw + shrinkage * np.diag(diagonal)
    covariance += relative_floor * scale * np.eye(len(diagonal))
    if drift_differences is not None:
        differences = np.asarray(drift_differences, dtype=np.complex128)
        if differences.ndim != 2 or differences.shape[0] < 2 or differences.shape[1] != repeats.shape[1]:
            raise ValueError("Drift differences need two or more matched controls")
        if not np.all(np.isfinite(differences)):
            raise ValueError("Nonfinite drift control")
        covariance += np.atleast_2d(np.cov(interleaved_real_imag(differences),
                                           rowvar=False, ddof=1))
    # An ill-conditioned covariance can spuriously claim many decimals of
    # calibration accuracy. Force the caller to increase the documented floor.
    GaussianObservation(covariance)
    return covariance
