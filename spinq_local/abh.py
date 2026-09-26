"""Local numerical methods A, B, H for exported NMR observations.

This module has no hardware imports or network access. All times are seconds,
frequencies Hz, SDK amplitudes percent, and mathematical phases radians.
The methods are NMR adaptations of Gerster et al. (PRX Quantum 3, 020350),
Beracha et al. (MRM 90, 839), and the public Boulder Opal calibration notebook.
They are not the authors' device software or a claim about Gemini performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, minimize
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from scipy.stats import norm


TAU = 2.0 * np.pi


def _wrap_phase(value):
    return (np.asarray(value) + np.pi) % TAU - np.pi


def _covariance_2d(covariance):
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        raise ValueError("Complex observation needs a finite 2x2 Re/Im covariance")
    if np.linalg.eigvalsh(cov)[0] <= 0:
        raise ValueError("Re/Im covariance must be positive definite")
    return cov


@dataclass(frozen=True)
class CalibrationSetting:
    """A commanded single-spin probe; duration and free evolution in seconds."""

    family: str
    duration_s: float
    amplitude_pct: float
    phase_deg: float = 0.0
    free_s: float = 0.0
    expected_wall_s: float = 1.0

    def __post_init__(self):
        if self.family not in {"rabi", "phase", "ramsey"}:
            raise ValueError("Unknown calibration family")
        for value in (self.duration_s, self.amplitude_pct, self.phase_deg,
                      self.free_s, self.expected_wall_s):
            if not np.isfinite(value):
                raise ValueError("Calibration setting contains nonfinite value")
        if self.duration_s <= 0 or self.amplitude_pct <= 0 or self.free_s < 0 or self.expected_wall_s <= 0:
            raise ValueError("Calibration duration, amplitude and cost must be positive")
        if self.family == "ramsey" and self.free_s <= 0:
            raise ValueError("Ramsey probe needs verified free-evolution interval")


def _bloch_rotate(vector, omega_xyz_hz, duration_s):
    """Rodrigues rotation for K=Omega_x Ix+Omega_y Iy+delta Iz in Hz."""
    vector = np.asarray(vector, float)
    omega = np.asarray(omega_xyz_hz, float)
    magnitude = np.linalg.norm(omega, axis=-1)
    axis = omega / np.maximum(magnitude[..., None], 1e-300)
    angle = TAU * magnitude * duration_s
    c = np.cos(angle)[..., None]
    s = np.sin(angle)[..., None]
    return vector * c + np.cross(axis, vector) * s + axis * np.sum(axis * vector, axis=-1)[..., None] * (1-c)


def predict_calibration(setting: CalibrationSetting, theta, *, rf_hz_at_100pct: float,
                        receiver_gain: complex = 1 + 0j):
    """Complex transverse response from theta=(detuning Hz, RF scale, phase rad).

    `receiver_gain` must be fixed from a common pilot. Phase is relative to that
    receiver gauge. The Ramsey family uses two equal-width pulses separated by
    a verified idle; a caller must compile precisely that sequence.
    """
    particles = np.atleast_2d(np.asarray(theta, dtype=float))
    if particles.shape[1] != 3 or not np.all(np.isfinite(particles)):
        raise ValueError("theta must be finite Nx3 (delta_hz, rf_scale, phase_rad)")
    if np.any(particles[:, 1] <= 0) or rf_hz_at_100pct <= 0:
        raise ValueError("RF scale and pilot RF rate must be positive")
    delta, scale, offset = particles.T
    rate = rf_hz_at_100pct * setting.amplitude_pct / 100.0 * scale
    phase = offset + np.deg2rad(setting.phase_deg)
    r = np.broadcast_to(np.array([0., 0., 1.]), (len(particles), 3)).copy()
    if setting.family == "ramsey":
        first = np.stack((rate*np.cos(offset), rate*np.sin(offset), delta), axis=1)
        r = _bloch_rotate(r, first, setting.duration_s)
        idle = np.stack((np.zeros_like(delta), np.zeros_like(delta), delta), axis=1)
        r = _bloch_rotate(r, idle, setting.free_s)
        second = np.stack((rate*np.cos(phase), rate*np.sin(phase), delta), axis=1)
        r = _bloch_rotate(r, second, setting.duration_s)
    else:
        drive = np.stack((rate*np.cos(phase), rate*np.sin(phase), delta), axis=1)
        r = _bloch_rotate(r, drive, setting.duration_s)
        if setting.free_s:
            idle = np.stack((np.zeros_like(delta), np.zeros_like(delta), delta), axis=1)
            r = _bloch_rotate(r, idle, setting.free_s)
    response = complex(receiver_gain) * (r[:, 0] + 1j*r[:, 1])
    return response[0] if np.asarray(theta).ndim == 1 else response


class CalibrationParticleFilter:
    """Log-weight SMC with Liu-West rejuvenation in Hz/log-scale/circular phase.

    The particles always represent an absolute device parameter. Changing a
    commanded pulse changes the likelihood, never resets the posterior mean.
    """

    def __init__(self, particles, *, rf_hz_at_100pct: float, seed: int = 0,
                 receiver_gain: complex = 1 + 0j):
        self.particles = np.asarray(particles, float).copy()
        if self.particles.ndim != 2 or self.particles.shape[1] != 3 or len(self.particles) < 16:
            raise ValueError("At least 16 Nx3 particles are required")
        if not np.all(np.isfinite(self.particles)) or np.any(self.particles[:, 1] <= 0):
            raise ValueError("Particles must be finite with positive RF scale")
        if rf_hz_at_100pct <= 0:
            raise ValueError("Pilot RF rate must be positive")
        self.particles[:, 2] = _wrap_phase(self.particles[:, 2])
        self.rf_hz_at_100pct = float(rf_hz_at_100pct)
        self.receiver_gain = complex(receiver_gain)
        self.log_weights = np.full(len(self.particles), -math.log(len(self.particles)))
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    @property
    def weights(self):
        return np.exp(self.log_weights)

    @property
    def ess(self):
        return float(1.0 / np.sum(self.weights**2))

    def predict(self, setting: CalibrationSetting):
        return predict_calibration(setting, self.particles,
                                   rf_hz_at_100pct=self.rf_hz_at_100pct,
                                   receiver_gain=self.receiver_gain)

    def _unwrapped(self, weights=None):
        weights = self.weights if weights is None else np.asarray(weights)
        center = np.angle(np.sum(weights*np.exp(1j*self.particles[:, 2])))
        if abs(np.sum(weights*np.exp(1j*self.particles[:, 2]))) < 1e-6:
            # Circular mean is undefined for strongly multimodal phases.
            center = float(self.particles[np.argmax(weights), 2])
        return np.column_stack((self.particles[:, 0], np.log(self.particles[:, 1]),
                                center + _wrap_phase(self.particles[:, 2]-center)))

    def summary(self):
        w = self.weights
        unwrapped = self._unwrapped()
        mean = np.sum(w[:, None]*unwrapped, axis=0)
        variance = np.sum(w[:, None]*(unwrapped-mean)**2, axis=0)
        phase_resultant = abs(np.sum(w*np.exp(1j*self.particles[:, 2])))
        return {"delta_hz": float(np.sum(w*self.particles[:, 0])),
                "delta_sd_hz": float(np.sqrt(np.sum(w*(self.particles[:, 0]-np.sum(w*self.particles[:, 0]))**2))),
                "rf_scale": float(np.sum(w*self.particles[:, 1])),
                "rf_scale_sd": float(np.sqrt(np.sum(w*(self.particles[:, 1]-np.sum(w*self.particles[:, 1]))**2))),
                "phase_deg": float(np.rad2deg(_wrap_phase(mean[2]))),
                "phase_sd_deg": float(np.rad2deg(np.sqrt(variance[2]))),
                "phase_resultant": float(phase_resultant),
                "phase_identifiable": bool(phase_resultant > 0.7),
                "ess": self.ess, "particle_count": len(self.particles)}

    def normalized_variance(self, tolerances: Sequence[float], weights=None):
        tol = np.asarray(tolerances, float)
        if tol.shape != (3,) or np.any(tol <= 0) or not np.all(np.isfinite(tol)):
            raise ValueError("Tolerances must be positive (Hz, RF fraction, radians)")
        w = self.weights if weights is None else np.asarray(weights, float)
        p = self._unwrapped(w)
        center = np.sum(w[:, None]*p, axis=0)
        return float(np.sum(np.sum(w[:, None]*(p-center)**2, axis=0)/tol**2))

    def update(self, setting: CalibrationSetting, observation: complex,
               covariance_re_im, *, resample_at_fraction: float = 0.5, a: float = 0.98):
        cov = _covariance_2d(covariance_re_im)
        if not np.isfinite(observation.real) or not np.isfinite(observation.imag):
            raise ValueError("Observation is not finite")
        if not 0 < resample_at_fraction < 1:
            raise ValueError("Resampling fraction must be in (0,1)")
        inv = np.linalg.inv(cov)
        beta = 0.0
        stages = 0
        # Annealed importance updates avoid the single-observation particle
        # collapse otherwise common with high-SNR complex FID coefficients.
        while beta < 1.0-1e-12:
            predicted = self.predict(setting)
            error = np.column_stack((observation.real-predicted.real,
                                     observation.imag-predicted.imag))
            ll = -0.5*np.einsum("ni,ij,nj->n", error, inv, error)
            remaining = 1.0-beta
            proposed = self.log_weights+remaining*ll
            proposed -= logsumexp(proposed)
            proposed_ess = 1.0/np.sum(np.exp(proposed)**2)
            if proposed_ess >= resample_at_fraction*len(self.particles):
                fraction = remaining
            else:
                lo, hi = 0.0, remaining
                for _ in range(32):
                    mid = (lo+hi)/2
                    trial = self.log_weights+mid*ll
                    trial -= logsumexp(trial)
                    ess = 1.0/np.sum(np.exp(trial)**2)
                    if ess >= resample_at_fraction*len(self.particles):
                        lo = mid
                    else:
                        hi = mid
                fraction = max(lo, min(remaining, 1e-12))
            updated = self.log_weights+fraction*ll
            self.log_weights = updated-logsumexp(updated)
            beta += fraction
            stages += 1
            if beta < 1.0-1e-12:
                self._liu_west(a)
            if stages >= 128 and beta < 1.0-1e-12:
                raise ValueError("Particle update did not bridge the high-SNR likelihood")
        self.updates += 1
        before = self.ess
        if before < resample_at_fraction*len(self.particles):
            self._liu_west(a)
        return {"ess_before_resample": before, "resampled": stages > 1 or before < resample_at_fraction*len(self.particles),
                "tempering_stages": stages,
                "posterior": self.summary()}

    def _liu_west(self, a: float):
        if not 0 < a < 1:
            raise ValueError("Liu-West shrinkage a must be in (0,1)")
        w = self.weights
        transformed = self._unwrapped()
        ancestors = self.rng.choice(len(w), len(w), p=w)
        scales = np.maximum(np.std(transformed, axis=0), [1e-6, 1e-6, 1e-6])
        coords = np.column_stack((transformed[:, 0]/scales[0], transformed[:, 1]/scales[1],
                                  np.cos(transformed[:, 2]), np.sin(transformed[:, 2])))
        neighbors = cKDTree(coords).query(coords[ancestors], k=min(32, len(w)))[1]
        draw = np.empty_like(transformed)
        for j, (ancestor, indices) in enumerate(zip(ancestors, np.atleast_2d(neighbors))):
            local = transformed[indices].copy()
            local[:, 2] = transformed[ancestor, 2] + _wrap_phase(local[:, 2]-transformed[ancestor, 2])
            mean = np.mean(local, axis=0)
            centered = local-mean
            covariance = (1-a*a)*(centered.T@centered/max(len(local)-1, 1) +
                                   np.diag([1e-14, 1e-14, 1e-14]))
            draw[j] = a*transformed[ancestor]+(1-a)*mean+self.rng.multivariate_normal(np.zeros(3), covariance)
        self.particles[:, 0] = draw[:, 0]
        self.particles[:, 1] = np.exp(draw[:, 1])
        self.particles[:, 2] = _wrap_phase(draw[:, 2])
        self.log_weights.fill(-math.log(len(w)))

    def _expected_score(self, setting: CalibrationSetting, covariance, tolerances,
                        *, n_outcomes: int = 32):
        cov = _covariance_2d(covariance)
        prediction = self.predict(setting)
        latent = self.rng.choice(len(self.particles), n_outcomes, p=self.weights)
        noise = self.rng.multivariate_normal(np.zeros(2), cov, n_outcomes)
        synthetic = prediction[latent] + noise[:, 0]+1j*noise[:, 1]
        inv = np.linalg.inv(cov)
        scores = []
        for outcome in synthetic:
            err = np.column_stack((outcome.real-prediction.real,
                                   outcome.imag-prediction.imag))
            logw = self.log_weights - 0.5*np.einsum("ni,ij,nj->n", err, inv, err)
            logw -= logsumexp(logw)
            scores.append(self.normalized_variance(tolerances, np.exp(logw)))
        return float(np.mean(scores))

    def _threshold_allows(self, setting: CalibrationSetting):
        summary = self.summary()
        if setting.family == "ramsey":
            if TAU*2*summary["delta_sd_hz"]*setting.free_s >= np.pi:
                return False
        rate_sd = self.rf_hz_at_100pct*setting.amplitude_pct/100*summary["rf_scale_sd"]
        if TAU*2*rate_sd*setting.duration_s >= np.pi:
            return False
        return True

    def select(self, settings: Sequence[CalibrationSetting], covariance_re_im,
               tolerances: Sequence[float], *, policy: str = "variance",
               last_family: str | None = None, n_outcomes: int = 32):
        """Monte-Carlo expected variance or thresholded reduction per wall second."""
        if policy not in {"variance", "thresholded_per_second"}:
            raise ValueError("Unknown design policy")
        eligible = [s for s in settings if policy == "variance" or self._threshold_allows(s)]
        if policy == "thresholded_per_second" and last_family:
            alternating = [s for s in eligible if (s.family == "rabi") != (last_family == "rabi")]
            if alternating:
                eligible = alternating
        if not eligible:
            raise ValueError("No alias-safe calibrated measurement setting")
        current = self.normalized_variance(tolerances)
        ranked = []
        for setting in eligible:
            expected = self._expected_score(setting, covariance_re_im, tolerances,
                                            n_outcomes=n_outcomes)
            if policy == "variance":
                score = expected
            else:
                score = -(current-expected)/setting.expected_wall_s
            ranked.append((score, expected, setting))
        ranked.sort(key=lambda row: row[0])
        best = ranked[0]
        return best[2], {"current_normalized_variance": current,
                         "expected_normalized_variance": best[1],
                         "expected_drop_per_wall_second": (current-best[1])/best[2].expected_wall_s,
                         "policy": policy, "eligible_count": len(eligible)}


def calibration_particle_prior(delta_range_hz, rf_scale_range, phase_range_deg,
                               *, count=2048, seed=0):
    """Uniform common pilot prior; phase interval must span <=360 degrees."""
    if count < 16:
        raise ValueError("Too few particles")
    for interval in (delta_range_hz, rf_scale_range, phase_range_deg):
        if len(interval) != 2 or not np.all(np.isfinite(interval)) or interval[0] >= interval[1]:
            raise ValueError("Invalid prior range")
    if rf_scale_range[0] <= 0 or phase_range_deg[1]-phase_range_deg[0] > 360:
        raise ValueError("Invalid RF or phase prior")
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(*delta_range_hz, count),
                            rng.uniform(*rf_scale_range, count),
                            _wrap_phase(np.deg2rad(rng.uniform(*phase_range_deg, count)))))


def fixed_calibration_schedule(settings: Sequence[CalibrationSetting], count: int):
    """Outcome-blind balanced setting order for the shared-fit/fixed-posterior baselines."""
    if count < 1 or not settings:
        raise ValueError("A nonempty fixed calibration budget is required")
    by_family = {family: sorted((s for s in settings if s.family == family),
                                key=lambda s: (s.duration_s+s.free_s, s.phase_deg))
                 for family in ("rabi", "phase", "ramsey")}
    nonempty = [family for family, values in by_family.items() if values]
    used = {family: 0 for family in nonempty}
    output = []
    for i in range(count):
        family = nonempty[i % len(nonempty)]
        output.append(by_family[family][used[family] % len(by_family[family])])
        used[family] += 1
    return output


def coarse_fine_calibration_schedule(settings: Sequence[CalibrationSetting], count: int,
                                     *, preliminary_rf_scale: float,
                                     rf_hz_at_100pct: float):
    """Predeclared coarse pilot, then points around estimated quarter turn.

    Only the first stage may be used to infer preliminary_rf_scale. Later
    outcomes must not be fed back into the frozen coarse/fine schedule.
    """
    if preliminary_rf_scale <= 0 or rf_hz_at_100pct <= 0:
        raise ValueError("A valid independent coarse pilot is required")
    coarse_count = max(1, count//2)
    coarse = fixed_calibration_schedule(settings, coarse_count)
    t90_at_100 = 1/(4*rf_hz_at_100pct*preliminary_rf_scale)
    fine = sorted(settings, key=lambda s: abs(s.duration_s*s.amplitude_pct/100-t90_at_100))
    output = coarse + [fine[i % len(fine)] for i in range(count-coarse_count)]
    return output


def fit_calibration_joint(observations, *, rf_hz_at_100pct, receiver_gain=1+0j,
                          initial=(0., 1., 0.), delta_bounds_hz=(-1e4, 1e4),
                          scale_bounds=(0.1, 10.), max_starts=12):
    """Shared multistart complex least-squares baseline for fixed/coarse scans.

    observations: iterable of (CalibrationSetting, complex, 2x2 covariance).
    Identifiability uses whitened Jacobian rank and condition, not a fit flag.
    """
    rows = list(observations)
    if len(rows) < 3:
        raise ValueError("Joint calibration needs at least three observations")
    whitening = [np.linalg.inv(np.linalg.cholesky(_covariance_2d(c))) for _, _, c in rows]
    def residual(theta):
        return np.concatenate([white @ np.array([(pred.real-y.real), (pred.imag-y.imag)])
            for (setting, y, _), white in zip(rows, whitening)
            for pred in [predict_calibration(setting, theta, rf_hz_at_100pct=rf_hz_at_100pct,
                                              receiver_gain=receiver_gain)]])
    low = [delta_bounds_hz[0], scale_bounds[0], -np.pi]
    high = [delta_bounds_hz[1], scale_bounds[1], np.pi]
    starts = [np.asarray(initial, float)]
    for d in (delta_bounds_hz[0], 0.0, delta_bounds_hz[1]):
        for phase in (-np.pi/2, 0., np.pi/2):
            starts.append(np.array([d, initial[1], phase]))
    solutions = []
    for guess in starts[:max_starts]:
        guess = np.clip(guess, np.asarray(low)+1e-9, np.asarray(high)-1e-9)
        fit = least_squares(residual, guess, bounds=(low, high), max_nfev=400)
        solutions.append(fit)
    best = min(solutions, key=lambda fit: np.sum(fit.fun**2))
    singular = np.linalg.svd(best.jac, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else math.inf
    identifiable = bool(np.isfinite(condition) and condition < 1e6 and
                        np.all(best.x > np.asarray(low)+1e-6) and
                        np.all(best.x < np.asarray(high)-1e-6))
    covariance = np.linalg.pinv(best.jac.T @ best.jac) if identifiable else None
    return {"delta_hz": float(best.x[0]), "rf_scale": float(best.x[1]),
            "phase_deg": float(np.rad2deg(_wrap_phase(best.x[2]))),
            "t90_us_at_command_amplitude_100pct": float(1e6/(4*rf_hz_at_100pct*best.x[1])),
            "sd_delta_hz": float(np.sqrt(covariance[0, 0])) if covariance is not None else None,
            "sd_rf_scale": float(np.sqrt(covariance[1, 1])) if covariance is not None else None,
            "sd_phase_deg": float(np.rad2deg(np.sqrt(covariance[2, 2]))) if covariance is not None else None,
            "whitened_residual_rms": float(np.sqrt(np.mean(best.fun**2))),
            "jacobian_condition": condition, "identifiable": identifiable,
            "n_observations": len(rows)}


class TimingNotVerified(RuntimeError):
    """An echo sequence cannot be treated as T2 without checked pulse timing."""


def hahn_echo_timing(*, width_90_s, width_180_s, inter_pulse_gap_s,
                     acquisition_start_s, acquisition_end_s, timing_verified: bool):
    """Pulse-center TE and expected echo center after finite-width 90/180 pulses."""
    if not timing_verified:
        raise TimingNotVerified("Zero-amplitude delays and segment ordering are not verified")
    vals = [width_90_s, width_180_s, inter_pulse_gap_s, acquisition_start_s, acquisition_end_s]
    if not np.all(np.isfinite(vals)) or width_90_s <= 0 or width_180_s <= 0 or inter_pulse_gap_s < 0:
        raise ValueError("Invalid echo timing")
    center_90 = width_90_s/2
    center_180 = width_90_s+inter_pulse_gap_s+width_180_s/2
    echo_center = 2*center_180-center_90
    if not acquisition_start_s <= echo_center <= acquisition_end_s:
        raise ValueError("Expected echo center is outside the acquisition window")
    return {"TE_s": float(echo_center-center_90),
            "excitation_center_s": float(center_90),
            "refocus_center_s": float(center_180),
            "echo_center_s": float(echo_center),
            "acquisition_start_s": float(acquisition_start_s),
            "acquisition_end_s": float(acquisition_end_s)}


def echo_fisher_information(te_s: Sequence[float], amplitude: float, t2_s: float,
                            noise_sd: Sequence[float] | float):
    """Fisher matrix for A exp(-TE/T2), including nuisance amplitude A."""
    te = np.atleast_1d(np.asarray(te_s, float))
    sigma = np.broadcast_to(np.asarray(noise_sd, float), te.shape)
    if np.any(te < 0) or t2_s <= 0 or amplitude <= 0 or np.any(sigma <= 0):
        raise ValueError("Invalid echo model parameters")
    attenuation = np.exp(-te/t2_s)
    derivatives = np.column_stack((attenuation, amplitude*attenuation*te/t2_s**2))
    return (derivatives/sigma[:, None]).T @ (derivatives/sigma[:, None])


class EchoGridPosterior:
    """2D amplitude/T2 posterior and posterior-averaged CRLB design."""

    def __init__(self, amplitudes, t2_seconds, *, prior_weights=None):
        aa = np.asarray(amplitudes, float)
        tt = np.asarray(t2_seconds, float)
        if aa.ndim != 1 or tt.ndim != 1 or len(aa) < 3 or len(tt) < 3:
            raise ValueError("Echo posterior needs amplitude and T2 grids")
        if np.any(aa <= 0) or np.any(tt <= 0) or not np.all(np.isfinite(aa)) or not np.all(np.isfinite(tt)):
            raise ValueError("Amplitude and T2 grids must be finite and positive")
        self.amplitudes, self.t2_seconds = np.meshgrid(aa, tt, indexing="ij")
        if prior_weights is None:
            weights = np.ones(self.amplitudes.shape)
        else:
            weights = np.asarray(prior_weights, float)
            if weights.shape != self.amplitudes.shape or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("Invalid echo prior weights")
        self.log_weights = np.log(np.maximum(weights, 1e-300))-math.log(float(weights.sum()))
        self.prior_weights = np.exp(self.log_weights)
        self.history: list[tuple[float, float]] = []

    @property
    def weights(self):
        return np.exp(self.log_weights)

    def update(self, te_s: float, echo_amplitude: float, noise_sd: float):
        if te_s < 0 or noise_sd <= 0 or not all(np.isfinite([te_s, echo_amplitude, noise_sd])):
            raise ValueError("Invalid echo observation")
        prediction = self.amplitudes*np.exp(-te_s/self.t2_seconds)
        logw = self.log_weights - 0.5*((echo_amplitude-prediction)/noise_sd)**2
        self.log_weights = logw-logsumexp(logw)
        self.history.append((te_s, noise_sd))
        return self.summary()

    def summary(self):
        w = self.weights
        mean_t2 = float(np.sum(w*self.t2_seconds))
        mean_a = float(np.sum(w*self.amplitudes))
        return {"T2_s": mean_t2,
                "T2_sd_s": float(np.sqrt(np.sum(w*(self.t2_seconds-mean_t2)**2))),
                "amplitude": mean_a,
                "amplitude_sd": float(np.sqrt(np.sum(w*(self.amplitudes-mean_a)**2))),
                "observations": len(self.history)}

    def _prior_information(self):
        p = np.column_stack((self.amplitudes.ravel(), self.t2_seconds.ravel()))
        w = self.prior_weights.ravel()
        mean = np.sum(w[:, None]*p, axis=0)
        d = p-mean
        cov = (w[:, None]*d).T @ d
        return np.linalg.pinv(cov)

    def expected_relative_crlb(self, proposed_te_s: float | None = None,
                               *, proposed_noise_sd: float | None = None,
                               support_points: int = 128):
        """Posterior expectation of sqrt(CRLB(T2))/T2; a design proxy."""
        history = list(self.history)
        if proposed_te_s is not None:
            if proposed_noise_sd is None or proposed_noise_sd <= 0:
                raise ValueError("Proposed TE needs measured noise SD")
            history.append((proposed_te_s, proposed_noise_sd))
        weights = self.weights.ravel()
        # Deterministic systematic weighted quantiles avoid a huge grid loop.
        cdf = np.cumsum(weights)
        indexes = np.searchsorted(cdf, (np.arange(support_points)+0.5)/support_points)
        baseline = self._prior_information()
        result = []
        for index in indexes:
            amp = float(self.amplitudes.ravel()[index]); t2 = float(self.t2_seconds.ravel()[index])
            fisher = baseline.copy()
            for te, sd in history:
                fisher += echo_fisher_information([te], amp, t2, sd)
            bound = np.linalg.pinv(fisher)[1, 1]
            result.append(np.sqrt(max(bound, 0.))/t2)
        return float(np.mean(result))

    def choose_te(self, candidates_s: Sequence[float], *, noise_sd: float,
                  preparation_s: float, recovery_s: float, overhead_s: float = 0.,
                  policy: str = "adaptive_per_second"):
        if policy not in {"adaptive_per_second", "minimum_crlb"}:
            raise ValueError("Unknown TE design policy")
        current = self.expected_relative_crlb()
        ranked = []
        for te in candidates_s:
            if te <= 0:
                continue
            after = self.expected_relative_crlb(float(te), proposed_noise_sd=noise_sd)
            wall = float(preparation_s+recovery_s+overhead_s+te)
            if wall <= 0:
                raise ValueError("Acquisition cost must be positive")
            score = -(current-after)/wall if policy == "adaptive_per_second" else after
            ranked.append((score, te, after, wall))
        if not ranked:
            raise ValueError("No valid TE candidates")
        _, te, after, wall = min(ranked)
        return {"TE_s": float(te), "expected_relative_crlb": float(after),
                "expected_crlb_reduction_per_wall_s": float((current-after)/wall),
                "expected_wall_s": float(wall), "policy": policy}


def static_echo_design(amplitudes, t2_seconds, *, candidates_s, count,
                       noise_sd, preparation_s, recovery_s, wall_budget_s=None):
    """Freeze a prior-optimized TE list before any adaptive outcome is seen.

    If supplied, wall_budget_s is a hard expected cost cap and design is
    selected by CRLB reduction per wall second, matching an adaptive budget.
    """
    posterior = EchoGridPosterior(amplitudes, t2_seconds)
    design = []
    spent = 0.0
    for _ in range(count):
        eligible = [float(te) for te in candidates_s if wall_budget_s is None or
                    spent+preparation_s+recovery_s+float(te) <= wall_budget_s]
        if not eligible:
            break
        choice = posterior.choose_te(eligible, noise_sd=noise_sd,
                                     preparation_s=preparation_s, recovery_s=recovery_s,
                                     policy="minimum_crlb" if wall_budget_s is None else "adaptive_per_second")
        design.append(choice["TE_s"])
        spent += choice["expected_wall_s"]
        posterior.history.append((choice["TE_s"], noise_sd))
    return design


def conventional_log_echo_design(min_te_s, max_te_s, count):
    if min_te_s <= 0 or max_te_s <= min_te_s or count < 2:
        raise ValueError("Invalid logarithmic TE design")
    return np.geomspace(min_te_s, max_te_s, count).tolist()


def choose_fid_acquisition_plan(pilot, *, target_se_hz, max_repeats=12):
    """Use actual independent acquisition scatter, never FID points as shots.

    Pilot rows: {sample_count, estimate_hz, wall_seconds}. Only sample counts
    already physically tested at least twice are considered. This plan is a
    prediction; a later independent validation still decides success.
    """
    if target_se_hz <= 0 or max_repeats < 2:
        raise ValueError("Invalid FID tolerance or budget")
    groups: dict[int, list] = {}
    for row in pilot:
        n = int(row["sample_count"])
        f = float(row["estimate_hz"]); wall = float(row["wall_seconds"])
        if n < 2 or wall <= 0 or not np.isfinite(f):
            raise ValueError("Invalid physical FID pilot row")
        groups.setdefault(n, []).append((f, wall))
    independent_means = [float(np.mean([row[0] for row in rows]))
                         for rows in groups.values() if len(rows) >= 2]
    if len(independent_means) >= 2:
        spread = max(independent_means)-min(independent_means)
        # Large sample-length dependence is evidence of an aliased/changed
        # spectral component or misspecified local fit, not improved precision.
        se_limit = max(float(np.std([row[0] for row in rows], ddof=1))/math.sqrt(len(rows))
                       for rows in groups.values() if len(rows) >= 2)
        if spread > max(3*se_limit, 2*target_se_hz):
            return {"status": "MODEL_MISMATCH", "reason":
                    "Pilot frequency depends on FID length; verify component identity and axis",
                    "between_length_spread_hz": float(spread)}
    candidates = []
    for n, rows in groups.items():
        if len(rows) < 2:
            continue
        sd = float(np.std([x[0] for x in rows], ddof=1))
        # Zero scatter from two repeats is not proof of zero uncertainty.
        if sd <= 0:
            continue
        repeat_count = max(2, math.ceil((sd/target_se_hz)**2))
        if repeat_count > max_repeats:
            continue
        cost = repeat_count*float(np.mean([x[1] for x in rows]))
        candidates.append((cost, n, repeat_count, sd/math.sqrt(repeat_count)))
    if not candidates:
        return {"status": "BUDGET_EXHAUSTED", "reason": "No empirically tested FID length reaches target"}
    wall, n, repeats, predicted = min(candidates)
    return {"status": "PLAN_ONLY", "sample_count": n, "physical_acquisitions": repeats,
            "predicted_frequency_se_hz": float(predicted), "target_se_hz": float(target_se_hz),
            "predicted_wall_s": float(wall), "requires_independent_validation": True}


class RabiRateMap:
    """Measured SDK amplitude (%) to effective RF rate (Hz) for x and y phases."""

    def __init__(self, amplitude_pct, rate_x_hz, rate_y_hz):
        amp = np.asarray(amplitude_pct, float)
        rx = np.asarray(rate_x_hz, float); ry = np.asarray(rate_y_hz, float)
        if amp.ndim != 1 or len(amp) < 3 or rx.shape != amp.shape or ry.shape != amp.shape:
            raise ValueError("At least three paired x/y Rabi pilot points required")
        if not np.all(np.diff(amp) > 0) or np.any(amp < 0) or np.any(rx < 0) or np.any(ry < 0):
            raise ValueError("Rabi pilot amplitude/rates must be sorted and nonnegative")
        if not np.all(np.isfinite(amp)) or not np.all(np.isfinite(rx)) or not np.all(np.isfinite(ry)):
            raise ValueError("Nonfinite Rabi pilot")
        self.amplitude_pct = amp
        self.x = PchipInterpolator(amp, rx, extrapolate=False)
        self.y = PchipInterpolator(amp, ry, extrapolate=False)
        self.non_linearity = {}
        for label, measured in (("x", rx), ("y", ry)):
            slope = np.dot(amp, measured)/max(np.dot(amp, amp), 1e-30)
            residual = measured-slope*amp
            self.non_linearity[label] = float(np.sqrt(np.mean(residual**2))/max(np.max(measured), 1e-30))

    def rate_hz(self, amplitude_pct, phase_deg):
        amplitude = float(amplitude_pct)
        if amplitude < self.amplitude_pct[0] or amplitude > self.amplitude_pct[-1]:
            raise ValueError("RF command outside measured Rabi map")
        phase = np.deg2rad(phase_deg)
        return float(np.hypot(float(self.x(amplitude))*np.cos(phase),
                              float(self.y(amplitude))*np.sin(phase)))


def iq_to_sdk(i_values_pct, q_values_pct, *, sample_scale, relative_scale,
              duration_s, verified_headroom_pct):
    """Exact u=Samp(Srel I+iQ), returned as SDK amplitudes/phases per segment.

    These effective controls are not a measurement of independent DAC I/Q
    branches unless separate hardware tests establish that interpretation.
    No clipping is performed, because clipping changes the requested pulse.
    """
    i = np.atleast_1d(np.asarray(i_values_pct, float))
    q = np.atleast_1d(np.asarray(q_values_pct, float))
    if i.shape != q.shape or i.ndim != 1 or len(i) == 0 or not np.all(np.isfinite(i+q)):
        raise ValueError("I and Q segment arrays must have the same finite shape")
    if duration_s <= 0 or sample_scale <= 0 or relative_scale <= 0 or verified_headroom_pct <= 0:
        raise ValueError("Positive scales, segment duration and verified headroom required")
    command = sample_scale*(relative_scale*i+1j*q)
    amplitude = np.abs(command)
    if np.max(amplitude) > verified_headroom_pct+1e-10:
        raise ValueError("IQ command exceeds independently verified RF headroom")
    return [{"duration_s": float(duration_s), "amplitude_pct": float(amp),
             "phase_deg": float(np.rad2deg(np.angle(z)) % 360.)}
            for amp, z in zip(amplitude, command)]


def notebook_repetition_cost(measured_bloch, target_bloch):
    """Q-CTRL notebook repetition cost; this is NOT process fidelity."""
    r = np.asarray(measured_bloch, float); target = np.asarray(target_bloch, float)
    if r.shape != target.shape or r.ndim != 2 or r.shape[1] != 3 or len(r) == 0:
        raise ValueError("Bloch arrays must have matching Nx3 shape")
    if not np.all(np.isfinite(r+target)):
        raise ValueError("Nonfinite Bloch data")
    return float(np.mean(1.-((1.+np.sum(r*target, axis=1))/2.)**2))


def repetition_resource_count(repetitions: Sequence[int], readout_bases: Sequence[str],
                              *, independent_repeats: int = 1):
    """Count excitations and requested RF gate repetitions for one cost point."""
    if not repetitions or not readout_bases or independent_repeats < 1:
        raise ValueError("Empty repetition design")
    if any(n < 1 or n % 4 != 1 for n in repetitions):
        raise ValueError("X90 error-amplification repetitions should be 1 mod 4")
    acquisitions = len(repetitions)*len(readout_bases)*independent_repeats
    rf_gate_uses = sum(repetitions)*len(readout_bases)*independent_repeats
    return {"physical_acquisitions": acquisitions, "rf_gate_uses": rf_gate_uses,
            "readout_bases": list(readout_bases), "repetitions": list(repetitions)}


class RFGPOptimizer:
    """Local Matérn GP with measured per-point cost variance and expected improvement."""

    def __init__(self, bounds, *, seed=0):
        bounds = np.asarray(bounds, float)
        if bounds.shape != (2, 2) or np.any(bounds[:, 0] <= 0) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Bounds must be [[Samp_min,max],[Srel_min,max]] from pilot/headroom")
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.points: list[np.ndarray] = []
        self.costs: list[float] = []
        self.variances: list[float] = []

    def observe(self, sample_scale, relative_scale, *, cost, cost_variance):
        p = np.array([sample_scale, relative_scale], float)
        if np.any(p < self.bounds[:, 0]) or np.any(p > self.bounds[:, 1]):
            raise ValueError("RF point outside frozen pilot bounds")
        if not np.isfinite(cost) or not np.isfinite(cost_variance) or cost_variance < 0:
            raise ValueError("Cost and measured variance must be finite")
        self.points.append(p)
        self.costs.append(float(cost))
        self.variances.append(float(cost_variance))

    def propose(self, *, candidate_count=1024, exploration=0.01):
        if candidate_count < 32:
            raise ValueError("Candidate pool too small")
        if not self.points:
            return np.mean(self.bounds, axis=1).tolist(), {"method": "initial_center"}
        if len(self.points) < 5:
            # Space-filling startup, avoiding the fiction of a fitted GP from 1-3 points.
            fixed = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
            proposed = self.bounds[:, 0]+fixed[len(self.points)-1]*(self.bounds[:, 1]-self.bounds[:, 0])
            return proposed.tolist(), {"method": "space_filling"}
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        from sklearn.exceptions import ConvergenceWarning
        x = (np.asarray(self.points)-self.bounds[:, 0])/(self.bounds[:, 1]-self.bounds[:, 0])
        y = np.asarray(self.costs)
        variance = np.asarray(self.variances)
        alpha = np.maximum(variance, max(np.var(y)*1e-6, 1e-12))
        kernel = ConstantKernel(max(float(np.var(y)), 1e-6), (1e-8, 1e3))*Matern(
            length_scale=[.3, .3], length_scale_bounds=(.05, 2.), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                      n_restarts_optimizer=2, random_state=int(self.rng.integers(2**31)))
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            gp.fit(x, y)
        candidates = self.rng.uniform(size=(candidate_count, 2))
        candidates = np.vstack((candidates, np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [.5, .5]])))
        mu, sd = gp.predict(candidates, return_std=True)
        best = float(np.min(y))
        improvement = best-mu-exploration*max(np.std(y), 1e-12)
        z = improvement/np.maximum(sd, 1e-12)
        ei = improvement*norm.cdf(z)+sd*norm.pdf(z)
        distance = np.linalg.norm(candidates[:, None, :]-x[None, :, :], axis=2).min(axis=1)
        ei[distance < 1e-4] = -np.inf
        ix = int(np.argmax(ei))
        if not np.isfinite(ei[ix]):
            raise ValueError("No new candidate inside RF bounds")
        point = self.bounds[:, 0]+candidates[ix]*(self.bounds[:, 1]-self.bounds[:, 0])
        return point.tolist(), {"method": "matern_gp_expected_improvement",
                                "expected_improvement": float(ei[ix]),
                                "predicted_cost": float(mu[ix]),
                                "predicted_cost_sd": float(sd[ix]),
                                "gp_fit_warnings": [str(w.message) for w in fit_warnings]}


def rf_coarse_grid(bounds, levels=3):
    bounds = np.asarray(bounds, float)
    if bounds.shape != (2, 2) or levels < 2:
        raise ValueError("Invalid RF grid")
    return [(float(x), float(y))
            for x in np.linspace(*bounds[0], levels)
            for y in np.linspace(*bounds[1], levels)]


def rf_sequential_scan(bounds, levels=5, *, center=(1., 1.)):
    """Frozen coordinate scan baseline; first Samp, then Srel at same center."""
    bounds = np.asarray(bounds, float)
    if bounds.shape != (2, 2) or levels < 2:
        raise ValueError("Invalid RF scan")
    return [(float(x), float(center[1])) for x in np.linspace(*bounds[0], levels)] + [
        (float(center[0]), float(y)) for y in np.linspace(*bounds[1], levels)]


def rf_nelder_mead(evaluate: Callable[[float, float], float], bounds,
                   *, max_evaluations: int, start=(1., 1.),
                   initial_step_fraction=.1):
    """Bounded local Nelder–Mead RF reference with an exact evaluation cap.

    `evaluate` is supplied by the Windows runner and must return a local
    measured cost. Each call has the same fixed repetition/readout design.
    This numerical function itself never contacts the device.
    """
    limits = np.asarray(bounds, float)
    origin = np.asarray(start, float)
    if limits.shape != (2, 2) or origin.shape != (2,) or max_evaluations < 3:
        raise ValueError("Invalid Nelder-Mead bounds, start or budget")
    if np.any(origin < limits[:, 0]) or np.any(origin > limits[:, 1]):
        raise ValueError("Nelder-Mead start lies outside RF pilot bounds")
    if not 0 < initial_step_fraction < .5:
        raise ValueError("Initial simplex fraction must be in (0,0.5)")
    steps = initial_step_fraction*(limits[:, 1]-limits[:, 0])
    simplex = np.vstack((origin, origin+[steps[0], 0.], origin+[0., steps[1]]))
    for j in (1, 2):
        if np.any(simplex[j] > limits[:, 1]):
            simplex[j] = origin - (simplex[j]-origin)
    history = []
    def objective(point):
        if len(history) >= max_evaluations:
            raise RuntimeError("Nelder-Mead evaluation budget exhausted")
        value = float(evaluate(float(point[0]), float(point[1])))
        if not np.isfinite(value):
            raise ValueError("Measured RF cost is nonfinite")
        history.append({"sample_scale": float(point[0]),
                        "relative_scale": float(point[1]), "cost": value})
        return value
    result = minimize(objective, origin, method="Nelder-Mead",
                      bounds=[tuple(row) for row in limits],
                      options={"initial_simplex": simplex, "maxfev": max_evaluations,
                               "xatol": 1e-3, "fatol": 1e-4})
    winner = min(history, key=lambda row: row["cost"])
    return {"best": winner, "evaluations": len(history), "history": history,
            "optimizer_message": str(result.message),
            "converged_before_budget": bool(result.success)}
