"""Numerically stable particle inference for short-pulse NMR calibration.

Particles are `[FID frequency shift Hz, t90 us, relative phase radians]`.
Only the latter is circular.  Resampling uses local transformed-coordinate
Liu-West kernels so a phase boundary or separated Rabi aliases are not
collapsed into an invalid global arithmetic mean.  Drift may broaden an
existing posterior; changing a command never resets the physical posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .likelihood import Candidate, GaussianObservation, NMRModel, predict_complex


def wrap_phase(phase_rad: np.ndarray | float) -> np.ndarray:
    """Map phases to [-pi, pi), including values crossing the branch cut."""
    return (np.asarray(phase_rad) + np.pi) % (2 * np.pi) - np.pi


def phase_difference(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return wrap_phase(np.asarray(a) - np.asarray(b))


def normalized_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize in log space and return the log normalizer."""
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(np.isnan(values)):
        raise ValueError("Expected finite or -inf one-dimensional log weights")
    peak = float(np.max(values))
    if not np.isfinite(peak):
        raise ValueError("Every particle has zero likelihood")
    shifted = values - peak
    log_sum = math.log(float(np.exp(shifted).sum()))
    # Form the normalized weights from the shifted values directly.  Using
    # `values - (peak + log_sum)` loses low-order bits when peak is ~-1e9.
    return shifted - log_sum, peak + log_sum


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid particle weights")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Zero particle weight")
    unit = values / total
    return float(1 / np.dot(unit, unit))


def systematic_resample_indices(weights: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling with one uniform draw; no dependence on phase."""
    weights = np.asarray(weights, dtype=np.float64)
    effective_sample_size(weights)  # validates the distribution
    cumulative = np.cumsum(weights / weights.sum())
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(cumulative, positions, side="right")


@dataclass(frozen=True)
class PriorBounds:
    delta_hz: tuple[float, float]
    t90_us: tuple[float, float]
    phase_center_rad: float = 0.0
    phase_half_width_rad: float = math.pi

    def __post_init__(self) -> None:
        values = (*self.delta_hz, *self.t90_us,
                  self.phase_center_rad, self.phase_half_width_rad)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite pilot prior")
        if not self.delta_hz[0] < self.delta_hz[1]:
            raise ValueError("Frequency prior needs an interval")
        if not 0 < self.t90_us[0] < self.t90_us[1]:
            raise ValueError("t90 prior must be positive and nondegenerate")
        if not 0 < self.phase_half_width_rad <= math.pi:
            raise ValueError("Circular phase half-width must be in (0, pi]")


def weighted_posterior_moments(particles: np.ndarray, weights: np.ndarray) -> dict:
    """Means/variances with a circular phase; phase identity is separate."""
    particles = np.asarray(particles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[1] != 3 or len(particles) != len(weights):
        raise ValueError("Particle/weight shape mismatch")
    effective_sample_size(weights)
    w = weights / weights.sum()
    linear_mean = np.sum(w[:, None] * particles[:, :2], axis=0)
    linear_var = np.sum(w[:, None] * (particles[:, :2] - linear_mean)**2, axis=0)
    resultant_complex = np.sum(w * np.exp(1j * particles[:, 2]))
    resultant = float(abs(resultant_complex))
    phase_mean = float(np.angle(resultant_complex)) if resultant > 1e-12 else None
    # This intrinsic squared angular deviation remains useful to design even
    # for a broad posterior.  It is never interpreted as phase identified
    # when the circular resultant is weak.
    center = phase_mean if phase_mean is not None else 0.0
    phase_var = float(np.sum(w * phase_difference(particles[:, 2], center)**2))
    return {"mean_delta_hz": float(linear_mean[0]),
            "mean_t90_us": float(linear_mean[1]),
            "mean_phase_rad": phase_mean,
            "var_delta_hz2": float(linear_var[0]),
            "var_t90_us2": float(linear_var[1]),
            "var_phase_rad2": phase_var,
            "phase_resultant": resultant,
            "phase_identified": resultant >= 0.5}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       probability: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(probability, cumulative, values[order]))


def _reflect(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    span = upper - lower
    shifted = (values - lower) % (2 * span)
    return lower + np.where(shifted <= span, shifted, 2 * span - shifted)


@dataclass(frozen=True)
class UpdateStats:
    log_evidence: float
    ess_before_resampling: float
    ess_after_update: float
    resampled: bool
    posterior_score_hint: dict = field(default_factory=dict)
    tempering_stages: int = 1
    move_acceptance: float = 0.0
    minimum_stage_ess: float | None = None


@dataclass
class ParticleFilter:
    particles: np.ndarray
    log_weights: np.ndarray
    bounds: PriorBounds
    rng: np.random.Generator = field(repr=False)
    updates: int = 0
    resampling_count: int = 0
    history: list[tuple[np.ndarray, Candidate, NMRModel, np.ndarray]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.particles = np.asarray(self.particles, dtype=np.float64)
        self.log_weights = np.asarray(self.log_weights, dtype=np.float64)
        if self.particles.ndim != 2 or self.particles.shape[1] != 3 or len(self.particles) < 8:
            raise ValueError("Particle filter needs at least eight 3D particles")
        if self.log_weights.shape != (len(self.particles),):
            raise ValueError("One log weight per particle required")
        if not np.all(np.isfinite(self.particles)) or np.any(self.particles[:, 1] <= 0):
            raise ValueError("Nonfinite or nonphysical particles")
        self.log_weights, _ = normalized_log_weights(self.log_weights)
        self.particles[:, 2] = wrap_phase(self.particles[:, 2])

    @classmethod
    def from_prior(cls, bounds: PriorBounds, n_particles: int,
                   rng: np.random.Generator) -> "ParticleFilter":
        """Draw a broad pilot-derived box prior; no hidden +20/200 Hz range."""
        if n_particles < 8:
            raise ValueError("Particle convergence needs at least eight particles")
        particles = np.column_stack((
            rng.uniform(*bounds.delta_hz, size=n_particles),
            rng.uniform(*bounds.t90_us, size=n_particles),
            wrap_phase(bounds.phase_center_rad + rng.uniform(
                -bounds.phase_half_width_rad, bounds.phase_half_width_rad, size=n_particles))))
        return cls(particles, np.full(n_particles, -math.log(n_particles)), bounds, rng)

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.log_weights)

    @property
    def ess(self) -> float:
        return effective_sample_size(self.weights)

    def _enforce_bounds(self) -> None:
        self.particles[:, 0] = _reflect(self.particles[:, 0], *self.bounds.delta_hz)
        lower_log, upper_log = np.log(self.bounds.t90_us)
        self.particles[:, 1] = np.exp(_reflect(np.log(self.particles[:, 1]),
                                                lower_log, upper_log))
        if self.bounds.phase_half_width_rad < math.pi:
            relative = phase_difference(self.particles[:, 2], self.bounds.phase_center_rad)
            relative = _reflect(relative, -self.bounds.phase_half_width_rad,
                                self.bounds.phase_half_width_rad)
            self.particles[:, 2] = wrap_phase(self.bounds.phase_center_rad + relative)
        else:
            self.particles[:, 2] = wrap_phase(self.particles[:, 2])

    def propagate_drift(self, *, delta_std_hz: float = 0.0,
                        log_t90_std: float = 0.0,
                        phase_std_rad: float = 0.0) -> None:
        """Broaden an *unused prior* using measured drift, without resetting it.

        Within a live Bayesian comparison, time-correlated drift belongs in
        the pilot/return-control covariance. A state-space transition after
        observations needs a different history-aware smoother; applying this
        simple broadening then replaying all old likelihoods would be wrong.
        """
        if self.history:
            raise RuntimeError("Use return-control drift covariance after observations, not prior broadening")
        scales = np.array([delta_std_hz, log_t90_std, phase_std_rad], dtype=float)
        if not np.all(np.isfinite(scales)) or np.any(scales < 0):
            raise ValueError("Drift standard deviations must be nonnegative")
        increments = self.rng.standard_normal(self.particles.shape) * scales
        self.particles[:, 0] += increments[:, 0]
        self.particles[:, 1] *= np.exp(increments[:, 1])
        self.particles[:, 2] += increments[:, 2]
        self._enforce_bounds()

    def resample(self, *, bandwidth: float = 0.15) -> None:
        """Systematic resampling plus *local* Liu-West rejuvenation.

        The kernel uses nearest neighbors in `(delta, log(t90), circular phase)`
        rather than a global mean.  This retains distinct Rabi aliases and
        treats +pi/-pi as neighbors.  Pilot bounds are reflected, not clipped.
        """
        if not 0 <= bandwidth < 1:
            raise ValueError("Liu-West bandwidth must be in [0, 1)")
        chosen = systematic_resample_indices(self.weights, self.rng)
        selected = self.particles[chosen].copy()
        if bandwidth:
            transformed = np.column_stack((selected[:, 0], np.log(selected[:, 1]),
                                           selected[:, 2]))
            scale = np.array([self.bounds.delta_hz[1] - self.bounds.delta_hz[0],
                              math.log(self.bounds.t90_us[1] / self.bounds.t90_us[0]),
                              2 * self.bounds.phase_half_width_rad])
            neighbor_count = min(len(selected), max(8, int(math.sqrt(len(selected)))))
            a = math.sqrt(1 - bandwidth**2)
            renewed = transformed.copy()
            for i in range(len(selected)):
                difference = transformed - transformed[i]
                difference[:, 2] = phase_difference(transformed[:, 2], transformed[i, 2])
                distances = np.sum((difference / scale)**2, axis=1)
                near = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
                local = difference[near]
                local_mean = local.mean(axis=0)
                local_cov = np.cov(local, rowvar=False, ddof=1)
                # If many identical ancestors were selected, preserve a tiny
                # physically bounded kernel; otherwise a degenerate empirical
                # covariance could make future exploration impossible.
                local_cov += np.diag((1e-5 * scale)**2)
                jitter = self.rng.multivariate_normal(np.zeros(3), local_cov)
                renewed[i] += (1 - a) * local_mean + bandwidth * jitter
            selected[:, 0] = renewed[:, 0]
            selected[:, 1] = np.exp(renewed[:, 1])
            selected[:, 2] = renewed[:, 2]
        self.particles = selected
        self._enforce_bounds()
        self.log_weights[:] = -math.log(len(selected))
        self.resampling_count += 1

    def _history_log_likelihood(self, particles: np.ndarray) -> np.ndarray:
        total = np.zeros(len(particles), dtype=np.float64)
        for observed, candidate, model, covariance in self.history:
            total += GaussianObservation(covariance).logpdf(
                observed, predict_complex(particles, candidate, model))
        return total

    def _metropolis_move(self, likelihood: np.ndarray, temperature: float,
                         observed: np.ndarray, candidate: Candidate,
                         model: NMRModel, gaussian: GaussianObservation,
                         *, sweeps: int = 4) -> tuple[np.ndarray, float]:
        """Posterior-preserving differential/jitter moves after tempering.

        These random-walk proposals are symmetric in transformed coordinates.
        A uniform physical t90 prior has density proportional to t90 in
        log(t90) coordinates; its Jacobian appears in the acceptance ratio.
        The donor population is frozen for each sweep so multimodal Rabi
        aliases may persist or exchange mass instead of being averaged.
        """
        count = len(self.particles)
        accepted_total = 0
        previous_log_likelihood = self._history_log_likelihood(self.particles)
        scales = np.array([self.bounds.delta_hz[1] - self.bounds.delta_hz[0],
                           math.log(self.bounds.t90_us[1] / self.bounds.t90_us[0]),
                           2 * self.bounds.phase_half_width_rad])
        for sweep in range(sweeps):
            # Split the population. Each target half proposes using only the
            # *opposite*, frozen half; thus donor selection and local step
            # scales do not depend on the target's old or proposed location.
            order = self.rng.permutation(count)
            groups = (order[:count // 2], order[count // 2:])
            for target, donor in ((groups[0], groups[1]), (groups[1], groups[0])):
                old = np.column_stack((self.particles[target, 0],
                                       np.log(self.particles[target, 1]),
                                       self.particles[target, 2]))
                donor_values = np.column_stack((self.particles[donor, 0],
                                                np.log(self.particles[donor, 1]),
                                                self.particles[donor, 2]))
                proposed = old.copy()
                if sweep % 2 == 0:
                    donors_a = self.rng.integers(len(donor), size=len(target))
                    donors_b = self.rng.integers(len(donor) - 1, size=len(target))
                    donors_b += donors_b >= donors_a
                    difference = donor_values[donors_a] - donor_values[donors_b]
                    difference[:, 2] = phase_difference(
                        donor_values[donors_a, 2], donor_values[donors_b, 2])
                    proposed += (2.38 / math.sqrt(2 * 3)) * difference
                    proposed += self.rng.normal(size=old.shape) * (0.002 * scales)
                else:
                    # Symmetric local random walk based on opposite-half
                    # scatter, including phase distance across ±pi.
                    phase_center = float(np.angle(np.mean(np.exp(1j * donor_values[:, 2]))))
                    phase_std = np.std(phase_difference(donor_values[:, 2], phase_center))
                    spread = np.array([np.std(donor_values[:, 0]),
                                       np.std(donor_values[:, 1]), phase_std])
                    step = 0.35 * spread + 0.002 * scales
                    proposed += self.rng.normal(size=old.shape) * step
                proposed[:, 0] = _reflect(proposed[:, 0], *self.bounds.delta_hz)
                lower_log, upper_log = np.log(self.bounds.t90_us)
                proposed[:, 1] = _reflect(proposed[:, 1], lower_log, upper_log)
                if self.bounds.phase_half_width_rad < math.pi:
                    relative = phase_difference(proposed[:, 2], self.bounds.phase_center_rad)
                    relative = _reflect(relative, -self.bounds.phase_half_width_rad,
                                        self.bounds.phase_half_width_rad)
                    proposed[:, 2] = wrap_phase(self.bounds.phase_center_rad + relative)
                else:
                    proposed[:, 2] = wrap_phase(proposed[:, 2])
                physical = np.column_stack((proposed[:, 0], np.exp(proposed[:, 1]),
                                            proposed[:, 2]))
                candidate_likelihood = gaussian.logpdf(
                    observed, predict_complex(physical, candidate, model))
                candidate_previous = self._history_log_likelihood(physical)
                log_ratio = (candidate_previous - previous_log_likelihood[target] +
                             temperature * (candidate_likelihood - likelihood[target]) +
                             proposed[:, 1] - old[:, 1])
                accepted = np.log(self.rng.random(len(target))) < np.minimum(log_ratio, 0.)
                moved = target[accepted]
                self.particles[moved] = physical[accepted]
                likelihood[moved] = candidate_likelihood[accepted]
                previous_log_likelihood[moved] = candidate_previous[accepted]
                accepted_total += len(moved)
        return likelihood, accepted_total / max(1, sweeps * count)

    def update(self, observed: np.ndarray, candidate: Candidate, model: NMRModel,
               covariance: np.ndarray, *, resample_below_ess_fraction: float = 0.5,
               rejuvenation_bandwidth: float = 0.15,
               max_tempering_stages: int = 80) -> UpdateStats:
        if not 0 < resample_below_ess_fraction <= 1:
            raise ValueError("ESS threshold must lie in (0, 1]")
        if max_tempering_stages < 1:
            raise ValueError("At least one tempering stage is required")
        # `rejuvenation_bandwidth` remains an exposed sensitivity parameter
        # for legacy callers, but the live update uses MH-corrected jitter
        # rather than unconditional Liu-West shrinkage.  The latter can lock
        # in a wrong particle after a sharply informative first FID.
        if not 0 <= rejuvenation_bandwidth < 1:
            raise ValueError("Invalid rejuvenation bandwidth")
        snapshot = (self.particles.copy(), self.log_weights.copy(),
                    self.updates, self.resampling_count, len(self.history))
        try:
            gaussian = GaussianObservation(covariance)
            likelihood = gaussian.logpdf(observed,
                                         predict_complex(self.particles, candidate, model))
            target_ess = max(resample_below_ess_fraction, 0.65) * len(self.particles)
            initially_resampled = self.ess < target_ess
            if initially_resampled:
                # Standard SMC approximation of the previous posterior;
                # subsequent MH sweeps use *all* historical likelihoods.
                self.resample(bandwidth=0.0)
                likelihood = gaussian.logpdf(
                    observed, predict_complex(self.particles, candidate, model))
            beta = 0.0
            log_evidence = 0.0
            stages = 0
            minimum_ess = float(len(self.particles))
            resampled = initially_resampled
            acceptances = []
            while beta < 1 - 1e-12:
                stages += 1
                if stages > max_tempering_stages:
                    raise RuntimeError("Particle tempering did not converge; replay with more particles")

                def tempered_ess(next_beta: float) -> float:
                    proposed, _ = normalized_log_weights(
                        self.log_weights + (next_beta - beta) * likelihood)
                    return effective_sample_size(np.exp(proposed))

                if tempered_ess(1.0) >= target_ess:
                    next_beta = 1.0
                else:
                    lower, upper = beta, 1.0
                    for _ in range(34):
                        middle = (lower + upper) / 2
                        if tempered_ess(middle) >= target_ess:
                            lower = middle
                        else:
                            upper = middle
                    next_beta = lower
                    if next_beta - beta < 1e-10:
                        raise RuntimeError("Likelihood is too concentrated for this particle cloud")
                self.log_weights, increment = normalized_log_weights(
                    self.log_weights + (next_beta - beta) * likelihood)
                log_evidence += increment
                beta = next_beta
                stage_ess = self.ess
                minimum_ess = min(minimum_ess, stage_ess)
                if beta < 1 - 1e-12 or stage_ess < resample_below_ess_fraction * len(self.particles):
                    self.resample(bandwidth=0.0)
                    resampled = True
                    likelihood = gaussian.logpdf(
                        observed, predict_complex(self.particles, candidate, model))
                    likelihood, acceptance = self._metropolis_move(
                        likelihood, beta, observed, candidate, model, gaussian)
                    acceptances.append(acceptance)
            self.updates += 1
            self.history.append((np.asarray(observed, complex).copy(), candidate,
                                 model, np.asarray(covariance, float).copy()))
            return UpdateStats(log_evidence=float(log_evidence),
                               ess_before_resampling=float(minimum_ess),
                               ess_after_update=float(self.ess),
                               resampled=resampled,
                               posterior_score_hint=self.summary(),
                               tempering_stages=stages,
                               move_acceptance=float(np.mean(acceptances)) if acceptances else 0.,
                               minimum_stage_ess=float(minimum_ess))
        except Exception:
            (self.particles, self.log_weights, self.updates,
             self.resampling_count, history_length) = snapshot
            del self.history[history_length:]
            raise

    def summary(self) -> dict:
        """Physical estimates and uncertainty; phase may remain unidentifiable."""
        moments = weighted_posterior_moments(self.particles, self.weights)
        weights = self.weights
        freq_ci = [_weighted_quantile(self.particles[:, 0], weights, q)
                   for q in (0.025, 0.975)]
        t90_ci = [_weighted_quantile(self.particles[:, 1], weights, q)
                  for q in (0.025, 0.975)]
        phase_ci = None
        if moments["phase_identified"]:
            center = moments["mean_phase_rad"]
            differences = phase_difference(self.particles[:, 2], center)
            phase_ci = [float(wrap_phase(center + _weighted_quantile(differences, weights, q)))
                        for q in (0.025, 0.975)]
        return {**moments,
                "ci95_delta_hz": freq_ci,
                "ci95_t90_us": t90_ci,
                "ci95_phase_rad": phase_ci,
                "n_particles": len(self.particles),
                "ess": self.ess,
                "updates": self.updates,
                "resampling_count": self.resampling_count}


def particle_count_convergence(filters: dict[int, ParticleFilter], *,
                               delta_tolerance_hz: float,
                               t90_tolerance_us: float,
                               phase_tolerance_deg: float,
                               mean_fraction: float = 0.35,
                               std_fraction: float = 0.35,
                               minimum_ess_fraction: float = 0.2) -> dict:
    """Compare independent pilot replays at increasing particle counts.

    Each filter must have been updated with *the same pilot observations*.
    A count is acceptable only if it agrees with every larger trial in
    tolerance-scaled means and standard deviations and retains useful ESS.
    This reports numerical convergence, not physical model validation.
    """
    counts = sorted(filters)
    if len(counts) < 2 or any(count != len(filters[count].particles) for count in counts):
        raise ValueError("Need at least two correctly labeled particle counts")
    if len({filters[count].updates for count in counts}) != 1:
        raise ValueError("Particle-count trials replayed a different number of observations")
    scales = np.array([delta_tolerance_hz, t90_tolerance_us,
                       math.radians(phase_tolerance_deg)], dtype=float)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Convergence scales must be finite and positive")
    summaries = {count: filters[count].summary() for count in counts}
    comparisons = []
    for left, right in zip(counts[:-1], counts[1:]):
        a, b = summaries[left], summaries[right]
        both_phase_identified = a["phase_identified"] and b["phase_identified"]
        phase_mean_delta = (abs(float(phase_difference(a["mean_phase_rad"],
                                                       b["mean_phase_rad"])))
                            if both_phase_identified else 0.0)
        means = np.array([abs(a["mean_delta_hz"] - b["mean_delta_hz"]),
                          abs(a["mean_t90_us"] - b["mean_t90_us"]),
                          phase_mean_delta]) / scales
        stds_a = np.sqrt([a["var_delta_hz2"], a["var_t90_us2"], a["var_phase_rad2"]])
        stds_b = np.sqrt([b["var_delta_hz2"], b["var_t90_us2"], b["var_phase_rad2"]])
        stds = np.abs(stds_a - stds_b) / scales
        phase_identity_consistent = a["phase_identified"] == b["phase_identified"]
        ess_ok = (a["ess"] / left >= minimum_ess_fraction and
                  b["ess"] / right >= minimum_ess_fraction)
        converged = bool(np.max(means) <= mean_fraction and
                         np.max(stds) <= std_fraction and
                         phase_identity_consistent and ess_ok)
        comparisons.append({"counts": [left, right],
                            "mean_difference_per_tolerance": means.tolist(),
                            "std_difference_per_tolerance": stds.tolist(),
                            "phase_identity_consistent": bool(phase_identity_consistent),
                            "ess_fraction": [a["ess"] / left, b["ess"] / right],
                            "converged": converged})
    selected = next((counts[i] for i in range(len(comparisons))
                     if all(row["converged"] for row in comparisons[i:])), None)
    return {"selected_particles": selected,
            "counts_tried": counts,
            "same_pilot_updates": filters[counts[0]].updates,
            "comparisons": comparisons,
            "status": "CONVERGED" if selected is not None else "NOT_CONVERGED",
            "meaning": "numeric particle-count agreement only; physical validation separate"}
