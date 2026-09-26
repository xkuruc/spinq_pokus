"""Monte Carlo experiment choice on normalized *posterior* uncertainty.

This is an NMR adaptation of the Bayesian design principle in Gerster et al.;
their trapped-ion Hamiltonian, shot likelihood, and settings are not reused.
Each hypothetical observation is a correlated continuous complex FID feature
vector.  Time-aware choice uses the entire empirically estimated task time,
including relaxation and overhead, rather than just pulse width.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .likelihood import Candidate, GaussianObservation, NMRModel, interleaved_real_imag, predict_complex
from .smc import ParticleFilter, weighted_posterior_moments


@dataclass(frozen=True)
class Tolerances:
    delta_hz: float
    t90_us: float
    phase_deg: float

    def __post_init__(self) -> None:
        values = (self.delta_hz, self.t90_us, self.phase_deg)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("All physical tolerances must be finite and positive")

    @property
    def phase_rad(self) -> float:
        return math.radians(self.phase_deg)


def normalized_variance_score(particles: np.ndarray, weights: np.ndarray,
                              tolerances: Tolerances) -> float:
    """Sum Var(theta_i)/tol_i², with angular variance across ±pi handled."""
    moments = weighted_posterior_moments(particles, weights)
    return float(moments["var_delta_hz2"] / tolerances.delta_hz**2 +
                 moments["var_t90_us2"] / tolerances.t90_us**2 +
                 moments["var_phase_rad2"] / tolerances.phase_rad**2)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    score_before: float
    expected_score_after: float
    expected_reduction: float
    utility: float
    estimated_wall_seconds: float | None
    mc_samples: int


@dataclass(frozen=True)
class DesignChoice:
    candidate: Candidate
    evaluations: tuple[CandidateEvaluation, ...]
    time_weighted: bool
    fallback_used: bool


def _hypothetical_log_likelihoods(predictions: np.ndarray,
                                  observations: np.ndarray,
                                  gaussian: GaussianObservation) -> np.ndarray:
    """Shape `(n_mc, n_particles)` without refactoring covariance per draw."""
    residual = observations[:, None, :] - predictions[None, :, :]
    interleaved = interleaved_real_imag(residual)
    flat = interleaved.reshape(-1, interleaved.shape[-1])
    solved = np.linalg.solve(gaussian._cholesky, flat.T)
    quadratic = np.sum(solved**2, axis=0).reshape(residual.shape[:2])
    return gaussian._log_normalizer - 0.5 * quadratic


def expected_posterior_score(particle_filter: ParticleFilter, candidate: Candidate,
                             model: NMRModel, covariance: np.ndarray,
                             tolerances: Tolerances, *, mc_samples: int = 32,
                             rng: np.random.Generator,
                             common_uniforms: np.ndarray | None = None,
                             common_normals: np.ndarray | None = None) -> float:
    """Integrate expected posterior variance over continuous predictive data.

    A predictive particle and one *joint* correlated Gaussian FID vector are
    drawn per Monte Carlo replicate.  This uses pre-update particles and never
    mutates the online filter.  A fixed seed/common random numbers make the
    candidate ranking reproducible but do not turn it into a measured result.
    """
    if mc_samples < 2:
        raise ValueError("At least two predictive integrations are required")
    predictions = predict_complex(particle_filter.particles, candidate, model)
    gaussian = GaussianObservation(covariance)
    if common_uniforms is None:
        uniforms = rng.random(mc_samples)
    else:
        uniforms = np.asarray(common_uniforms, dtype=np.float64)
        if uniforms.shape != (mc_samples,) or np.any((uniforms < 0) | (uniforms >= 1)):
            raise ValueError("Common uniforms have the wrong shape/range")
    if common_normals is None:
        normals = rng.standard_normal((mc_samples, 2 * predictions.shape[1]))
    else:
        normals = np.asarray(common_normals, dtype=np.float64)
        if normals.shape != (mc_samples, 2 * predictions.shape[1]):
            raise ValueError("Common Gaussian draws have the wrong shape")
    ancestry = np.searchsorted(np.cumsum(particle_filter.weights), uniforms, side="right")
    ancestry = np.minimum(ancestry, len(predictions) - 1)
    perturbation = normals @ gaussian._cholesky.T
    observations = (predictions[ancestry] + perturbation[:, 0::2] +
                    1j * perturbation[:, 1::2])
    likelihoods = _hypothetical_log_likelihoods(predictions, observations, gaussian)
    prior_log = particle_filter.log_weights
    scores = []
    for log_likelihood in likelihoods:
        log_posterior = prior_log + log_likelihood
        peak = float(np.max(log_posterior))
        if not math.isfinite(peak):
            raise ValueError("Predictive observation has no supported particle")
        posterior = np.exp(log_posterior - peak)
        posterior /= posterior.sum()
        scores.append(normalized_variance_score(particle_filter.particles,
                                                posterior, tolerances))
    return float(np.mean(scores))


def choose_candidate(particle_filter: ParticleFilter, candidates: list[Candidate] | tuple[Candidate, ...],
                     model: NMRModel, covariance: np.ndarray, tolerances: Tolerances,
                     *, rng: np.random.Generator, mc_samples: int = 32,
                     time_weighted: bool = False,
                     fallback_candidate: Candidate | None = None) -> DesignChoice:
    """Choose minimum expected score or maximum reduction / whole-task time.

    A caller may supply a less ambiguous return-control candidate. If every
    noisy MC estimate predicts no gain, the fallback is chosen instead of
    chasing a negative estimated improvement. Hardware capability checks and
    model-mismatch gates remain the live driver's responsibility.
    """
    if not candidates:
        raise ValueError("No physically admissible candidate")
    if time_weighted and any(c.estimated_wall_seconds is None for c in candidates):
        raise ValueError("Time-aware design requires full measured task time for each candidate")
    dimensions = len(candidates[0].feature_windows_s)
    if any(len(c.feature_windows_s) != dimensions for c in candidates):
        raise ValueError("All candidates must share the same frozen feature dimension")
    before = normalized_variance_score(particle_filter.particles,
                                       particle_filter.weights, tolerances)
    uniforms = rng.random(mc_samples)
    normals = rng.standard_normal((mc_samples, 2 * dimensions))
    evaluations = []
    for candidate in candidates:
        score_after = expected_posterior_score(
            particle_filter, candidate, model, covariance, tolerances,
            mc_samples=mc_samples, rng=rng,
            common_uniforms=uniforms, common_normals=normals)
        gain = before - score_after
        utility = (gain / candidate.estimated_wall_seconds if time_weighted else -score_after)
        evaluations.append(CandidateEvaluation(candidate, before, score_after,
                                                gain, float(utility),
                                                candidate.estimated_wall_seconds,
                                                mc_samples))
    best = max(evaluations, key=lambda item: item.utility)
    fallback_used = False
    if fallback_candidate is not None and all(row.expected_reduction <= 0 for row in evaluations):
        if fallback_candidate not in candidates:
            raise ValueError("Fallback must be one of the admissible candidates")
        best = next(row for row in evaluations if row.candidate == fallback_candidate)
        fallback_used = True
    return DesignChoice(best.candidate, tuple(evaluations), time_weighted, fallback_used)


def meets_uncertainty_and_validation(particle_filter: ParticleFilter,
                                     tolerances: Tolerances, *,
                                     model_validated: bool,
                                     independent_reference_validated: bool,
                                     control_rotation_validated: bool) -> bool:
    """Budget exhaustion alone is never calibration success."""
    if not (model_validated and independent_reference_validated and
            control_rotation_validated):
        return False
    moments = weighted_posterior_moments(particle_filter.particles,
                                         particle_filter.weights)
    return bool(moments["phase_identified"] and
                math.sqrt(moments["var_delta_hz2"]) <= tolerances.delta_hz and
                math.sqrt(moments["var_t90_us2"]) <= tolerances.t90_us and
                math.sqrt(moments["var_phase_rad2"]) <= tolerances.phase_rad)
