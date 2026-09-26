"""Local NMR adaptations of phase GRAPE, graybox ID, and pulse generation.

All physical inputs (RF Hz/SDK %, detunings, channel operators, readout gain)
must come from a separate pilot.  Nothing here contacts a SpinQ device.
The convention throughout is I=sigma/2, K=H/h in Hz, U=exp(-2*pi*i*K*t).
Simulation process overlap is never reported as an experimental fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, cos, pi, sin
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import least_squares, minimize


_ID2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.diag([1, -1]).astype(complex)


def spin_operator(axis: str, spin: int, n_spins: int) -> np.ndarray:
    """Tensor-embedded I_axis, where I=sigma/2; leftmost spin is index zero."""
    if axis not in "xyz" or not 0 <= spin < n_spins or n_spins < 1:
        raise ValueError("invalid spin operator")
    pauli = {"x": _SX, "y": _SY, "z": _SZ}[axis]
    factors = [pauli / 2 if i == spin else _ID2 for i in range(n_spins)]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _hermitian_traceless(value: np.ndarray, name: str, dimension: int) -> np.ndarray:
    array = np.asarray(value, dtype=complex)
    if array.shape != (dimension, dimension) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {dimension}x{dimension} matrix")
    if not np.allclose(array, array.conj().T, atol=1e-10):
        raise ValueError(f"{name} must be Hermitian")
    if abs(np.trace(array)) > 1e-8:
        raise ValueError(f"{name} must be traceless")
    return array


@dataclass(frozen=True)
class SpinHamiltonian:
    """Calibrated rotating-frame model for one *known* RF channel.

    ``rf_hz_per_percent`` is measured rotation frequency per SDK amplitude
    percent. ``bx``/``by`` contain spin I operators and actual channel
    addressing. ``detuning_op`` defines which measured transition is swept.
    No extra spin or coupling is silently inserted.
    """

    n_spins: int
    drift_hz: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    detuning_op: np.ndarray
    rf_hz_per_percent: float
    channel: str = "H"

    def __post_init__(self) -> None:
        if self.n_spins not in (1, 2, 3):
            raise ValueError("only explicitly modeled 1/2/3-spin systems are supported")
        d = 2**self.n_spins
        for name in ("drift_hz", "bx", "by", "detuning_op"):
            object.__setattr__(self, name, _hermitian_traceless(getattr(self, name), name, d))
        if not np.isfinite(self.rf_hz_per_percent) or self.rf_hz_per_percent <= 0:
            raise ValueError("measured RF Hz per SDK percent must be positive")
        if not self.channel:
            raise ValueError("actual channel must be named")

    @property
    def dimension(self) -> int:
        return 2**self.n_spins


def single_spin_model(detuning_hz: float, rf_hz_per_percent: float, channel: str = "H") -> SpinHamiltonian:
    iz = spin_operator("z", 0, 1)
    return SpinHamiltonian(1, float(detuning_hz) * iz, spin_operator("x", 0, 1),
                           spin_operator("y", 0, 1), iz, rf_hz_per_percent, channel)


def coupled_spin_model(
    detunings_hz: Sequence[float],
    couplings_hz: Mapping[tuple[int, int], float],
    target_spin: int,
    drive_weights: Sequence[float],
    rf_hz_per_percent: float,
    *,
    weak_coupling_verified: bool,
    channel: str = "H",
) -> SpinHamiltonian:
    """Build a weak-coupling K; all pair J values and addressing are explicit.

    Zero is accepted only as an explicit, externally justified value.  This
    function does not establish weak coupling or measured spectator identity.
    """
    n = len(detunings_hz)
    if n not in (2, 3) or not weak_coupling_verified:
        raise ValueError("multi-spin ZZ model requires verified weak-coupling regime")
    if len(drive_weights) != n or not 0 <= target_spin < n:
        raise ValueError("channel addressing and target spin required")
    expected = {(i, j) for i in range(n) for j in range(i + 1, n)}
    if set(couplings_hz) != expected:
        raise ValueError("every pair coupling must be explicitly supplied")
    d = 2**n
    drift = np.zeros((d, d), dtype=complex)
    for i, delta in enumerate(detunings_hz):
        drift += float(delta) * spin_operator("z", i, n)
    for (i, j), coupling in couplings_hz.items():
        drift += float(coupling) * (spin_operator("z", i, n) @ spin_operator("z", j, n))
    bx = sum((float(w) * spin_operator("x", i, n) for i, w in enumerate(drive_weights)),
             np.zeros((d, d), dtype=complex))
    by = sum((float(w) * spin_operator("y", i, n) for i, w in enumerate(drive_weights)),
             np.zeros((d, d), dtype=complex))
    return SpinHamiltonian(n, drift, bx, by, spin_operator("z", target_spin, n),
                           rf_hz_per_percent, channel)


@dataclass(frozen=True)
class PulseProgram:
    duration_s: tuple[float, ...]
    amplitude_percent: tuple[float, ...]
    phase_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.duration_s)
        if n < 1 or len(self.amplitude_percent) != n or len(self.phase_rad) != n:
            raise ValueError("pulse arrays must have the same nonzero length")
        if not all(np.isfinite(self.duration_s)) or min(self.duration_s) <= 0:
            raise ValueError("all segment durations must be positive")
        if not all(np.isfinite(self.amplitude_percent)) or min(self.amplitude_percent) < 0:
            raise ValueError("amplitudes must be finite, nonnegative SDK percent")
        if not all(np.isfinite(self.phase_rad)):
            raise ValueError("phases must be finite radians")

    @property
    def rf_duration_s(self) -> float:
        return float(sum(t for t, a in zip(self.duration_s, self.amplitude_percent) if a > 0))

    @property
    def rf_integral_percent_s(self) -> float:
        return float(sum(t * a for t, a in zip(self.duration_s, self.amplitude_percent)))

    def serialized(self) -> list[dict[str, float]]:
        return [{"duration_s": float(t), "amplitude_percent": float(a),
                 "phase_deg": float(np.rad2deg(p) % 360)}
                for t, a, p in zip(self.duration_s, self.amplitude_percent, self.phase_rad)]


def quantize_program(program: PulseProgram, verified_tick_s: float) -> PulseProgram:
    """Round to an externally verified hardware duration quantum, then reevaluate.

    This does not establish that the SDK supports independent adjacent slots.
    The caller must perform that live timing contract before sending pulses.
    """
    if not np.isfinite(verified_tick_s) or verified_tick_s <= 0:
        raise ValueError("verified positive timing quantum required")
    ticks = [int(np.floor(dt / verified_tick_s + .5)) for dt in program.duration_s]
    if min(ticks) < 1:
        raise ValueError("a pulse segment falls below verified timing quantum")
    return PulseProgram(tuple(tick * verified_tick_s for tick in ticks),
                        program.amplitude_percent, program.phase_rad)


@dataclass(frozen=True)
class EnsemblePoint:
    detuning_shift_hz: float = 0.0
    rf_scale: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.all(np.isfinite((self.detuning_shift_hz, self.rf_scale, self.weight))):
            raise ValueError("ensemble values must be finite")
        if self.rf_scale <= 0 or self.weight <= 0:
            raise ValueError("RF scale and ensemble weight must be positive")


def _validate_target(model: SpinHamiltonian, target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=complex)
    d = model.dimension
    if target.shape != (d, d) or not np.allclose(target.conj().T @ target, np.eye(d), atol=1e-8):
        raise ValueError("target must be unitary on the actual modeled Hilbert space")
    return target


def rotation(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """SU(2) exp(-i angle n·I), determinant one."""
    vec = np.asarray(axis, dtype=float)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)) or np.linalg.norm(vec) == 0:
        raise ValueError("rotation axis must be a nonzero 3-vector")
    vec = vec / np.linalg.norm(vec)
    return cos(angle_rad / 2) * _ID2 - 1j * sin(angle_rad / 2) * (
        vec[0] * _SX + vec[1] * _SY + vec[2] * _SZ)


def embed_single_spin_gate(gate: np.ndarray, spin: int, n_spins: int) -> np.ndarray:
    gate = np.asarray(gate, dtype=complex)
    if gate.shape != (2, 2) or not 0 <= spin < n_spins:
        raise ValueError("single-spin target and actual spin index required")
    result = gate if spin == 0 else _ID2
    for i in range(1, n_spins):
        result = np.kron(result, gate if i == spin else _ID2)
    return result


def _segment_hz(model: SpinHamiltonian, amp: float, phase: float, point: EnsemblePoint) -> np.ndarray:
    drive = point.rf_scale * model.rf_hz_per_percent * amp
    return (model.drift_hz + point.detuning_shift_hz * model.detuning_op
            + drive * (cos(phase) * model.bx + sin(phase) * model.by))


def propagate(model: SpinHamiltonian, program: PulseProgram,
              point: EnsemblePoint = EnsemblePoint()) -> np.ndarray:
    total = np.eye(model.dimension, dtype=complex)
    for dt, amp, phase in zip(program.duration_s, program.amplitude_percent, program.phase_rad):
        total = expm(-2j * pi * dt * _segment_hz(model, amp, phase, point)) @ total
    return total


def process_overlap(model: SpinHamiltonian, target: np.ndarray, program: PulseProgram,
                    point: EnsemblePoint = EnsemblePoint()) -> float:
    """SIMULATED |Tr(target† U)|²/d², not a measured process fidelity."""
    target = _validate_target(model, target)
    z = np.trace(target.conj().T @ propagate(model, program, point)) / model.dimension
    return float(np.clip(abs(z)**2, 0.0, 1.0))


def ensemble_infidelity(model: SpinHamiltonian, target: np.ndarray, program: PulseProgram,
                        ensemble: Sequence[EnsemblePoint]) -> float:
    points = tuple(ensemble)
    if not points:
        raise ValueError("measured design ensemble required")
    norm = sum(p.weight for p in points)
    return float(sum(p.weight * (1 - process_overlap(model, target, program, p))
                     for p in points) / norm)


def grape_loss_gradient(
    variables: np.ndarray,
    model: SpinHamiltonian,
    target: np.ndarray,
    duration_s: float,
    n_segments: int,
    amplitude_percent: float,
    ensemble: Sequence[EnsemblePoint],
    *,
    optimize_amplitude: bool = False,
    amplitude_penalty: float = 0.0,
    amplitude_limit_percent: float | None = None,
) -> tuple[float, np.ndarray]:
    """Exact Frechet forward/backward gradient of weighted gate infidelity."""
    target = _validate_target(model, target)
    points = tuple(ensemble)
    if not points or duration_s <= 0 or n_segments < 1 or amplitude_percent < 0:
        raise ValueError("valid duration, segments, calibration and measured ensemble required")
    if amplitude_limit_percent is not None and amplitude_limit_percent <= 0:
        raise ValueError("amplitude limit must be positive")
    x = np.asarray(variables, dtype=float)
    nvar = 2 * n_segments if optimize_amplitude else n_segments
    if x.shape != (nvar,) or not np.all(np.isfinite(x)):
        raise ValueError("wrong GRAPE variable vector")
    phases = x[:n_segments]
    amplitudes = x[n_segments:] if optimize_amplitude else np.full(n_segments, amplitude_percent)
    if np.any(amplitudes < 0) or (amplitude_limit_percent is not None and
                                   np.any(amplitudes > amplitude_limit_percent + 1e-9)):
        raise ValueError("optimizer proposed amplitude outside explicit bounds")
    dt = duration_s / n_segments
    norm = sum(p.weight for p in points)
    grad = np.zeros(nvar, dtype=float)
    loss = 0.0
    td = target.conj().T
    d = model.dimension
    eye = np.eye(d, dtype=complex)
    for point in points:
        matrices = []
        dphase = []
        damp = []
        for amp, phase in zip(amplitudes, phases):
            h = _segment_hz(model, amp, phase, point)
            a = -2j * pi * dt * h
            matrices.append(expm(a))
            drive = point.rf_scale * model.rf_hz_per_percent * amp
            ephase = -2j * pi * dt * drive * (-sin(phase) * model.bx + cos(phase) * model.by)
            dphase.append(expm_frechet(a, ephase, compute_expm=False))
            if optimize_amplitude:
                eamp = -2j * pi * dt * point.rf_scale * model.rf_hz_per_percent * (
                    cos(phase) * model.bx + sin(phase) * model.by)
                damp.append(expm_frechet(a, eamp, compute_expm=False))
        left = []
        prefix = eye
        for matrix in matrices:
            left.append(prefix)
            prefix = matrix @ prefix
        right = [eye] * n_segments
        suffix = eye
        for j in range(n_segments - 1, -1, -1):
            right[j] = suffix
            suffix = suffix @ matrices[j]
        z = np.trace(td @ prefix) / d
        weight = point.weight / norm
        loss += weight * (1 - abs(z)**2)
        for j in range(n_segments):
            dz = np.trace(td @ right[j] @ dphase[j] @ left[j]) / d
            grad[j] += -2 * weight * np.real(np.conj(z) * dz)
            if optimize_amplitude:
                dzamp = np.trace(td @ right[j] @ damp[j] @ left[j]) / d
                grad[n_segments + j] += -2 * weight * np.real(np.conj(z) * dzamp)
    if optimize_amplitude and amplitude_penalty:
        if not amplitude_limit_percent:
            raise ValueError("RF penalty requires explicit amplitude limit")
        loss += amplitude_penalty * float(np.mean((amplitudes / amplitude_limit_percent)**2))
        grad[n_segments:] += 2 * amplitude_penalty * amplitudes / (
            n_segments * amplitude_limit_percent**2)
    return float(loss), grad


@dataclass
class GRAPEResult:
    program: PulseProgram
    design_infidelity: float
    initial_infidelity: float
    iterations: int
    success: bool
    optimizer_message: str
    elapsed_s: float
    optimize_amplitude: bool
    gradient_max_error: float | None = None


def optimize_grape(
    model: SpinHamiltonian,
    target: np.ndarray,
    *,
    duration_s: float,
    n_segments: int,
    amplitude_percent: float,
    ensemble: Sequence[EnsemblePoint],
    initial_phases: Sequence[float] | None = None,
    initial_amplitudes: Sequence[float] | None = None,
    optimize_amplitude: bool = False,
    amplitude_limit_percent: float | None = None,
    amplitude_penalty: float = 0.0,
    maxiter: int = 100,
    check_gradient: bool = False,
    verified_tick_s: float | None = None,
) -> GRAPEResult:
    """Local phase-only or separate amplitude-phase SciPy L-BFGS-B GRAPE."""
    if maxiter < 1 or n_segments < 1 or duration_s <= 0:
        raise ValueError("positive GRAPE budget, segment count and duration required")
    if verified_tick_s is not None:
        duration_s = sum(quantize_program(
            PulseProgram((duration_s / n_segments,) * n_segments,
                         (float(amplitude_percent),) * n_segments,
                         (0.0,) * n_segments), verified_tick_s).duration_s)
    phases = np.zeros(n_segments) if initial_phases is None else np.asarray(initial_phases, dtype=float)
    if phases.shape != (n_segments,):
        raise ValueError("initial phases do not match segment count")
    amps = (np.full(n_segments, amplitude_percent) if initial_amplitudes is None
            else np.asarray(initial_amplitudes, dtype=float))
    if amps.shape != (n_segments,) or np.any(amps < 0):
        raise ValueError("invalid initial amplitudes")
    if initial_amplitudes is not None and not optimize_amplitude:
        raise ValueError("initial amplitudes require the amplitude-phase variant")
    if optimize_amplitude and (amplitude_limit_percent is None or
                               np.max(amps) > amplitude_limit_percent):
        raise ValueError("amplitude-phase GRAPE needs measured headroom bound")
    x0 = np.r_[phases, amps] if optimize_amplitude else phases.copy()
    kwargs = dict(optimize_amplitude=optimize_amplitude, amplitude_penalty=amplitude_penalty,
                  amplitude_limit_percent=amplitude_limit_percent)

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        return grape_loss_gradient(x, model, target, duration_s, n_segments,
                                   amplitude_percent, ensemble, **kwargs)

    initial_loss = objective(x0)[0]
    grad_error = None
    if check_gradient:
        analytic = objective(x0)[1]
        numeric = np.zeros_like(x0)
        eps = 1e-6
        for j in range(x0.size):
            xp, xm = x0.copy(), x0.copy()
            xp[j] += eps
            xm[j] -= eps
            if optimize_amplitude and j >= n_segments and xm[j] < 0:
                numeric[j] = (objective(xp)[0] - initial_loss) / eps
            elif (optimize_amplitude and j >= n_segments and
                  amplitude_limit_percent is not None and xp[j] > amplitude_limit_percent):
                numeric[j] = (initial_loss - objective(xm)[0]) / eps
            else:
                numeric[j] = (objective(xp)[0] - objective(xm)[0]) / (xp[j] - xm[j])
        grad_error = float(np.max(np.abs(analytic - numeric)))
        if grad_error > 1e-4:
            raise RuntimeError(f"GRAPE gradient check failed: {grad_error:g}")
    bounds = ([(None, None)] * n_segments + [(0.0, amplitude_limit_percent)] * n_segments
              if optimize_amplitude else None)
    start = perf_counter()
    fit = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-11})
    elapsed = perf_counter() - start
    outph = tuple(float(x % (2 * pi)) for x in fit.x[:n_segments])
    outamp = (tuple(float(a) for a in fit.x[n_segments:]) if optimize_amplitude
              else (float(amplitude_percent),) * n_segments)
    program = PulseProgram((duration_s / n_segments,) * n_segments, outamp, outph)
    return GRAPEResult(program, ensemble_infidelity(model, target, program, ensemble),
                       float(initial_loss), int(fit.nit), bool(fit.success), str(fit.message),
                       elapsed, optimize_amplitude, grad_error)


def rectangular_pulse(model: SpinHamiltonian, angle_rad: float, amplitude_percent: float,
                      phase_rad: float = 0.0) -> PulseProgram:
    """Calibrated on-resonance one-spin rectangle; no drift cancellation claimed."""
    if amplitude_percent <= 0 or angle_rad <= 0:
        raise ValueError("calibrated positive amplitude and rotation required")
    dt = angle_rad / (2 * pi * model.rf_hz_per_percent * amplitude_percent)
    return PulseProgram((dt,), (amplitude_percent,), (phase_rad,))


def bb1_pulse(model: SpinHamiltonian, angle_rad: float, amplitude_percent: float,
              phase_rad: float = 0.0) -> PulseProgram:
    """Symmetric BB1: theta/2, pi_phi, 2pi_3phi, pi_phi, theta/2.

    Designed chiefly for RF scale error; extra duration and RF integral are
    explicit.  No frequency robustness or hardware feasibility is assumed.
    """
    if angle_rad <= 0 or angle_rad > 4 * pi or amplitude_percent <= 0:
        raise ValueError("BB1 angle must lie in (0,4*pi] and amplitude must be positive")
    phi = acos(-angle_rad / (4 * pi))
    angles = (angle_rad / 2, pi, 2 * pi, pi, angle_rad / 2)
    phases = (phase_rad, phase_rad + phi, phase_rad + 3 * phi,
              phase_rad + phi, phase_rad)
    factor = 1 / (2 * pi * model.rf_hz_per_percent * amplitude_percent)
    return PulseProgram(tuple(float(a * factor) for a in angles),
                        (float(amplitude_percent),) * 5, tuple(phases))


def heldout_grid_metrics(model: SpinHamiltonian, target: np.ndarray,
                         programs: Mapping[str, PulseProgram],
                         heldout: Sequence[EnsemblePoint]) -> dict[str, dict[str, object]]:
    """Simulated grid only; hardware must repeat with independent input/readout."""
    if not heldout:
        raise ValueError("independent heldout grid required")
    out = {}
    for name, program in programs.items():
        errors = [1 - process_overlap(model, target, program, p) for p in heldout]
        out[name] = {"simulated_mean_infidelity": float(np.mean(errors)),
                     "simulated_worst_infidelity": float(np.max(errors)),
                     "simulated_grid_errors": [float(x) for x in errors],
                     "duration_s": float(sum(program.duration_s)),
                     "rf_duration_s": program.rf_duration_s,
                     "rf_integral_percent_s": program.rf_integral_percent_s}
    return out


# D: physics-constrained and non-neural/blackbox FID prediction.  The input
# is the exported complex FID.  No server FFT, fit or matrix is accepted.


@dataclass(frozen=True)
class FIDExample:
    program: PulseProgram
    initial_state: np.ndarray
    detector: np.ndarray
    times_s: np.ndarray
    fid: np.ndarray
    family: str
    session: str
    record_id: str

    def __post_init__(self) -> None:
        rho = np.asarray(self.initial_state, dtype=complex)
        detector = np.asarray(self.detector, dtype=complex)
        times = np.asarray(self.times_s, dtype=float)
        fid = np.asarray(self.fid, dtype=complex)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or detector.shape != rho.shape:
            raise ValueError("known initial state and detector matrices required")
        if not np.allclose(rho, rho.conj().T, atol=1e-9):
            raise ValueError("initial density/deviation state must be Hermitian")
        if times.ndim != 1 or fid.shape != times.shape or len(times) < 2:
            raise ValueError("paired acquisition times and complex FID required")
        if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0) or times[0] < 0:
            raise ValueError("acquisition seconds must be finite, ordered and nonnegative")
        if not np.all(np.isfinite(fid)) or not np.all(np.isfinite(rho)) or not np.all(np.isfinite(detector)):
            raise ValueError("FID and matrices must be finite")
        if not self.family or not self.session or not self.record_id:
            raise ValueError("family, session, and record identity required for heldout split")
        for attr, value in (("initial_state", rho), ("detector", detector),
                            ("times_s", times), ("fid", fid)):
            object.__setattr__(self, attr, value)


@dataclass(frozen=True)
class ReadoutPhysics:
    """Fixed readout gauge from an independent calibration; not fit per record."""

    acq_hamiltonian_hz: np.ndarray
    complex_gain: complex
    background: complex = 0j
    decay_s_inv: float = 0.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.acq_hamiltonian_hz, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("acquisition Hamiltonian must be square")
        _hermitian_traceless(matrix, "acq_hamiltonian_hz", matrix.shape[0])
        if not np.isfinite(self.complex_gain) or self.complex_gain == 0:
            raise ValueError("fixed, nonzero complex readout gain required")
        if not np.isfinite(self.background) or not np.isfinite(self.decay_s_inv) or self.decay_s_inv < 0:
            raise ValueError("invalid fixed background or nonnegative decay")
        object.__setattr__(self, "acq_hamiltonian_hz", matrix)


def split_examples_by_group(examples: Sequence[FIDExample], *,
                            heldout_families: Sequence[str] = (),
                            heldout_sessions: Sequence[str] = ()) -> tuple[list[FIDExample], list[FIDExample]]:
    """Exclude entire sequence families/sessions from training, not FID points."""
    families, sessions = set(heldout_families), set(heldout_sessions)
    if not families and not sessions:
        raise ValueError("select an independent sequence family or session")
    train, test = [], []
    seen = set()
    for example in examples:
        if example.record_id in seen:
            raise ValueError("duplicate record identity")
        seen.add(example.record_id)
        (test if example.family in families or example.session in sessions else train).append(example)
    if not train or not test:
        raise ValueError("group split must leave training and independent heldout records")
    return train, test


def _fid_subset(example: FIDExample, max_points: int) -> np.ndarray:
    if max_points < 2:
        raise ValueError("at least two FID times needed")
    return np.unique(np.linspace(0, len(example.times_s) - 1,
                                 min(len(example.times_s), max_points), dtype=int))


def _fid_from_after(rho_after: np.ndarray, detector: np.ndarray, times_s: np.ndarray,
                    readout: ReadoutPhysics) -> np.ndarray:
    vals, vectors = np.linalg.eigh(readout.acq_hamiltonian_hz)
    rho_e = vectors.conj().T @ rho_after @ vectors
    detector_e = vectors.conj().T @ detector @ vectors
    coeff = detector_e * rho_e.T
    differences = vals[None, :] - vals[:, None]
    oscillations = np.exp(-2j * pi * times_s[:, None, None] * differences)
    signal = np.sum(oscillations * coeff, axis=(1, 2))
    return (readout.complex_gain * np.exp(-readout.decay_s_inv * times_s) * signal
            + readout.background)


def predict_fid_physics(
    example: FIDExample, model: SpinHamiltonian, readout: ReadoutPhysics,
    correction: Callable[[float, float, float], tuple[float, float, float]] | None = None,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """FID under a fixed readout gauge and optional control correction.

    Correction returns relative RF scale, phase shift in radians, and
    detuning shift in Hz for each segment.  No unidentified J is introduced.
    """
    if example.initial_state.shape != (model.dimension, model.dimension):
        raise ValueError("example Hilbert space differs from physical model")
    if readout.acq_hamiltonian_hz.shape != example.initial_state.shape:
        raise ValueError("readout and example dimensions differ")
    total = np.eye(model.dimension, dtype=complex)
    for dt, amp, phase in zip(example.program.duration_s, example.program.amplitude_percent,
                              example.program.phase_rad):
        scale, dphase, ddelta = correction(amp, phase, dt) if correction else (1.0, 0.0, 0.0)
        if scale <= 0 or not np.all(np.isfinite((scale, dphase, ddelta))):
            raise ValueError("control correction produced nonphysical RF scale")
        drive = model.rf_hz_per_percent * amp * scale
        h = (model.drift_hz + ddelta * model.detuning_op
             + drive * (cos(phase + dphase) * model.bx + sin(phase + dphase) * model.by))
        total = expm(-2j * pi * dt * h) @ total
    rho_after = total @ example.initial_state @ total.conj().T
    times = example.times_s if indices is None else example.times_s[indices]
    return _fid_from_after(rho_after, example.detector, times, readout)


@dataclass
class PolynomialRFResult:
    coefficients: np.ndarray
    amplitude_scale_percent: float
    max_relative_rf: float
    max_phase_rad: float
    max_detuning_hz: float
    training_loss: float
    nfev: int
    success: bool
    train_record_ids: tuple[str, ...]

    def correction(self, amp: float, phase: float, dt: float) -> tuple[float, float, float]:
        x = amp / self.amplitude_scale_percent
        c = self.coefficients
        return 1 + c[0] * x + c[1] * x**2, c[2] * x, c[3] * x


def fit_polynomial_rf(train: Sequence[FIDExample], model: SpinHamiltonian,
                      readout: ReadoutPhysics, *, amplitude_scale_percent: float,
                      max_relative_rf: float, max_phase_rad: float,
                      max_detuning_hz: float, max_points: int = 48,
                      max_nfev: int = 100) -> PolynomialRFResult:
    """Flexible non-neural RF nonlinearity with the same FID training data."""
    if not train or amplitude_scale_percent <= 0 or min(max_relative_rf, max_phase_rad,
                                                         max_detuning_hz) <= 0:
        raise ValueError("pilot-derived scale and positive correction bounds required")
    if max_relative_rf >= 1:
        raise ValueError("RF correction cap must keep effective scale positive")
    subsets = [_fid_subset(ex, max_points) for ex in train]
    lower = [-max_relative_rf / 2, -max_relative_rf / 2, -max_phase_rad, -max_detuning_hz]
    upper = [-v for v in lower]

    def residual(c: np.ndarray) -> np.ndarray:
        fit = PolynomialRFResult(c, amplitude_scale_percent, max_relative_rf,
                                 max_phase_rad, max_detuning_hz, 0, 0, False, ())
        parts = []
        for ex, ids in zip(train, subsets):
            difference = predict_fid_physics(ex, model, readout, fit.correction, ids) - ex.fid[ids]
            weight = 1 / np.sqrt(len(ids) * len(train))
            parts.extend((weight * difference.real, weight * difference.imag))
        return np.concatenate(parts)

    fit = least_squares(residual, np.zeros(4), bounds=(lower, upper), max_nfev=max_nfev)
    return PolynomialRFResult(fit.x, amplitude_scale_percent, max_relative_rf,
                              max_phase_rad, max_detuning_hz,
                              float(np.dot(fit.fun, fit.fun)), int(fit.nfev),
                              bool(fit.success), tuple(ex.record_id for ex in train))


def _require_torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError("CPU PyTorch import failed; neural C/D/E stages unavailable") from exc
    return torch


@dataclass
class GrayboxResult:
    network: object
    model: SpinHamiltonian
    readout: ReadoutPhysics
    amplitude_scale_percent: float
    duration_scale_s: float
    correction_caps: tuple[float, float, float]
    fid_scale: float
    history: list[dict[str, float]]
    train_record_ids: tuple[str, ...]
    training_seconds: float


def _torch_graybox_fid(example: FIDExample, net: object, model: SpinHamiltonian,
                       readout: ReadoutPhysics, amp_scale: float, duration_scale: float,
                       caps: tuple[float, float, float], ids: np.ndarray | None = None):
    torch = _require_torch()
    dtype = torch.complex128
    unitary, corrections = _torch_graybox_unitary(example.program, net, model,
                                                  amp_scale, duration_scale, caps)
    rho = unitary @ torch.as_tensor(example.initial_state, dtype=dtype) @ unitary.conj().T
    times = example.times_s if ids is None else example.times_s[ids]
    times_t = torch.as_tensor(times, dtype=torch.float64)
    h_acq = torch.as_tensor(readout.acq_hamiltonian_hz, dtype=dtype)
    eigvals, vectors = torch.linalg.eigh(h_acq)
    rho_e = vectors.conj().T @ rho @ vectors
    detector_e = vectors.conj().T @ torch.as_tensor(example.detector, dtype=dtype) @ vectors
    coefficients = detector_e * rho_e.T
    frequencies = eigvals[None, :] - eigvals[:, None]
    oscillations = torch.exp(-2j * pi * times_t[:, None, None] * frequencies)
    signal = torch.sum(oscillations * coefficients, dim=(1, 2))
    pred = (readout.complex_gain * torch.exp(-readout.decay_s_inv * times_t) * signal
            + readout.background)
    return pred, corrections


def _torch_graybox_unitary(program: PulseProgram, net: object, model: SpinHamiltonian,
                           amp_scale: float, duration_scale: float,
                           caps: tuple[float, float, float], phases=None):
    torch = _require_torch()
    dtype = torch.complex128
    h0 = torch.as_tensor(model.drift_hz, dtype=dtype)
    bx = torch.as_tensor(model.bx, dtype=dtype)
    by = torch.as_tensor(model.by, dtype=dtype)
    iz = torch.as_tensor(model.detuning_op, dtype=dtype)
    cap = torch.as_tensor(caps, dtype=torch.float64)
    unitary = torch.eye(model.dimension, dtype=dtype)
    corrections = []
    if phases is None:
        phases = torch.as_tensor(program.phase_rad, dtype=torch.float64)
    for j, (dt, amp) in enumerate(zip(program.duration_s, program.amplitude_percent)):
        phase = phases[j]
        features = torch.stack((torch.as_tensor(amp / amp_scale, dtype=torch.float64),
                                torch.cos(phase), torch.sin(phase),
                                torch.as_tensor(dt / duration_scale, dtype=torch.float64)))
        correction = torch.tanh(net(features)) * cap
        scale, dphase, ddelta = correction.unbind()
        effective_phase = phase + dphase
        h = (h0 + ddelta * iz + model.rf_hz_per_percent * amp * (1 + scale)
             * (torch.cos(effective_phase) * bx + torch.sin(effective_phase) * by))
        unitary = torch.matrix_exp(-2j * pi * dt * h) @ unitary
        corrections.append(correction)
    return unitary, torch.stack(corrections)


def fit_graybox(train: Sequence[FIDExample], model: SpinHamiltonian,
                readout: ReadoutPhysics, *, amplitude_scale_percent: float,
                duration_scale_s: float, correction_caps: tuple[float, float, float],
                epochs: int = 60, hidden: int = 16, learning_rate: float = 0.01,
                regularization: float = 0.001, max_points: int = 48,
                seed: int = 0) -> GrayboxResult:
    """Youssry-inspired control→Hermitian Hamiltonian→evolution→complex FID.

    The small per-segment correction MLP is this project's NMR adaptation,
    not the authors' unpublished trained model.  Readout gain/phase is fixed
    by an independent calibration to avoid an arbitrary global gauge.
    """
    torch = _require_torch()
    if not train or min(amplitude_scale_percent, duration_scale_s, *correction_caps) <= 0:
        raise ValueError("pilot scales and positive model correction caps required")
    if correction_caps[0] >= 1 or epochs < 1 or hidden < 2 or learning_rate <= 0:
        raise ValueError("invalid graybox training settings")
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(4, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, 3)).double()
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    subsets = [_fid_subset(ex, max_points) for ex in train]
    fid_scale = float(np.median([np.sqrt(np.mean(np.abs(ex.fid[ids])**2))
                                 for ex, ids in zip(train, subsets)]))
    if not np.isfinite(fid_scale) or fid_scale <= 0:
        raise ValueError("nonzero complex FID scale required for graybox training")
    history = []
    start = perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        record_losses = []
        penalties = []
        for ex, ids in zip(train, subsets):
            pred, corrections = _torch_graybox_fid(ex, net, model, readout,
                                                   amplitude_scale_percent, duration_scale_s,
                                                   correction_caps, ids)
            target = torch.as_tensor(ex.fid[ids], dtype=torch.complex128)
            record_losses.append(torch.mean(torch.abs((pred - target) / fid_scale)**2))
            penalties.append(torch.mean((corrections / torch.as_tensor(
                correction_caps, dtype=torch.float64))**2))
        data_loss = torch.stack(record_losses).mean()
        penalty = torch.stack(penalties).mean()
        loss = data_loss + regularization * penalty
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite graybox loss; check FID scale and readout gauge")
        loss.backward()
        optimizer.step()
        history.append({"epoch": float(epoch + 1),
                        "training_normalized_fid_mse": float(data_loss.detach()),
                        "regularization": float(penalty.detach())})
    return GrayboxResult(net, model, readout, amplitude_scale_percent, duration_scale_s,
                         correction_caps, fid_scale, history, tuple(ex.record_id for ex in train),
                         perf_counter() - start)


def predict_fid_graybox(example: FIDExample, result: GrayboxResult,
                        indices: np.ndarray | None = None) -> np.ndarray:
    torch = _require_torch()
    result.network.eval()
    with torch.no_grad():
        pred, _ = _torch_graybox_fid(example, result.network, result.model,
                                     result.readout, result.amplitude_scale_percent,
                                     result.duration_scale_s, result.correction_caps, indices)
    return pred.cpu().numpy()


def predict_unitary_graybox(program: PulseProgram, result: GrayboxResult) -> np.ndarray:
    """Model-internal unitary; no experimental unitary is reconstructed here."""
    torch = _require_torch()
    result.network.eval()
    with torch.no_grad():
        unitary, _ = _torch_graybox_unitary(program, result.network, result.model,
                                            result.amplitude_scale_percent,
                                            result.duration_scale_s,
                                            result.correction_caps)
    return unitary.cpu().numpy()


def optimize_graybox_gate(result: GrayboxResult, target: np.ndarray,
                          initial_program: PulseProgram, *, maxiter: int = 40) -> dict[str, object]:
    """Differentiate learned-H propagators to design a new phase-only gate.

    Only the *model prediction* is returned. A fresh hardware experiment on
    independent initial/readout configurations is required for validation.
    """
    torch = _require_torch()
    target = _validate_target(result.model, target)
    if maxiter < 1:
        raise ValueError("positive graybox control budget required")
    td = torch.as_tensor(target.conj().T, dtype=torch.complex128)
    d = result.model.dimension
    result.network.eval()

    def objective(values: np.ndarray):
        phases = torch.as_tensor(values, dtype=torch.float64).clone().detach().requires_grad_(True)
        unitary, _ = _torch_graybox_unitary(initial_program, result.network,
                                            result.model, result.amplitude_scale_percent,
                                            result.duration_scale_s, result.correction_caps,
                                            phases)
        overlap = torch.trace(td @ unitary) / d
        loss = 1 - torch.abs(overlap)**2
        derivative = torch.autograd.grad(loss, phases)[0]
        return float(loss.detach()), derivative.detach().cpu().numpy()

    start = perf_counter()
    fit = minimize(objective, np.asarray(initial_program.phase_rad, dtype=float),
                   jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    program = PulseProgram(initial_program.duration_s, initial_program.amplitude_percent,
                           tuple(float(v % (2 * pi)) for v in fit.x))
    return {"program": program, "simulated_graybox_infidelity": float(fit.fun),
            "local_design_seconds": perf_counter() - start,
            "optimizer_iterations": int(fit.nit), "optimizer_success": bool(fit.success),
            "hardware_validation_status": "PENDING_INDEPENDENT_MEASUREMENT"}


@dataclass
class BlackboxResult:
    network: object
    max_slots: int
    amplitude_scale_percent: float
    duration_scale_s: float
    time_scale_s: float
    fid_scale: float
    history: list[dict[str, float]]
    train_record_ids: tuple[str, ...]
    training_seconds: float


def _blackbox_features(example: FIDExample, times: np.ndarray, *, max_slots: int,
                       amp_scale: float, duration_scale: float,
                       time_scale: float) -> np.ndarray:
    if len(example.program.duration_s) > max_slots:
        raise ValueError("blackbox maximum pulse slots exceeded")
    pulse = np.zeros((max_slots, 4), dtype=float)
    for i, (dt, amp, phase) in enumerate(zip(example.program.duration_s,
                                              example.program.amplitude_percent,
                                              example.program.phase_rad)):
        pulse[i] = (dt / duration_scale, amp / amp_scale, cos(phase), sin(phase))
    rho = example.initial_state.ravel()
    detector = example.detector.ravel()
    context = np.r_[pulse.ravel(), rho.real, rho.imag, detector.real, detector.imag]
    return np.column_stack((np.broadcast_to(context, (len(times), len(context))),
                            times / time_scale))


def fit_blackbox(train: Sequence[FIDExample], *, max_slots: int,
                 amplitude_scale_percent: float, duration_scale_s: float,
                 time_scale_s: float, epochs: int = 60, hidden: int = 32,
                 learning_rate: float = 0.01, max_points: int = 48,
                 seed: int = 0) -> BlackboxResult:
    """Same FID records, flexible nonphysical MLP reference."""
    torch = _require_torch()
    if not train or min(max_slots, amplitude_scale_percent, duration_scale_s, time_scale_s) <= 0:
        raise ValueError("blackbox dimensions and pilot scales required")
    torch.manual_seed(seed)
    subsets = [_fid_subset(ex, max_points) for ex in train]
    features = [torch.as_tensor(_blackbox_features(ex, ex.times_s[ids], max_slots=max_slots,
                                                   amp_scale=amplitude_scale_percent,
                                                   duration_scale=duration_scale_s,
                                                   time_scale=time_scale_s), dtype=torch.float64)
                for ex, ids in zip(train, subsets)]
    fid_scale = float(np.median([np.sqrt(np.mean(np.abs(ex.fid[ids])**2))
                                 for ex, ids in zip(train, subsets)]))
    if not np.isfinite(fid_scale) or fid_scale <= 0:
        raise ValueError("nonzero FID scale required for blackbox training")
    targets = [torch.as_tensor(np.column_stack((ex.fid[ids].real, ex.fid[ids].imag)) / fid_scale,
                               dtype=torch.float64) for ex, ids in zip(train, subsets)]
    net = torch.nn.Sequential(torch.nn.Linear(features[0].shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, 2)).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    history = []
    start = perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.stack([torch.mean(torch.sum((net(x) - y)**2, dim=1))
                            for x, y in zip(features, targets)]).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite blackbox loss")
        loss.backward()
        optimizer.step()
        history.append({"epoch": float(epoch + 1), "training_normalized_mse": float(loss.detach())})
    return BlackboxResult(net, max_slots, amplitude_scale_percent, duration_scale_s,
                          time_scale_s, fid_scale, history,
                          tuple(ex.record_id for ex in train), perf_counter() - start)


def predict_fid_blackbox(example: FIDExample, result: BlackboxResult,
                         indices: np.ndarray | None = None) -> np.ndarray:
    torch = _require_torch()
    times = example.times_s if indices is None else example.times_s[indices]
    features = _blackbox_features(example, times, max_slots=result.max_slots,
                                  amp_scale=result.amplitude_scale_percent,
                                  duration_scale=result.duration_scale_s,
                                  time_scale=result.time_scale_s)
    result.network.eval()
    with torch.no_grad():
        pred = result.network(torch.as_tensor(features, dtype=torch.float64)).cpu().numpy()
    return result.fid_scale * (pred[:, 0] + 1j * pred[:, 1])


def evaluate_fid_predictors(heldout: Sequence[FIDExample], model: SpinHamiltonian,
                            readout: ReadoutPhysics, polynomial: PolynomialRFResult,
                            graybox: GrayboxResult, blackbox: BlackboxResult,
                            *, max_points: int = 96) -> dict[str, dict[str, object]]:
    """Heldout complex FID MSE per *record*, preserving the statistical unit."""
    if not heldout:
        raise ValueError("independent heldout FID records required")
    train_ids = set(polynomial.train_record_ids) | set(graybox.train_record_ids) | set(blackbox.train_record_ids)
    if any(ex.record_id in train_ids for ex in heldout):
        raise ValueError("training record leaked into heldout evaluation")
    outputs: dict[str, list[float]] = {key: [] for key in ("classical", "polynomial_rf",
                                                               "blackbox_mlp", "graybox")}
    for ex in heldout:
        ids = _fid_subset(ex, max_points)
        truth = ex.fid[ids]
        predictions = {
            "classical": predict_fid_physics(ex, model, readout, indices=ids),
            "polynomial_rf": predict_fid_physics(ex, model, readout, polynomial.correction, ids),
            "blackbox_mlp": predict_fid_blackbox(ex, blackbox, ids),
            "graybox": predict_fid_graybox(ex, graybox, ids),
        }
        for key, pred in predictions.items():
            outputs[key].append(float(np.mean(np.abs(pred - truth)**2)))
    return {key: {"heldout_record_mse": errors,
                  "heldout_mean_record_mse": float(np.mean(errors)),
                  "n_independent_records": len(errors),
                  "record_ids": [ex.record_id for ex in heldout]}
            for key, errors in outputs.items()}


def optimize_pulse_for_fid(template: FIDExample, desired_fid: np.ndarray,
                           predictor: Callable[[FIDExample], np.ndarray], *,
                           maxiter: int = 40) -> dict[str, object]:
    """Model-guided design for one declared readout task; requires hardware validation.

    This same phase-only objective can be used for calibrated physical,
    polynomial, graybox, and blackbox predictors.  It does not claim a
    blackbox model predicts arbitrary gate unitaries.
    """
    target = np.asarray(desired_fid, dtype=complex)
    if target.shape != template.fid.shape or maxiter < 1:
        raise ValueError("desired FID shape and optimization budget required")

    def objective(phases: np.ndarray) -> float:
        program = PulseProgram(template.program.duration_s, template.program.amplitude_percent,
                               tuple(float(p) for p in phases))
        candidate = FIDExample(program, template.initial_state, template.detector,
                               template.times_s, template.fid, template.family,
                               template.session, template.record_id)
        prediction = predictor(candidate)
        return float(np.mean(np.abs(prediction - target)**2))

    start = perf_counter()
    fit = minimize(objective, np.asarray(template.program.phase_rad), method="L-BFGS-B",
                   options={"maxiter": maxiter})
    program = PulseProgram(template.program.duration_s, template.program.amplitude_percent,
                           tuple(float(p % (2 * pi)) for p in fit.x))
    return {"program": program, "predicted_fid_mse": float(fit.fun),
            "optimizer_iterations": int(fit.nit), "optimizer_success": bool(fit.success),
            "local_design_seconds": perf_counter() - start,
            "hardware_validation_status": "PENDING_INDEPENDENT_MEASUREMENT"}


# E: supervised phase-only pulse generator.  Labels are *local* GRAPE results
# from one common initial phase vector.  Target split is by physical SU(2)
# gate, not by matrix's arbitrary global phase.


def canonical_gate_features(gate: np.ndarray) -> np.ndarray:
    """Canonical real unit quaternion for a physical single-spin gate.

    U and -U, or any other global-phase variant, return the same features.
    Boundary points at angle pi use a lexicographic sign convention.
    """
    matrix = np.asarray(gate, dtype=complex)
    if matrix.shape != (2, 2) or not np.allclose(matrix.conj().T @ matrix, _ID2, atol=1e-8):
        raise ValueError("physical single-spin unitary required")
    normalized = matrix / np.sqrt(np.linalg.det(matrix))
    quaternion = np.array([0.5 * np.trace(normalized).real,
                           (0.5j * np.trace(_SX @ normalized)).real,
                           (0.5j * np.trace(_SY @ normalized)).real,
                           (0.5j * np.trace(_SZ @ normalized)).real], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    for value in quaternion:
        if abs(value) > 1e-10:
            if value < 0:
                quaternion = -quaternion
            break
    return quaternion


def physical_gate_distance(first: np.ndarray, second: np.ndarray) -> float:
    """1-|Tr(U†V)|²/4: global-phase invariant one-spin process distance."""
    a = np.asarray(first, dtype=complex)
    b = np.asarray(second, dtype=complex)
    if a.shape != (2, 2) or b.shape != (2, 2):
        raise ValueError("single-spin gates required")
    return float(np.clip(1 - abs(np.trace(a.conj().T @ b) / 2)**2, 0, 1))


@dataclass(frozen=True)
class GateTarget:
    gate_id: str
    gate2: np.ndarray
    axis: tuple[float, float, float]
    angle_rad: float
    split: str

    def __post_init__(self) -> None:
        canonical_gate_features(self.gate2)
        if self.split not in ("train", "validation", "test") or not self.gate_id:
            raise ValueError("named target and declared split required")


def sample_gate_targets(n_train: int, n_validation: int, n_test: int,
                        *, seed: int, min_distance: float = 1e-5) -> list[GateTarget]:
    """Axes uniform on S², angles uniform in [0,pi]; explicitly NOT Haar SU(2)."""
    if min(n_train, n_validation, n_test) < 1 or min_distance <= 0:
        raise ValueError("nonempty train/validation/test splits required")
    rng = np.random.default_rng(seed)
    splits = ["train"] * n_train + ["validation"] * n_validation + ["test"] * n_test
    gates = []
    attempts = 0
    for split in splits:
        while True:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("could not draw distinct physical target gates")
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = float(rng.uniform(0.08, pi))
            gate = rotation(axis, angle)
            if all(physical_gate_distance(gate, old.gate2) > min_distance for old in gates):
                break
        gates.append(GateTarget(f"gate_{len(gates):04d}", gate,
                                tuple(float(v) for v in axis), angle, split))
    return gates


@dataclass
class PulseLabel:
    target: GateTarget
    program: PulseProgram
    simulated_design_infidelity: float
    label_optimization_seconds: float
    label_iterations: int


@dataclass
class LabelCollection:
    labels: list[PulseLabel]
    rejected: list[dict[str, object]]
    common_start_phases_rad: tuple[float, ...]
    total_generation_seconds: float
    attempts: int


def generate_grape_labels(targets: Sequence[GateTarget], model: SpinHamiltonian,
                          *, target_spin: int, duration_s: float,
                          amplitude_percent: float, common_start_phases_rad: Sequence[float],
                          ensemble: Sequence[EnsemblePoint], maxiter: int,
                          accept_infidelity: float) -> LabelCollection:
    """Generate supervised labels locally; rejects and costs are preserved."""
    phases = tuple(float(p) for p in common_start_phases_rad)
    if not phases or not 0 < accept_infidelity < 1:
        raise ValueError("common phase start and explicit quality threshold required")
    if not 0 <= target_spin < model.n_spins:
        raise ValueError("target spin outside actual model")
    labels: list[PulseLabel] = []
    rejected: list[dict[str, object]] = []
    start = perf_counter()
    for target in targets:
        embedded = embed_single_spin_gate(target.gate2, target_spin, model.n_spins)
        result = optimize_grape(model, embedded, duration_s=duration_s,
                                n_segments=len(phases), amplitude_percent=amplitude_percent,
                                ensemble=ensemble, initial_phases=phases, maxiter=maxiter)
        if result.design_infidelity <= accept_infidelity:
            labels.append(PulseLabel(target, result.program, result.design_infidelity,
                                     result.elapsed_s, result.iterations))
        else:
            rejected.append({"gate_id": target.gate_id, "split": target.split,
                             "simulated_design_infidelity": result.design_infidelity,
                             "optimization_seconds": result.elapsed_s,
                             "iterations": result.iterations,
                             "reason": "GRAPE_LABEL_QUALITY_NOT_REACHED"})
    return LabelCollection(labels, rejected, phases, perf_counter() - start, len(targets))


@dataclass
class PulseGeneratorResult:
    network: object
    n_segments: int
    duration_s: float
    amplitude_percent: float
    history: list[dict[str, float]]
    training_seconds: float
    train_gate_ids: tuple[str, ...]
    validation_gate_ids: tuple[str, ...]


def _generator_xy(labels: Sequence[PulseLabel]):
    x = np.vstack([canonical_gate_features(label.target.gate2) for label in labels])
    y = np.stack([np.column_stack((np.cos(label.program.phase_rad),
                                   np.sin(label.program.phase_rad))) for label in labels])
    return x, y


def train_pulse_generator(train_labels: Sequence[PulseLabel],
                          validation_labels: Sequence[PulseLabel], *,
                          hidden: int = 48, epochs: int = 200,
                          learning_rate: float = 0.003, seed: int = 0) -> PulseGeneratorResult:
    """Feed-forward SU(2)→phase generator with circular sin/cos target loss.

    Validation history is reported, but the final epoch is fixed in advance;
    heldout test gates never affect weights or selection.
    """
    torch = _require_torch()
    if not train_labels or not validation_labels or epochs < 1 or hidden < 2:
        raise ValueError("nonempty train/validation labels and training budget required")
    n = len(train_labels[0].program.phase_rad)
    durations = {tuple(label.program.duration_s) for label in [*train_labels, *validation_labels]}
    amplitudes = {tuple(label.program.amplitude_percent) for label in [*train_labels, *validation_labels]}
    if len(durations) != 1 or len(amplitudes) != 1 or len(next(iter(durations))) != n:
        raise ValueError("supervised generator requires common segment geometry")
    if len(set(next(iter(durations)))) != 1 or len(set(next(iter(amplitudes)))) != 1:
        raise ValueError("phase-only generator requires equal slots and constant RF amplitude")
    if any(label.target.split != "train" for label in train_labels):
        raise ValueError("training labels contain nontraining gates")
    if any(label.target.split != "validation" for label in validation_labels):
        raise ValueError("validation labels have wrong split")
    for train in train_labels:
        if any(physical_gate_distance(train.target.gate2, val.target.gate2) < 1e-9
               for val in validation_labels):
            raise ValueError("same physical gate appears in train and validation")
    tx, ty = _generator_xy(train_labels)
    vx, vy = _generator_xy(validation_labels)
    x = torch.as_tensor(tx, dtype=torch.float64)
    y = torch.as_tensor(ty, dtype=torch.float64)
    xv = torch.as_tensor(vx, dtype=torch.float64)
    yv = torch.as_tensor(vy, dtype=torch.float64)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(4, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, 2 * n)).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    def circular_loss(features, target):
        raw = net(features).reshape(-1, n, 2)
        unit = raw / torch.linalg.vector_norm(raw, dim=2, keepdim=True).clamp_min(1e-12)
        return torch.mean(torch.sum((unit - target)**2, dim=2))

    history = []
    start = perf_counter()
    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        loss = circular_loss(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite pulse generator loss")
        loss.backward()
        optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = circular_loss(xv, yv)
        history.append({"epoch": float(epoch + 1), "training_circular_mse": float(loss.detach()),
                        "validation_circular_mse": float(val_loss.detach())})
    duration = float(sum(next(iter(durations))))
    amplitude = float(next(iter(amplitudes))[0])
    return PulseGeneratorResult(net, n, duration, amplitude, history,
                                perf_counter() - start,
                                tuple(label.target.gate_id for label in train_labels),
                                tuple(label.target.gate_id for label in validation_labels))


def infer_generated_pulse(target: GateTarget, generator: PulseGeneratorResult) -> tuple[PulseProgram, float]:
    torch = _require_torch()
    features = canonical_gate_features(target.gate2)
    start = perf_counter()
    generator.network.eval()
    with torch.no_grad():
        raw = generator.network(torch.as_tensor(features, dtype=torch.float64)).cpu().numpy()
    circular = raw.reshape(generator.n_segments, 2)
    phases = np.arctan2(circular[:, 1], circular[:, 0]) % (2 * pi)
    result = PulseProgram((generator.duration_s / generator.n_segments,) * generator.n_segments,
                          (generator.amplitude_percent,) * generator.n_segments,
                          tuple(float(p) for p in phases))
    return result, perf_counter() - start


def compare_generated_pulses(
    test_targets: Sequence[GateTarget], train_labels: Sequence[PulseLabel],
    generator: PulseGeneratorResult, model: SpinHamiltonian, *, target_spin: int,
    ensemble: Sequence[EnsemblePoint], heldout_ensemble: Sequence[EnsemblePoint],
    common_start_phases_rad: Sequence[float], full_grape_iterations: int,
    refine_iterations: int,
) -> list[dict[str, object]]:
    """Five same-Windows simulated baselines; each program awaits hardware test."""
    if not test_targets or not train_labels or not heldout_ensemble:
        raise ValueError("independent targets, library, and heldout grid required")
    if any(t.split != "test" for t in test_targets):
        raise ValueError("final comparison accepts only test targets")
    if any(label.target.split != "train" for label in train_labels):
        raise ValueError("nearest library must contain only training gates")
    if full_grape_iterations < 1 or refine_iterations < 1:
        raise ValueError("explicit optimization budgets required")
    for target in test_targets:
        if any(physical_gate_distance(target.gate2, label.target.gate2) < 1e-9
               for label in train_labels):
            raise ValueError("same physical gate leaked into training library")
    output = []
    for target in test_targets:
        embedded = embed_single_spin_gate(target.gate2, target_spin, model.n_spins)
        nearest = min(train_labels, key=lambda lab: physical_gate_distance(
            target.gate2, lab.target.gate2))
        generated, inference_seconds = infer_generated_pulse(target, generator)
        fresh = optimize_grape(model, embedded, duration_s=generator.duration_s,
                               n_segments=generator.n_segments,
                               amplitude_percent=generator.amplitude_percent,
                               ensemble=ensemble, initial_phases=common_start_phases_rad,
                               maxiter=full_grape_iterations)
        warm = optimize_grape(model, embedded, duration_s=generator.duration_s,
                              n_segments=generator.n_segments,
                              amplitude_percent=generator.amplitude_percent,
                              ensemble=ensemble, initial_phases=nearest.program.phase_rad,
                              maxiter=refine_iterations)
        refined = optimize_grape(model, embedded, duration_s=generator.duration_s,
                                 n_segments=generator.n_segments,
                                 amplitude_percent=generator.amplitude_percent,
                                 ensemble=ensemble, initial_phases=generated.phase_rad,
                                 maxiter=refine_iterations)
        candidates = {"fresh_grape": (fresh.program, fresh.elapsed_s),
                      "nearest_library": (nearest.program, 0.0),
                      "warm_start_grape": (warm.program, warm.elapsed_s),
                      "network": (generated, inference_seconds),
                      "network_refined": (refined.program, inference_seconds + refined.elapsed_s)}
        for name, (program, design_seconds) in candidates.items():
            output.append({"gate_id": target.gate_id, "method": name,
                           "simulated_design_infidelity": ensemble_infidelity(
                               model, embedded, program, ensemble),
                           "simulated_heldout_infidelity": ensemble_infidelity(
                               model, embedded, program, heldout_ensemble),
                           "local_design_seconds": float(design_seconds),
                           "pulse_duration_s": float(sum(program.duration_s)),
                           "rf_integral_percent_s": program.rf_integral_percent_s,
                           "program": program.serialized(),
                           "hardware_validation_status": "PENDING_INDEPENDENT_MEASUREMENT"})
    return output


def break_even_uses(training_and_label_seconds: float, baseline_design_seconds: float,
                    generator_design_seconds: float, *, equivalent_quality: bool) -> int | None:
    """Return uses to recover offline cost only if quality and savings permit."""
    if not equivalent_quality or baseline_design_seconds <= generator_design_seconds:
        return None
    if training_and_label_seconds < 0:
        raise ValueError("offline cost cannot be negative")
    return int(np.ceil(training_and_label_seconds /
                       (baseline_design_seconds - generator_design_seconds)))
