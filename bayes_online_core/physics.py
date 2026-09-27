"""Independent two-spin physics and Bell-state evaluation for bayes_online.

Units are explicit: K = H/h is in Hz, time is in seconds, and
U(t) = exp(-i 2*pi*K*t).  The model is an intentionally small weak-coupling
model.  Its agreement with Gemini Lab, channel signs, and effective readout
operators must be checked with held-out measurements before physical claims.

This module does not issue hardware commands, infer calibration from an ideal
Bell state, or silently turn a thermal state into a pure state.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import expm


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PAULI_SINGLE = {"I": I2, "X": X, "Y": Y, "Z": Z}
PAULI_LABELS = tuple(
    a + b for a, b in product("IXYZ", repeat=2) if a + b != "II"
)
PAULIS = {label: np.kron(PAULI_SINGLE[label[0]], PAULI_SINGLE[label[1]])
          for label in PAULI_LABELS}
II = np.eye(4, dtype=complex)
XI, YI, ZI = PAULIS["XI"], PAULIS["YI"], PAULIS["ZI"]
IX, IY, IZ = PAULIS["IX"], PAULIS["IY"], PAULIS["IZ"]
ZZ = PAULIS["ZZ"]


@dataclass(frozen=True)
class SpinParams:
    """Effective rotating-frame parameters in Hz, never carrier frequencies."""

    df_h_hz: float = 0.0
    df_p_hz: float = 0.0
    j_hz: float = 0.0


@dataclass(frozen=True)
class Drive:
    """Actual signed Rabi frequency and phase at one channel during a segment.

    A caller maps sent RF amplitude to rabi_hz using its measured gain.  The
    transmitter carrier and receiver demodulation are separate from df_hz.
    """

    rabi_hz: float
    phase_rad: float = 0.0


@dataclass(frozen=True)
class CoherentSegment:
    """One explicit simultaneous H/P control interval, in execution order."""

    duration_s: float
    drive_h: Drive | None = None
    drive_p: Drive | None = None


def hamiltonian_hz(
    params: SpinParams,
    drive_h: Drive | None = None,
    drive_p: Drive | None = None,
) -> np.ndarray:
    """Weak-coupling K = df_H ZI/2 + df_P IZ/2 + J ZZ/4 + RF terms."""

    k = (params.df_h_hz * ZI / 2 + params.df_p_hz * IZ / 2
         + params.j_hz * ZZ / 4).astype(complex)
    if drive_h is not None:
        k += drive_h.rabi_hz * (
            np.cos(drive_h.phase_rad) * XI + np.sin(drive_h.phase_rad) * YI
        ) / 2
    if drive_p is not None:
        k += drive_p.rabi_hz * (
            np.cos(drive_p.phase_rad) * IX + np.sin(drive_p.phase_rad) * IY
        ) / 2
    return k


def segment_unitary(
    duration_s: float,
    params: SpinParams,
    drive_h: Drive | None = None,
    drive_p: Drive | None = None,
) -> np.ndarray:
    if not np.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and nonnegative")
    return expm(-2j * np.pi * duration_s * hamiltonian_hz(params, drive_h, drive_p))


def propagate(
    rho: np.ndarray,
    duration_s: float,
    params: SpinParams,
    drive_h: Drive | None = None,
    drive_p: Drive | None = None,
) -> np.ndarray:
    rho = _matrix4(rho)
    u = segment_unitary(duration_s, params, drive_h, drive_p)
    return u @ rho @ u.conj().T


def sequence_unitary(segments: Sequence[CoherentSegment], params: SpinParams) -> np.ndarray:
    """Compose finite-pulse and free-evolution intervals in given time order."""

    total = II.copy()
    for segment in segments:
        total = segment_unitary(segment.duration_s, params,
                                segment.drive_h, segment.drive_p) @ total
    return total


def propagate_sequence(
    rho: np.ndarray, segments: Sequence[CoherentSegment], params: SpinParams
) -> np.ndarray:
    rho = _matrix4(rho)
    total = sequence_unitary(segments, params)
    return total @ rho @ total.conj().T


def rx(angle_rad: float, qubit: str) -> np.ndarray:
    return _local(expm(-0.5j * angle_rad * X), qubit)


def ry(angle_rad: float, qubit: str) -> np.ndarray:
    return _local(expm(-0.5j * angle_rad * Y), qubit)


def rz(angle_rad: float, qubit: str) -> np.ndarray:
    return _local(expm(-0.5j * angle_rad * Z), qubit)


def _local(u: np.ndarray, qubit: str) -> np.ndarray:
    if qubit == "H":
        return np.kron(u, I2)
    if qubit == "P":
        return np.kron(I2, u)
    raise ValueError("qubit must be H or P")


def hadamard(qubit: str) -> np.ndarray:
    return _local((X + Z) / np.sqrt(2), qubit)


def hadamard_yz_decomposition(qubit: str) -> np.ndarray:
    """H = i Ry(pi/2) Rz(pi), with the rightmost operation applied first."""

    return 1j * ry(np.pi / 2, qubit) @ rz(np.pi, qubit)


def hadamard_xy_decomposition(qubit: str) -> np.ndarray:
    """H = i Rx(pi) Ry(pi/2); two transverse pulses, up to known phase."""

    return 1j * rx(np.pi, qubit) @ ry(np.pi / 2, qubit)


def rz_xy_decomposition(angle_rad: float, qubit: str) -> np.ndarray:
    """Virtual Z alternative using three ideal transverse rotations."""

    return rx(np.pi / 2, qubit) @ ry(angle_rad, qubit) @ rx(-np.pi / 2, qubit)


def cnot(control: str, target: str) -> np.ndarray:
    if (control, target) not in {("H", "P"), ("P", "H")}:
        raise ValueError("control and target must be distinct H/P spins")
    p0 = (I2 + Z) / 2
    p1 = (I2 - Z) / 2
    if control == "H":
        return np.kron(p0, I2) + np.kron(p1, X)
    return np.kron(I2, p0) + np.kron(X, p1)


def cz() -> np.ndarray:
    return np.diag([1, 1, 1, -1]).astype(complex)


def cz_from_zz(j_hz: float) -> tuple[float, np.ndarray]:
    """CZ from free ZZ and local Z rotations; returns duration and unitary.

    Requires verified coherent free evolution and independently known J sign.
    The matrix is mathematical, not evidence that a device performed it.
    """

    if not np.isfinite(j_hz) or j_hz == 0:
        raise ValueError("a nonzero, measured J in Hz is required")
    sign = np.sign(j_hz)
    duration_s = 1 / (2 * abs(j_hz))
    free = segment_unitary(duration_s, SpinParams(j_hz=j_hz))
    local_z = rz(-sign * np.pi / 2, "H") @ rz(-sign * np.pi / 2, "P")
    return duration_s, np.exp(-1j * sign * np.pi / 4) * local_z @ free


def cnot_from_cz(control: str, target: str) -> np.ndarray:
    return hadamard(target) @ cz() @ hadamard(target)


def verify_gate_conventions(atol: float = 1e-12) -> dict[str, float]:
    """Numerical unitary identities, with no fitted global phase."""

    checks = {
        f"hadamard_{q}": float(np.linalg.norm(hadamard_yz_decomposition(q) - hadamard(q)))
        for q in ("H", "P")
    }
    for qubit in ("H", "P"):
        checks[f"hadamard_xy_{qubit}"] = float(
            np.linalg.norm(hadamard_xy_decomposition(qubit) - hadamard(qubit))
        )
        checks[f"rz_xy_{qubit}"] = float(
            np.linalg.norm(rz_xy_decomposition(np.pi / 3, qubit) - rz(np.pi / 3, qubit))
        )
    checks["cz_positive_j"] = float(np.linalg.norm(cz_from_zz(40.0)[1] - cz()))
    checks["cz_negative_j"] = float(np.linalg.norm(cz_from_zz(-40.0)[1] - cz()))
    for control, target in (("H", "P"), ("P", "H")):
        checks[f"cnot_{control}_{target}"] = float(
            np.linalg.norm(cnot_from_cz(control, target) - cnot(control, target))
        )
    if max(checks.values()) > atol:
        raise AssertionError(f"gate identities failed: {checks}")
    return checks


@dataclass(frozen=True)
class TemporalBranch:
    name: str
    # Gates listed in execution order; total unitary is right-to-left product.
    gates: tuple[str, ...]
    unitary: np.ndarray


def temporal_pps_branches() -> tuple[TemporalBranch, TemporalBranch, TemporalBranch]:
    """Three |00>-preserving permutations of the excited populations.

    The CNOT matrices only specify a circuit.  Physical execution requires
    independently qualified local pulses, J evolution, and temporal order.
    """

    hp = cnot("H", "P")
    ph = cnot("P", "H")
    return (
        TemporalBranch("identity", (), II),
        TemporalBranch("cycle_forward", ("CNOT_H_P", "CNOT_P_H"), ph @ hp),
        TemporalBranch("cycle_reverse", ("CNOT_P_H", "CNOT_H_P"), hp @ ph),
    )


def temporal_average_deviation(
    branch_deviations: Sequence[np.ndarray],
    weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
) -> np.ndarray:
    """Average whole-branch measured deviations, without individual scaling.

    Input branches must share one externally frozen receiver scale and are
    each a complete independently acquired preparation/readout experiment.
    """

    if len(branch_deviations) != 3 or len(weights) != 3:
        raise ValueError("temporal PPS requires exactly three acquired branches")
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError("weights must be fixed nonnegative weights summing to one")
    mats = [_matrix4(m) for m in branch_deviations]
    if any(abs(np.trace(m)) > 1e-8 for m in mats):
        raise ValueError("each input must be a traceless deviation matrix")
    return sum((wi * mi for wi, mi in zip(w, mats)), np.zeros((4, 4), complex))


def pps_target_deviation() -> np.ndarray:
    """Unscaled |00> deviation target (ZI + IZ + ZZ)/4."""

    return (ZI + IZ + ZZ) / 4


@dataclass(frozen=True)
class TomographyDesign:
    """Frozen map from 15 real Pauli coefficients to complex FID components."""

    complex_matrix: np.ndarray
    real_matrix: np.ndarray
    measurement_labels: tuple[str, ...]
    readout_unitaries: tuple[np.ndarray, ...]
    rank: int
    condition: float
    gains: Mapping[str, complex]


def default_readout_unitaries() -> tuple[tuple[str, np.ndarray], ...]:
    """Nine ideal readout settings; a real evaluator must qualify its pulses."""

    axes = (("I", II),
            ("Rx90H", rx(np.pi / 2, "H")),
            ("Ry90H", ry(np.pi / 2, "H")))
    other = (("I", II),
             ("Rx90P", rx(np.pi / 2, "P")),
             ("Ry90P", ry(np.pi / 2, "P")))
    return tuple((f"{name_h}+{name_p}", up @ uh)
                 for name_h, uh in axes for name_p, up in other)


def tomography_design(
    readout_unitaries: Sequence[tuple[str, np.ndarray]] | None = None,
    complex_gains: Mapping[str, complex] | None = None,
) -> TomographyDesign:
    """Build the measurement operator map, then check rank and conditioning.

    The default measurement model is two resolved multiplet lines per spin:
    sigma_+ on the observed spin times |0>/<0| or |1>/<1| on the spectator.
    Whether Gemini Lab actually resolves these four complex coefficients,
    and their complex gains, requires an independent held-out qualification.
    Supply the *actual* frozen pulse propagators in readout_unitaries; labels
    alone do not define measurements.  The identity mapping is never inferred
    from a Bell target.
    """

    settings = tuple(readout_unitaries or default_readout_unitaries())
    if not settings:
        raise ValueError("at least one readout setting is required")
    gains = dict(complex_gains or {"H": 1 + 0j, "P": 1 + 0j})
    if set(gains) != {"H", "P"} or any(not np.isfinite(v) or abs(v) == 0 for v in gains.values()):
        raise ValueError("two finite nonzero frozen complex readout gains required")
    plus = (X + 1j * Y) / 2
    p0, p1 = (I2 + Z) / 2, (I2 - Z) / 2
    operators = (
        ("H|P0", np.kron(plus, p0), gains["H"]),
        ("H|P1", np.kron(plus, p1), gains["H"]),
        ("P|H0", np.kron(p0, plus), gains["P"]),
        ("P|H1", np.kron(p1, plus), gains["P"]),
    )
    labels = []
    rows = []
    unitaries = []
    for setting_name, u in settings:
        u = _matrix4(u)
        if not np.allclose(u.conj().T @ u, II, atol=1e-9):
            raise ValueError(f"readout {setting_name} is not unitary")
        unitaries.append(u)
        for operator_name, m, gain in operators:
            labels.append(f"{setting_name}:{operator_name}")
            rows.append([gain * np.trace(m @ u @ PAULIS[label] @ u.conj().T) / 4
                         for label in PAULI_LABELS])
    ac = np.asarray(rows, dtype=complex)
    ar = np.concatenate((ac.real, ac.imag), axis=0)
    rank = int(np.linalg.matrix_rank(ar))
    condition = float(np.linalg.cond(ar))
    return TomographyDesign(ac, ar, tuple(labels), tuple(unitaries),
                            rank, condition, gains)


def reconstruct_deviation(
    measured_coefficients: Sequence[complex],
    design: TomographyDesign,
    covariance: np.ndarray | None = None,
    max_condition: float = 1e8,
) -> dict[str, object]:
    """Linear tomography of an unknown state, without an ideal target prior.

    A measured coefficient is one independently fitted complex multiplet line
    under one readout; the caller must obtain it from exported FID with its
    uncertainty.  The covariance, when supplied, is for the stacked real
    and imaginary coefficients, including their correlation.
    """

    y = np.asarray(measured_coefficients, dtype=complex)
    if y.shape != (len(design.measurement_labels),) or not np.all(np.isfinite(y)):
        raise ValueError("one finite complex coefficient per measurement required")
    if design.rank != 15 or design.condition > max_condition:
        raise ValueError(f"tomography NONIDENTIFIABLE: rank={design.rank}/15, "
                         f"condition={design.condition:.3g}")
    b = np.concatenate((y.real, y.imag))
    a = design.real_matrix
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (len(b), len(b)) or not np.allclose(cov, cov.T):
            raise ValueError("covariance must be a symmetric 2N by 2N matrix")
        eig, vec = np.linalg.eigh(cov)
        if np.min(eig) <= 0:
            raise ValueError("covariance must be positive definite")
        invsqrt = (vec / np.sqrt(eig)) @ vec.T
        a_fit, b_fit = invsqrt @ a, invsqrt @ b
        parameter_covariance = np.linalg.pinv(a_fit.T @ a_fit)
    else:
        a_fit, b_fit = a, b
        parameter_covariance = None
    coeff, _, _, _ = np.linalg.lstsq(a_fit, b_fit, rcond=None)
    delta = sum((float(v) * PAULIS[label] / 4
                 for label, v in zip(PAULI_LABELS, coeff)),
                np.zeros((4, 4), dtype=complex))
    predicted = design.complex_matrix @ coeff
    return {
        "pauli_coefficients": dict(zip(PAULI_LABELS, map(float, coeff))),
        "delta_rho": delta,
        "predicted_coefficients": predicted,
        "residual_rms": float(np.sqrt(np.mean(abs(y - predicted) ** 2))),
        "rank": design.rank,
        "condition": design.condition,
        "conditional_pauli_covariance": parameter_covariance,
    }


def reconstruct_full_density(
    delta_rho: np.ndarray,
    polarization_scale: float,
    project_physical: bool = False,
) -> dict[str, object]:
    """Add a separately calibrated absolute scale to a traceless deviation.

    A full-state claim requires a calibrated polarization_scale from an
    independent common reference.  ``project_physical`` reports both the
    unprojected estimate and its nearest PSD trace-one matrix; projection is
    never silent.
    """

    delta = _matrix4(delta_rho)
    if not np.isfinite(polarization_scale) or polarization_scale <= 0:
        raise ValueError("an independent positive absolute scale is required")
    if abs(np.trace(delta)) > 1e-8 or not np.allclose(delta, delta.conj().T, atol=1e-8):
        raise ValueError("delta_rho must be traceless Hermitian")
    unprojected = II / 4 + polarization_scale * delta
    eig, vectors = np.linalg.eigh(unprojected)
    result = {
        "rho_unprojected": unprojected,
        "minimum_eigenvalue_unprojected": float(np.min(eig)),
        "rho_physical": None,
        "projection_frobenius": None,
        "polarization_scale": float(polarization_scale),
    }
    if project_physical:
        corrected = _simplex_projection(eig)
        projected = (vectors * corrected) @ vectors.conj().T
        result["rho_physical"] = projected
        result["projection_frobenius"] = float(np.linalg.norm(projected - unprojected))
    elif np.min(eig) >= -1e-9:
        result["rho_physical"] = unprojected
        result["projection_frobenius"] = 0.0
    return result


def _simplex_projection(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    active = np.nonzero(ordered - (cumulative - 1) / np.arange(1, len(values) + 1) > 0)[0]
    threshold = (cumulative[active[-1]] - 1) / (active[-1] + 1)
    return np.maximum(values - threshold, 0)


def bell_target(label: str) -> np.ndarray:
    """Ideal mathematical targets, intended only for an isolated scorer."""

    vectors = {
        "Phi+": np.array([1, 0, 0, 1], complex),
        "Phi-": np.array([1, 0, 0, -1], complex),
        "Psi+": np.array([0, 1, 1, 0], complex),
        "Psi-": np.array([0, 1, -1, 0], complex),
    }
    if label not in vectors:
        raise ValueError("Bell label must be Phi+, Phi-, Psi+, or Psi-")
    ket = vectors[label] / np.sqrt(2)
    return np.outer(ket, ket.conj())


def bell_preparation_gates(label: str) -> tuple[str, ...]:
    """Abstract gates in execution order, before hardware pulse compilation."""

    suffixes = {
        "Phi+": (),
        "Phi-": ("Z_H",),
        "Psi+": ("X_P",),
        "Psi-": ("X_P", "Z_H"),
    }
    if label not in suffixes:
        raise ValueError("Bell label must be Phi+, Phi-, Psi+, or Psi-")
    return ("H_H", "CNOT_H_P") + suffixes[label]


def ideal_gate_sequence(gates: Sequence[str]) -> np.ndarray:
    """Ideal unitary for a small explicit gate vocabulary, for numeric checks."""

    available = {
        "H_H": hadamard("H"), "H_P": hadamard("P"),
        "X_H": XI, "X_P": IX, "Z_H": ZI, "Z_P": IZ,
        "CNOT_H_P": cnot("H", "P"), "CNOT_P_H": cnot("P", "H"),
        "CZ": cz(),
    }
    u = II.copy()
    for gate in gates:
        if gate not in available:
            raise ValueError(f"unsupported ideal gate {gate}")
        u = available[gate] @ u
    return u


def bell_score(rho: np.ndarray, label: str) -> dict[str, object]:
    """State fidelity of an independently reconstructed normalized state.

    A pseudo-pure *effective* state score is not entanglement of the full
    thermal ensemble and a Bell score is not a CNOT process fidelity.
    """

    rho = _matrix4(rho)
    if not np.allclose(rho, rho.conj().T, atol=1e-8):
        raise ValueError("Bell scoring requires a Hermitian state")
    if not np.isclose(np.trace(rho), 1, atol=1e-8):
        raise ValueError("Bell scoring requires trace one")
    eig = np.linalg.eigvalsh(rho)
    if np.min(eig) < -1e-8:
        raise ValueError("Bell scoring requires a physical positive state")
    target = bell_target(label)
    fidelity = float(np.trace(rho @ target).real)
    pauli = {p: float(np.trace(m @ rho).real) for p, m in PAULIS.items()}
    target_pauli = {p: float(np.trace(m @ target).real) for p, m in PAULIS.items()}
    return {
        "label": label,
        "state_fidelity": fidelity,
        "pauli_coefficients": pauli,
        "pauli_difference": {p: pauli[p] - target_pauli[p] for p in PAULI_LABELS},
        "frobenius_error": float(np.linalg.norm(rho - target)),
        "minimum_eigenvalue": float(np.min(eig)),
    }


def _matrix4(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if arr.shape != (4, 4) or not np.all(np.isfinite(arr)):
        raise ValueError("a finite 4 by 4 complex matrix is required")
    return arr
