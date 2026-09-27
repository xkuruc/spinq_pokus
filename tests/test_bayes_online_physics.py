"""Offline mathematical checks; no device commands or claimed hardware data."""

import unittest
from itertools import combinations

import numpy as np

from bayes_online_core.physics import (
    CoherentSegment,
    Drive,
    II,
    PAULI_LABELS,
    PAULIS,
    SpinParams,
    bell_score,
    bell_preparation_gates,
    bell_target,
    default_readout_unitaries,
    minimum_readout_unitaries,
    ideal_gate_sequence,
    pps_target_deviation,
    propagate,
    reconstruct_deviation,
    reconstruct_full_density,
    rx,
    segment_unitary,
    sequence_unitary,
    temporal_average_deviation,
    temporal_pps_branches,
    tomography_design,
    verify_gate_conventions,
)


class SpinPhysicsTests(unittest.TestCase):
    def test_finite_pulse_retains_detuning_and_j_evolution(self):
        duration_s = 0.002
        rabi_hz = 1 / (4 * duration_s)
        ideal = segment_unitary(duration_s, SpinParams(), Drive(rabi_hz, 0))
        np.testing.assert_allclose(ideal, rx(np.pi / 2, "H"), atol=1e-12)
        coupled = segment_unitary(
            duration_s, SpinParams(df_h_hz=32, df_p_hz=-13, j_hz=70),
            Drive(rabi_hz, 0),
        )
        self.assertGreater(np.linalg.norm(coupled - ideal), 0.1)
        np.testing.assert_allclose(coupled.conj().T @ coupled, II, atol=1e-12)
        rho = np.diag([1, 0, 0, 0]).astype(complex)
        evolved = propagate(rho, duration_s, SpinParams(j_hz=70), Drive(rabi_hz))
        self.assertAlmostEqual(np.trace(evolved).real, 1.0)
        pieces = (CoherentSegment(duration_s, Drive(rabi_hz)),
                  CoherentSegment(duration_s / 2),
                  CoherentSegment(duration_s, drive_p=Drive(rabi_hz, np.pi / 2)))
        composed = sequence_unitary(pieces, SpinParams(j_hz=70))
        manual = II.copy()
        for piece in pieces:
            manual = segment_unitary(piece.duration_s, SpinParams(j_hz=70),
                                     piece.drive_h, piece.drive_p) @ manual
        np.testing.assert_allclose(composed, manual, atol=1e-12)

    def test_ideal_gate_decompositions(self):
        self.assertLess(max(verify_gate_conventions().values()), 1e-12)

    def test_temporal_pps_is_three_complete_branches_and_not_one_unitary(self):
        probabilities = np.array([0.55, 0.25, 0.15, 0.05])
        thermal = np.diag(probabilities).astype(complex)
        branches = temporal_pps_branches()
        self.assertEqual(len(branches), 3)
        deviations = []
        for branch in branches:
            np.testing.assert_allclose(branch.unitary[:, 0], [1, 0, 0, 0])
            transformed = branch.unitary @ thermal @ branch.unitary.conj().T
            np.testing.assert_allclose(np.linalg.eigvalsh(transformed),
                                       np.linalg.eigvalsh(thermal))
            deviations.append(transformed - II / 4)
        average = temporal_average_deviation(deviations)
        expected_excited = probabilities[1:].mean()
        np.testing.assert_allclose(np.diag(average + II / 4),
                                   [probabilities[0]] + [expected_excited] * 3)
        np.testing.assert_allclose(average,
                                   (probabilities[0] - expected_excited) *
                                   pps_target_deviation())
        self.assertFalse(any(np.allclose(average, one) for one in deviations))

    def test_tomography_rank_recovery_and_condition_gate(self):
        design = tomography_design()
        self.assertEqual(design.rank, 15)
        self.assertLess(design.condition, 2)
        state = bell_target("Phi+")
        coefficients = np.array([np.trace(PAULIS[name] @ state).real
                                 for name in PAULI_LABELS])
        measured = design.complex_matrix @ coefficients
        reconstruction = reconstruct_deviation(measured, design)
        np.testing.assert_allclose(reconstruction["delta_rho"], state - II / 4,
                                   atol=1e-12)
        self.assertLess(reconstruction["residual_rms"], 1e-12)
        incomplete = tomography_design(default_readout_unitaries()[:1])
        self.assertLess(incomplete.rank, 15)
        with self.assertRaisesRegex(ValueError, "NONIDENTIFIABLE"):
            reconstruct_deviation(np.zeros(len(incomplete.measurement_labels)),
                                  incomplete)

    def test_minimum_physical_readout_subset_is_rank_15_and_well_conditioned(self):
        physical = default_readout_unitaries()[1:]
        self.assertEqual(len(physical), 8)
        self.assertNotIn("I+I", [name for name, _ in physical])
        for size in range(1, 4):
            self.assertTrue(all(tomography_design(group).rank < 15
                                for group in combinations(physical, size)))
        four_setting_designs = [
            (group, tomography_design(group))
            for group in combinations(physical, 4)
        ]
        full_rank = [(group, design) for group, design in four_setting_designs
                     if design.rank == 15]
        self.assertEqual(len(full_rank), 45)
        selected = minimum_readout_unitaries()
        self.assertEqual([name for name, _ in selected],
                         ["I+Rx90P", "I+Ry90P", "Rx90H+I", "Ry90H+I"])
        design = tomography_design(selected)
        self.assertEqual(design.rank, 15)
        self.assertAlmostEqual(design.condition, 2.0)
        self.assertAlmostEqual(design.condition,
                               min(candidate.condition for _, candidate in full_rank))
        cheapest = min(sum(name.count("90") for name, _ in group)
                       for group, candidate in full_rank
                       if np.isclose(candidate.condition, design.condition))
        self.assertEqual(sum(name.count("90") for name, _ in selected), cheapest)
        self.assertEqual(design.complex_matrix.shape, (16, 15))
        self.assertEqual(design.real_matrix.shape, (32, 15))
        self.assertTrue(all(name.count("90") == 1 for name, _ in selected))
        state = bell_target("Psi-")
        coefficients = np.array([np.trace(PAULIS[name] @ state).real
                                 for name in PAULI_LABELS])
        reconstructed = reconstruct_deviation(design.complex_matrix @ coefficients,
                                              design)
        np.testing.assert_allclose(reconstructed["delta_rho"], state - II / 4,
                                   atol=1e-12)

    def test_absolute_scale_and_projection_are_explicit(self):
        state = bell_target("Psi-")
        delta = state - II / 4
        with self.assertRaisesRegex(ValueError, "independent positive absolute"):
            reconstruct_full_density(delta, 0)
        exact = reconstruct_full_density(delta, 1)
        np.testing.assert_allclose(exact["rho_physical"], state)
        impossible = reconstruct_full_density(delta, 1.2, project_physical=True)
        self.assertLess(impossible["minimum_eigenvalue_unprojected"], 0)
        self.assertGreater(impossible["projection_frobenius"], 0)
        self.assertGreaterEqual(
            np.min(np.linalg.eigvalsh(impossible["rho_physical"])), -1e-12
        )

    def test_bell_scorer_does_not_accept_nonphysical_deviation(self):
        zero = np.diag([1, 0, 0, 0]).astype(complex)
        for label in ("Phi+", "Phi-", "Psi+", "Psi-"):
            unitary = ideal_gate_sequence(bell_preparation_gates(label))
            np.testing.assert_allclose(unitary @ zero @ unitary.conj().T,
                                       bell_target(label), atol=1e-12)
        score = bell_score(bell_target("Phi+"), "Phi+")
        self.assertAlmostEqual(score["state_fidelity"], 1)
        self.assertAlmostEqual(score["pauli_coefficients"]["XX"], 1)
        self.assertAlmostEqual(score["pauli_coefficients"]["YY"], -1)
        self.assertAlmostEqual(score["pauli_coefficients"]["ZZ"], 1)
        with self.assertRaisesRegex(ValueError, "trace one"):
            bell_score(pps_target_deviation(), "Phi+")
        with self.assertRaisesRegex(ValueError, "positive state"):
            bell_score(II / 4 + 1.2 * (bell_target("Phi+") - II / 4), "Phi+")


if __name__ == "__main__":
    unittest.main()
