"""Numerical C/D/E contracts; these are synthetic, never hardware claims."""

import math
import unittest

import numpy as np
from scipy.linalg import expm

from spinq_local.cde import (
    BlackboxResult, EnsemblePoint, FIDExample, GateTarget, PulseLabel,
    PulseProgram, ReadoutPhysics, bb1_pulse, break_even_uses,
    canonical_gate_features, compare_generated_pulses, coupled_spin_model,
    embed_single_spin_gate, ensemble_infidelity, evaluate_fid_predictors,
    fit_blackbox, fit_graybox, fit_polynomial_rf, generate_grape_labels,
    grape_loss_gradient, heldout_grid_metrics, infer_generated_pulse,
    optimize_grape, optimize_graybox_gate, physical_gate_distance, predict_fid_physics,
    predict_unitary_graybox,
    process_overlap, quantize_program, rectangular_pulse, rotation, sample_gate_targets,
    single_spin_model, spin_operator, split_examples_by_group,
    train_pulse_generator,
)


def _synthetic_fids():
    model = single_spin_model(350, 20000)
    readout = ReadoutPhysics(model.drift_hz, 2 + 0.4j, 0.01 - 0.02j, 20)
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    detector = np.array([[0, 2], [0, 0]], dtype=complex)
    times = np.arange(16) * 1e-4
    records = []
    for j in range(6):
        program = PulseProgram((7e-6, 8e-6), (.7 + .04 * j, .8),
                               (.1 + .25 * j, -.2 + .1 * j))
        empty = FIDExample(program, rho, detector, times, np.zeros(len(times), complex),
                           f"family_{j // 2}", f"session_{j // 2}", f"record_{j}")
        fid = predict_fid_physics(empty, model, readout)
        records.append(FIDExample(program, rho, detector, times, fid,
                                  empty.family, empty.session, empty.record_id))
    return model, readout, records


class GRAPETests(unittest.TestCase):
    def test_spin_units_and_bb1(self):
        model = single_spin_model(0, 25000)
        target = rotation((1, 0, 0), math.pi / 2)
        rectangular = rectangular_pulse(model, math.pi / 2, 1)
        bb1 = bb1_pulse(model, math.pi / 2, 1)
        self.assertAlmostEqual(rectangular.duration_s[0], 10e-6)
        self.assertAlmostEqual(process_overlap(model, target, rectangular), 1, places=12)
        self.assertAlmostEqual(process_overlap(model, target, bb1), 1, places=12)
        self.assertGreater(bb1.rf_integral_percent_s, rectangular.rf_integral_percent_s)
        error_rect = 1 - process_overlap(model, target, rectangular, EnsemblePoint(0, 1.15))
        error_bb1 = 1 - process_overlap(model, target, bb1, EnsemblePoint(0, 1.15))
        self.assertLess(error_bb1, error_rect)
        self.assertEqual(rectangular.serialized()[0]["phase_deg"], 0)
        snapped = quantize_program(PulseProgram((1.49e-6, 2.51e-6), (1, 1), (0, 0)),
                                   1e-6)
        np.testing.assert_allclose(snapped.duration_s, (1e-6, 3e-6))
        with self.assertRaises(ValueError):
            quantize_program(PulseProgram((.3e-6,), (1,), (0,)), 1e-6)

    def test_two_spin_model_requires_measured_pair_and_addressing(self):
        with self.assertRaises(ValueError):
            coupled_spin_model([0, 200], {}, 0, [1, 0], 25000,
                               weak_coupling_verified=True)
        with self.assertRaises(ValueError):
            coupled_spin_model([0, 200], {(0, 1): 80}, 0, [1, 0], 25000,
                               weak_coupling_verified=False)
        model = coupled_spin_model([0, 200], {(0, 1): 80}, 0, [1, 0], 25000,
                                   weak_coupling_verified=True)
        self.assertEqual(model.dimension, 4)
        self.assertAlmostEqual(float(np.trace(model.bx @ model.bx).real), 1)
        with self.assertRaises(ValueError):
            process_overlap(model, rotation((1, 0, 0), math.pi / 2),
                            PulseProgram((1e-5,), (1,), (0,)))

    def test_phase_and_amplitude_gradients_match_finite_difference(self):
        model = coupled_spin_model([150, 400], {(0, 1): 70}, 0, [1, .15], 23000,
                                   weak_coupling_verified=True)
        target = embed_single_spin_gate(rotation((1, 0, 0), math.pi / 2), 0, 2)
        ensemble = [EnsemblePoint(-150, .95, 2), EnsemblePoint(120, 1.06, 1)]
        x = np.array([.25, -.44, .73, .9, 1.1, .8])
        for amplitude in (False, True):
            args = dict(optimize_amplitude=amplitude,
                        amplitude_limit_percent=1.5 if amplitude else None)
            v = x if amplitude else x[:3]
            loss, gradient = grape_loss_gradient(v, model, target, 18e-6, 3, 1,
                                                  ensemble, **args)
            numerical = []
            for i in range(len(v)):
                xp, xm = v.copy(), v.copy()
                xp[i] += 1e-6
                xm[i] -= 1e-6
                fp = grape_loss_gradient(xp, model, target, 18e-6, 3, 1,
                                         ensemble, **args)[0]
                fm = grape_loss_gradient(xm, model, target, 18e-6, 3, 1,
                                         ensemble, **args)[0]
                numerical.append((fp - fm) / 2e-6)
            self.assertLess(max(abs(gradient - numerical)), 2e-6)
            self.assertTrue(math.isfinite(loss))

    def test_grape_optimizes_and_heldout_grid_remains_separate(self):
        model = single_spin_model(400, 25000)
        target = rotation((1, 0, 0), math.pi / 2)
        design = [EnsemblePoint(-200, .95), EnsemblePoint(200, 1.05)]
        fit = optimize_grape(model, target, duration_s=12e-6, n_segments=3,
                             amplitude_percent=1, ensemble=design,
                             initial_phases=[.6, -.4, .2], maxiter=25,
                             check_gradient=True)
        self.assertLess(fit.design_infidelity, fit.initial_infidelity)
        self.assertLess(fit.gradient_max_error, 1e-4)
        heldout = [EnsemblePoint(-100, .98), EnsemblePoint(100, 1.02)]
        metrics = heldout_grid_metrics(model, target, {"grape": fit.program}, heldout)
        self.assertEqual(len(metrics["grape"]["simulated_grid_errors"]), 2)
        self.assertAlmostEqual(metrics["grape"]["duration_s"], 12e-6)


class GrayboxTests(unittest.TestCase):
    def test_exported_fid_physics_matches_direct_propagation(self):
        model, readout, records = _synthetic_fids()
        record = records[0]
        total = np.eye(2, dtype=complex)
        for dt, amp, phase in zip(record.program.duration_s,
                                  record.program.amplitude_percent,
                                  record.program.phase_rad):
            h = (model.drift_hz + amp * model.rf_hz_per_percent
                 * (np.cos(phase) * model.bx + np.sin(phase) * model.by))
            total = expm(-2j * math.pi * h * dt) @ total
        rho = total @ record.initial_state @ total.conj().T
        direct = []
        for t in record.times_s:
            u = expm(-2j * math.pi * readout.acq_hamiltonian_hz * t)
            direct.append(readout.complex_gain * np.exp(-readout.decay_s_inv * t)
                          * np.trace(record.detector @ u @ rho @ u.conj().T)
                          + readout.background)
        np.testing.assert_allclose(predict_fid_physics(record, model, readout),
                                   direct, atol=1e-12)

    def test_group_split_and_control_fit(self):
        model, readout, records = _synthetic_fids()
        train, test = split_examples_by_group(records, heldout_families=["family_2"])
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 2)
        with self.assertRaises(ValueError):
            split_examples_by_group(records)
        poly = fit_polynomial_rf(train, model, readout,
                                 amplitude_scale_percent=1,
                                 max_relative_rf=.2, max_phase_rad=.3,
                                 max_detuning_hz=200, max_nfev=5, max_points=8)
        self.assertLess(poly.training_loss, 1e-15)
        self.assertTrue(set(poly.train_record_ids).isdisjoint({e.record_id for e in test}))

    def test_torch_graybox_and_blackbox_are_real_trainable_models(self):
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("CPU PyTorch unavailable")
        model, readout, records = _synthetic_fids()
        train, test = split_examples_by_group(records, heldout_sessions=["session_2"])
        poly = fit_polynomial_rf(train, model, readout, amplitude_scale_percent=1,
                                 max_relative_rf=.2, max_phase_rad=.3,
                                 max_detuning_hz=200, max_nfev=5, max_points=8)
        gray = fit_graybox(train, model, readout, amplitude_scale_percent=1,
                           duration_scale_s=10e-6, correction_caps=(.2, .3, 200),
                           epochs=3, max_points=8)
        black = fit_blackbox(train, max_slots=2, amplitude_scale_percent=1,
                             duration_scale_s=10e-6, time_scale_s=.002,
                             epochs=3, max_points=8)
        self.assertEqual(len(gray.history), 3)
        self.assertEqual(len(black.history), 3)
        scores = evaluate_fid_predictors(test, model, readout, poly, gray, black,
                                          max_points=8)
        self.assertEqual(set(scores), {"classical", "polynomial_rf", "blackbox_mlp", "graybox"})
        self.assertLess(scores["classical"]["heldout_mean_record_mse"], 1e-20)
        self.assertTrue(math.isfinite(scores["graybox"]["heldout_mean_record_mse"]))
        unitary = predict_unitary_graybox(test[0].program, gray)
        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(2), atol=1e-10)
        target = rotation((1, 0, 0), math.pi / 2)
        designed = optimize_graybox_gate(gray, target, test[0].program, maxiter=3)
        self.assertEqual(designed["hardware_validation_status"],
                         "PENDING_INDEPENDENT_MEASUREMENT")
        self.assertEqual(len(designed["program"].phase_rad), 2)
        with self.assertRaises(ValueError):
            evaluate_fid_predictors(train[:1], model, readout, poly, gray, black)


class GeneratorTests(unittest.TestCase):
    def test_gate_features_ignore_global_phase_and_split_is_disjoint(self):
        gate = rotation((0, 1, 0), math.pi / 3)
        np.testing.assert_allclose(canonical_gate_features(gate),
                                   canonical_gate_features(np.exp(.76j) * gate))
        self.assertLess(physical_gate_distance(gate, np.exp(.76j) * gate), 1e-12)
        targets = sample_gate_targets(4, 2, 2, seed=3)
        self.assertEqual([t.split for t in targets].count("test"), 2)
        self.assertGreater(min(physical_gate_distance(a.gate2, b.gate2)
                               for i, a in enumerate(targets) for b in targets[i + 1:]), 1e-5)

    def test_common_start_labels_and_generator_baselines(self):
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("CPU PyTorch unavailable")
        model = single_spin_model(0, 25000)
        target = GateTarget("x90", rotation((1, 0, 0), math.pi / 2), (1, 0, 0),
                            math.pi / 2, "train")
        generated = generate_grape_labels([target], model, target_spin=0,
                                           duration_s=10e-6, amplitude_percent=1,
                                           common_start_phases_rad=(0, 0),
                                           ensemble=[EnsemblePoint()], maxiter=3,
                                           accept_infidelity=.05)
        self.assertEqual(len(generated.labels), 1)
        self.assertEqual(generated.attempts, 1)

        def label(gate_id, axis, phase, split):
            gate = GateTarget(gate_id, rotation(axis, math.pi / 2), tuple(axis),
                              math.pi / 2, split)
            pulse = PulseProgram((5e-6, 5e-6), (1, 1), (phase, phase))
            return PulseLabel(gate, pulse, 0.0, 0.1, 2)

        train = [label("x", (1, 0, 0), 0, "train"),
                 label("y", (0, 1, 0), math.pi / 2, "train")]
        validation = [label("diag", (1 / math.sqrt(2), 1 / math.sqrt(2), 0),
                            math.pi / 4, "validation")]
        generator = train_pulse_generator(train, validation, hidden=8, epochs=3,
                                          learning_rate=.01)
        self.assertEqual(len(generator.history), 3)
        test = GateTarget("heldout", rotation((.8, .6, 0), math.pi / 2),
                          (.8, .6, 0), math.pi / 2, "test")
        predicted, elapsed = infer_generated_pulse(test, generator)
        self.assertEqual(len(predicted.phase_rad), 2)
        self.assertGreaterEqual(elapsed, 0)
        comparison = compare_generated_pulses([test], train, generator, model,
                                                target_spin=0, ensemble=[EnsemblePoint()],
                                                heldout_ensemble=[EnsemblePoint(100, 1.03)],
                                                common_start_phases_rad=(0, 0),
                                                full_grape_iterations=3,
                                                refine_iterations=2)
        self.assertEqual(len(comparison), 5)
        self.assertEqual({row["method"] for row in comparison},
                         {"fresh_grape", "nearest_library", "warm_start_grape",
                          "network", "network_refined"})
        self.assertTrue(all(row["hardware_validation_status"] ==
                            "PENDING_INDEPENDENT_MEASUREMENT" for row in comparison))

    def test_generator_rejects_leaked_equivalent_gate(self):
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("CPU PyTorch unavailable")
        gate = rotation((1, 0, 0), math.pi / 2)
        program = PulseProgram((1e-5,), (1,), (0,))
        train = PulseLabel(GateTarget("train", gate, (1, 0, 0), math.pi / 2, "train"),
                           program, 0, 0, 0)
        val = PulseLabel(GateTarget("val", np.exp(.7j) * gate, (1, 0, 0),
                                    math.pi / 2, "validation"), program, 0, 0, 0)
        with self.assertRaises(ValueError):
            train_pulse_generator([train], [val], epochs=1)
        self.assertEqual(break_even_uses(100, 10, 5, equivalent_quality=True), 20)
        self.assertIsNone(break_even_uses(100, 10, 5, equivalent_quality=False))


if __name__ == "__main__":
    unittest.main()
