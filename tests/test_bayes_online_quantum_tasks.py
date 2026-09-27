"""Numerical task checks use synthetic callbacks, never claimed device results."""

import unittest

import numpy as np

from bayes_online_core.physics import (
    II, PAULI_LABELS, PAULIS, bell_preparation_gates,
    default_readout_unitaries, ideal_gate_sequence, minimum_readout_unitaries,
    temporal_pps_branches, tomography_design,
)
from bayes_online_core.quantum_tasks import run_bell, run_coupling, run_pps
from bayes_online_core.transport import AcquisitionResult, _schedule


def anchors():
    return {
        "h90_width_us": 40, "p90_width_us": 42,
        "h90_amplitude_pct": 10, "p90_amplitude_pct": 10,
        "sample_count": 4000, "sample_frequency_hz": 10000,
        "relaxation_delay_s": 15, "coherent_window_s": 0.008,
        "coupling_delays_us": (0, 4000, 8000, 12000, 16000, 20000, 24000, 28000),
        "coupling_center_bands_hz": {"H": (300, 350), "P": (-150, -100)},
        "coupling_j_search_hz": (20, 100), "j_sign": 1,
    }


def synthetic_result(request, fid):
    t = np.arange(request.sample_count) / request.sample_frequency_hz
    return AcquisitionResult(
        key=request.key, task_id=request.key, status="COMPLETED", time_s=t,
        fid_complex=fid, vendor_reference={}, requested_payload={}, sent_payload={},
        timing_evidence={}, elapsed_s=1, device_snapshot={},
    )


class QuantumTaskTests(unittest.TestCase):
    def test_unverified_primitives_submit_no_tasks(self):
        called = []
        def acquire(request):
            called.append(request)
            raise AssertionError("should not acquire")
        self.assertEqual(run_coupling(acquire, anchors(), {}, "c")["status"],
                         "UNSUPPORTED_REQUIRED_PRIMITIVE")
        self.assertEqual(run_pps(acquire, anchors(), {}, "p")["status"],
                         "UNSUPPORTED_REQUIRED_PRIMITIVE")
        self.assertEqual(run_bell(acquire, anchors(), {}, "b")["status"],
                         "UNSUPPORTED_REQUIRED_PRIMITIVE")
        self.assertEqual(called, [])
        unqualified = run_coupling(
            acquire, anchors(),
            {"idle_gap_verified_by_channel": {"H": True, "P": True},
             "verified_idle_us_by_channel": {
                 "H": (500, 1000, 1500, 2000, 3000),
                 "P": (500, 1000, 1500, 2000, 3000),
             }}, "c")
        self.assertEqual(unqualified["status"], "UNSUPPORTED_REQUIRED_PRIMITIVE")
        self.assertEqual(called, [])

    def test_coupling_uses_completed_delay_fids_and_heldout(self):
        sent = []
        def acquire(request):
            sent.append(request)
            path = request.sample_path
            base = 300 + 25 if path == 0 else -150 + 25
            first = request.pulses[0]
            delay_s = (request.pulses[1].width_us / 1e6
                       if len(request.pulses) > 1 else 0.0)
            t = np.arange(request.sample_count) / request.sample_frequency_hz
            total_t = t + delay_s
            fid = (
                (1 + 0.2j) * np.exp((-120 + 2j*np.pi*(base-27.5))*total_t)
                + (0.7 - 0.1j) * np.exp((-120 + 2j*np.pi*(base+27.5))*total_t)
                + 0.01 + 0.02j
            )
            return synthetic_result(request, fid)
        result = run_coupling(acquire, anchors(),
                              {"idle_gap_verified": True,
                               "cross_channel_alignment_verified": True,
                               "j_sign_verified": True,
                               "verified_idle_us_by_channel": {
                                   "H": anchors()["coupling_delays_us"][1:],
                                   "P": anchors()["coupling_delays_us"][1:],
                               }}, "synthetic")
        self.assertEqual(result["status"], "TARGET_REACHED", result)
        self.assertEqual(result["acquisitions"], 16)
        self.assertAlmostEqual(result["effective_j_abs_hz"], 55, places=1)
        self.assertEqual(result["heldout_delay_us"], 28000)
        self.assertTrue(all(not request.initialize_state for request in sent))
        sent.clear()
        unsigned = run_coupling(acquire, anchors(),
                                {"idle_gap_verified_by_channel": {"H": True, "P": True},
                                 "verified_idle_us_by_channel": {
                                     "H": anchors()["coupling_delays_us"][1:],
                                     "P": anchors()["coupling_delays_us"][1:],
                                 }},
                                "unsigned")
        self.assertEqual(unsigned["status"], "TARGET_REACHED", unsigned)
        self.assertIsNone(unsigned["effective_j_hz"])
        self.assertFalse(unsigned["sign_validated"])
        self.assertEqual(len(sent), 16)
        self.assertTrue(all({p.path for p in request.pulses} == {request.sample_path}
                            for request in sent))
        self.assertTrue(all(len(request.pulses) == (1 if request.key.endswith("_00") else 2)
                            for request in sent))
        self.assertTrue(all(request.pulses[-1].width_us in anchors()["coupling_delays_us"]
                            for request in sent if len(request.pulses) == 2))
        for request in sent:
            segments, _ = _schedule(request)
            idle = [segment for segment in segments if segment["am"] == 0]
            expected = [] if len(request.pulses) == 1 else [request.pulses[-1].width_us]
            self.assertEqual([segment["width"] for segment in idle], expected)

    def test_bell_requires_frozen_readout_and_absolute_scale(self):
        a = anchors()
        a.update({"readout_unitaries": dict(default_readout_unitaries()),
                  "readout_gains": {"H": 1+0j, "P": 1+0j},
                  "line_frequencies_hz": {"H": (270, 380), "P": (-180, -60)},
                  "line_decay_per_s": {"H": 100, "P": 100},
                  "max_sequence_us": 1_000_000,
                  "max_rf_us_per_acquisition": 50_000})
        evidence = {"idle_gap_verified": True, "cross_channel_alignment_verified": True,
                    "cnot_verified": True, "line_resolved_readout_verified": True,
                    "frozen_readout_verified": True}
        coupling = {"status": "TARGET_REACHED", "effective_j_hz": 55}
        def acquire(_):
            raise AssertionError("missing scale must block before acquisition")
        self.assertEqual(run_bell(acquire, a, evidence, coupling=coupling)["status"],
                         "UNSUPPORTED_REQUIRED_PRIMITIVE")

    def test_qualified_temporal_pps_path_counts_every_branch_readout(self):
        a = anchors()
        settings = minimum_readout_unitaries()
        a.update({"readout_unitaries": dict(settings),
                  "readout_gains": {"H": 1+0j, "P": 1+0j},
                  "line_frequencies_hz": {"H": (270, 380), "P": (-180, -60)},
                  "line_decay_per_s": {"H": 100, "P": 100},
                  "max_fid_model_residual_rms": {"H": 1e-4, "P": 1e-4},
                  "max_sequence_us": 1_000_000,
                  "max_rf_us_per_acquisition": 50_000,
                  "pps_reference_signal_norm": 1})
        evidence = {"idle_gap_verified": True, "cross_channel_alignment_verified": True,
                    "cnot_verified": True, "line_resolved_readout_verified": True,
                    "frozen_readout_verified": True}
        coupling = {"status": "TARGET_REACHED", "effective_j_hz": 55}
        design = tomography_design(settings)
        thermal = np.diag([0.55, 0.25, 0.15, 0.05]).astype(complex)
        branch_lookup = {branch.name: branch for branch in temporal_pps_branches()}
        sent = []
        def acquire(request):
            sent.append(request)
            self.assertFalse(request.initialize_state)
            tokens = request.key.split("_")
            # Key has synthetic prefix, then branch name, setting index, channel.
            branch_name = "_".join(tokens[1:-2])
            setting_index = int(tokens[-2])
            channel = tokens[-1]
            branch = branch_lookup[branch_name]
            state = branch.unitary @ thermal @ branch.unitary.conj().T
            coefficients = np.array([np.trace(PAULIS[label] @ state).real
                                     for label in PAULI_LABELS])
            responses = design.complex_matrix @ coefficients
            start = 4 * setting_index + (0 if channel == "H" else 2)
            pair = responses[start:start+2]
            t = np.arange(request.sample_count) / request.sample_frequency_hz
            f0, f1 = a["line_frequencies_hz"][channel]
            fid = pair[0]*np.exp((-100+2j*np.pi*f0)*t) + \
                  pair[1]*np.exp((-100+2j*np.pi*f1)*t) + (0.02+0.01j)
            return synthetic_result(request, fid)
        result = run_pps(acquire, a, evidence, key_prefix="synthetic", coupling=coupling)
        self.assertEqual(result["status"], "EVALUATED", result.get("reason"))
        self.assertEqual(result["acquisitions"], 24)
        self.assertEqual(len(sent), 24)
        self.assertEqual(result["tomography_rank"], 15)
        self.assertAlmostEqual(result["tomography_condition"], 2.0)
        self.assertLess(result["shape_error"], 1e-9)
        self.assertEqual(result["preparation_kind"], "TEMPORAL_AVERAGED_EFFECTIVE_PPS")
        self.assertTrue(any("pps_cycle_forward" in p.role for request in sent
                            for p in request.pulses))

    def test_bell_tomography_scores_measured_mixed_state_not_ideal_target(self):
        a = anchors()
        settings = minimum_readout_unitaries()
        a.update({"readout_unitaries": dict(settings),
                  "readout_gains": {"H": 1+0j, "P": 1+0j},
                  "line_frequencies_hz": {"H": (270, 380), "P": (-180, -60)},
                  "line_decay_per_s": {"H": 100, "P": 100},
                  "max_fid_model_residual_rms": {"H": 1e-4, "P": 1e-4},
                  "max_sequence_us": 1_000_000,
                  "max_rf_us_per_acquisition": 50_000,
                  "polarization_scale": 1,
                  "max_projection_frobenius": 0.01})
        evidence = {"idle_gap_verified": True, "cross_channel_alignment_verified": True,
                    "cnot_verified": True, "line_resolved_readout_verified": True,
                    "frozen_readout_verified": True, "absolute_scale_verified": True}
        coupling = {"status": "TARGET_REACHED", "effective_j_hz": 55}
        design = tomography_design(settings)
        thermal = np.diag([0.55, 0.25, 0.15, 0.05]).astype(complex)
        branches = {branch.name: branch for branch in temporal_pps_branches()}
        label_lookup = {"Phiplus": "Phi+", "Phiminus": "Phi-",
                        "Psiplus": "Psi+", "Psiminus": "Psi-"}
        def acquire(request):
            words = request.key.split("_")
            label = label_lookup[words[1]]
            branch = branches["_".join(words[2:-2])]
            setting_index = int(words[-2])
            channel = words[-1]
            bell_u = ideal_gate_sequence(bell_preparation_gates(label))
            prepared = bell_u @ branch.unitary @ thermal @ branch.unitary.conj().T @ bell_u.conj().T
            pauli = np.array([np.trace(PAULIS[p] @ prepared).real for p in PAULI_LABELS])
            values = design.complex_matrix @ pauli
            index = 4 * setting_index + (0 if channel == "H" else 2)
            pair = values[index:index + 2]
            t = np.arange(request.sample_count) / request.sample_frequency_hz
            f0, f1 = a["line_frequencies_hz"][channel]
            fid = pair[0]*np.exp((-100+2j*np.pi*f0)*t) + \
                  pair[1]*np.exp((-100+2j*np.pi*f1)*t) + 0.02
            return synthetic_result(request, fid)
        result = run_bell(acquire, a, evidence, key_prefix="synthetic",
                          coupling=coupling)
        self.assertEqual(result["status"], "EVALUATED", result)
        self.assertEqual(result["acquisitions"], 96)
        for row in result["states"].values():
            self.assertEqual(row["status"], "EVALUATED", row)
            self.assertAlmostEqual(row["state_fidelity"], 0.55, places=8)


if __name__ == "__main__":
    unittest.main()
