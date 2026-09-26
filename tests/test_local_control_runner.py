"""Synthetic acquisition-callback tests; no device connection."""

import unittest

import numpy as np

from spinq_local.cde import EnsemblePoint, PulseProgram, single_spin_model
from spinq_local.control_runner import (
    HardwareCondition, ObservableReading, ReadoutTask, run_c_physical,
    run_g_physical, run_paired_control_comparison,
)
from spinq_local.core import Capabilities, RawFIDRecord


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.reference = PulseProgram((10e-6,), (100,), (0,))
        self.task = ReadoutTask("z_to_transverse", self.reference,
                                "independent reference control, synthetic test",
                                reference_uncertainty=.01, tolerance=.5,
                                max_reference_drift=.2)
        self.cap = Capabilities(segment_sequence_verified=True)

    def acquire(self, key, sequence):
        self.calls.append(key)
        n = 64
        # A deterministic synthetic complex response of the compiled pulse.
        total = sum(s.amplitude_pct * s.duration_us for s in sequence.segments)
        signal = (total / 1000) + .1j
        return RawFIDRecord(key=key, task_id=key, group="0", path="0", qubit="0",
                            step="NMRSIG", axis_original=np.arange(n) * .1,
                            time_seconds=np.arange(n) / 10000,
                            re=np.full(n, signal.real), im=np.full(n, signal.imag),
                            parameters_sent={"sampleFre": 10000, "sampleCount": n})

    @staticmethod
    def observable(record):
        return ObservableReading(record.fid[0], .01, "local synthetic readout")

    def test_budget_is_checked_before_any_acquisition(self):
        programs = {"basic": self.reference,
                    "classical_expert": PulseProgram((20e-6,), (50,), (0,))}
        result = run_paired_control_comparison(self.acquire, programs, [self.task],
                                                [HardwareCondition("nominal")],
                                                self.observable, capabilities=self.cap,
                                                blocks=3, max_acquisitions=11,
                                                baseline_name="classical_expert")
        self.assertEqual(result["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(result["planned_acquisitions"], 12)
        self.assertEqual(self.calls, [])

    def test_c_runs_mixed_order_with_bracketed_references(self):
        model = single_spin_model(0, 250)
        result = run_c_physical(self.acquire, model, [self.task],
                                [HardwareCondition("nominal")], self.observable,
                                design_ensemble=[EnsemblePoint()],
                                capabilities=self.cap, amplitude_percent=100,
                                n_segments=2, duration_s=20e-6,
                                verified_tick_s=1e-6, blocks=3,
                                max_acquisitions=15, max_grape_iterations=3)
        self.assertEqual(result["planned_acquisitions"], 15)
        self.assertEqual(result["physical_acquisitions_requested"], 15)
        self.assertEqual(len(result["rows"]), 9)
        self.assertEqual({r["method"] for r in result["rows"]},
                         {"rectangle", "BB1", "phase_GRAPE"})
        self.assertTrue(all(not r["hardware_gate_fidelity_inferred"]
                            for r in result["rows"]))
        self.assertEqual(len(set(self.calls)), 15)

    def test_unfeasible_bb1_is_explicit_and_other_c_methods_continue(self):
        model = single_spin_model(0, 62.5)  # calibrated t90=40 us at 100 %
        reference = PulseProgram((40e-6,), (100,), (0,))
        task = ReadoutTask("readout", reference, "independent synthetic reference",
                           .01, .5, .2)
        result = run_c_physical(self.acquire, model, [task],
                                [HardwareCondition("nominal")], self.observable,
                                design_ensemble=[EnsemblePoint()],
                                capabilities=self.cap, amplitude_percent=100,
                                n_segments=4, duration_s=40e-6,
                                verified_tick_s=1e-6, blocks=3,
                                max_acquisitions=12, max_grape_iterations=3)
        self.assertIn("BB1", result["skipped"])
        self.assertEqual({r["method"] for r in result["rows"]},
                         {"rectangle", "phase_GRAPE"})
        self.assertEqual(len(self.calls), 12)

    def test_g_ablation_uses_same_raw_fid_service(self):
        programs = {"basic": self.reference,
                    "classical_expert": PulseProgram((20e-6,), (50,), (0,))}
        result = run_g_physical(self.acquire, programs, [self.task],
                                [HardwareCondition("nominal")], self.observable,
                                capabilities=self.cap, blocks=3, max_acquisitions=12)
        self.assertEqual(result["physical_acquisitions_requested"], 12)
        self.assertEqual(result["ablation_scope"],
                         "physical H-only pulse comparison before G4 readout correction")
        self.assertFalse(result["two_qubit_verified"])


if __name__ == "__main__":
    unittest.main()
