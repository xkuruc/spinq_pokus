"""Bounded *physical* C/G comparisons over one raw-FID acquisition callback.

This module does not own a connection.  ``acquire(key, SequenceIR)`` is the
single serialized hardware worker supplied by the Windows runner.  All
reported errors are of independently calibrated FID observables, never a
measured quantum process fidelity inferred from a lone NMR signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np

from .cde import (EnsemblePoint, PulseProgram, SpinHamiltonian, bb1_pulse,
                  optimize_grape, quantize_program, rectangular_pulse, rotation)
from .core import (Capabilities, CapabilityUnavailable, RawFIDRecord, Segment,
                   SequenceIR, compile_sequence)
from .signal import MultipletSpec, NoiseModel, fit_complex_multiplet


@dataclass(frozen=True)
class ObservableReading:
    coefficient: complex
    uncertainty: float
    method: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.coefficient) or not np.isfinite(self.uncertainty):
            raise ValueError("readout coefficient and uncertainty must be finite")
        if self.uncertainty < 0 or not self.method:
            raise ValueError("nonnegative uncertainty and local method required")


def fixed_multiplet_observable(spec: MultipletSpec, component_index: int, *,
                              noise: NoiseModel | None = None,
                              coefficient_uncertainty: float) -> Callable[[RawFIDRecord], ObservableReading]:
    """Freeze pilot component identity; vendor FFT/fit is never consulted."""
    if component_index < 0 or coefficient_uncertainty <= 0:
        raise ValueError("pilot component index and independent coefficient uncertainty required")

    def extract(record: RawFIDRecord) -> ObservableReading:
        result = fit_complex_multiplet(record, spec, noise)
        if result["status"] != "FIT_COMPLETED" or component_index >= len(result["modes"]):
            raise ValueError("FID multiplet fit failed frozen pilot identity check")
        mode = result["modes"][component_index]
        return ObservableReading(complex(mode["coefficient_re"], mode["coefficient_im"]),
                                 coefficient_uncertainty,
                                 "local complex multiplet fit; frozen pilot bands")

    return extract


@dataclass(frozen=True)
class ReadoutTask:
    """One independently calibrated input/readout context for the target gate."""

    name: str
    reference_program: PulseProgram
    reference_provenance: str
    reference_uncertainty: float
    tolerance: float
    max_reference_drift: float
    preparation: PulseProgram | None = None
    readout: PulseProgram | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.reference_provenance:
            raise ValueError("named reference with provenance required")
        if min(self.reference_uncertainty, self.tolerance, self.max_reference_drift) <= 0:
            raise ValueError("frozen positive reference uncertainty/tolerance/drift cap required")
        if self.reference_uncertainty >= self.tolerance:
            raise ValueError("reference uncertainty is too large for the target tolerance")


@dataclass(frozen=True)
class HardwareCondition:
    """A heldout *commanded* perturbation, not assumed equal to model detuning."""

    name: str
    rf_command_scale: float = 1.0
    frequency_command_shift_hz: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not np.all(np.isfinite((self.rf_command_scale,
                                                  self.frequency_command_shift_hz))):
            raise ValueError("named finite heldout command required")
        if self.rf_command_scale <= 0:
            raise ValueError("RF command scale must be positive")


def _safe_label(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    if not safe or len(safe) > 50:
        raise ValueError("unsafe or oversized record key component")
    return safe


def compose_sequence(task: ReadoutTask, candidate: PulseProgram,
                     condition: HardwareCondition, *, label: str,
                     sample_count: int = 16000, sample_hz: int = 10000) -> SequenceIR:
    """Apply perturbation to gate only, keeping preparation/readout calibrated."""
    segments: list[Segment] = []
    cursor_us = 0.0
    for part, perturbed in ((task.preparation, False), (candidate, True),
                            (task.readout, False)):
        if part is None:
            continue
        for dt_s, amplitude, phase in zip(part.duration_s, part.amplitude_percent,
                                           part.phase_rad):
            amplitude_pct = amplitude * (condition.rf_command_scale if perturbed else 1)
            shift = condition.frequency_command_shift_hz if perturbed else 0.0
            segments.append(Segment(start_us=cursor_us, duration_us=dt_s * 1e6,
                                    channel="H", amplitude_pct=amplitude_pct,
                                    phase_deg=float(np.rad2deg(phase) % 360),
                                    detuning_hz=shift))
            cursor_us += dt_s * 1e6
    return SequenceIR(tuple(segments), sample_count=sample_count,
                      sample_hz=sample_hz, sample_path=0,
                      state_initialization=True, label=label)


def _bootstrap_paired_blocks(rows: list[dict], baseline_name: str,
                             *, seed: int = 3, draws: int = 1000) -> dict:
    """Bootstrap whole blocks; FID points never masquerade as repetitions."""
    by_block: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        if row["reference_status"] != "VALID":
            continue
        by_block.setdefault(row["block"], {}).setdefault(row["method"], []).append(row["absolute_error"])
    methods = sorted({row["method"] for row in rows})
    result = {}
    rng = np.random.default_rng(seed)
    for method in methods:
        if method == baseline_name:
            continue
        pairs = []
        for block, values in by_block.items():
            if method in values and baseline_name in values:
                pairs.append(float(np.mean(values[baseline_name]) - np.mean(values[method])))
        if len(pairs) < 3:
            result[method] = {"paired_blocks": len(pairs), "status": "INSUFFICIENT_BLOCKS"}
            continue
        data = np.asarray(pairs)
        draws_index = rng.integers(0, len(data), size=(draws, len(data)))
        boot = np.mean(data[draws_index], axis=1)
        result[method] = {"paired_blocks": len(pairs),
                          "mean_error_reduction": float(np.mean(data)),
                          "bootstrap_ci95": [float(v) for v in np.quantile(boot, [.025, .975])],
                          "status": "EXPLORATORY_PAIRED_BLOCK_CI"}
    return result


def run_paired_control_comparison(
    acquire: Callable[[str, SequenceIR], RawFIDRecord],
    programs: Mapping[str, PulseProgram],
    tasks: Sequence[ReadoutTask], conditions: Sequence[HardwareCondition],
    observable: Callable[[RawFIDRecord], ObservableReading],
    *,
    capabilities: Capabilities,
    blocks: int,
    max_acquisitions: int,
    baseline_name: str,
    seed: int = 42,
    sample_count: int = 16000,
    key_prefix: str = "ctrl",
) -> dict:
    """Measure all methods with one service and bracket each block by a reference.

    Planning/compilation happens before the first RF task.  A failure in the
    acquisition callback stops the run; the owner handles lock, Ctrl+C,
    cooldown and resumptions.  Internal instrument repetitions are UNKNOWN.
    """
    if blocks < 3 or max_acquisitions < 1 or not programs or not tasks or not conditions:
        raise ValueError("three blocks, finite budget, candidates, tasks and conditions required")
    if baseline_name not in programs:
        raise ValueError("declared classical baseline absent")
    key_prefix=_safe_label(key_prefix)
    if len(set(programs)) != len(programs) or len({t.name for t in tasks}) != len(tasks):
        raise ValueError("candidate and task names must be unique")
    if len({c.name for c in conditions}) != len(conditions):
        raise ValueError("heldout condition names must be unique")
    for name in programs:
        _safe_label(name)
    for task in tasks:
        _safe_label(task.name)
    for condition in conditions:
        _safe_label(condition.name)
    neutral = HardwareCondition("nominal")
    precompiled: dict[tuple[str, str, str], SequenceIR] = {}
    skipped: dict[str, str] = {}
    for task in tasks:
        ref = compose_sequence(task, task.reference_program, neutral,
                               label=f"reference_{task.name}", sample_count=sample_count)
        compile_sequence(ref, capabilities)
        precompiled[(task.name, "reference", "nominal")] = ref
    for name, program in programs.items():
        try:
            for task in tasks:
                for condition in conditions:
                    seq = compose_sequence(task, program, condition,
                                           label=f"{name}_{task.name}_{condition.name}",
                                           sample_count=sample_count)
                    compile_sequence(seq, capabilities)
                    precompiled[(task.name, name, condition.name)] = seq
        except (CapabilityUnavailable, ValueError) as exc:
            skipped[name] = f"UNVERIFIED_OR_OUTSIDE_TESTED_ENVELOPE: {exc}"
            for key in list(precompiled):
                if key[1] == name:
                    del precompiled[key]
    active = [name for name in programs if name not in skipped]
    if baseline_name not in active or len(active) < 2:
        return {"status": "DEPENDENCY_FAILED", "reason": "fewer than two feasible physical methods",
                "skipped": skipped, "rows": [], "planned_acquisitions": 0,
                "physical_acquisitions_requested": 0}
    per_task_block = 2 + len(active) * len(conditions)
    planned = blocks * len(tasks) * per_task_block
    if planned > max_acquisitions:
        return {"status": "BUDGET_EXHAUSTED", "planned_acquisitions": planned,
                "max_acquisitions": max_acquisitions, "rows": [],
                "skipped": skipped,
                "physical_acquisitions_requested": 0}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    requested = 0
    total_reference_wall_s = 0.0

    def measure(key: str, seq: SequenceIR) -> tuple[ObservableReading, float, float]:
        start = perf_counter()
        record = acquire(key, seq)
        acquisition_seconds = perf_counter() - start
        start = perf_counter()
        reading = observable(record)
        analysis_seconds = perf_counter() - start
        return reading, acquisition_seconds, analysis_seconds

    for block in range(blocks):
        task_order = list(tasks)
        rng.shuffle(task_order)
        for task in task_order:
            ref_seq = precompiled[(task.name, "reference", "nominal")]
            key_before = f"{key_prefix}_b{block:03d}_{_safe_label(task.name)}_ref_before"
            before, before_wall, _ = measure(key_before, ref_seq)
            requested += 1
            candidates = [(name, condition) for name in active for condition in conditions]
            rng.shuffle(candidates)
            observations = []
            for position, (name, condition) in enumerate(candidates, start=1):
                seq = precompiled[(task.name, name, condition.name)]
                key = (f"{key_prefix}_b{block:03d}_{_safe_label(task.name)}_"
                       f"{_safe_label(name)}_{_safe_label(condition.name)}")
                observation, acquisition_seconds, analysis_seconds = measure(key, seq)
                requested += 1
                observations.append((position, name, condition, key, observation, seq,
                                     acquisition_seconds, analysis_seconds))
            key_after = f"{key_prefix}_b{block:03d}_{_safe_label(task.name)}_ref_after"
            after, after_wall, _ = measure(key_after, ref_seq)
            requested += 1
            total_reference_wall_s += before_wall + after_wall
            drift = abs(after.coefficient - before.coefficient)
            ref_status = ("VALID" if drift <= task.max_reference_drift else
                          "REFERENCE_DRIFT_EXCEEDED")
            for (position, name, condition, key, observation, seq,
                 acquisition_seconds, analysis_seconds) in observations:
                alpha = position / (len(candidates) + 1)
                reference = before.coefficient * (1 - alpha) + after.coefficient * alpha
                error = abs(observation.coefficient - reference)
                uncertainty = float(np.hypot(observation.uncertainty,
                                              task.reference_uncertainty))
                rows.append({"block": block, "task": task.name, "method": name,
                             "condition": condition.name, "record_key": key,
                             "reference_before_key": key_before,
                             "reference_after_key": key_after,
                             "reference_source": task.reference_provenance,
                             "observable_method": observation.method,
                             "observable_re": float(observation.coefficient.real),
                             "observable_im": float(observation.coefficient.imag),
                             "reference_re": float(reference.real),
                             "reference_im": float(reference.imag),
                             "reference_drift": float(drift),
                             "reference_status": ref_status,
                             "absolute_error": float(error),
                             "physical_task_wall_s": float(acquisition_seconds),
                             "local_analysis_s": float(analysis_seconds),
                             "combined_uncertainty": uncertainty,
                             "tolerance": task.tolerance,
                             "within_tolerance": bool(error + 1.96 * uncertainty <= task.tolerance
                                                      and ref_status == "VALID"),
                             "rf_duration_us": float(sum(s.duration_us for s in seq.segments)),
                             "rf_integral_percent_us": float(sum(
                                 s.duration_us * s.amplitude_pct for s in seq.segments)),
                             "hardware_gate_fidelity_inferred": False,
                             "internal_excitations": "UNKNOWN"})
    paired = _bootstrap_paired_blocks(rows, baseline_name, seed=seed)
    valid = [row for row in rows if row["reference_status"] == "VALID"]
    improved = [name for name, result in paired.items()
                if result.get("bootstrap_ci95", [float("-inf")])[0] > 0
                and all(row["within_tolerance"] for row in valid if row["method"] == name)]
    status = ("REFERENCE_INADEQUATE" if len(valid) != len(rows) else
              "FEASIBILITY_PILOT" if blocks < 10 else
              "SUCCESS_VALIDATED" if improved else "VALID_NEGATIVE_RESULT")
    return {"status": status, "claim_scope": "heldout calibrated complex FID observable",
            "gate_process_fidelity_measured": False,
            "methods_with_positive_paired_ci_and_tolerance": improved,
            "evidence_strength": "feasibility_only" if blocks < 10 else "main_block_comparison",
            "planned_acquisitions": planned, "physical_acquisitions_requested": requested,
            "reference_task_wall_s": total_reference_wall_s,
            "internal_excitations": "UNKNOWN", "rows": rows, "skipped": skipped,
            "paired_block_comparisons": paired}


def run_c_physical(
    acquire: Callable[[str, SequenceIR], RawFIDRecord],
    model: SpinHamiltonian,
    tasks: Sequence[ReadoutTask], conditions: Sequence[HardwareCondition],
    observable: Callable[[RawFIDRecord], ObservableReading],
    *, design_ensemble: Sequence[EnsemblePoint], capabilities: Capabilities,
    amplitude_percent: float, n_segments: int, duration_s: float,
    verified_tick_s: float, blocks: int, max_acquisitions: int,
    max_grape_iterations: int = 80, seed: int = 42,
) -> dict:
    """H-only X90: calibrated rectangle, BB1, phase-only GRAPE on real FIDs."""
    if model.n_spins != 1 or model.channel != "H":
        raise ValueError("C physical runner currently requires calibrated H-only 1q model")
    target = rotation((1, 0, 0), pi / 2)
    rectangle = quantize_program(rectangular_pulse(model, pi / 2, amplitude_percent),
                                  verified_tick_s)
    bb1 = quantize_program(bb1_pulse(model, pi / 2, amplitude_percent),
                           verified_tick_s)
    grape = optimize_grape(model, target, duration_s=duration_s, n_segments=n_segments,
                           amplitude_percent=amplitude_percent,
                           ensemble=design_ensemble, maxiter=max_grape_iterations,
                           verified_tick_s=verified_tick_s)
    programs = {"rectangle": rectangle, "BB1": bb1, "phase_GRAPE": grape.program}
    measured = run_paired_control_comparison(acquire, programs, tasks, conditions,
                                              observable, capabilities=capabilities,
                                              blocks=blocks, max_acquisitions=max_acquisitions,
                                              baseline_name="rectangle", seed=seed,key_prefix="C")
    measured["design"] = {"model_scope": "calibrated H-only single-spin",
                           "simulated_grape_design_infidelity": grape.design_infidelity,
                           "simulated_grape_initial_infidelity": grape.initial_infidelity,
                           "grape_design_seconds": grape.elapsed_s,
                           "grape_iterations": grape.iterations,
                           "programs": {name: program.serialized()
                                        for name, program in programs.items()}}
    return measured


def run_g_physical(
    acquire: Callable[[str, SequenceIR], RawFIDRecord],
    ablation_programs: Mapping[str, PulseProgram],
    tasks: Sequence[ReadoutTask], conditions: Sequence[HardwareCondition],
    observable: Callable[[RawFIDRecord], ObservableReading],
    *, capabilities: Capabilities, blocks: int, max_acquisitions: int,
    seed: int = 42,
) -> dict:
    """Measure externally compiled G ablations with a common FID readout.

    G1/G2/G3 pulse variants must be genuine low-level programs supplied by the
    caller. G4 postprocessing must be evaluated on these saved *same* raw FIDs,
    separately from physical control error. 2q claims require their own
    verified model/compiler and are not produced by this H-only runner.
    """
    if "classical_expert" not in ablation_programs or "basic" not in ablation_programs:
        raise ValueError("G needs basic and classical_expert physical baselines")
    measured = run_paired_control_comparison(acquire, ablation_programs, tasks,
                                              conditions, observable,
                                              capabilities=capabilities, blocks=blocks,
                                              max_acquisitions=max_acquisitions,
                                              baseline_name="classical_expert", seed=seed,key_prefix="G")
    measured["ablation_scope"] = "physical H-only pulse comparison before G4 readout correction"
    measured["two_qubit_verified"] = False
    return measured
