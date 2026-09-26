"""One low-level acquisition path and a lossless, metadata-aware FID record.

The physical connection is owned by spinq_benchmark.hardware.LiveHardware.
This module reads decoded SDK events before the SDK's simplified graph loses
task/group/path/qubit/step information. It never treats exported FID as ADC.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from spinq_audit.common import atomic_json, redact
from spinq_benchmark.hardware import LiveHardware, physical_request


class CapabilityUnavailable(RuntimeError):
    """A requested pulse sequence needs a physical capability not verified yet."""


class IncompleteFID(RuntimeError):
    """No interpolation or vendor-result fallback is permitted."""


@dataclass(frozen=True)
class Segment:
    start_us: float
    duration_us: float
    channel: str = "H"
    amplitude_pct: float = 100.0
    phase_deg: float = 90.0
    detuning_hz: float = 0.0


@dataclass(frozen=True)
class SequenceIR:
    segments: tuple[Segment, ...]
    sample_count: int = 16000
    sample_hz: int = 10000
    sample_path: int = 0
    state_initialization: bool = True
    label: str = ""


@dataclass(frozen=True)
class Capabilities:
    h_path_verified: bool = True
    p_path_verified: bool = False
    zero_amplitude_delay_verified: bool = False
    segment_sequence_verified: bool = False
    maximum_task_span_us: float = 200.0  # historical software envelope, not manufacturer rating
    maximum_segments: int = 8


@dataclass(frozen=True)
class ExperimentSpec:
    payload: dict[str, Any]
    sequence: SequenceIR
    rf_duration_us: float
    sequence_duration_us: float
    idle_probe: bool = False


@dataclass
class RawFIDRecord:
    key: str
    task_id: str
    group: str
    path: str
    qubit: str
    step: str
    axis_original: np.ndarray
    time_seconds: np.ndarray
    re: np.ndarray
    im: np.ndarray
    parameters_sent: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fid(self) -> np.ndarray:
        return self.re + 1j * self.im

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / f"{self.key}.npz", axis_original=self.axis_original,
                            time_seconds=self.time_seconds, re=self.re, im=self.im)
        atomic_json(root / f"{self.key}.json", {
            "key": self.key, "task_id": self.task_id, "group": self.group,
            "path": self.path, "qubit": self.qubit, "step": self.step,
            "parameters_sent": self.parameters_sent, "metadata": self.metadata,
            "numeric_file": f"{self.key}.npz"})

    @classmethod
    def load(cls, root: Path, key: str) -> "RawFIDRecord":
        data = json.loads((root / f"{key}.json").read_text(encoding="utf-8"))
        with np.load(root / data["numeric_file"], allow_pickle=False) as numeric:
            arrays = {name: numeric[name].copy() for name in
                      ("axis_original", "time_seconds", "re", "im")}
        return cls(key=key, task_id=data["task_id"], group=data["group"],
                   path=data["path"], qubit=data["qubit"], step=data["step"],
                   parameters_sent=data["parameters_sent"], metadata=data["metadata"], **arrays)


def compile_sequence(sequence: SequenceIR, capabilities: Capabilities,
                     *, idle_probe: bool = False) -> ExperimentSpec:
    """Compile serial H pulses; uncertain timing or P addressing fails closed.

    The SDK splits H and P lists, so concurrent or interleaved cross-channel
    timing cannot be established by list order. A zero-amplitude segment is
    only emitted after a dedicated controlled probe verifies its delay effect.
    """
    if not sequence.segments:
        raise ValueError("At least one explicit pulse is required")
    if sequence.sample_path != 0 or any(s.channel != "H" for s in sequence.segments):
        raise CapabilityUnavailable("P/multichannel timing and readout are not verified for this device")
    if not capabilities.h_path_verified:
        raise CapabilityUnavailable("H path is not verified")
    if not sequence.state_initialization:
        raise CapabilityUnavailable("Preparation-off repeatability has not been verified")
    if sequence.sample_hz != 10000 or sequence.sample_count not in (4000, 8000, 16000):
        raise CapabilityUnavailable("Requested acquisition mode is outside the completed H envelope")
    segments = sorted(sequence.segments, key=lambda s: s.start_us)
    pulses: list[dict[str, float]] = []
    cursor = 0.0
    rf_us = 0.0
    for segment in segments:
        values = (segment.start_us, segment.duration_us, segment.amplitude_pct,
                  segment.phase_deg, segment.detuning_hz)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Nonfinite pulse field")
        if segment.start_us < cursor - 1e-6 or segment.duration_us < 5:
            raise ValueError("Pulse overlap or invalid duration")
        if not 0 <= segment.amplitude_pct <= 100 or not 0 <= segment.phase_deg < 360:
            raise ValueError("Pulse amplitude/phase outside tested command range")
        if segment.amplitude_pct == 0 and not (capabilities.zero_amplitude_delay_verified or idle_probe):
            raise CapabilityUnavailable("UNVERIFIED_TIMING: zero-amplitude SDK delay not verified")
        if abs(segment.detuning_hz) > 20:
            raise ValueError("Pulse detuning outside historical test range")
        gap = segment.start_us - cursor
        if gap > 1e-6:
            if not (capabilities.zero_amplitude_delay_verified or idle_probe):
                raise CapabilityUnavailable("UNVERIFIED_TIMING: zero-amplitude SDK delay not verified")
            if gap < 5:
                raise CapabilityUnavailable("Gap below tested SDK pulse resolution")
            pulses.append({"width": float(gap), "am": 0.0, "phase": 0.0, "freshift": 0.0})
        pulses.append({"width": float(segment.duration_us),
                       "am": float(segment.amplitude_pct),
                       "phase": float(segment.phase_deg),
                       "freshift": float(segment.detuning_hz)})
        cursor = segment.start_us + segment.duration_us
        if segment.amplitude_pct > 0:
            rf_us += segment.duration_us
    if len(pulses) > capabilities.maximum_segments or cursor > capabilities.maximum_task_span_us:
        raise CapabilityUnavailable("Sequence exceeds historical software envelope")
    if len(segments) > 1 and not (capabilities.segment_sequence_verified or idle_probe):
        raise CapabilityUnavailable("Segment order/effect has not been verified")
    payload = physical_request(pulses=pulses, sample_count=sequence.sample_count,
                               sample_hz=sequence.sample_hz)
    return ExperimentSpec(payload=payload, sequence=sequence, rf_duration_us=rf_us,
                          sequence_duration_us=cursor,
                          idle_probe=idle_probe or any(p["am"] == 0 for p in pulses))


def _chart_series(events: Iterable[dict[str, Any]], task_id: str) -> tuple[dict[tuple, list], set[tuple], bool]:
    """Choose full-chart replacements and concatenate only disjoint fragments."""
    charts: dict[tuple, list] = {}
    finished: set[tuple] = set()
    task_finished = False
    for event in events:
        kind = event.get("kind") or event.get("msg_id")
        payload = event.get("payload", event)
        body = payload.get("chart_data") or payload.get("json_data") or {}
        if str(body.get("taskId")) != str(task_id):
            continue
        group = str(body.get("group", ""))
        path = str(body.get("path", ""))
        qubit = str(body.get("qubit", ""))
        step = str(body.get("step", ""))
        if kind == "s_post_exp_finished":
            task_finished = True
        elif kind == "s_post_exp_chart_updated_finished":
            finished.add((group, path, step))
        elif kind == "s_post_exp_chart_updated":
            name = str(body.get("chart_name", ""))
            points = body.get("points")
            if not isinstance(points, list) or not points:
                raise IncompleteFID(f"Empty chart update {name}")
            key = (group, path, qubit, step, name)
            previous = charts.get(key)
            if previous:
                # The SDK often sends a fresh full curve with the same first x.
                if points[0][0] == previous[0][0] and len(points) >= len(previous):
                    charts[key] = points
                elif points[0][0] > previous[-1][0]:
                    charts[key] = previous + points
                else:
                    raise IncompleteFID(f"Ambiguous overlapping chart update {key}")
            else:
                charts[key] = points
    return charts, finished, task_finished


def uniform_axis_step(axis: np.ndarray) -> tuple[float, float, float]:
    """Check spacing against an endpoint slope, allowing float32 chart rounding.

    SpinQ exports each x value as a float32. At 16,000 points the median of
    adjacent rounded differences is biased, even when every ideal x is on a
    perfectly uniform grid. The endpoint slope avoids accumulating that bias.
    A deviation above 0.2% of one sample step still fails closed.
    """
    if axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis)):
        raise ValueError("Original chart axis is missing or nonfinite")
    if np.any(np.diff(axis) <= 0):
        raise ValueError("Original chart axis is not increasing")
    step = float((axis[-1] - axis[0]) / (len(axis) - 1))
    predicted = axis[0] + np.arange(len(axis), dtype=np.float64) * step
    residual = float(np.max(np.abs(axis - predicted)))
    tolerance = 0.002 * step
    if residual > tolerance:
        raise ValueError("Exported axis inconsistent with uniform requested sampling")
    return step, residual, tolerance


def assemble_fid(events: Iterable[dict[str, Any]], task_id: str, path: str = "0",
                 qubit: str = "0", step: str = "NMRSIG", *, key: str = "",
                 parameters_sent: dict | None = None, metadata: dict | None = None) -> RawFIDRecord:
    """Pair exactly matching Re/Im from a completed task and chart update."""
    charts, finished, terminal = _chart_series(events, task_id)
    choices = [(k, pts) for k, pts in charts.items() if k[1:] == (path, qubit, step, "fidRe")]
    if len(choices) != 1:
        raise IncompleteFID(f"Expected one fidRe for task/path/qubit/step; found {len(choices)}")
    re_key, real = choices[0]
    imag = charts.get(re_key[:-1] + ("fidIm",))
    if imag is None or len(real) != len(imag):
        raise IncompleteFID("Missing or length-mismatched fidIm")
    if (re_key[0], path, step) not in finished or not terminal:
        raise IncompleteFID("FID chart or task has no terminal update")
    axis = np.asarray([point[0] for point in real], dtype=np.float64)
    imag_axis = np.asarray([point[0] for point in imag], dtype=np.float64)
    if not np.array_equal(axis, imag_axis):
        raise IncompleteFID("fidRe/fidIm axes differ")
    re = np.asarray([point[1] for point in real], dtype=np.float64)
    im = np.asarray([point[1] for point in imag], dtype=np.float64)
    if len(axis) < 64 or not all(np.all(np.isfinite(a)) for a in (axis, re, im)):
        raise IncompleteFID("FID short or nonfinite")
    params = parameters_sent or {}
    fs = int(params.get("sampleFre", 0))
    if fs <= 0:
        raise IncompleteFID("Actual request sampling frequency unavailable")
    expected = int(params.get("sampleCount", len(axis)))
    if len(axis) not in (expected, expected - 1):
        raise IncompleteFID(f"Chart has {len(axis)} points; request asked {expected}")
    try:
        step, axis_residual, axis_tolerance = uniform_axis_step(axis)
    except ValueError as exc:
        raise IncompleteFID(str(exc)) from exc
    # Use this experiment's requested clock for local analysis. The exported
    # axis supports a consistent inferred unit, not independent ADC proof.
    scale_s = (1.0 / fs) / step
    record_meta = dict(metadata or {})
    record_meta["axis_contract"] = {
        "configured_sample_hz": fs, "original_step_endpoint": step,
        "original_step_median": float(np.median(np.diff(axis))),
        "uniformity_max_deviation": axis_residual,
        "uniformity_tolerance": axis_tolerance,
        "seconds_per_original_axis_unit_inferred": scale_s,
        "time_seconds_origin": "requested sampleFre and uniform original axis consistency",
        "physical_ADC_clock_independently_verified": False,
        "chart_point_count": len(axis), "requested_point_count": expected,
        "exported_signal_type": "complex FID after unknown vendor preprocessing"}
    return RawFIDRecord(key=key, task_id=str(task_id), group=re_key[0], path=path,
                        qubit=qubit, step=step, axis_original=axis,
                        time_seconds=np.arange(len(axis), dtype=np.float64) / fs,
                        re=re, im=im, parameters_sent=params, metadata=record_meta)


def _events_for_task(path: Path, task_id: str) -> list[dict[str, Any]]:
    events = []
    source_path = path if path.exists() else path.with_suffix(path.suffix + ".gz")
    if not source_path.exists():
        return events
    opener = gzip.open if source_path.suffix == ".gz" else open
    with opener(source_path, "rt", encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            body = event.get("payload", {}).get("chart_data") or event.get("payload", {}).get("json_data") or {}
            if str(body.get("taskId")) == str(task_id):
                events.append(event)
    return events


def _save_vendor(charts: dict[tuple, list], record: RawFIDRecord, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta = {"task_id": record.task_id, "source": "vendor decoded chart events, reference only",
            "not_primary_model_input": True, "charts": []}
    for (group, path, qubit, step, name), points in charts.items():
        if group != record.group or path != record.path or qubit != record.qubit or step != record.step:
            continue
        if name in {"fidRe", "fidIm"}:
            continue
        safe_name = "".join(ch for ch in name if ch.isalnum() or ch == "_")
        if not safe_name:
            continue
        arr = np.asarray(points, dtype=np.float64)
        arrays[safe_name] = arr
        meta["charts"].append(safe_name)
    if arrays:
        np.savez_compressed(root / f"{record.key}.npz", **arrays)
    atomic_json(root / f"{record.key}.json", meta)


def run_raw(spec: ExperimentSpec, *, key: str, hardware: LiveHardware,
            output: Path) -> RawFIDRecord:
    """Run one real physical-layer task; resumptions reuse saved raw records."""
    raw_root = output / "raw"
    target = raw_root / f"{key}.json"
    if target.exists():
        prior = RawFIDRecord.load(raw_root, key)
        if prior.parameters_sent != spec.payload:
            raise ValueError("Resume payload differs from the original physical request")
        return prior
    row = hardware.measure(key, spec.payload, allow_idle_probe=spec.idle_probe)
    task_id = row["task_id"]
    events = _events_for_task(hardware.data / "events.jsonl", task_id)
    if not events:
        atomic_json(raw_root / f"{key}.error.json", {"task_id": task_id,
                    "error": "No task events captured; simplified SDK graph not substituted"})
        raise IncompleteFID("No task events captured")
    try:
        event_times={}
        for event in events:
            kind=event.get("kind","")
            stamp=event.get("received_monotonic_ns")
            if not isinstance(stamp,int): continue
            body=event.get("payload",{}).get("chart_data") or {}
            if kind=="s_post_exp_chart_updated" and body.get("chart_name") in ("fidRe","fidIm"):
                event_times.setdefault("first_fid_chart_received_monotonic_ns",stamp)
                event_times["last_fid_chart_received_monotonic_ns"]=stamp
            elif kind=="s_post_exp_chart_updated_finished":
                event_times["chart_finished_received_monotonic_ns"]=stamp
            elif kind=="s_post_exp_finished":
                event_times["task_finished_received_monotonic_ns"]=stamp
        record = assemble_fid(events, task_id, key=key,
                              parameters_sent=spec.payload,
                              metadata={"preflight": row.get("preflight"),
                                        "wall_seconds": row.get("wall_seconds"),
                                        "finished_utc": row.get("finished_utc"),
                                        "local_event_timing": event_times,
                                        "internal_repetitions": "UNKNOWN",
                                        "server_processing_offload_unverified": True})
    except IncompleteFID as exc:
        atomic_json(raw_root / f"{key}.error.json", {"task_id": task_id,
                    "error": str(exc), "events_received": len(events)})
        raise
    record.save(raw_root)
    previous_error = raw_root / f"{key}.error.json"
    if previous_error.exists():
        recovery = json.loads(previous_error.read_text(encoding="utf-8"))
        recovery["status"] = "RECOVERED_FROM_SAVED_EVENTS"
        recovery["resolution"] = "FID reassembled and validated from original decoded task events"
        atomic_json(previous_error, recovery)
    charts, _, _ = _chart_series(events, task_id)
    _save_vendor(charts, record, output / "vendor_reference")
    return record
