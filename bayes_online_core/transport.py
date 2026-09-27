"""SpinQLabLink 1.0.2 physical-layer transport for the independent online study.

No connection is made on import.  This module intentionally uses only mode 0
and keeps the exported complex FID separate from the vendor's FFT products.
The SDK exposes no documented hardware clock readback or task-cancel command;
the evidence fields state those limits rather than inventing confirmations.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import isfinite
import re
from threading import Lock
from time import monotonic, sleep
from typing import Any, Callable, Optional

import numpy as np


class TransportError(RuntimeError):
    """A request was rejected before or during a physical acquisition."""

    def __init__(self, message: str, *, task_id: str = "", events: Optional[list] = None):
        super().__init__(message)
        self.task_id = task_id
        self.events = events or []


class UnsupportedTiming(TransportError):
    """The SDK/server timing primitive has not been experimentally qualified."""


class TaskUncertain(TransportError):
    """A task may still be running; this transport must submit nothing further."""


class IncompleteData(TransportError):
    """The task ended but its required exported FID was incomplete."""


@dataclass(frozen=True)
class PulseSpec:
    path: int
    start_us: float
    width_us: float
    amplitude_pct: float
    phase_deg: float
    detuning_hz: float = 0.0
    role: str = ""


@dataclass(frozen=True)
class AcquisitionRequest:
    key: str
    pulses: tuple[PulseSpec, ...]
    sample_path: int
    sample_count: int = 16000
    sample_frequency_hz: int = 10000
    sample_delay_us: float = 0
    # A literal baseline value.  SDK source labels this µs, published example
    # labels it seconds; the working value is passed unchanged, not converted.
    relaxation_delay_s: float = 15
    h_frequency_shift_hz: float = 0
    p_frequency_shift_hz: float = 0
    h_demodulation_hz: float = 0
    p_demodulation_hz: float = 0
    initialize_state: bool = False


@dataclass
class AcquisitionResult:
    key: str
    task_id: str
    status: str
    time_s: np.ndarray
    fid_complex: np.ndarray
    vendor_reference: dict[str, Any]
    requested_payload: dict[str, Any]
    sent_payload: dict[str, Any]
    timing_evidence: dict[str, Any]
    elapsed_s: float
    device_snapshot: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    exported_axis: np.ndarray = field(default_factory=lambda: np.empty(0, float))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe full-resolution record; FID is never reduced to a peak."""
        return {
            "key": self.key,
            "task_id": self.task_id,
            "status": self.status,
            "time_s": self.time_s.tolist(),
            "fid_re": self.fid_complex.real.tolist(),
            "fid_im": self.fid_complex.imag.tolist(),
            "exported_axis": self.exported_axis.tolist(),
            "vendor_reference": _json_plain(self.vendor_reference),
            "requested_payload": _json_plain(self.requested_payload),
            "sent_payload": _json_plain(self.sent_payload),
            "timing_evidence": _json_plain(self.timing_evidence),
            "elapsed_s": self.elapsed_s,
            "device_snapshot": _json_plain(self.device_snapshot),
            "events": _json_plain(self.events),
        }


def _json_plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_plain(v) for v in value]
    return value


def _finite_number(label: str, value: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TransportError(f"{label} must be numeric") from exc
    if not isfinite(out):
        raise TransportError(f"{label} must be finite")
    return out


def _integer_microseconds(label: str, value: float) -> int:
    out = _finite_number(label, value)
    if out < 0 or abs(out - round(out)) > 1e-9:
        raise UnsupportedTiming(f"{label} must be a nonnegative integer µs; no hardware sub-µs grid is verified")
    return int(round(out))


def _qualification_key(prefix: str, base: str) -> str:
    if prefix == "":
        return base
    if not isinstance(prefix, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", prefix) is None:
        raise TransportError("Qualification key_prefix must use 1..80 letters, digits, underscores or hyphens")
    return f"{prefix}_{base}"


def _validate_request(request: AcquisitionRequest) -> None:
    if not request.key or not isinstance(request.key, str):
        raise TransportError("Nonempty acquisition key required")
    if request.sample_path not in (0, 1):
        raise UnsupportedTiming("Dual-channel exported step-to-channel mapping has not been verified; acquire H/P separately")
    if not 4000 <= request.sample_count <= 16000:
        raise TransportError("sample_count must be within documented 4000..16000")
    if request.sample_frequency_hz not in range(10000, 100001, 10000):
        raise TransportError("sample_frequency_hz must use a documented 10 kHz step")
    if _integer_microseconds("sample_delay_us", request.sample_delay_us) > 2_000_000:
        raise TransportError("sample_delay_us exceeds SDK range")
    if not 0 <= _finite_number("relaxation_delay_s", request.relaxation_delay_s) <= 25_000_000:
        raise TransportError("relaxation_delay_s outside SDK numeric range")
    for key in ("h_frequency_shift_hz", "p_frequency_shift_hz", "h_demodulation_hz", "p_demodulation_hz"):
        x = _finite_number(key, getattr(request, key))
        if x != round(x) or not -1000 <= x <= 1000:
            raise TransportError(f"{key} must be an integer Hz in SDK range -1000..1000")
    for pulse in request.pulses:
        if pulse.path not in (0, 1):
            raise TransportError("Pulse path must be H=0 or P=1")
        start = _integer_microseconds("pulse.start_us", pulse.start_us)
        width = _integer_microseconds("pulse.width_us", pulse.width_us)
        if width <= 0 or start + width > 20_000_000:
            raise TransportError("Pulse width or scheduled end outside SDK range")
        amplitude = _finite_number("pulse.amplitude_pct", pulse.amplitude_pct)
        if not 0 <= amplitude <= 100:
            raise TransportError("Pulse amplitude outside SDK range 0..100%; no clipping is applied")
        _finite_number("pulse.phase_deg", pulse.phase_deg)
        detuning = _finite_number("pulse.detuning_hz", pulse.detuning_hz)
        if not -10000 <= detuning <= 10000:
            raise TransportError("Pulse detuning outside SDK range -10000..10000 Hz")


def _schedule(request: AcquisitionRequest) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Compile separate H/P lists with explicit zero-amplitude padding.

    This describes the serialized command, not a measured FPGA timing grid.
    """
    _validate_request(request)
    by_path: dict[int, list[PulseSpec]] = {0: [], 1: []}
    for pulse in request.pulses:
        by_path[pulse.path].append(pulse)
    segments: list[dict[str, float]] = []
    timing: dict[str, Any] = {
        "clock_grid": "integer_microseconds_requested; hardware_actual_unverified",
        "requested_starts_us": [float(p.start_us) for p in request.pulses],
        "requested_widths_us": [float(p.width_us) for p in request.pulses],
        "uses_zero_amplitude_padding": False,
        "has_idle_segment": False,
        "cross_channel_alignment_verified": False,
        "idle_gap_verified": False,
        "relaxation_delay_unit": "SDK numeric literal; published docs/source disagree on unit",
        "sample_delay_role": "acquisition_delay_only",
    }
    for path in (0, 1):
        cursor = 0
        for pulse in sorted(by_path[path], key=lambda p: p.start_us):
            start = int(round(pulse.start_us))
            width = int(round(pulse.width_us))
            if start < cursor:
                raise UnsupportedTiming("Overlapping pulses on one channel cannot be serialized")
            if start > cursor:
                segments.append({"path": path, "width": start - cursor, "am": 0.0,
                                 "phase": 0.0, "freshift": 0.0, "role": "padding"})
                timing["uses_zero_amplitude_padding"] = True
                timing["has_idle_segment"] = True
            if float(pulse.amplitude_pct) == 0:
                timing["has_idle_segment"] = True
            segments.append({"path": path, "width": width, "am": float(pulse.amplitude_pct),
                             "phase": float(pulse.phase_deg), "freshift": float(pulse.detuning_hz),
                             "role": pulse.role})
            cursor = start + width
    return segments, timing


def _prepare_graph(graph: Any, sample_count: int, sample_frequency_hz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    if not isinstance(graph, list) or len(graph) != 1 or not isinstance(graph[0], dict):
        raise IncompleteData("Expected exactly one labeled physical-layer graph step for one sample path")
    step = graph[0]
    try:
        re = np.asarray(step["fidRe"], dtype=float)
        im = np.asarray(step["fidIm"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise IncompleteData("Completed task lacks valid fidRe and fidIm") from exc
    if re.shape != (sample_count, 2) or im.shape != (sample_count, 2):
        raise IncompleteData(f"FID incomplete: expected {sample_count} Re/Im points, received {re.shape}/{im.shape}")
    if not np.isfinite(re).all() or not np.isfinite(im).all():
        raise IncompleteData("Non-finite exported FID or x-axis values")
    x = re[:, 0]
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise IncompleteData("Exported FID x-axis is not strictly increasing")
    if not np.allclose(re[:, 0], im[:, 0], atol=max(abs(float(np.median(dx))) * 1e-5, 1e-9), rtol=1e-9):
        raise IncompleteData("fidRe and fidIm x-axes disagree")
    median_dx = float(np.median(dx))
    uniform = bool(np.allclose(dx, median_dx, rtol=0.02, atol=0))
    if not uniform:
        raise IncompleteData("Exported FID x-axis is nonuniform")
    expected_s = 1.0 / sample_frequency_hz
    axis_scale = None
    axis_unit = "UNKNOWN"
    for label, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if abs(median_dx * scale / expected_s - 1.0) <= 0.02:
            axis_scale, axis_unit = scale, label
            break
    if axis_scale is None:
        # Retain the export intact.  The time axis below is the requested-rate
        # index axis, never misrepresented as a measured server timestamp.
        time_s = np.arange(sample_count, dtype=float) * expected_s
        axis_status = "REQUESTED_RATE_ONLY_EXPORTED_AXIS_MISMATCH"
    else:
        time_s = (x - x[0]) * axis_scale
        axis_status = "EXPORT_MATCHES_REQUESTED_RATE"
    evidence = {"axis_status": axis_status, "exported_axis_unit_inferred": axis_unit,
                "exported_x0": float(x[0]), "exported_dx": median_dx,
                "requested_sample_period_s": expected_s}
    vendor = {key: deepcopy(value) for key, value in step.items() if key not in ("fidRe", "fidIm")}
    return time_s, re[:, 1] + 1j * im[:, 1], x.copy(), vendor, evidence


class PhysicalTransport:
    """One-owner synchronous physical-layer executor; never runs on import."""

    def __init__(self, host: str = "172.19.20.100", port: int = 8181,
                 account: str = "anyword", password: str = "anyword", *,
                 exclusive_use_confirmed: bool = False, timeout_s: float = 120):
        self.host = host
        self.port = int(port)
        self.account = account
        self._password = password
        self.exclusive_use_confirmed = bool(exclusive_use_confirmed)
        self.timeout_s = float(timeout_s)
        if not isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self._client: Any = None
        self._queue: Optional[list] = None
        self._queue_at = 0.0
        self._queue_lock = Lock()
        self._execution_lock = Lock()
        self._uncertain = False
        self._active = False
        self._last_completed_id = ""
        self._verified_paths: set[int] = set()
        self._verified_idle_us: dict[int, set[int]] = {0: set(), 1: set()}
        self._qualification: dict[int, dict[str, Any]] = {}

    @property
    def timing_verified(self) -> bool:
        """True only if contiguous segments were physically qualified on H and P."""
        return self._verified_paths == {0, 1}

    @property
    def timing_qualification(self) -> dict[int, dict[str, Any]]:
        return deepcopy(self._qualification)

    def idle_gap_verified(self, path: int, gap_us: float) -> bool:
        return int(gap_us) in self._verified_idle_us.get(path, set()) and float(gap_us) == int(gap_us)

    def __enter__(self) -> "PhysicalTransport":
        self.connect()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def connect(self) -> "PhysicalTransport":
        if self._client is not None:
            return self
        from spinqlablink import SpinQLabLink
        from spinqlablink.utils.types import MachineType

        client = SpinQLabLink(self.host, self.port, self.account, self._password)

        def capture_queue(message: dict[str, Any]) -> None:
            queue = message.get("queue")
            if isinstance(queue, list):
                with self._queue_lock:
                    # Queue contents may refer to other users: retain in memory
                    # for the admission gate, never include in output records.
                    self._queue = deepcopy(queue)
                    self._queue_at = monotonic()

        # SDK's default queue handler dereferences current_experiment even at
        # login.  Install a read-only guard before the socket thread starts.
        client.handler_map[MachineType.MSG_POST_EXP_QUEUE_UPDATE] = capture_queue
        try:
            result = client.connect()
            if result is False or not client.wait_for_login(timeout=min(20, max(2, int(self.timeout_s)))):
                raise TransportError("SpinQLabLink login did not complete")
            self._client = client
            return self
        except Exception:
            try:
                client.disconnect()
            except Exception:
                pass
            raise

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass

    def device_snapshot(self) -> dict[str, Any]:
        if self._client is None:
            raise TransportError("Transport is not connected")
        status = self._client.get_device_status()
        device = self._client.get_device()
        params = device.params.to_dict() if device is not None else {}
        return {
            "status": deepcopy(status),
            "frequencies_hz": deepcopy(self._client.get_device_frequencies()),
            "pulse_param": deepcopy(params.get("pulseParam", {})),
            "pps_param": deepcopy(params.get("ppsParam", {})),
            "sample_param": deepcopy(params.get("sampleParam", {})),
        }

    def _admit(self) -> dict[str, Any]:
        if self._client is None:
            raise TransportError("Transport is not connected")
        if self._uncertain:
            raise TaskUncertain("Previous task state is uncertain; reconnect and inspect it before any new submission")
        if self._active:
            raise TransportError("Only one active hardware task is allowed")
        if not self.exclusive_use_confirmed:
            raise TransportError("Exclusive device use has not been confirmed; no task submitted")
        deadline = monotonic() + min(5.0, self.timeout_s / 5)
        snapshot = self.device_snapshot()
        status = snapshot["status"]
        while (status.get("connected") is not True or status.get("lock_state") is not True) and monotonic() < deadline:
            sleep(0.1)
            snapshot = self.device_snapshot()
            status = snapshot["status"]
        if status.get("connected") is not True or status.get("lock_state") is not True:
            raise TransportError(f"Device not confirmed connected and locked: {status}")
        # The SDK does not offer a public queue-query API.  Use its passive
        # updates if received; explicit human exclusive-use confirmation is
        # required only when an update is unavailable/stale.
        with self._queue_lock:
            fresh = self._queue is not None and monotonic() - self._queue_at < 30.0
            queue = deepcopy(self._queue) if fresh else None
        if queue is not None:
            busy = [item for item in queue if not (
                self._last_completed_id and isinstance(item, dict) and
                item.get("id") == self._last_completed_id)]
            if busy:
                raise TransportError("Device queue reports another task; no task submitted")
        return snapshot

    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        return self._acquire(request, qualification=False)

    def _acquire(self, request: AcquisitionRequest, *, qualification: bool) -> AcquisitionResult:
        if not self._execution_lock.acquire(blocking=False):
            raise TransportError("Another task already owns this transport")
        try:
            return self._acquire_locked(request, qualification=qualification)
        finally:
            self._execution_lock.release()

    def _acquire_locked(self, request: AcquisitionRequest, *, qualification: bool) -> AcquisitionResult:
        from spinqlablink import ExperimentType, Pulse
        from spinqlablink.utils.types import ExperimentState, MachineType
        import json

        segments, timing = _schedule(request)
        nonzero = [x for x in segments if x["am"] > 0]
        paths = {x["path"] for x in nonzero}
        if not qualification:
            idle = [x for x in segments if x["am"] == 0]
            if idle and (not paths or any(int(x["width"]) not in self._verified_idle_us[x["path"]] for x in idle)):
                raise UnsupportedTiming("Zero-amplitude idle gap duration has not been physically qualified on this path")
            if len(nonzero) > 1 and not paths.issubset(self._verified_paths):
                raise UnsupportedTiming("Contiguous multi-segment order is unverified for requested H/P path")
            if len({x["path"] for x in segments}) > 1:
                raise UnsupportedTiming("Concurrent H/P alignment is not independently verified")
        device_before = self._admit()
        client = self._client
        assert client is not None
        exp, para = client.register_experiment(ExperimentType.PHYSICAL_LAYER_EXPERIMENT)
        events: list[dict[str, Any]] = []
        event_lock = Lock()
        started = monotonic()
        task_started = False
        task_id = str(exp.id)
        completed = False
        handler_originals: dict[Any, Any] = {}
        handler_wrappers: dict[Any, Any] = {}
        try:
            para.type_setting = 0
            para.state_initialization = bool(request.initialize_state)
            para.samplePath = request.sample_path
            para.sampleCount = request.sample_count
            para.sampleFre = request.sample_frequency_hz
            para.sampleDelay = int(round(request.sample_delay_us))
            para.relaxation_delay = request.relaxation_delay_s
            para.h_freShift = int(round(request.h_frequency_shift_hz))
            para.p_freShift = int(round(request.p_frequency_shift_hz))
            para.h_freDemo = int(round(request.h_demodulation_hz))
            para.p_freDemo = int(round(request.p_demodulation_hz))
            para.gradients = []
            para.pulses = [Pulse(path=int(s["path"]), width=s["width"],
                                 amplitude=s["am"], phase=s["phase"],
                                 detuning=s["freshift"]) for s in segments]
            expected = para.get_parameters()
            if expected.get("compute_type") != 0 or expected.get("makePps") != request.initialize_state:
                raise TransportError("Physical-layer mode 0 / PPS flag serializer mismatch")
            if expected.get("pulse", {}) != {
                    "hPulse": [{"width": s["width"], "am": s["am"], "phase": s["phase"],
                                "freshift": s["freshift"]} for s in segments if s["path"] == 0],
                    "pPulse": [{"width": s["width"], "am": s["am"], "phase": s["phase"],
                                "freshift": s["freshift"]} for s in segments if s["path"] == 1]}:
                raise TransportError("Pulse payload differs from compiled H/P lists")
            requested = {"request": asdict(request), "compiled_segments": segments}
            sent: dict[str, Any] = {}
            event_kinds = (
                (MachineType.MSG_RES_ADD_EXP_TASK_RES, "ack"),
                (MachineType.MSG_POST_EXP_STARTED, "started"),
                (MachineType.MSG_POST_EXP_STEP_CHANGED, "step"),
                (MachineType.MSG_POST_EXP_CHART_UPDATED_STARTED, "chart_started"),
                (MachineType.MSG_POST_EXP_CHART_UPDATED, "chart_updated"),
                (MachineType.MSG_POST_EXP_CHART_UPDATED_FINISHED, "chart_finished"),
                (MachineType.MSG_POST_EXP_FINISHED, "finished"),
                (MachineType.MSG_POST_EXP_TERMINATED, "terminated"),
                (MachineType.MSG_POST_EXP_REMOVED, "removed"),
            )
            for event_code, kind in event_kinds:
                original = client.exp_handler_map[event_code]
                handler_originals[event_code] = original

                def capture(message: dict[str, Any], *, _original=original, _kind=kind) -> None:
                    nonlocal task_id
                    message_id = str(message.get("taskId", ""))
                    if _kind == "ack":
                        _original(message)
                        if message.get("code") == 0:
                            task_id = str(exp.id)
                        return
                    if message_id and message_id != str(exp.id):
                        return
                    if _kind in ("chart_started", "chart_updated", "chart_finished"):
                        payload = deepcopy(message)
                    else:
                        payload = {"taskId": message_id, "step": message.get("step"),
                                   "state": message.get("state")}
                    with event_lock:
                        events.append({"kind": _kind, "elapsed_s": monotonic() - started,
                                       "task_id": message_id, "payload": payload})
                    _original(message)

                client.exp_handler_map[event_code] = capture
                handler_wrappers[event_code] = capture

            send_original = client._send_message

            def capture_sent(msg_id: str, payload: dict[str, Any]) -> Any:
                if msg_id == MachineType.MSG_REQ_ADD_EXP_TASK_REQ:
                    sent.update(json.loads(payload["params"]))
                return send_original(msg_id, payload)

            client._send_message = capture_sent
            try:
                # Once add-task sending begins, a socket exception cannot prove
                # that the server did not enqueue it.  Keep the state uncertain.
                task_started = True
                client.run_experiment()
            finally:
                client._send_message = send_original
            if sent != expected:
                self._uncertain = task_started
                raise TaskUncertain("Actual add-task payload did not match preflight serialization",
                                    task_id=task_id, events=deepcopy(events))
            self._active = True
            deadline = monotonic() + self.timeout_s
            while monotonic() < deadline:
                if client.get_expMgr().current_experiment is not exp:
                    self._uncertain = True
                    raise TaskUncertain("SDK deregistered task unexpectedly; remote state unknown",
                                        task_id=task_id, events=deepcopy(events))
                state = exp.get_status()
                if state in (ExperimentState.COMPLETED, ExperimentState.FAILED):
                    completed = True
                    break
                if not client.is_connected or not client.is_logged_in:
                    self._uncertain = True
                    raise TaskUncertain("Connection lost before a definitive task state",
                                        task_id=task_id, events=deepcopy(events))
                sleep(0.2)
            if not completed:
                self._uncertain = True
                raise TaskUncertain("Experiment timeout; no cancel primitive is documented, state must be inspected",
                                    task_id=str(exp.id), events=deepcopy(events))
            # Task completion and chart delivery are separate callbacks.  Give
            # Re/Im a bounded grace interval after the terminal state.
            graph_error: Optional[IncompleteData] = None
            graph_deadline = monotonic() + min(10.0, max(2.0, self.timeout_s / 10))
            while True:
                info = client.get_experiment_result()
                task_id = str(info.get("id", exp.id))
                if task_id != str(exp.id) or info.get("state") != ExperimentState.COMPLETED:
                    raise TransportError(f"Experiment ended without completed matching task ID: {info.get('state')}",
                                         task_id=task_id, events=deepcopy(events))
                result = info.get("result")
                graph = result.get("graph") if isinstance(result, dict) else None
                try:
                    time_s, fid, exported_x, vendor_graph, axis_evidence = _prepare_graph(
                        graph, request.sample_count, request.sample_frequency_hz)
                    graph_error = None
                    break
                except IncompleteData as exc:
                    graph_error = exc
                    if monotonic() >= graph_deadline:
                        break
                    sleep(0.2)
            if graph_error is not None:
                raise IncompleteData(str(graph_error), task_id=task_id, events=deepcopy(events))
            timing.update(axis_evidence)
            timing["contiguous_segment_qualification"] = {
                path: self._qualification.get(path, {"status": "UNVERIFIED"}) for path in paths}
            timing["idle_gap_qualification"] = {
                path: sorted(self._verified_idle_us[path]) for path in paths}
            timing["hardware_actual_pulse_timing"] = "UNAVAILABLE_FROM_SDK"
            timing["hardware_actual_rf_frequency"] = "UNAVAILABLE_FROM_SDK"
            vendor = {"graph_step_0": vendor_graph}
            if isinstance(result, dict):
                vendor.update({key: deepcopy(value) for key, value in result.items() if key != "graph"})
            with event_lock:
                event_copy = deepcopy(events)
            return AcquisitionResult(
                key=request.key, task_id=task_id, status="COMPLETED", time_s=time_s,
                fid_complex=fid, vendor_reference=vendor, requested_payload=requested,
                sent_payload=sent, timing_evidence=timing, elapsed_s=monotonic() - started,
                device_snapshot=device_before, events=event_copy, exported_axis=exported_x)
        finally:
            self._active = False
            if task_started and not completed:
                self._uncertain = True
            if completed and not self._uncertain:
                self._last_completed_id = str(exp.id)
                try:
                    client.deregister_experiment()
                except Exception:
                    self._uncertain = True
            elif not task_started:
                try:
                    client.deregister_experiment()
                except Exception:
                    pass
            # register_experiment installs fresh bound handlers.  Restore this
            # task's exact originals so repeated acquisitions do not retain
            # closures holding FIDs, events, or previous experiment objects.
            with event_lock:
                for event_code, original in handler_originals.items():
                    if client.exp_handler_map.get(event_code) is handler_wrappers.get(event_code):
                        client.exp_handler_map[event_code] = original

    def verify_two_segment_equivalence(self, reference: PulseSpec, *,
                                       sample_count: int = 4000,
                                       sample_frequency_hz: int = 10000,
                                       key_prefix: str = "",
                                       on_acquisition: Optional[Callable[[AcquisitionResult], None]] = None) -> dict[str, Any]:
        """Qualify contiguous single-channel serialization using real FIDs.

        This bounded test submits four acquisitions, using no larger pulse than
        the supplied known-working reference.  It cannot validate idle gaps,
        cross-channel alignment, gradients, or absolute hardware timestamps.
        ``on_acquisition`` receives each completed FID before the next task.
        """
        if reference.path not in (0, 1) or reference.start_us != 0:
            raise UnsupportedTiming("Qualification reference must start at zero on one channel")
        _qualification_key(key_prefix, "preflight")
        if reference.amplitude_pct <= 0:
            raise UnsupportedTiming("Qualification reference needs a nonzero known-working RF amplitude")
        width = _integer_microseconds("reference.width_us", reference.width_us)
        if width < 2 or width % 2:
            raise UnsupportedTiming("Qualification requires even integer reference width >=2 µs")
        _validate_request(AcquisitionRequest("timing_reference", (reference,), reference.path,
                                             sample_count, sample_frequency_hz))
        half = width // 2
        part_a = PulseSpec(reference.path, 0, half, reference.amplitude_pct,
                           reference.phase_deg, reference.detuning_hz, "qualification")
        part_b = PulseSpec(reference.path, half, half, reference.amplitude_pct,
                           reference.phase_deg, reference.detuning_hz, "qualification")
        opposite = PulseSpec(reference.path, half, half, reference.amplitude_pct,
                             reference.phase_deg + 180, reference.detuning_hz, "qualification")
        suffix = "h" if reference.path == 0 else "p"
        specs = (
            (_qualification_key(key_prefix, f"timing_{suffix}_single_a"), (reference,)),
            (_qualification_key(key_prefix, f"timing_{suffix}_split"), (part_a, part_b)),
            (_qualification_key(key_prefix, f"timing_{suffix}_inverse"), (part_a, opposite)),
            (_qualification_key(key_prefix, f"timing_{suffix}_single_b"), (reference,)),
        )
        results = []
        for key, pulses in specs:
            result = self._acquire(AcquisitionRequest(key, pulses, reference.path,
                                   sample_count, sample_frequency_hz,
                                   initialize_state=True), qualification=True)
            if on_acquisition is not None:
                # The caller writes/counts every physical FID before another
                # qualification task may be submitted.  Sink errors propagate.
                on_acquisition(result)
            results.append(result)
        # Use the early high-SNR interval, not a tail presumed to be pure noise.
        n = min(256, sample_count, max(32, int(0.005 * sample_frequency_hz)))
        a, split, inverse, b = (x.fid_complex[:n] for x in results)
        anchor = (a + b) / 2
        scale = max(float(np.sqrt(np.mean(abs(anchor) ** 2))), 1e-12)
        drift = float(np.sqrt(np.mean(abs(a - b) ** 2))) / scale
        split_error = float(np.sqrt(np.mean(abs(split - anchor) ** 2))) / scale
        inverse_contrast = float(np.sqrt(np.mean(abs(split - inverse) ** 2))) / scale
        accepted = bool(split_error <= max(0.12, 3 * drift) and inverse_contrast >= max(0.2, 2 * drift))
        report = {"status": "CONTIGUOUS_SEGMENTS_VERIFIED" if accepted else "UNVERIFIED_TIMING",
                  "path": reference.path, "task_ids": [x.task_id for x in results],
                  "split_relative_error": split_error, "control_drift": drift,
                  "inverse_relative_contrast": inverse_contrast,
                  "criterion": "split_error <= max(0.12,3*drift); inverse_contrast >= max(0.2,2*drift)",
                  "scope": "contiguous same-channel segments only; no idle, H/P alignment or gradient proof",
                  "acquisition_count": len(results), "elapsed_s": sum(x.elapsed_s for x in results)}
        self._qualification[reference.path] = report
        if accepted:
            self._verified_paths.add(reference.path)
        else:
            self._verified_paths.discard(reference.path)
        return deepcopy(report)

    def verify_idle_gap(self, reference: PulseSpec, gap_us: int = 1000, *,
                        sample_count: int = 4000,
                        sample_frequency_hz: int = 10000,
                        key_prefix: str = "",
                        on_acquisition: Optional[Callable[[AcquisitionResult], None]] = None) -> dict[str, Any]:
        """Test whether a zero-RF segment acts as measured coherent idle time.

        A same-pulse baseline, 1×/2× padded FID, and repeat baseline are
        acquired.  Delayed FIDs must match the *unscaled* baseline FID shifted
        by the requested number of acquired sample periods, and differ from
        the unshifted FID beyond natural baseline drift.  Only tested lengths
        are admitted afterward; no conclusion is drawn for cross-channel J.
        ``on_acquisition`` receives each completed FID before the next task.
        """
        if reference.path not in (0, 1) or reference.start_us != 0 or reference.amplitude_pct <= 0:
            raise UnsupportedTiming("Idle qualification needs a known-working nonzero H/P pulse at t=0")
        _qualification_key(key_prefix, "preflight")
        _validate_request(AcquisitionRequest("idle_reference", (reference,), reference.path,
                                             sample_count, sample_frequency_hz))
        width = _integer_microseconds("reference.width_us", reference.width_us)
        gap = _integer_microseconds("gap_us", gap_us)
        if gap < 1 or width + 2 * gap > 20_000_000:
            raise UnsupportedTiming("Idle gap outside supported serialized pulse range")
        step = gap * sample_frequency_hz / 1_000_000
        if step < 1 or abs(step - round(step)) > 1e-9:
            raise UnsupportedTiming("Idle qualification gap must equal an integer number of requested FID samples")
        shift = int(round(step))
        n = min(int(round(0.003 * sample_frequency_hz)), sample_count - 2 * shift - 1)
        if n < max(20, 2 * shift):
            raise UnsupportedTiming("Insufficient early FID points for a two-gap comparison")
        suffix = "h" if reference.path == 0 else "p"
        specs = (
            (_qualification_key(key_prefix, f"idle_{suffix}_reference_a"), (reference,)),
            (_qualification_key(key_prefix, f"idle_{suffix}_{gap}us"), (reference, PulseSpec(reference.path, width, gap, 0, 0, 0, "qualified_idle"))),
            (_qualification_key(key_prefix, f"idle_{suffix}_{2*gap}us"), (reference, PulseSpec(reference.path, width, 2*gap, 0, 0, 0, "qualified_idle"))),
            (_qualification_key(key_prefix, f"idle_{suffix}_reference_b"), (reference,)),
        )
        measurements = []
        for key, pulses in specs:
            result = self._acquire(AcquisitionRequest(key, pulses, reference.path,
                                   sample_count, sample_frequency_hz,
                                   initialize_state=True), qualification=True)
            if on_acquisition is not None:
                on_acquisition(result)
            measurements.append(result)
        if any(x.timing_evidence.get("axis_status") != "EXPORT_MATCHES_REQUESTED_RATE" for x in measurements):
            accepted = False
            drift = mismatch_1 = mismatch_2 = contrast_1 = contrast_2 = float("nan")
            reason = "Exported FID x-axis does not corroborate requested sample period"
        else:
            baseline_a, delayed_1, delayed_2, baseline_b = (x.fid_complex for x in measurements)
            anchor = (baseline_a + baseline_b) / 2
            scale = max(float(np.sqrt(np.mean(abs(anchor[:n]) ** 2))), 1e-12)
            drift = float(np.sqrt(np.mean(abs(baseline_a[:n] - baseline_b[:n]) ** 2))) / scale
            mismatch_1 = float(np.sqrt(np.mean(abs(delayed_1[:n] - anchor[shift:shift+n]) ** 2))) / scale
            mismatch_2 = float(np.sqrt(np.mean(abs(delayed_2[:n] - anchor[2*shift:2*shift+n]) ** 2))) / scale
            contrast_1 = float(np.sqrt(np.mean(abs(delayed_1[:n] - anchor[:n]) ** 2))) / scale
            contrast_2 = float(np.sqrt(np.mean(abs(delayed_2[:n] - anchor[:n]) ** 2))) / scale
            allowed = max(0.18, 3 * drift)
            required = max(0.15, 3 * drift)
            accepted = bool(drift < 0.2 and mismatch_1 < allowed and mismatch_2 < allowed
                            and contrast_1 > required and contrast_2 > required)
            reason = "shift matches and differs from no-delay control" if accepted else "shift/contrast did not clear conservative control checks"
        report = {"status": "SAME_CHANNEL_IDLE_VERIFIED" if accepted else "UNVERIFIED_TIMING",
                  "path": reference.path, "tested_gap_us": [gap, 2*gap],
                  "task_ids": [x.task_id for x in measurements],
                  "baseline_drift": drift, "shift_mismatch_1": mismatch_1,
                  "shift_mismatch_2": mismatch_2, "no_shift_contrast_1": contrast_1,
                  "no_shift_contrast_2": contrast_2, "reason": reason,
                  "scope": "same-channel idle only; no cross-channel alignment or J identification",
                  "acquisition_count": 4, "elapsed_s": sum(x.elapsed_s for x in measurements)}
        if accepted:
            self._verified_idle_us[reference.path].update((gap, 2*gap))
        return report
