"""Version-pinned receive instrumentation for SpinQLabLink 1.0.2.

No vendor files are modified. A borrowed connection is never disconnected.
"""

from __future__ import annotations

import struct
import time
from inspect import signature
from pathlib import Path
from typing import Any

from .discovery import REFERENCE, SourceSet
from .recorder import EventRecorder

TELEMETRY = {"s_post_device_param", "s_post_lock_data", "s_post_device_info",
             "s_post_sample_calibration_data", "s_post_exp_queue_update"}
PASSIVE_OUT = {"c_user_login_req", "heartbeat_req", "c_user_logout_req"}
ACTIVE_OUT = PASSIVE_OUT | {"c_add_exp_task_req"}
EXPERIMENT = {"s_add_exp_task_res", "s_terminate_exp_task_res", "s_post_exp_removed",
              "s_post_exp_started", "s_post_exp_step_changed", "s_post_exp_data_updated",
              "s_post_exp_chart_updated_started", "s_post_exp_chart_updated",
              "s_post_exp_chart_updated_finished", "s_post_exp_finished"}
PINNED_FILES = ("spinqlablink/spinqlablink.py", "spinqlablink/connection/protocol.py",
                "spinqlablink/connection/connection.py", "spinqlablink/experiment/exp_layer_physical.py",
                "spinqlablink/experiment/experiment_base.py", "spinqlablink/experiment/ExperimentManager.py")


def verify_installed_sdk() -> None:
    sources = SourceSet()
    if sources.origin != "installed_distribution" or sources.version != "1.0.2":
        raise RuntimeError("Živý režim vyžaduje nainštalovaný SpinQLabLink 1.0.2 v tomto interpreteri.")
    for path in PINNED_FILES:
        source = sources.read(path)
        if not source or not all(snippet in source for snippet in REFERENCE["required_snippets"].get(path, [])):
            raise RuntimeError(f"SDK súbor nemá štruktúru očakávanú adaptérom: {path}")


class AuditAdapter:
    def __init__(self, client: Any, recorder: EventRecorder, *, mode: str,
                 own_task_ids: set[str] | None = None, owns_connection: bool = False,
                 fake_transport: bool = False):
        if not fake_transport:
            verify_installed_sdk()
        if mode not in {"passive", "active"}:
            raise ValueError("Adapter is only for passive/active")
        self.client = client
        self.recorder = recorder
        self.mode = mode
        self.owns_connection = owns_connection
        self.own_task_ids = own_task_ids if own_task_ids is not None else set()
        self.pending_own_ack = False
        self.pending_sequence_id: int | None = None
        self.ack_mismatch = False
        self.latest: dict[str, tuple[int, dict[str, Any]]] = {}
        self.queue: tuple[int, dict[str, Any]] | None = None
        self.lock_lost_observed = False
        self.decoder_failures = 0
        self.outgoing: list[str] = []
        self._original_decode = None
        self._original_serialize = None
        self._original_callback = None
        self._decode_wrapper = None
        self._serialize_wrapper = None
        self._callback_wrapper = None
        self._original_queue_handler = None
        self._queue_wrapper = None

    def attach(self) -> "AuditAdapter":
        if not callable(getattr(self.client.protocol, "deserialize_message", None)):
            raise RuntimeError("SDK nemá očakávaný dekóder.")
        if not callable(getattr(self.client.protocol, "serialize_message", None)):
            raise RuntimeError("SDK nemá očakávaný serializér.")
        if len(signature(self.client.protocol.deserialize_message).parameters) != 1 or len(
                signature(self.client.protocol.serialize_message).parameters) != 3:
            raise RuntimeError("Signatúra SDK dekódera/serializéra sa zmenila.")
        if self.owns_connection and (not callable(getattr(self.client.connection, "_message_callback", None)) or
                                     not hasattr(self.client.protocol, "remaining_data")):
            raise RuntimeError("SDK callback/buffer nemá overenú štruktúru.")
        self._original_decode = self.client.protocol.deserialize_message
        self._original_serialize = self.client.protocol.serialize_message

        def observed_decode(data: bytes):
            result = self._original_decode(data)
            success, message = result
            if success and isinstance(message, dict):
                if (self.mode == "active" and message.get("msg_id") == "s_add_exp_task_res" and
                    self.pending_own_ack and message.get("metadata", {}).get("sequence_id") != self.pending_sequence_id):
                    self.ack_mismatch = True
                    return False, {}  # Prevent SDK from attributing another task's ACK to ours.
                self._observe(message)
            return result

        def guarded_serialize(msg_id: str, metadata: dict, data: dict):
            allowed = ACTIVE_OUT if self.mode == "active" else PASSIVE_OUT
            if msg_id not in allowed:
                raise RuntimeError(f"Zablokovaná odchádzajúca správa: {msg_id}")
            if msg_id == "c_add_exp_task_req":
                self.pending_sequence_id = metadata.get("sequence_id")
            self.outgoing.append(msg_id)
            packed = self._original_serialize(msg_id, metadata, data)
            if not packed:
                raise RuntimeError(f"SDK nevytvorilo rámec pre povolenú správu {msg_id}")
            return packed

        self._decode_wrapper = observed_decode
        self._serialize_wrapper = guarded_serialize
        self.client.protocol.deserialize_message = observed_decode
        # SDK 1.0.2 crashes on a queue update without a locally registered
        # experiment. Preserve its behavior when one exists.
        self._original_queue_handler = self.client.handler_map.get("s_post_exp_queue_update")
        if self._original_queue_handler is not None:
            def guarded_queue(data):
                manager = getattr(self.client, "expMgr", None)
                if isinstance(data.get("queue"), list) and getattr(manager, "current_experiment", None) is not None:
                    self._original_queue_handler(data)
            self._queue_wrapper = guarded_queue
            self.client.handler_map["s_post_exp_queue_update"] = guarded_queue
        # Borrowed clients may have another owner sending commands. Do not
        # alter that owner's behavior; the auditor itself never sends.
        if self.owns_connection:
            self.client.protocol.serialize_message = guarded_serialize
            self._original_callback = self.client.connection._message_callback

            def drained_callback(data: bytes):
                self._original_callback(data)
                for _ in range(100):
                    remaining = self.client.protocol.remaining_data
                    if len(remaining) < 8:
                        return
                    magic, length = struct.unpack(">II", remaining[:8])
                    if magic != 0xCAFEBABE:
                        self.decoder_failures += 1
                        return
                    if len(remaining) < 8 + length:
                        return
                    before = len(remaining)
                    self._original_callback(b"")
                    if len(self.client.protocol.remaining_data) >= before:
                        self.decoder_failures += 1
                        return
                self.decoder_failures += 1

            self._callback_wrapper = drained_callback
            self.client.connection._message_callback = drained_callback
        return self

    def _observe(self, message: dict[str, Any]) -> None:
        msg_id = message.get("msg_id")
        now = time.monotonic_ns()
        body = message.get("chart_data") or message.get("json_data") or {}
        if not isinstance(body, dict):
            return
        # Keep server metadata's original timestamp and sequence, but redaction
        # in EventRecorder removes account/session identifiers before disk.
        if msg_id in TELEMETRY:
            self.latest[msg_id] = (now, body)
            if msg_id == "s_post_device_info" and body.get("lockState") is False:
                self.lock_lost_observed = True
            if msg_id == "s_post_exp_queue_update":
                self.queue = (now, body)
                # Do not record identifiers/names of other people's tasks.
                safe_body = {"queue_length": len(body.get("queue", [])),
                             "queue_present": "queue" in body}
                self.recorder.record(msg_id, {"metadata": message.get("metadata", {}),
                                              "json_data": safe_body})
            else:
                self.recorder.record(msg_id, message)
            return
        if self.mode != "active" or msg_id not in EXPERIMENT:
            return
        task_id = body.get("taskId")
        if msg_id == "s_add_exp_task_res" and self.pending_own_ack:
            self.pending_own_ack = False
            if body.get("code") == 0 and task_id:
                self.own_task_ids.add(str(task_id))
        if task_id is not None and str(task_id) in self.own_task_ids:
            self.recorder.record(msg_id, message)

    def detach(self) -> None:
        if self._original_decode is not None and self.client.protocol.deserialize_message is self._decode_wrapper:
            self.client.protocol.deserialize_message = self._original_decode
        if self._original_serialize is not None and self.client.protocol.serialize_message is self._serialize_wrapper:
            self.client.protocol.serialize_message = self._original_serialize
        if self._original_callback is not None and self.client.connection._message_callback is self._callback_wrapper:
            self.client.connection._message_callback = self._original_callback
        if self._original_queue_handler is not None and self.client.handler_map.get("s_post_exp_queue_update") is self._queue_wrapper:
            self.client.handler_map["s_post_exp_queue_update"] = self._original_queue_handler

    def fresh(self, msg_id: str, max_age_seconds: float) -> dict[str, Any] | None:
        record = self.latest.get(msg_id)
        if record is None or (time.monotonic_ns() - record[0]) / 1e9 > max_age_seconds:
            return None
        return record[1]


def audit_existing_client(client: Any, *, out: str | Path, duration: float = 60,
                          owns_connection: bool = False, own_task_ids: set[str] | None = None,
                          fake_transport: bool = False) -> dict[str, Any]:
    """Observe an already connected client without taking over its session.

    A borrowed client is never disconnected and no experiment is registered.
    Only telemetry is captured unless the caller explicitly supplies owned
    task IDs and the active adapter is separately approved.
    """
    if not 0 <= duration <= 3600:
        raise ValueError("duration must be 0..3600 seconds")
    recorder = EventRecorder(Path(out))
    adapter = AuditAdapter(client, recorder, mode="passive", own_task_ids=own_task_ids,
                           owns_connection=owns_connection, fake_transport=fake_transport)
    try:
        adapter.attach()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            time.sleep(min(0.2, deadline - time.monotonic()))
    finally:
        adapter.detach()
        status = recorder.close()
    return {"recorder": status, "decoder_failures": adapter.decoder_failures,
            "outgoing_by_auditor": [], "owns_connection": owns_connection}
