"""Live entry points. Never imported by offline/replay execution paths."""

from __future__ import annotations

import getpass
import json
import os
import time
from pathlib import Path
from typing import Any

from .adapter import AuditAdapter, verify_installed_sdk
from .common import atomic_json, canonical, redact, utc_now
from .recorder import EventRecorder
from .safety import HardwareLock, validate_payload, verify_approval


def _password(config: dict[str, Any]) -> str:
    key = config.get("password_env", "SPINQ_AUDIT_PASSWORD")
    value = os.environ.get(key)
    if value is None:
        value = getpass.getpass("SpinQ heslo (iba v pamäti): ")
    return value


def _new_client(config: dict[str, Any]):
    verify_installed_sdk()
    from spinqlablink import SpinQLabLink
    host = config.get("host")
    port = config.get("port", 8181)
    account = config.get("account")
    if not host or not account or type(port) is not int or not 1 <= port <= 65535:
        raise RuntimeError("Konfigurácia potrebuje host, account a platný port.")
    return SpinQLabLink(host, port, account, _password(config))


def _snapshot(adapter: AuditAdapter) -> dict[str, Any]:
    now = time.monotonic_ns()
    snapshots = {}
    for msg_id, (stamp, body) in adapter.latest.items():
        if msg_id == "s_post_exp_queue_update":
            body = {"queue_length": len(body.get("queue", [])), "queue_present": "queue" in body}
        snapshots[msg_id] = {"age_seconds": (now - stamp) / 1e9, "as_received": redact(body)}
    return snapshots


def passive(config: dict[str, Any], out: Path, duration: float) -> dict[str, Any]:
    if not 1 <= duration <= 3600:
        raise ValueError("passive duration musí byť 1..3600 sekúnd")
    link = _new_client(config)
    recorder = EventRecorder(out)
    adapter = AuditAdapter(link, recorder, mode="passive", owns_connection=True)
    connected = False
    try:
        adapter.attach()
        atomic_json(out / "connection_status.json", {"attempted": True, "connected_once": False,
                                                     "attempted_utc": utc_now(), "mode": "passive"})
        if not link.connect() or not link.wait_for_login(timeout=10):
            raise RuntimeError("Pasívne prihlásenie zlyhalo; experiment sa neposlal.")
        connected = True
        atomic_json(out / "connection_status.json", {"connected_once": True, "connected_utc": utc_now(),
                                                     "mode": "passive"})
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if not link.get_connection():
                raise RuntimeError("Pasívne spojenie sa prerušilo.")
            time.sleep(min(.2, deadline - time.monotonic()))
        snapshots = _snapshot(adapter)
        atomic_json(out / "state_snapshots.json", snapshots)
        return {"connection_success": True, "hardware_experiments_executed": 0,
                "outgoing_message_types": sorted(set(adapter.outgoing)),
                "decoder_failures": adapter.decoder_failures,
                "telemetry_types_received": sorted(adapter.latest),
                "queue_length_last": len(adapter.queue[1].get("queue", [])) if adapter.queue else None}
    finally:
        adapter.detach()
        recorder.close()
        if connected and link.get_connection():
            link.disconnect()  # Only the client created here.


def _configure_physical(params: Any, requested: dict[str, Any]):
    from spinqlablink import Pulse
    params.type_setting = requested["compute_type"]
    params.relaxation_delay = requested["relaxation_time"]
    params.stepList = requested["stepList"][:]
    params.samplePath = requested["samplePath"]
    for name in ("h_freShift", "p_freShift", "h_freDemo", "p_freDemo"):
        setattr(params, name, requested[name])
    params.state_initialization = requested["makePps"]
    params.sampleFre = requested["sampleFre"]
    params.sampleCount = requested["sampleCount"]
    params.sampleDelay = requested["sampleDelay"]
    params.gradients = []
    params.pulses = []
    for path, channel in ((0, "hPulse"), (1, "pPulse")):
        for pulse in requested["pulse"][channel]:
            params.pulses.append(Pulse(path=path, width=pulse["width"],
                                       amplitude=pulse["am"], phase=pulse["phase"],
                                       detuning=pulse["freshift"]))
    actual = params.get_parameters()
    if canonical(actual) != canonical(requested):
        raise RuntimeError("SDK serializoval odlišný finálny payload; nič sa neposlalo.")


def wait_terminal(experiment: Any, connected: Any, recorder: EventRecorder,
                  seconds: float, on_poll: Any = None) -> str:
    """Deadline-bound state wait. FAILED is never counted as success."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = experiment.get_status()
        if state in ("COMPLETED", "FAILED"):
            return state
        if not connected():
            raise RuntimeError("Spojenie stratené; stav úlohy neznámy. Bez retry.")
        if not recorder.status()["complete"]:
            raise RuntimeError("Recorder stratil dáta; ďalšie merania sa neposielajú.")
        if on_poll:
            on_poll(state)
        time.sleep(min(.2, max(0, deadline - time.monotonic())))
    raise TimeoutError("Deadline vypršal; úloha môže pokračovať na hardvéri. Bez retry/abort.")


def active(config: dict[str, Any], baseline: dict[str, Any], plan: dict[str, Any],
           approval: dict[str, Any], out: Path, *, allow_hardware: bool,
           max_experiments: int) -> dict[str, Any]:
    if allow_hardware is not True:
        raise RuntimeError("Chýba explicitné --allow-hardware.")
    verify_approval(approval, plan, config, baseline, max_experiments)
    verify_installed_sdk()
    # One local process owns writes. A remote operator may still exist;
    # the fresh queue condition below is an independent gate.
    lock_path = Path(config.get("hardware_lock_path", "~/.spinq_audit_hardware.lock"))
    with HardwareLock(lock_path):
        from spinqlablink import ExperimentType
        link = _new_client(config)
        recorder = EventRecorder(out)
        adapter = AuditAdapter(link, recorder, mode="active", owns_connection=True)
        submitted = 0
        records: list[dict[str, Any]] = []
        connected = False
        try:
            adapter.attach()
            atomic_json(out / "connection_status.json", {"attempted": True, "connected_once": False,
                                                         "attempted_utc": utc_now(), "mode": "active"})
            if not link.connect() or not link.wait_for_login(timeout=10):
                raise RuntimeError("Prihlásenie zlyhalo; žiadny experiment sa neposlal.")
            connected = True
            atomic_json(out / "connection_status.json", {"connected_once": True, "connected_utc": utc_now(),
                                                         "mode": "active"})
            # Queue/status messages are server push notifications. Absence or
            # staleness blocks writes; no undocumented polling is performed.
            wait_deadline = time.monotonic() + 15
            while time.monotonic() < wait_deadline and (
                    adapter.fresh("s_post_device_info", 15) is None or adapter.queue is None):
                time.sleep(.2)
            for test_index, test in enumerate(plan["tests"]):
                record = {"id": test["id"], "phase": "not_submitted", "created_utc": utc_now()}
                records.append(record)
                atomic_json(out / "experiments.json", records)
                if test_index:
                    pause = config["limits"]["min_inter_experiment_seconds"]
                    time.sleep(pause)
                if submitted >= max_experiments:
                    raise RuntimeError("Rozpočet experimentov sa vyčerpal.")
                problems = validate_payload(test["params"], baseline, config,
                                            planned_count=len(plan["tests"]))
                if problems:
                    raise RuntimeError("Finálny payload neprešiel kontrolou: " + ", ".join(problems))
                status = adapter.fresh("s_post_device_info", config.get("status_max_age_seconds", 15))
                if adapter.lock_lost_observed:
                    raise RuntimeError("Počas auditu sa stratil lock; ďalšie merania vyžadujú zásah obsluhy.")
                if not status or status.get("connected") is not True or status.get("lockState") is not True:
                    raise RuntimeError("Stav alebo lock chýba, je starý alebo nevyhovuje; nové meranie zablokované.")
                temperature = status.get("temperature")
                if type(temperature) not in (int, float) or not (
                    config["limits"]["min_temperature_c"] <= temperature <= config["limits"]["max_temperature_c"]):
                    raise RuntimeError("Teplota chýba alebo je mimo schváleného intervalu; nové meranie zablokované.")
                if adapter.queue is None or (time.monotonic_ns() - adapter.queue[0]) / 1e9 > config.get("queue_max_age_seconds", 15):
                    raise RuntimeError("Chýba čerstvý stav fronty; nové meranie zablokované.")
                if adapter.queue[1].get("queue") != []:
                    raise RuntimeError("Fronta nie je prázdna; cudzie úlohy sa neovládajú.")
                if not recorder.status()["complete"] or adapter.decoder_failures:
                    raise RuntimeError("Neúplný záznam alebo parser chyba; nové meranie zablokované.")
                experiment, params = link.register_experiment(ExperimentType.PHYSICAL_LAYER_EXPERIMENT)
                try:
                    _configure_physical(params, test["params"])
                    payload = experiment.get_experiment_parameter()
                    if canonical(json.loads(payload["params"])) != canonical(test["params"]):
                        raise RuntimeError("Finálny experiment má iné params než schválený plán.")
                    if payload["params"] != test["params_wire_json"]:
                        raise RuntimeError("Finálny serializovaný JSON sa od schváleného plánu líši.")
                    adapter.own_task_ids.add(str(experiment.id))
                    adapter.pending_own_ack = True
                    record["phase"] = "submission_attempted_unconfirmed"
                    atomic_json(out / "experiments.json", records)
                    link.run_experiment()
                    submitted += 1
                    record["phase"] = "send_enqueued_server_unconfirmed"
                    def poll(_state):
                        if adapter.ack_mismatch:
                            raise RuntimeError("Odpoveď tasku nemá naše sequence_id; vlastníctvo úlohy je nejasné. Bez retry.")
                        if str(experiment.id) in adapter.own_task_ids and not adapter.pending_own_ack:
                            record["phase"] = "server_accepted_or_running"
                    try:
                        state = wait_terminal(experiment, link.get_connection, recorder,
                                              min(float(test["deadline_seconds"]), 300), poll)
                    except TimeoutError:
                        record["phase"] = "deadline_exceeded_task_may_continue"
                        atomic_json(out / "experiments.json", records)
                        raise
                    record["phase"] = "confirmed_finished" if state == "COMPLETED" else "confirmed_failed"
                    record["state"] = state
                    record["task_id"] = str(experiment.id)
                    record["result_summary"] = {"graph_blocks": len(experiment.get_result().get("result", {}).get("graph", []))}
                    atomic_json(out / "experiments.json", records)
                    if record["state"] != "COMPLETED":
                        raise RuntimeError("Experiment FAILED; ďalšie merania zablokované.")
                finally:
                    # Deregistration is local only, and only safe after final state.
                    if record.get("phase") in {"confirmed_finished", "confirmed_failed"}:
                        link.deregister_experiment()
                # A fresh empty queue is required before the next iteration.
            atomic_json(out / "state_snapshots.json", _snapshot(adapter))
            return {"hardware_experiments_executed": submitted, "records": records,
                    "outgoing_message_types": sorted(set(adapter.outgoing)),
                    "decoder_failures": adapter.decoder_failures}
        finally:
            atomic_json(out / "experiments.json", records)
            atomic_json(out / "state_snapshots.json", _snapshot(adapter))
            adapter.detach()
            recorder.close()
            if connected and link.get_connection():
                link.disconnect()  # Closing client is NOT an abort of hardware.
