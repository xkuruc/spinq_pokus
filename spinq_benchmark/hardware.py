"""The only module allowed to submit real SpinQLabLink experiments."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

from spinq_audit.adapter import AuditAdapter, verify_installed_sdk
from spinq_audit.common import atomic_json, redact, utc_now
from spinq_audit.probes import _configure_physical, wait_terminal
from spinq_audit.recorder import EventRecorder
from spinq_live_suite import HISTORICAL_PHYSICAL_BASELINE, wait_recorded


class HardwareUncertain(RuntimeError):
    """Task might still be on the device; stop every further submission."""


class QueuePreflightUnavailable(HardwareUncertain):
    """No task was sent: remote queue ownership cannot be established yet."""


class PreSubmissionFailure(HardwareUncertain):
    """Local checkpoint failed before run_experiment; this key was not sent."""


def same_physical_payload(expected, actual, *, rel_tol=1e-6, abs_tol=1e-6):
    """Compare serialized physical fields without JSON order or int/float noise."""
    if isinstance(expected,dict) and isinstance(actual,dict):
        return set(expected)==set(actual) and all(same_physical_payload(expected[k],actual[k],
            rel_tol=rel_tol,abs_tol=abs_tol) for k in expected)
    if isinstance(expected,list) and isinstance(actual,list):
        return len(expected)==len(actual) and all(same_physical_payload(a,b,
            rel_tol=rel_tol,abs_tol=abs_tol) for a,b in zip(expected,actual))
    if type(expected) is bool or type(actual) is bool:
        return expected is actual
    if isinstance(expected,(int,float)) and isinstance(actual,(int,float)):
        return math.isfinite(expected) and math.isfinite(actual) and math.isclose(
            expected,actual,rel_tol=rel_tol,abs_tol=abs_tol)
    return expected==actual


def physical_request(*, pulses=None, sample_count=16000, sample_hz=10000,
                     detuning_hz=0, phase_deg=90., amplitude_pct=100., width_us=40.):
    p=copy.deepcopy(HISTORICAL_PHYSICAL_BASELINE)
    p["sampleCount"]=int(sample_count)
    p["sampleFre"]=int(sample_hz)
    p["pulse"]["hPulse"]=(copy.deepcopy(pulses) if pulses is not None else
        [{"width":float(width_us),"am":float(amplitude_pct),"phase":float(phase_deg)%360,
          "freshift":float(detuning_hz)}])
    return p


def validate_request(p, cumulative_rf_us, max_rf_us=12000., *, allow_idle_probe=False):
    """Finite study envelope derived from operator's completed 40-200 us H runs.

    This is a software research budget, NOT a manufacturer safety rating.
    Unknown preparation RF is uncounted and reported as such.
    """
    base=HISTORICAL_PHYSICAL_BASELINE
    for key in ("compute_type","relaxation_time","stepList","samplePath","p_freShift",
                "p_freDemo","h_freShift","h_freDemo","makePps","sampleDelay","gradient"):
        if p.get(key)!=base[key]: raise ValueError(f"Unsupported physical field change: {key}")
    if set(p)!=set(base) or set(p.get("pulse",{}))!={"hPulse","pPulse"} or p["pulse"]["pPulse"]:
        raise ValueError("Only H physical type 0 without gradients is enabled")
    if p["sampleFre"]!=10000 or p["sampleCount"] not in (4000,8000,16000):
        raise ValueError("Only previously used sampling rate and documented 4k/8k/16k lengths")
    pulses=p["pulse"]["hPulse"]
    if not 1<=len(pulses)<=8: raise ValueError("No verified H pulse sequence")
    rf=0.
    for q in pulses:
        if set(q)!={"width","am","phase","freshift"} or any(type(q[k]) not in (int,float) or not math.isfinite(q[k]) for k in q):
            raise ValueError("Malformed pulse")
        if not (5<=q["width"]<=200 and (0<=q["am"]<=100 if allow_idle_probe else 0<q["am"]<=100)
                and 0<=q["phase"]<360 and abs(q["freshift"])<=20):
            raise ValueError("Pulse outside completed-study envelope")
        if q["am"]==0 and not allow_idle_probe:
            raise ValueError("Zero-amplitude delay not verified")
        if q["am"]>0: rf+=q["width"]
    if sum(q["width"] for q in pulses)>200 or rf>200 or cumulative_rf_us+rf>max_rf_us:
        raise ValueError("Requested RF time exceeds finite study budget")
    return rf


def paired_fid_graphs(graph, requested_count):
    """Preserve matched decoded charts, including the observed one-point truncation."""
    pairs=[]
    for block,g in enumerate(graph):
        real,imag=g.get("fidRe"),g.get("fidIm")
        if real is None or imag is None: continue
        if (len(real)!=len(imag) or len(real) not in (requested_count,requested_count-1)
                or any(a[0]!=b[0] for a,b in zip(real,imag))):
            continue
        pairs.append({"block":block,"axis_as_received":[a[0] for a in real],
                      "re_im":[[a[1],b[1]] for a,b in zip(real,imag)],
                      "actual_sample_count":len(real),
                      "requested_sample_count":requested_count})
    return pairs


def stored_graph_fields(graph, requested_count, *, compact=False):
    """Keep one event journal and raw NPZ as the local FID evidence.

    Legacy callers retain their full SDK graph export. The local benchmark
    reads the recorded chart events and saves its own validated FID NPZ, so
    another copy of every chart in each task JSON is unnecessary.
    """
    if compact:
        return {"sdk_graph_blocks": len(graph),
                "decoded_charts_in_event_journal": True,
                "duplicate_sdk_graphs_stored": False}
    return {"fid_pairs": paired_fid_graphs(graph, requested_count),
            "all_decoded_graphs": graph}


def last_complete_event_offset(path: Path) -> int:
    """Return the byte immediately after the final complete JSONL newline.

    A large chart line can be mid-write while the main thread checkpoints a
    new task. Starting at the current file size could then land inside JSON.
    Including that preceding partial line is harmless because its task ID is
    filtered; starting in its middle is not.
    """
    if not path.exists():
        return 0
    with path.open("rb") as source:
        end = source.seek(0, os.SEEK_END)
        cursor = end
        while cursor:
            size = min(4096, cursor)
            cursor -= size
            source.seek(cursor)
            block = source.read(size)
            index = block.rfind(b"\n")
            if index >= 0:
                return cursor + index + 1
    return 0


def wait_completed_fid_events(recorder, event_path: Path, task_id: str,
                              payload: dict, start_byte_offset: int, *,
                              seconds: float = 10., settle_seconds: float = .15) -> dict:
    """Boundedly drain delayed chart callbacks after SDK terminal completion.

    This is a receive-only wait. It never invokes run_experiment or a retry.
    Offset polling reads each complete JSONL line once; a partial concurrent
    write is retried from its first byte on the next poll. `assemble_fid`
    validates final task/group/path/qubit/step, matching axes and chart end.
    """
    from spinq_local.core import IncompleteFID, assemble_fid, read_task_events

    deadline = time.monotonic() + max(0., seconds)
    offset = start_byte_offset
    events = []
    reason = "No complete FID chart pair received"
    first_complete = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining < 0:
            return {"fid_capture_status": "INCOMPLETE", "fid_capture_reason": reason,
                    "event_start_byte_offset": start_byte_offset,
                    "event_end_byte_offset": offset, "captured_task_events": len(events)}
        wait_recorded(recorder, seconds=min(1., max(.05, remaining)))
        if not recorder.status()["complete"]:
            raise RuntimeError("Recorder lost events after terminal hardware state")
        new_events, offset = read_task_events(event_path, task_id, offset)
        if new_events:
            events.extend(new_events)
            first_complete = None
        try:
            assemble_fid(events, task_id, parameters_sent=payload)
        except IncompleteFID as exc:
            reason = str(exc)
            first_complete = None
        else:
            if first_complete is None:
                first_complete = time.monotonic()
            if time.monotonic() - first_complete >= settle_seconds:
                return {"fid_capture_status": "COMPLETE",
                        "event_start_byte_offset": start_byte_offset,
                        "event_end_byte_offset": offset,
                        "captured_task_events": len(events)}
        time.sleep(min(.05, max(0., deadline - time.monotonic())))


class LiveHardware:
    def __init__(self, out: Path, *, host="172.19.20.100", port=8181, account="anyword",
                 timeout_seconds=180, pause_seconds=2., max_tasks=180,
                 max_requested_rf_us=12000., exclusive_use_confirmed=True,
                 compact_result=False):
        self.out=out
        self.data=out/"data"
        self.data.mkdir(parents=True,exist_ok=True)
        self.host,self.port,self.account=host,port,account
        self.timeout,self.pause=timeout_seconds,pause_seconds
        self.max_tasks,self.max_rf=max_tasks,max_requested_rf_us
        self.exclusive=exclusive_use_confirmed
        self.compact_result=compact_result
        self.link=self.adapter=self.recorder=None
        self.last_finished=0.
        self.halted=False
        self.journal_path=self.data/"hardware_journal.json"
        self.journal=json.loads(self.journal_path.read_text(encoding="utf-8")) if self.journal_path.exists() else {}
        if any(v.get("phase") in ("submission_attempted_unconfirmed","sent_unconfirmed","unknown") for v in self.journal.values()):
            raise HardwareUncertain("Previous session has an uncertain task; do not resume blindly")
        self.task_count=sum(v.get("phase")=="completed" for v in self.journal.values())
        self.rf_us=sum(float(v.get("requested_rf_us",0)) for v in self.journal.values() if v.get("phase") in ("completed","failed"))

    def __enter__(self):
        verify_installed_sdk()
        if importlib.metadata.version("spinqlablink")!="1.0.2": raise RuntimeError("SDK version changed")
        from spinqlablink import SpinQLabLink
        password=os.environ.get("SPINQ_AUDIT_PASSWORD") or ("anyword" if self.account=="anyword" else None)
        if password is None: raise RuntimeError("Missing SPINQ_AUDIT_PASSWORD")
        try:
            self.recorder=EventRecorder(self.data,max_events=512)
            self.link=SpinQLabLink(self.host,self.port,self.account,password)
            self.adapter=AuditAdapter(self.link,self.recorder,mode="active",owns_connection=True)
            self.adapter.attach()
            self.link.connect()
            if not self.link.get_connection() or not self.link.wait_for_login(timeout=10):
                raise RuntimeError("SpinQLabLink login failed")
            deadline=time.monotonic()+20
            while time.monotonic()<deadline and self.adapter.fresh("s_post_device_info",120) is None:
                time.sleep(.1)
            atomic_json(self.data/"initial_telemetry.json",{k:redact(v[1]) for k,v in self.adapter.latest.items()
                                                     if k!="s_post_exp_queue_update"})
            return self
        except Exception:
            self.__exit__(None,None,None)
            raise

    def __exit__(self, *_):
        if self.adapter: self.adapter.detach()
        if self.recorder: self.recorder.close()
        if self.link and self.link.get_connection():
            try:self.link.disconnect()
            except Exception:pass

    def _preflight(self):
        if self.halted: raise HardwareUncertain("Hardware submission halted after uncertain state")
        if not self.link.get_connection(): raise HardwareUncertain("Connection lost")
        status=self.adapter.fresh("s_post_device_info",120)
        if not status or status.get("connected") is not True or status.get("lockState") is not True:
            raise HardwareUncertain("Fresh connected/lock status unavailable")
        if not isinstance(status.get("temperature"),(int,float)) or not math.isfinite(status["temperature"]):
            raise HardwareUncertain("Fresh finite temperature unavailable")
        if self.adapter.lock_lost_observed or self.adapter.decoder_failures or not self.recorder.status()["complete"]:
            raise HardwareUncertain("Lock/decoder/recorder failure")
        queue=self.adapter.queue
        fresh=bool(queue and (time.monotonic_ns()-queue[0])/1e9<=120)
        if fresh and queue[1].get("queue")!=[]:
            raise QueuePreflightUnavailable("Server queue is occupied")
        if not fresh and not self.exclusive:
            raise QueuePreflightUnavailable("Queue unavailable and exclusive use not confirmed")
        return {"temperature":status["temperature"],"queue_fresh":fresh,
                "queue_empty":fresh and queue[1].get("queue")==[],
                "exclusive_use_confirmed":self.exclusive,
                "queue_basis":"fresh server push" if fresh else
                              "runtime operator assertion; no fresh server queue push"}

    def measure(self,key,p,*,allow_idle_probe=False):
        """Configure -> submit -> terminal state -> paired FID and metadata.

        Completed keys are read from disk for idempotent continuation. An
        uncertain submission blocks every new task until manually resolved.
        """
        target=self.data/(key+".json")
        prior=self.journal.get(key)
        if prior and prior.get("phase")=="completed" and target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
        if prior: raise HardwareUncertain(f"Unresolved prior task {key}: {prior.get('phase')}")
        if self.task_count>=self.max_tasks: raise ValueError("Study task budget exhausted")
        rf=validate_request(p,self.rf_us,self.max_rf,allow_idle_probe=allow_idle_probe)
        preflight=self._preflight()
        wait=max(0.,self.pause-(time.monotonic()-self.last_finished))
        if wait:time.sleep(wait)
        # A queue update may arrive during the relaxation pause. Avoid even
        # registering a local experiment if ownership has since changed.
        preflight=self._preflight()
        from spinqlablink import ExperimentType
        exp,pars=self.link.register_experiment(ExperimentType.PHYSICAL_LAYER_EXPERIMENT)
        terminal=False
        submission_attempted=False
        start=time.monotonic()
        try:
            _configure_physical(pars,p,check_serialization=False)
            wire=exp.get_experiment_parameter()
            actual_payload=json.loads(wire["params"])
            if not same_physical_payload(p,actual_payload):
                atomic_json(self.data/(key+".payload_mismatch.json"),
                            {"requested":p,"sdk_serialized":actual_payload})
                raise RuntimeError("Final SDK physical payload mismatch")
            # The inter-task pause and local SDK serialization take time.
            # Recheck the live queue/status after both, immediately before
            # the durable submission checkpoint and run_experiment().
            preflight=self._preflight()
            self.adapter.own_task_ids.add(str(exp.id))
            self.adapter.pending_own_ack=True
            self.adapter.ack_mismatch=False
            # Journal the active task's seek point before run_experiment. The
            # writer may still drain older notifications, which task-ID
            # filtering safely ignores; no event for this task can precede
            # the command submission below.
            event_path = self.data / "events.jsonl"
            event_start = last_complete_event_offset(event_path)
            self.journal[key]={"phase":"submission_attempted_unconfirmed","requested_rf_us":rf,
                               "task_id_before_ack":str(exp.id),"params":p,"utc":utc_now(),
                               "event_start_byte_offset":event_start}
            try:
                atomic_json(self.journal_path,self.journal)
            except Exception as exc:
                # The SDK sends only in run_experiment(), which is below this
                # durable checkpoint. No command for this key reached it.
                self.halted=True
                self.journal.pop(key,None)
                raise PreSubmissionFailure(
                    f"Local journal write failed before submission for {key}; "
                    f"no experiment was sent ({type(exc).__name__}: {redact(str(exc))})"
                ) from exc
            self.task_count+=1
            self.rf_us+=rf
            try:
                submission_attempted=True
                self.link.run_experiment()
                self.journal[key]["phase"]="sent_unconfirmed"
                atomic_json(self.journal_path,self.journal)
                state=wait_terminal(exp,self.link.get_connection,self.recorder,self.timeout,
                    on_poll=lambda _: (_ for _ in ()).throw(HardwareUncertain("ACK mismatch"))
                    if self.adapter.ack_mismatch else None)
            except Exception as exc:
                self.halted=True
                self.journal[key]["phase"]="unknown"
                self.journal[key]["error"]=str(redact(str(exc)))
                atomic_json(self.journal_path,self.journal)
                raise HardwareUncertain(str(exc)) from exc
            terminal=True
            self.last_finished=time.monotonic()
            if state!="COMPLETED":
                self.journal[key]["phase"]="failed"
                atomic_json(self.journal_path,self.journal)
                self.halted=True
                raise HardwareUncertain(f"Hardware task {key} FAILED; stop submissions")
            try:
                capture = wait_completed_fid_events(
                    self.recorder, event_path, str(exp.id), p, event_start,
                    seconds=min(10., max(2., self.timeout / 10.)))
                # SDK result is an original decoded chart export, alongside event journal.
                result=self.link.get_experiment_result()
                graph=result.get("result",{}).get("graph",[])
                row={"key":key,"state":state,"task_id":str(exp.id),"params":p,"preflight":preflight,
                     "wall_seconds":time.monotonic()-start,"finished_utc":utc_now(),
                     "internal_repetitions":"UNKNOWN","requested_rf_us":rf,
                     "raw_adc_confirmed":False}
                row.update(capture)
                row.update(stored_graph_fields(graph,p["sampleCount"],compact=self.compact_result))
                atomic_json(target,row)
                self.journal[key]["phase"]="completed"
                self.journal[key]["result_file"]=str(target.relative_to(self.out))
                self.journal[key]["event_end_byte_offset"]=capture["event_end_byte_offset"]
                self.journal[key]["fid_capture_status"]=capture["fid_capture_status"]
                atomic_json(self.journal_path,self.journal)
                return row
            except Exception as exc:
                self.halted=True
                self.journal[key]["phase"]="unknown"
                self.journal[key]["error"]=str(redact(str(exc)))
                atomic_json(self.journal_path,self.journal)
                raise HardwareUncertain("Completed task output not captured; stop submissions") from exc
        finally:
            if terminal or not submission_attempted:
                if not submission_attempted:
                    self.adapter.own_task_ids.discard(str(exp.id))
                    self.adapter.pending_own_ack=False
                active_error=sys.exc_info()[0] is not None
                try:
                    self.link.deregister_experiment()
                except Exception as exc:
                    if not active_error: raise
                    print(f"LOCAL CLEANUP WARNING: {type(exc).__name__}: {redact(str(exc))}",flush=True)


def first_fid(row):
    pairs=row.get("fid_pairs") or []
    if not pairs: raise ValueError(f"No paired complex FID for {row.get('key')}")
    import numpy as np
    pair=pairs[0]
    return np.asarray([complex(re,im) for re,im in pair["re_im"]]),pair["axis_as_received"]
