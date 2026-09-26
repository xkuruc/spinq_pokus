"""The only module allowed to submit real SpinQLabLink experiments."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

from spinq_audit.adapter import AuditAdapter, verify_installed_sdk
from spinq_audit.common import atomic_json, redact, utc_now
from spinq_audit.probes import _configure_physical, wait_terminal
from spinq_audit.recorder import EventRecorder
from spinq_live_suite import HISTORICAL_PHYSICAL_BASELINE, read_new_events, wait_recorded


class HardwareUncertain(RuntimeError):
    """Task might still be on the device; stop every further submission."""


def physical_request(*, pulses=None, sample_count=16000, sample_hz=10000,
                     detuning_hz=0, phase_deg=90., amplitude_pct=100., width_us=40.):
    p=copy.deepcopy(HISTORICAL_PHYSICAL_BASELINE)
    p["sampleCount"]=int(sample_count)
    p["sampleFre"]=int(sample_hz)
    p["pulse"]["hPulse"]=(copy.deepcopy(pulses) if pulses is not None else
        [{"width":float(width_us),"am":float(amplitude_pct),"phase":float(phase_deg)%360,
          "freshift":float(detuning_hz)}])
    return p


def validate_request(p, cumulative_rf_us, max_rf_us=12000.):
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
        if not (5<=q["width"]<=200 and 0<q["am"]<=100 and 0<=q["phase"]<360 and abs(q["freshift"])<=20):
            raise ValueError("Pulse outside completed-study envelope")
        rf+=q["width"]
    if rf>200 or cumulative_rf_us+rf>max_rf_us:
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


class LiveHardware:
    def __init__(self, out: Path, *, host="172.19.20.100", port=8181, account="anyword",
                 timeout_seconds=180, pause_seconds=2., max_tasks=180,
                 max_requested_rf_us=12000., exclusive_use_confirmed=True):
        self.out=out
        self.data=out/"data"
        self.data.mkdir(parents=True,exist_ok=True)
        self.host,self.port,self.account=host,port,account
        self.timeout,self.pause=timeout_seconds,pause_seconds
        self.max_tasks,self.max_rf=max_tasks,max_requested_rf_us
        self.exclusive=exclusive_use_confirmed
        self.link=self.adapter=self.recorder=None
        self.last_finished=0.
        self.halted=False
        self.journal_path=self.data/"hardware_journal.json"
        self.journal=json.loads(self.journal_path.read_text(encoding="utf-8")) if self.journal_path.exists() else {}
        if any(v.get("phase") in ("submission_attempted_unconfirmed","sent_unconfirmed","unknown") for v in self.journal.values()):
            raise HardwareUncertain("Previous session has an uncertain task; do not resume blindly")
        self.task_count=sum(v.get("phase")=="completed" for v in self.journal.values())
        self.rf_us=sum(float(v.get("requested_rf_us",0)) for v in self.journal.values() if v.get("phase") in ("completed","failed"))
        self.file_offset=0

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
            raise HardwareUncertain("Server queue is occupied")
        if not fresh and not self.exclusive:
            raise HardwareUncertain("Queue unavailable and exclusive use not confirmed")
        return {"temperature":status["temperature"],"queue_fresh":fresh,
                "queue_empty":fresh and queue[1].get("queue")==[]}

    def measure(self,key,p):
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
        rf=validate_request(p,self.rf_us,self.max_rf)
        preflight=self._preflight()
        wait=max(0.,self.pause-(time.monotonic()-self.last_finished))
        if wait:time.sleep(wait)
        from spinqlablink import ExperimentType
        exp,pars=self.link.register_experiment(ExperimentType.PHYSICAL_LAYER_EXPERIMENT)
        terminal=False
        start=time.monotonic()
        try:
            _configure_physical(pars,p)
            wire=exp.get_experiment_parameter()
            if json.loads(wire["params"])!=p: raise RuntimeError("Final SDK payload mismatch")
            self.adapter.own_task_ids.add(str(exp.id))
            self.adapter.pending_own_ack=True
            self.adapter.ack_mismatch=False
            self.journal[key]={"phase":"submission_attempted_unconfirmed","requested_rf_us":rf,
                               "task_id_before_ack":str(exp.id),"params":p,"utc":utc_now()}
            atomic_json(self.journal_path,self.journal)
            self.task_count+=1
            self.rf_us+=rf
            try:
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
                wait_recorded(self.recorder)
                # SDK result is an original decoded chart export, alongside event journal.
                result=self.link.get_experiment_result()
                graph=result.get("result",{}).get("graph",[])
                pairs=paired_fid_graphs(graph,p["sampleCount"])
                row={"key":key,"state":state,"task_id":str(exp.id),"params":p,"preflight":preflight,
                     "wall_seconds":time.monotonic()-start,"finished_utc":utc_now(),
                     "fid_pairs":pairs,"all_decoded_graphs":graph,
                     "internal_repetitions":"UNKNOWN","requested_rf_us":rf,
                     "raw_adc_confirmed":False}
                atomic_json(target,row)
                self.journal[key]["phase"]="completed"
                self.journal[key]["result_file"]=str(target.relative_to(self.out))
                atomic_json(self.journal_path,self.journal)
                return row
            except Exception as exc:
                self.halted=True
                self.journal[key]["phase"]="unknown"
                self.journal[key]["error"]=str(redact(str(exc)))
                atomic_json(self.journal_path,self.journal)
                raise HardwareUncertain("Completed task output not captured; stop submissions") from exc
        finally:
            if terminal:
                self.link.deregister_experiment()


def first_fid(row):
    pairs=row.get("fid_pairs") or []
    if not pairs: raise ValueError(f"No paired complex FID for {row.get('key')}")
    import numpy as np
    pair=pairs[0]
    return np.asarray([complex(re,im) for re,im in pair["re_im"]]),pair["axis_as_received"]
