"""Entry point for real Gemini Lab experiments on the connected Windows PC.

Importing or offline rebuilding this module never contacts the instrument.
The single LiveHardware instance serializes every RF command and checks lock,
queue, payload, timeouts, cooldown, and total study budget.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from publish_results import publish_results
from spinq_audit.common import atomic_json, redact, utc_now
from spinq_audit.safety import HardwareLock
from spinq_benchmark.hardware import HardwareUncertain, LiveHardware
from spinq_local.core import CapabilityUnavailable, IncompleteFID, Segment, SequenceIR
from spinq_local.report import Results, bundle_complete, plots_from_results
from spinq_local.session import LocalSession


ROOT=Path(__file__).resolve().parent
DOCS=("computation_map.md","reproduction_scope.md","sources.md")


def _args():
    parser=argparse.ArgumentParser(description="Gemini Lab: real local low-level benchmark")
    parser.add_argument("--host",default="172.19.20.100")
    parser.add_argument("--port",type=int,default=8181)
    parser.add_argument("--blocks",type=int,default=10,help="3 = feasibility pilot; 10 = main block comparison")
    parser.add_argument("--seed",type=int,default=260926)
    parser.add_argument("--max-tasks",type=int,default=280)
    parser.add_argument("--max-requested-rf-us",type=float,default=20000.)
    parser.add_argument("--preflight",type=Path,default=ROOT/".benchmark-preflight.json")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--offline-rebuild",action="store_true")
    parser.add_argument("--no-upload",action="store_true")
    args=parser.parse_args()
    if not 3<=args.blocks<=20:parser.error("--blocks must be 3..20")
    if args.max_tasks<30 or not 0<args.max_requested_rf_us<=30000:
        parser.error("study software budget must be 30+ tasks and <=30000 us requested RF")
    if args.offline_rebuild and args.resume is None:
        parser.error("--offline-rebuild requires --resume")
    return args


def _copy_docs(out:Path):
    for name in DOCS:
        source=ROOT/name
        if source.is_file():shutil.copyfile(source,out/name)


def _record_status(results:Results,module:str,exc:Exception):
    if isinstance(exc,CapabilityUnavailable):status="UNVERIFIED_TIMING"
    elif isinstance(exc,IncompleteFID):status="DEPENDENCY_FAILED"
    elif isinstance(exc,ValueError) and "budget" in str(exc).lower():status="BUDGET_EXHAUSTED"
    else:status="METHOD_FAILED"
    reason=f"{type(exc).__name__}: {exc}"
    results.data["errors"].append(f"{module}: {reason}")
    results.module(module,status,reason)
    print(f"{module}: {status}: {reason}",flush=True)


def _run_guarded(results:Results,module:str,fn):
    try:return fn()
    except HardwareUncertain:raise
    except KeyboardInterrupt:raise
    except Exception as exc:
        _record_status(results,module,exc)
        return None


def _physical_control(session:LocalSession,results:Results):
    """C and 1q G use real heldout FID observables when timing was verified."""
    from spinq_local.cde import EnsemblePoint, PulseProgram, single_spin_model
    from spinq_local.control_runner import (HardwareCondition, ReadoutTask,
        fixed_multiplet_observable, run_c_physical, run_g_physical)
    if session.t90_us is None or session.spec is None or session.noise is None:
        raise ValueError("Calibrated pilot Rabi and FID components required")
    if not session.caps.segment_sequence_verified:
        results.module("C","UNVERIFIED_TIMING","Multiple H segments failed the live timing contract; no GRAPE task submitted")
        results.module("G","UNVERIFIED_TIMING","Compiler/control ablation needs verified H segment order")
        return
    rf_hz_per_percent=1/(4*session.t90_us*1e-6*100.)
    model=single_spin_model(0.,rf_hz_per_percent)
    # This uses an independent measured Rabi point as a *relative* FID
    # reference, never an invented ideal-gate process fidelity.
    pilot_coeff=np.array([session.coefficient(session.pilot_records[f"pilot_40_r{i}"])
                          for i in range(3)])
    reference_sigma=float(max(np.std(pilot_coeff,ddof=1),1e-4))
    tolerance=max(6*reference_sigma,.15*float(np.mean(np.abs(pilot_coeff))))
    if reference_sigma>=tolerance:
        results.module("C","REFERENCE_INADEQUATE","Independent pilot reference scatter exceeds frozen tolerance")
        results.module("G","REFERENCE_INADEQUATE","Independent pilot reference scatter exceeds frozen tolerance")
        return
    reference_program=PulseProgram((session.t90_us*1e-6,),(100.,),(math.pi/2,))
    task=ReadoutTask("relative_H_X90_FID",reference_program,
       "three independent pilot_40 records; relative complex component, not ideal gate truth",
       reference_sigma,tolerance,max(6*reference_sigma,.25*abs(pilot_coeff.mean())))
    observable=fixed_multiplet_observable(session.spec,session.primary_component_index,noise=session.noise,
                                          coefficient_uncertainty=reference_sigma)
    conditions=[HardwareCondition("nominal")]
    def acquire(key:str,sequence:SequenceIR):
        return session.acquire(key,sequence.segments,count=sequence.sample_count,
                               role="train",family="control")
    ensemble=[EnsemblePoint(0.,1.,.5),EnsemblePoint(5.,.95,.25),
              EnsemblePoint(-5.,1.05,.25)]
    c=run_c_physical(acquire,model,[task],conditions,observable,
        design_ensemble=ensemble,capabilities=session.caps,
        amplitude_percent=100.,n_segments=4,
        duration_s=min(160.,4*session.t90_us)*1e-6,
        verified_tick_s=10e-6,blocks=session.blocks,
        max_acquisitions=6*session.blocks,seed=session.seed)
    atomic_json(results.out/"models"/"C_physical.json",c)
    for row in c.get("rows",[]):
        results.row(module="C",method=row["method"],baseline="rectangle",
            task=row["record_key"],block=row["block"],
            data_source="raw complex FID; relative pilot reference",
            acquisitions=1,wall_seconds=row["physical_task_wall_s"],
            analysis_seconds=row["local_analysis_s"],
            rf_duration_us=row["rf_duration_us"],error=row["absolute_error"],
            ci_low=max(0.,row["absolute_error"]-1.96*row["combined_uncertainty"]),
            ci_high=row["absolute_error"]+1.96*row["combined_uncertainty"],
            tolerance=row["tolerance"],status="REFERENCE_INADEQUATE" if
                row["reference_status"]!="VALID" else "RUNNING",
            reason="relative observable, not measured gate fidelity")
    cstatus=("BUDGET_EXHAUSTED" if c["status"]=="BUDGET_EXHAUSTED" else
             "DEPENDENCY_FAILED" if c["status"]=="DEPENDENCY_FAILED" else
             "REFERENCE_INADEQUATE")
    results.module("C",cstatus,
        "Physical H pulse comparison measured; ideal 1q gate truth/readout normalization unverified",
        physical_status=c["status"],paired=c.get("paired_block_comparisons"),
        skipped=c.get("skipped"))
    # G1: a deliberately small 1q low-level ablation. G2/G3/G4 are
    # implemented locally but cannot claim the absent calibrated 2q context.
    from spinq_local.cde import rectangular_pulse
    base=rectangular_pulse(model,math.pi/2,100.)
    split=PulseProgram((session.t90_us*1e-6/2,)*2,(100.,100.),
                       (math.pi/2,math.pi/2))
    programs={"basic":base,"classical_expert":split}
    if c.get("design",{}).get("programs",{}).get("phase_GRAPE") and c.get("status")!="DEPENDENCY_FAILED":
        try:
            from spinq_local.cde import optimize_grape,rotation
            design=optimize_grape(model,rotation((1,0,0),math.pi/2),
                duration_s=min(160.,4*session.t90_us)*1e-6,n_segments=4,
                amplitude_percent=100.,ensemble=ensemble,maxiter=50,
                verified_tick_s=10e-6)
            programs["G2_local_design"]=design.program
        except Exception as exc:
            results.data["errors"].append(f"G local design: {type(exc).__name__}: {exc}")
    g=run_g_physical(acquire,programs,[task],conditions,observable,
        capabilities=session.caps,blocks=session.blocks,
        max_acquisitions=6*session.blocks,seed=session.seed+1)
    atomic_json(results.out/"models"/"G_physical.json",g)
    for row in g.get("rows",[]):
        results.row(module="G",method=row["method"],baseline="classical_expert",
            task=row["record_key"],block=row["block"],
            data_source="raw complex FID; 1q relative control ablation",
            acquisitions=1,wall_seconds=row["physical_task_wall_s"],
            analysis_seconds=row["local_analysis_s"],
            rf_duration_us=row["rf_duration_us"],error=row["absolute_error"],
            tolerance=row["tolerance"],status="RUNNING",
            reason="G3 2q DD and G4 physical response require separate calibration")
    gstatus=("BUDGET_EXHAUSTED" if g["status"]=="BUDGET_EXHAUSTED" else
             "DEPENDENCY_FAILED" if g["status"]=="DEPENDENCY_FAILED" else
             "REFERENCE_INADEQUATE")
    results.module("G",gstatus,
        "Physical 1q ablation measured; calibrated 2q coupling, contextual DD and readout basis unavailable",
        physical_status=g["status"],skipped=g.get("skipped"),
        paired=g.get("paired_block_comparisons"))


def _offline_analysis(session:LocalSession,results:Results,preflight:dict):
    from spinq_local.analysis_runner import analyze_de_fg
    analyze_de_fg(session.role_records,results.data["pilot"],results.out,
                  preflight,results)


def _physical_e(session:LocalSession,results:Results):
    """Measure generated and classical heldout pulses on the same H FID path."""
    from spinq_local.cde import PulseProgram
    from spinq_local.control_runner import (HardwareCondition, ReadoutTask,
        fixed_multiplet_observable, run_paired_control_comparison)
    path=results.out/"models"/"E_model_only.json"
    if not path.is_file():
        return
    if not session.caps.segment_sequence_verified:
        results.module("E","UNVERIFIED_TIMING",
            "Local generator may have trained, but physical multi-segment timing was not verified")
        return
    saved=json.loads(path.read_text(encoding="utf-8"))
    comparison=saved.get("comparison",[])
    gate_ids=sorted({r["gate_id"] for r in comparison})
    if not gate_ids:
        return
    gate=gate_ids[0]  # frozen first heldout gate, independent of physical outcomes
    rows={r["method"]:r for r in comparison if r["gate_id"]==gate}
    needed=("fresh_grape","nearest_library","network","network_refined")
    if not all(name in rows for name in needed):
        raise ValueError("E heldout comparison lacks required classical and generated programs")
    def program(name):
        slots=rows[name]["program"]
        return PulseProgram(tuple(float(p["duration_s"]) for p in slots),
            tuple(float(p["amplitude_percent"]) for p in slots),
            tuple(math.radians(float(p["phase_deg"])) for p in slots))
    reference_sigma=max(float(np.sqrt(np.trace(session.noise.re_im_covariance))),1e-4)
    tolerance=max(6*reference_sigma,.2*abs(session.coefficient(session.pilot_records["pilot_40_r0"])))
    if reference_sigma>=tolerance:
        results.module("E","REFERENCE_INADEQUATE","Measured FID reference scatter too large")
        return
    task=ReadoutTask("heldout_1q_target",program("fresh_grape"),
        "fresh full-GRAPE simulated candidate for heldout gate; not ideal-gate truth",
        reference_sigma,tolerance,max(6*reference_sigma,.3*tolerance))
    programs={name:program(name) for name in needed}
    def acquire(key,sequence:SequenceIR):
        return session.acquire(key,sequence.segments,count=sequence.sample_count,
            role="train",family="E_generated_heldout")
    observable=fixed_multiplet_observable(session.spec,session.primary_component_index,noise=session.noise,
                                          coefficient_uncertainty=reference_sigma)
    measured=run_paired_control_comparison(acquire,programs,[task],
        [HardwareCondition("nominal")],observable,capabilities=session.caps,
        blocks=3,max_acquisitions=18,baseline_name="fresh_grape",
        seed=session.seed+8,key_prefix="E")
    atomic_json(results.out/"models"/"E_physical.json",measured)
    for row in measured.get("rows",[]):
        results.row(module="E",method=row["method"],baseline="fresh_grape",
            task=row["record_key"],block=row["block"],
            data_source="real heldout generated H pulse FID; relative reference",
            acquisitions=1,wall_seconds=row["physical_task_wall_s"],
            analysis_seconds=row["local_analysis_s"],
            rf_duration_us=row["rf_duration_us"],error=row["absolute_error"],
            tolerance=row["tolerance"],status="REFERENCE_INADEQUATE",
            reason="relative FID agreement, not experimental gate fidelity")
    status=("BUDGET_EXHAUSTED" if measured["status"]=="BUDGET_EXHAUSTED" else
            "DEPENDENCY_FAILED" if measured["status"]=="DEPENDENCY_FAILED" else
            "REFERENCE_INADEQUATE")
    results.module("E",status,
        "Generated pulse and GRAPE/library baselines measured on heldout 1q task; independent gate truth unavailable",
        physical_status=measured["status"],gate_id=gate,
        paired=measured.get("paired_block_comparisons"))


def _finalize(out:Path,results:Results,*,upload:bool,mark_finished:bool=True):
    _copy_docs(out)
    if mark_finished:results.data["finished_utc"]=utc_now()
    results.save()
    plots_from_results(out,results.data)
    archive=bundle_complete(out)
    if upload:
        results.data["upload"]=publish_results(ROOT,archive,f"benchmark/{out.name}")
        results.save()
    print(f"Report: {out/'REPORT.md'}",flush=True)
    print(f"Archive: {archive}",flush=True)
    print(f"Upload: {results.data['upload']['status']}",flush=True)


def main():
    args=_args()
    if args.offline_rebuild:
        out=args.resume.resolve()
        data=json.loads((out/"results.json").read_text(encoding="utf-8"))
        results=Results(out,data["config"],data.get("preflight",{}))
        _finalize(out,results,upload=not args.no_upload and data.get("state")!="RUNNING",
                  mark_finished=False)
        return 0
    preflight=json.loads(args.preflight.read_text(encoding="utf-8"))
    if not preflight.get("sdk",{}).get("ready") or not preflight.get("numeric",{}).get("ready"):
        raise RuntimeError("SDK/numeric preflight did not pass in a separate process")
    out=(args.resume or ROOT/"results"/datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")).resolve()
    config={"host":args.host,"port":args.port,"blocks":args.blocks,"seed":args.seed,
        "max_tasks":args.max_tasks,"max_requested_rf_us":args.max_requested_rf_us,
        "sdk_required":"spinqlablink==1.0.2","sample_hz":10000,
        "per_task_historical_span_us":200.,
        "preparation":"makePps=True, existing relaxation_time=15; internal RF UNKNOWN",
        "limits_origin":"finite software study envelope, not certified device rating",
        "analysis_location":"same Windows host CPU"}
    results=Results(out,config,preflight)
    results.data["torch_optional_ready"]=bool(preflight.get("torch",{}).get("ready"))
    results.save()
    failed=False
    session=None
    try:
        with HardwareLock(Path("~/.spinq_live_gemini.lock")),LiveHardware(
            out,host=args.host,port=args.port,max_tasks=args.max_tasks,
            max_requested_rf_us=args.max_requested_rf_us) as hardware:
            session=LocalSession(hardware,results,blocks=args.blocks,seed=args.seed)
            print("Pilot: independent FID, noise, Rabi and acquisition lengths",flush=True)
            pilot=_run_guarded(results,"A",session.pilot)
            if pilot is not None:
                try:session.vendor_fft_control()
                except Exception as exc:
                    results.data["errors"].append(
                        f"Separate vendor FFT replica check unavailable: {type(exc).__name__}: {exc}")
                    results.save()
                _run_guarded(results,"B",session.timing_probe)
                if session.caps.zero_amplitude_delay_verified:
                    _run_guarded(results,"B",session.module_b_echo_probe)
                _run_guarded(results,"H",session.h_map)
                for block in range(args.blocks):
                    print(f"Measurement block {block+1}/{args.blocks}",flush=True)
                    for module,fn in (("A",session.module_a_block),
                                      ("B",session.module_b_block),
                                      ("H",session.module_h_block)):
                        _run_guarded(results,module,lambda fn=fn:fn(block))
                    _run_guarded(results,"F",lambda:session.collect_f_block(block))
                session.finish_a_b_h()
                _run_guarded(results,"C",lambda:_physical_control(session,results))
                _run_guarded(results,"D",lambda:_offline_analysis(session,results,preflight))
                _run_guarded(results,"E",lambda:_physical_e(session,results))
            else:
                for module in "BCDEFGH":
                    results.module(module,"DEPENDENCY_FAILED",
                        "Primary pilot failed; no valid frozen physical model")
    except (KeyboardInterrupt,HardwareUncertain) as exc:
        failed=True
        results.data["state"]="STOPPED_UNCERTAIN"
        results.data["errors"].append(f"STOP: {type(exc).__name__}: {redact(str(exc))}")
        for module in "ABCDEFGH":
            if results.data["modules"][module]["status"] in ("PENDING","RUNNING"):
                results.module(module,"STOPPED_UNCERTAIN",str(exc))
    except Exception as exc:
        failed=True
        results.data["state"]="STOPPED_UNCERTAIN"
        results.data["errors"].append(f"GLOBAL: {type(exc).__name__}: {redact(str(exc))}")
        results.data["errors"].append(traceback.format_exc(limit=4))
    else:
        results.data["state"]="COMPLETED_WITH_EXPLICIT_LIMITATIONS"
    finally:
        for module in "ABCDEFGH":
            if results.data["modules"][module]["status"] in ("PENDING","RUNNING"):
                results.module(module,"DEPENDENCY_FAILED","No validated result produced before finalization")
        _finalize(out,results,upload=not args.no_upload and not failed)
    return 2 if failed else 0


if __name__=="__main__":
    raise SystemExit(main())
