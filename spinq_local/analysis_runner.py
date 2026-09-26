"""Offline D/E/F/G analysis of saved Gemini Lab exported complex FIDs.

This module has no SpinQLabLink import and makes no hardware/network calls.
Only `records_by_role` can supply measured data. Simulated target unitaries,
label optimizations and ideal circuit rewrites are recorded as model-only;
they never become experimental gate fidelities in comparison.csv.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from spinq_audit.common import atomic_json
from .core import RawFIDRecord
from .signal import estimate_noise, validate_axis
from . import cde, fg


def _record_groups(records: Sequence[RawFIDRecord]):
    groups=defaultdict(list)
    for record in records:
        session=str(record.metadata.get("measurement_block", ""))
        family=str(record.metadata.get("setting_family", ""))
        if not session or not family:
            raise ValueError(f"{record.key}: measurement_block and setting_family metadata required")
        groups[(session,family)].append(record)
    return groups


def _validated_roles(records_by_role):
    roles={role:list(records_by_role.get(role,())) for role in ("pilot","train","validation","test")}
    keys={}
    for role,records in roles.items():
        for record in records:
            if not isinstance(record,RawFIDRecord):
                raise TypeError("analysis_runner accepts only saved RawFIDRecord objects")
            validate_axis(record)
            if record.key in keys:
                raise ValueError(f"{record.key}: same acquisition assigned to {keys[record.key]} and {role}")
            keys[record.key]=role
            if record.metadata.get("measurement_block") is None or not record.metadata.get("setting_family"):
                raise ValueError(f"{record.key}: independent session/family labels missing")
    return roles


def _complex_matrix(data, name):
    if not isinstance(data,Mapping) or "re" not in data or "im" not in data:
        raise ValueError(f"{name} needs independently calibrated re/im matrices")
    value=np.asarray(data["re"],float)+1j*np.asarray(data["im"],float)
    if value.ndim!=2 or value.shape[0]!=value.shape[1] or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite square complex matrix")
    return value


def _complex_scalar(value, name):
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError(f"{name} needs measured [Re, Im]")
    scalar=complex(float(value[0]),float(value[1]))
    if not np.isfinite(scalar):
        raise ValueError(f"{name} is not finite")
    return scalar


def _calibrated_one_spin_model(pilot: Mapping):
    settings=pilot.get("d_model")
    if not isinstance(settings,Mapping):
        raise ValueError("D requires pilot.d_model with measured 1q state, detector and receiver gauge")
    if int(settings.get("n_spins",0))!=1:
        raise ValueError("D only enables the 1q model until the full measured multi-spin model is supplied")
    rf=float(settings["rf_hz_per_percent"])
    delta=float(settings["detuning_hz"])
    model=cde.single_spin_model(delta,rf)
    initial=_complex_matrix(settings.get("initial_state"),"initial_state")
    detector=_complex_matrix(settings.get("detector"),"detector")
    acq=_complex_matrix(settings.get("acq_hamiltonian_hz"),"acq_hamiltonian_hz")
    if any(matrix.shape!=(2,2) for matrix in (initial,detector,acq)):
        raise ValueError("1q state, detector and acquisition Hamiltonian must be 2x2")
    readout=cde.ReadoutPhysics(acq,_complex_scalar(settings.get("readout_gain"),"readout_gain"),
        _complex_scalar(settings.get("readout_background",[0,0]),"readout_background"),
        float(settings["decay_s_inv"]))
    if not settings.get("preparation_identifiability_verified") or not settings.get("receiver_gauge_verified"):
        raise ValueError("Initial-state identifiability and receiver phase/scale gauge must be verified")
    return model,readout,initial,detector,settings


def _program_from_record(record: RawFIDRecord):
    pulses=record.parameters_sent.get("pulse",{}).get("hPulse")
    if not isinstance(pulses,list) or not pulses:
        raise ValueError(f"{record.key}: original H pulse payload missing")
    if any(abs(float(p.get("freshift",0)))>1e-12 for p in pulses):
        raise ValueError(f"{record.key}: per-segment detuning unsupported in this D model")
    return cde.PulseProgram(tuple(float(p["width"])*1e-6 for p in pulses),
        tuple(float(p["am"]) for p in pulses),
        tuple(math.radians(float(p["phase"])) for p in pulses))


def _fid_example(record, initial, detector):
    return cde.FIDExample(_program_from_record(record), initial, detector,
        record.time_seconds,record.fid,str(record.metadata["setting_family"]),
        str(record.metadata["measurement_block"]),record.key)


def _single_h_pulse(record):
    pulses=record.parameters_sent.get("pulse",{}).get("hPulse",())
    if len(pulses)!=1 or float(pulses[0].get("am",0))<=0 or abs(float(pulses[0].get("freshift",0)))>1e-12:
        return None
    return pulses[0]


def _effective_one_spin_model(roles,pilot,out):
    """Fit an explicitly assumed 1q response from pilot FID only.

    |0><0|, Ix+iIy and zero pulse detuning are gauge/model choices, never
    independent state or receiver calibration. Held-out phase/amplitude pilot
    records must pass before D may use this predictor on role train/test FIDs.
    """
    rabi=pilot.get("rabi",{})
    if not isinstance(rabi,Mapping) or "t90_us" not in rabi or "amplitude_pct" not in rabi:
        raise ValueError("Effective D needs measured pilot Rabi t90 and its commanded amplitude")
    t90=float(rabi["t90_us"]);amplitude=float(rabi["amplitude_pct"])
    if not 0<t90<1e6 or not 0<amplitude<=100:
        raise ValueError("Effective D pilot Rabi values invalid")
    rate=1/(4*t90*1e-6*amplitude)
    candidates=[];holdout=[]
    for record in roles["pilot"]:
        pulse=_single_h_pulse(record)
        if pulse is None:continue
        width=float(pulse["width"]);phase=float(pulse["phase"])%360
        amp=float(pulse["am"])
        family=str(record.metadata.get("setting_family",""))
        if (family in ("repeat","rabi") and abs(amp-amplitude)<1e-6
                and abs(phase-90)<1e-6 and abs(width-120)>1e-6):
            candidates.append(record)
        elif family=="rf_map" and (abs(amp-amplitude)>1e-6 or abs(phase-90)>1e-6):
            holdout.append(record)
    if len({float(_single_h_pulse(r)["width"]) for r in candidates})<3:
        raise ValueError("Effective D needs >=3 distinct pilot Rabi widths excluding 120us test family")
    conditions={(round(float(_single_h_pulse(r)["am"]),4),
                 round(float(_single_h_pulse(r)["phase"])%360,4)) for r in holdout}
    if len(conditions)<2:
        raise ValueError("Effective D needs >=2 held-out pilot phase/amplitude conditions")
    noise=pilot.get("noise",{})
    covariance=np.asarray(noise.get("covariance"),float) if isinstance(noise,Mapping) else np.asarray([])
    if covariance.shape!=(2,2) or np.linalg.eigvalsh(covariance).min()<=0:
        raise ValueError("Effective D needs independent-repeat Re/Im noise covariance")
    noise_rms=math.sqrt(float(np.trace(covariance)))
    model=cde.single_spin_model(0.,rate)
    initial=np.array([[1.,0.],[0.,0.]],complex)
    detector=cde.spin_operator("x",0,1)+1j*cde.spin_operator("y",0,1)
    fit_examples=[_fid_example(r,initial,detector) for r in candidates]
    validation_examples=[_fid_example(r,initial,detector) for r in holdout]
    sample_hz=validate_axis(candidates[0]).sample_hz
    if any(not np.isclose(validate_axis(r).sample_hz,sample_hz,rtol=1e-8)
           for r in [*candidates,*holdout]):
        raise ValueError("Effective D pilot and validation FIDs use different sample clocks")
    center=float(pilot.get("reference_frequency_hz",float("nan")))
    if not np.isfinite(center):
        raise ValueError("Effective D needs independently fitted pilot FID frequency")
    band=None
    for item in pilot.get("multiplet_bands_hz",()):
        if len(item)==2 and float(item[0])<=center<=float(item[1]):
            band=(float(item[0]),float(item[1]));break
    if band is None or not -sample_hz/2<band[0]<band[1]<sample_hz/2:
        raise ValueError("Effective D needs frozen pilot component band around frequency")

    def indices(example):
        n=len(example.fid)
        return np.unique(np.r_[np.linspace(0,min(n,2048)-1,min(192,n),dtype=int),
                                  np.linspace(0,n-1,min(192,n),dtype=int)])

    fit_indices=[indices(ex) for ex in fit_examples]
    validation_indices=[indices(ex) for ex in validation_examples]
    whiten=np.linalg.inv(np.linalg.cholesky(covariance))

    def linear_solve(freq,decay):
        readout=cde.ReadoutPhysics(freq*cde.spin_operator("z",0,1),1+0j,0j,decay)
        basis=np.concatenate([cde.predict_fid_physics(ex,model,readout,indices=ix)
                              for ex,ix in zip(fit_examples,fit_indices)])
        target=np.concatenate([ex.fid[ix] for ex,ix in zip(fit_examples,fit_indices)])
        coeff=np.linalg.lstsq(np.column_stack((basis,np.ones(len(basis)))),target,rcond=None)[0]
        return coeff,basis,target

    def residual(parameters):
        freq,decay=parameters
        (gain,background),basis,target=linear_solve(freq,decay)
        difference=gain*basis+background-target
        return (np.column_stack((difference.real,difference.imag))@whiten.T).ravel()

    fits=[]
    for start_frequency in (center,band[0]+.25*(band[1]-band[0]),band[0]+.75*(band[1]-band[0])):
        fits.append(least_squares(residual,[start_frequency,80.],
            bounds=([band[0],0.],[band[1],2000.]),max_nfev=50))
    best=min(fits,key=lambda f:float(np.dot(f.fun,f.fun)))
    frequency,decay=map(float,best.x)
    if min(frequency-band[0],band[1]-frequency)<.01*(band[1]-band[0]):
        raise ValueError("Effective D acquisition-frequency fit hit frozen band edge")
    (gain,background),_,_=linear_solve(frequency,decay)
    if abs(gain)<1e-9:
        raise ValueError("Effective D receiver scale collapsed")
    readout=cde.ReadoutPhysics(frequency*cde.spin_operator("z",0,1),
                               complex(gain),complex(background),decay)
    heldout=[]
    for record,example,ix in zip(holdout,validation_examples,validation_indices):
        observed=example.fid[ix]
        predicted=cde.predict_fid_physics(example,model,readout,indices=ix)
        error=float(np.sqrt(np.mean(np.abs(predicted-observed)**2)))
        signal=float(np.sqrt(np.mean(np.abs(observed-background)**2)))
        threshold=max(4*noise_rms,.15*signal)
        passes=signal>=5*noise_rms and error<=threshold
        heldout.append({"record_id":record.key,"amplitude_pct":float(_single_h_pulse(record)["am"]),
            "phase_deg":float(_single_h_pulse(record)["phase"]),"complex_fid_rmse":error,
            "signal_rms":signal,"threshold":threshold,"passes":bool(passes)})
    artifact={"scope":"effective relative one-spin FID predictor; physical preparation/gate fidelity unverified",
        "assumed_initial_state":"|0><0| model gauge, not independently reconstructed rho",
        "assumed_detector":"Ix+iIy, not independently measured detector matrix",
        "assumed_control_detuning_hz":0.,"rf_hz_per_percent_from_pilot_t90":rate,
        "calibration_record_ids":[r.key for r in candidates],"validation_rows":heldout,
        "acquisition_frequency_hz":frequency,"decay_per_s":decay,
        "gain_re":float(gain.real),"gain_im":float(gain.imag),
        "background_re":float(background.real),"background_im":float(background.imag),
        "noise_rms_from_independent_repeats":noise_rms,
        "fit_whitened_rss":float(np.dot(best.fun,best.fun)),
        "validation_rule":"each distinct pilot phase/amplitude: background-subtracted signal>=5 noise RMS, residual<=max(4 noise RMS, 15% signal RMS)"}
    atomic_json(out/"models"/"D_effective_pilot_model.json",artifact)
    if not all(row["passes"] for row in heldout):
        raise ValueError("Effective 1q model failed held-out phase/amplitude FID residual gate")
    settings={"amplitude_scale_percent":amplitude,"max_relative_rf":.2,
              "max_phase_rad":.2,"max_detuning_hz":20.,
              "graybox_epochs":20,"blackbox_epochs":20,
              "bounds_scope":"local model correction capacity, not hardware operating limits"}
    return model,readout,initial,detector,settings,artifact


def _row(results, module, method, baseline, task, block, error, *, acquisitions,
         analysis_seconds=0., reason="", status="SUCCESS_VALIDATED", data_source="measured_exported_FID"):
    results.row(module=module,method=method,baseline=baseline,task=task,block=block,
        data_source=data_source,acquisitions=acquisitions,wall_seconds=None,
        analysis_seconds=float(analysis_seconds),design_seconds=None,rf_duration_us=None,
        error=float(error),ci_low=None,ci_high=None,tolerance=None,
        status=status,reason=reason)


def _analyze_d(roles, pilot, out, preflight, results):
    try:
        effective_artifact=None
        if isinstance(pilot.get("d_model"),Mapping):
            model,readout,initial,detector,settings=_calibrated_one_spin_model(pilot)
            model_scope="independently calibrated one-spin state, detector and receiver gauge"
            train_records=list(roles["train"])
            test_records=list(roles["test"])
        else:
            (model,readout,initial,detector,settings,effective_artifact)=_effective_one_spin_model(
                roles,pilot,out)
            model_scope=("effective relative one-spin FID predictor; assumed |0><0| and Ix+iIy; "
                         "preparation identity and experimental gate fidelity unknown")
            train_records=[r for r in [*roles["train"],*roles["validation"]]
                           if str(r.metadata.get("setting_family")) in ("F_train_40","F_validation_80")]
            test_records=[r for r in roles["test"]
                          if str(r.metadata.get("setting_family"))=="F_test_120"]
        if not train_records or not test_records:
            raise ValueError("D requires separate F train/validation widths 40/80 and held-out test width 120")
        train=[_fid_example(r,initial,detector) for r in train_records]
        test=[_fid_example(r,initial,detector) for r in test_records]
        if {e.session for e in train}&{e.session for e in test} or {e.family for e in train}&{e.family for e in test}:
            raise ValueError("D train/test sessions or setting families overlap")
        amp_scale=float(settings["amplitude_scale_percent"])
        caps=(float(settings["max_relative_rf"]),float(settings["max_phase_rad"]),
              float(settings["max_detuning_hz"]))
        max_slots=max(len(ex.program.duration_s) for ex in [*train,*test])
        duration_scale=max(dt for ex in train for dt in ex.program.duration_s)
        time_scale=max(float(ex.times_s[-1]) for ex in train)
        t0=perf_counter()
        polynomial=cde.fit_polynomial_rf(train,model,readout,
            amplitude_scale_percent=amp_scale,max_relative_rf=caps[0],
            max_phase_rad=caps[1],max_detuning_hz=caps[2],max_points=48,max_nfev=50)
        train_seconds=perf_counter()-t0
        np.savez_compressed(out/"models"/"D_polynomial_rf.npz",coefficients=polynomial.coefficients,
            train_record_ids=np.asarray(polynomial.train_record_ids),
            rf_hz_per_percent=model.rf_hz_per_percent,
            detuning_hz=float(settings.get("detuning_hz",0.)),model_scope=model_scope)
        graybox=blackbox=None
        neural_failure=None
        if preflight.get("torch",{}).get("ready"):
            try:
                graybox=cde.fit_graybox(train,model,readout,
                    amplitude_scale_percent=amp_scale,duration_scale_s=duration_scale,
                    correction_caps=caps,epochs=int(settings.get("graybox_epochs",30)),max_points=48)
                blackbox=cde.fit_blackbox(train,max_slots=max_slots,
                    amplitude_scale_percent=amp_scale,duration_scale_s=duration_scale,
                    time_scale_s=time_scale,epochs=int(settings.get("blackbox_epochs",30)),max_points=48)
                import torch
                torch.save({"network_state":graybox.network.state_dict(),
                    "history":graybox.history,"train_record_ids":graybox.train_record_ids,
                    "model_scope":model_scope},
                    out/"models"/"D_graybox.pt")
                torch.save({"network_state":blackbox.network.state_dict(),
                    "history":blackbox.history,"train_record_ids":blackbox.train_record_ids,
                    "model_scope":"unconstrained local FID predictor baseline"},
                    out/"models"/"D_blackbox.pt")
            except (RuntimeError,ValueError) as exc:
                neural_failure=f"{type(exc).__name__}: {exc}"
        else:
            neural_failure=preflight.get("torch",{}).get("reason","CPU PyTorch preflight unavailable")
        predictions=[]
        for record,ex in zip(test_records,test):
            ids=np.unique(np.linspace(0,len(ex.fid)-1,min(128,len(ex.fid)),dtype=int))
            target=ex.fid[ids]
            candidate={"fixed_physics":cde.predict_fid_physics(ex,model,readout,indices=ids),
                       "polynomial_rf":cde.predict_fid_physics(ex,model,readout,polynomial.correction,ids)}
            if graybox is not None:
                candidate["graybox"]=cde.predict_fid_graybox(ex,graybox,ids)
            if blackbox is not None:
                candidate["blackbox_mlp"]=cde.predict_fid_blackbox(ex,blackbox,ids)
            for method,estimated in candidate.items():
                error=float(np.mean(np.abs(estimated-target)**2))
                predictions.append({"record_id":record.key,"measurement_block":ex.session,
                    "family":ex.family,"method":method,"complex_fid_mse":error,
                    "points":len(ids),"source":"held-out measured exported FID"})
                _row(results,"D",method,"fixed_physics","heldout_fid_prediction:"+record.key,
                    ex.session,error,acquisitions=len(train)+1,
                    analysis_seconds=train_seconds if method=="polynomial_rf" else 0.,
                    reason="independent held-out FID prediction; no experimental gate fidelity; "+model_scope)
        atomic_json(out/"models"/"D_fid_prediction.json",{
            "test_rows":predictions,"model":model_scope,
            "training_record_ids":[r.key for r in train_records],
            "validation_record_ids":([r["record_id"] for r in effective_artifact["validation_rows"]]
                                     if effective_artifact else []),
            "polynomial_training_record_ids":list(polynomial.train_record_ids),
            "graybox_training_record_ids":list(graybox.train_record_ids) if graybox else [],
            "neural_failure":neural_failure,
            "hardware_gate_validation":"NOT_PERFORMED_BY_OFFLINE_ANALYSIS"})
        status="DEPENDENCY_FAILED"  # predictive validation alone is not physical gate control validation
        results.module("D",status,
            "Held-out real FID prediction computed; control improvement requires independent pulse measurements"
            if graybox is not None and blackbox is not None else
            "Classical held-out FID fit completed; neural dependency or control validation absent",
            prediction_rows=len(predictions),neural_failure=neural_failure,
            hardware_gate_validation="PENDING",model_scope=model_scope,
            effective_pilot_validation=(effective_artifact["validation_rows"] if effective_artifact else None))
        return {"status":status,"prediction_rows":len(predictions),"neural_failure":neural_failure}
    except (KeyError,TypeError,ValueError,RuntimeError) as exc:
        reason=f"{type(exc).__name__}: {exc}"
        results.module("D","DEPENDENCY_FAILED",reason)
        return {"status":"DEPENDENCY_FAILED","reason":reason}


def _measured_rf_model(pilot, roles):
    settings=pilot.get("e_model",{})
    if not isinstance(settings,Mapping):settings={}
    rabi=pilot.get("rabi",{})
    if not isinstance(rabi,Mapping):rabi={}
    all_records=[*roles["pilot"],*roles["train"]]
    if not all_records:raise ValueError("E pulse geometry needs a completed physical H record")
    pulse_candidates=[p for record in all_records for p in record.parameters_sent.get("pulse",{}).get("hPulse",[])
                      if float(p.get("am",0))>0]
    if not pulse_candidates:raise ValueError("No completed H pulse provides E geometry")
    if "rf_hz_per_percent" in settings:
        rate=float(settings["rf_hz_per_percent"])
        provenance="pilot.e_model.rf_hz_per_percent"
    elif "rf_map" in pilot and isinstance(pilot["rf_map"],Mapping) and "rf_hz_per_percent" in pilot["rf_map"]:
        rate=float(pilot["rf_map"]["rf_hz_per_percent"])
        provenance="pilot.rf_map.rf_hz_per_percent"
    elif "d_model" in pilot and "rf_hz_per_percent" in pilot["d_model"]:
        rate=float(pilot["d_model"]["rf_hz_per_percent"])
        provenance="pilot.d_model.rf_hz_per_percent"
    elif "t90_us" in rabi or "t90_us" in pilot:
        t90=float(rabi.get("t90_us",pilot.get("t90_us")))
        if "amplitude_pct" in rabi:
            rabi_amp=float(rabi["amplitude_pct"])
            amp_source="pilot.rabi.amplitude_pct"
        elif "t90_amplitude_percent" in pilot:
            rabi_amp=float(pilot["t90_amplitude_percent"])
            amp_source="pilot.t90_amplitude_percent"
        else:
            observed={float(p["am"]) for p in pulse_candidates}
            if len(observed)!=1:
                raise ValueError("Rabi t90 command amplitude ambiguous across completed H records")
            rabi_amp=observed.pop()
            amp_source="unique completed H pulse amplitude; Rabi amplitude attribution unverified"
        rate=1/(4*t90*1e-6*rabi_amp)
        provenance=f"pilot.rabi.t90_us and {amp_source}; effective model RF rate, not independent RF calibration"
    else:
        raise ValueError("E needs measured RF rate or t90 with its command amplitude")
    if not np.isfinite(rate) or rate<=0:
        raise ValueError("Measured RF rate invalid")
    detuning_measured="detuning_hz" in settings or "detuning_hz" in pilot
    delta=float(settings.get("detuning_hz",pilot.get("detuning_hz",0.)))
    if not np.isfinite(delta):raise ValueError("Measured detuning invalid")
    if not detuning_measured:
        provenance+="; rotating-frame detuning=0 is a MODEL-ONLY ASSUMPTION, not measured resonance"
    model=cde.single_spin_model(delta,rate)
    chosen=min(pulse_candidates,key=lambda p:abs(float(p["width"])-40))
    amplitude=float(chosen["am"])
    duration=float(chosen["width"])*1e-6
    if duration<=0 or amplitude<=0:raise ValueError("Measured pulse geometry invalid")
    return model,amplitude,duration,provenance


def _analyze_e(roles,pilot,out,preflight,results):
    try:
        if not preflight.get("torch",{}).get("ready"):
            raise RuntimeError("CPU PyTorch preflight failed: "+str(preflight.get("torch",{}).get("reason")))
        model,amplitude,duration,provenance=_measured_rf_model(pilot,roles)
        settings=pilot.get("e_model",{})
        settings=settings if isinstance(settings,Mapping) else {}
        counts=settings.get("target_counts",[8,3,3])
        if len(counts)!=3 or any(int(v)<1 for v in counts):
            raise ValueError("E needs train/validation/test target counts")
        n_segments=int(settings.get("segments",4))
        if n_segments<1 or duration/n_segments<5e-6:
            raise ValueError("E segment geometry below prior completed 5 us software envelope")
        targets=cde.sample_gate_targets(*[int(v) for v in counts],seed=int(settings.get("seed",37)))
        detuning_span=float(settings.get("pilot_detuning_span_hz",1.))
        rf_span=float(settings.get("pilot_rf_scale_span",.03))
        ensemble_provenance=("measured pilot uncertainty range" if
            "pilot_detuning_span_hz" in settings and "pilot_rf_scale_span" in settings
            else "illustrative model-only uncertainty grid; no measured robustness claim")
        if not (0<rf_span<.3 and 0<detuning_span<1000):
            raise ValueError("E model ensemble needs bounded pilot-derived ranges")
        ensemble=[cde.EnsemblePoint(d,s) for d in (-detuning_span,0,detuning_span)
                  for s in (1-rf_span,1,1+rf_span)]
        heldout=[cde.EnsemblePoint(d,s) for d in (-.6*detuning_span,.6*detuning_span)
                 for s in (1-.6*rf_span,1+.6*rf_span)]
        start=tuple(float(v) for v in np.linspace(0,2*math.pi,n_segments,endpoint=False))
        collection=cde.generate_grape_labels(targets,model,target_spin=0,
            duration_s=duration,amplitude_percent=amplitude,common_start_phases_rad=start,
            ensemble=ensemble,maxiter=int(settings.get("label_maxiter",25)),
            accept_infidelity=float(settings.get("label_max_infidelity",.9)))
        train=[x for x in collection.labels if x.target.split=="train"]
        val=[x for x in collection.labels if x.target.split=="validation"]
        test_targets=[x for x in targets if x.split=="test"]
        if not train or not val:
            raise ValueError("GRAPE label quality threshold left no train/validation targets")
        generator=cde.train_pulse_generator(train,val,epochs=int(settings.get("epochs",60)),
            seed=int(settings.get("seed",37)))
        comparison=cde.compare_generated_pulses(test_targets,train,generator,model,
            target_spin=0,ensemble=ensemble,heldout_ensemble=heldout,
            common_start_phases_rad=start,full_grape_iterations=int(settings.get("full_grape_iter",25)),
            refine_iterations=int(settings.get("refine_iter",10)))
        import torch
        torch.save({"network_state":generator.network.state_dict(),"history":generator.history,
            "train_gate_ids":generator.train_gate_ids,"validation_gate_ids":generator.validation_gate_ids,
            "rf_rate_provenance":provenance,"training_seconds":generator.training_seconds,
            "scope":"local simulated 1q SU(2) targets; not measured process fidelity"},
            out/"models"/"E_generator.pt")
        atomic_json(out/"models"/"E_model_only.json",{
            "scope":"simulated one-spin labels and holdout gates on Windows; hardware validation pending",
            "rf_rate_provenance":provenance,"label_generation_seconds":collection.total_generation_seconds,
            "label_attempts":collection.attempts,"label_rejections":collection.rejected,
            "ensemble_provenance":ensemble_provenance,
            "gate_target_distribution":"axis uniform sphere; angle uniform [0.08,pi], not Haar SU(2)",
            "comparison":comparison})
        results.module("E","DEPENDENCY_FAILED",
            "Generator trained and compared on held-out simulated 1q gates; independent physical gate/readout measurements absent",
            simulated_targets=len(targets),simulated_comparison_rows=len(comparison),
            labels_rejected=len(collection.rejected),hardware_gate_validation="PENDING")
        return {"status":"DEPENDENCY_FAILED","simulation_rows":len(comparison),
                "model_saved":True,"reason":"independent physical gate validation absent"}
    except (KeyError,TypeError,ValueError,RuntimeError) as exc:
        reason=f"{type(exc).__name__}: {exc}"
        results.module("E","DEPENDENCY_FAILED",reason)
        return {"status":"DEPENDENCY_FAILED","reason":reason}


def _same_acquisition_mode(records):
    if not records:return False
    first=records[0]
    try:
        fs=validate_axis(first).sample_hz
        for record in records[1:]:
            if validate_axis(record).sample_hz!=fs or len(record.fid)!=len(first.fid):
                return False
            if not np.allclose(record.time_seconds,first.time_seconds,rtol=1e-8,atol=1e-12):
                return False
            if record.parameters_sent.get("pulse")!=first.parameters_sent.get("pulse"):
                return False
        return True
    except (KeyError,ValueError):
        return False


def _pilot_noise_covariance(roles,pilot):
    provided=pilot.get("f_noise_covariance_re_im")
    if provided is not None:
        if pilot.get("f_noise_covariance_source")!="independent_measured_repeats":
            raise ValueError("F noise covariance must name independent measured-repeat provenance")
        cov=np.asarray(provided,float)
        if cov.shape!=(2,2) or np.linalg.eigvalsh(cov).min()<=0:
            raise ValueError("F pilot noise covariance is not positive definite")
        return cov,"pilot independent measured repeats"
    for group in _record_groups([*roles["pilot"],*roles["train"]]).values():
        if len(group)>=3 and _same_acquisition_mode(group):
            model=estimate_noise(group)
            return model.re_im_covariance,model.source+" (pilot/train only)"
    raise ValueError("F controlled-noise training needs >=3 independent same-setting pilot/train repeats")


def _role_reference_examples(records, role, covariance, count, rng):
    examples=[]
    chol=np.linalg.cholesky(covariance)
    for (session,family),group in _record_groups(records).items():
        if len(group)<3 or not _same_acquisition_mode(group):
            continue
        target=fg.average_fids([r.fid[:count] for r in group],
            axes=[r.time_seconds[:count] for r in group])
        reference_ids=[r.key for r in group]
        for replicate in range(2):
            eps=rng.normal(size=(count,2))@chol.T
            noisy=target+eps[:,0]+1j*eps[:,1]
            examples.append({"acquisition_id":f"synthetic_{role}_{session}_{family}_{replicate}",
                "session_id":session,"setting_family":family,
                "noisy_fid":noisy,"target_fid":target,
                "reference_kind":"synthetic_corruption_of_independent_measured_average",
                "reference_ids":reference_ids})
    return examples


def _bootstrap_paired_difference(values, *, seed=109, draws=1000):
    # One value per independent session/family block, never one per FID point.
    x=np.asarray(values,float)
    if len(x)<3:return None
    rng=np.random.default_rng(seed)
    boot=np.mean(x[rng.integers(0,len(x),(draws,len(x)))],axis=1)
    return [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]


def _analyze_f(roles,pilot,out,preflight,results):
    try:
        if not roles["test"]:
            raise ValueError("F needs held-out real repeated FID acquisitions")
        test_groups={k:v for k,v in _record_groups(roles["test"]).items()
                     if len(v)>=3 and _same_acquisition_mode(v)}
        if not test_groups:
            raise ValueError("F independent test reference needs >=3 same-setting repeats")
        all_records=[r for recs in roles.values() for r in recs]
        count=min(2048,min(len(r.fid) for r in all_records))
        if count<64:raise ValueError("F common verified FID prefix too short")
        try:
            covariance,noise_source=_pilot_noise_covariance(roles,pilot)
        except ValueError as exc:
            covariance=None
            noise_source=f"UNAVAILABLE: {exc}"
        rng=np.random.default_rng(int(pilot.get("f_seed",42)))
        train=(_role_reference_examples(roles["train"],"train",covariance,count,rng)
               if covariance is not None else [])
        validation=(_role_reference_examples(roles["validation"],"validation",covariance,count,rng)
                    if covariance is not None else [])
        neural_failure=None
        if validation:
            pairs=[(np.fft.fft(x["noisy_fid"],norm="ortho"),
                    np.fft.fft(x["target_fid"],norm="ortho")) for x in validation]
            tv=fg.select_tv_lambda(pairs,pilot.get("f_tv_candidates",(.01,.1,1.)))
            hankel=fg.select_hankel_rank([(x["noisy_fid"],x["target_fid"])
                for x in validation],pilot.get("f_hankel_ranks",(1,2,3)))
            tv_lambda=tv["lambda"];hankel_rank=hankel["rank"]
            parameter_source="separate measured-reference validation groups with controlled noise"
        else:
            # Explicit exploratory pilot heuristics, never tuned to the test reference.
            tv_lambda=(float(np.sqrt(np.trace(covariance))) if covariance is not None
                       else float(pilot.get("f_exploratory_tv_lambda",.1)))
            hankel_rank=int(pilot.get("f_pilot_component_count",1))
            tv={"lambda":tv_lambda,"status":"PILOT_NOISE_HEURISTIC_NOT_VALIDATED"}
            hankel={"rank":hankel_rank,"status":"PILOT_COMPONENT_COUNT_NOT_VALIDATED"}
            parameter_source=("pilot noise heuristic; no independent validation groups" if covariance is not None
                              else "predeclared exploratory defaults; no independent pilot noise estimate")
        model_paths={}
        model_training_seconds={}
        train_sessions={x["session_id"] for x in train};train_families={x["setting_family"] for x in train}
        val_sessions={x["session_id"] for x in validation};val_families={x["setting_family"] for x in validation}
        test_sessions={key[0] for key in test_groups};test_families={key[1] for key in test_groups}
        split_ok=not ((train_sessions&val_sessions) or (train_sessions&test_sessions) or
                      (val_sessions&test_sessions) or (train_families&val_families) or
                      (train_families&test_families) or (val_families&test_families))
        if preflight.get("torch",{}).get("ready") and train and validation and split_ok:
            for variant,conditioned in (("real",True),("complex",True),("complex",False)):
                name=f"{variant}_{'conditioned' if conditioned else 'plain_unet'}"
                try:
                    start_training=perf_counter()
                    info=fg.train_tvcondnet(train,validation,out/"models"/f"F_{name}.pt",
                        variant=variant,conditioned=conditioned,tv_lambda=tv_lambda,
                        epochs=int(pilot.get("f_epochs",20)),seed=int(pilot.get("f_seed",42)))
                    model_paths[name]=info["model_path"]
                    model_training_seconds[name]=perf_counter()-start_training
                except (ValueError,RuntimeError) as exc:
                    neural_failure=f"{name}: {type(exc).__name__}: {exc}"
        else:
            neural_failure=("CPU PyTorch preflight failed" if not preflight.get("torch",{}).get("ready")
                else "insufficient independent train/validation measured references or split leakage")
        frequency=pilot.get("f_tracked_frequency_hz")
        fs=validate_axis(next(iter(test_groups.values()))[0]).sample_hz
        records=[]; adequacy=[]
        for (session,family),group in test_groups.items():
            scores=defaultdict(list); times=defaultdict(list);reference_ratios=[]
            for record in group:
                reference=[r for r in group if r.key!=record.key]
                target=fg.average_fids([r.fid[:count] for r in reference],
                    axes=[r.time_seconds[:count] for r in reference])
                noisy=record.fid[:count]
                reference_residuals=np.stack([r.fid[:count] for r in reference])-target
                reference_sem=float(np.sqrt(np.mean(np.abs(reference_residuals)**2)/len(reference)))
                raw_rmse=float(np.sqrt(np.mean(np.abs(noisy-target)**2)))
                reference_ratios.append(reference_sem/max(raw_rmse,1e-12))
                algorithms={
                    "unchanged":lambda:noisy,
                    "complex_TV":lambda:np.fft.ifft(fg.tv_denoise(
                        np.fft.fft(noisy,norm="ortho"),tv_lambda),norm="ortho"),
                    "randomized_Hankel":lambda:fg.hankel_low_rank_fid(noisy,hankel_rank),
                }
                if frequency is not None:
                    algorithms["fixed_pilot_complex_fit"] = lambda:fg.fit_exponential_fid(
                        noisy,fs,[float(frequency)])["reconstruction"]
                for name,path in model_paths.items():
                    algorithms[name]=lambda path=path:fg.apply_tvcondnet(noisy,Path(path))["fid"]
                for method,algorithm in algorithms.items():
                    t0=perf_counter();estimated=algorithm();elapsed=perf_counter()-t0
                    score=fg.denoising_metrics(estimated,target)
                    scores[method].append(score["complex_rmse"])
                    times[method].append(elapsed)
            adequate=len(group)>=4 and max(reference_ratios)<.5
            adequacy.append(adequate)
            for method,values in scores.items():
                error=float(np.mean(values))
                row={"measurement_block":session,"setting_family":family,"method":method,
                    "complex_fid_rmse":error,"analysis_seconds_mean":float(np.mean(times[method])),
                    "reference_acquisitions":len(group)-1,"test_acquisitions":len(group),
                    "reference_sem_to_raw_error_max":max(reference_ratios),
                    "reference_adequate":adequate,"sample_count_analyzed":count,
                    "source":"independent real exported FID, leave-one-out reference"}
                records.append(row)
                physical_cost=(len(group) if method=="unchanged" else
                    len(group)+len(roles["train"])+len(roles["validation"]))
                _row(results,"F",method,"unchanged",f"denoise:{family}",session,error,
                    acquisitions=physical_cost,analysis_seconds=row["analysis_seconds_mean"],
                    reason=("paired real FID; independent reference average, whole block is unit"
                            if adequate else "reference uncertainty too large for superiority claim"),
                    status="SUCCESS_VALIDATED" if adequate else "REFERENCE_INADEQUATE")
        primary="complex_conditioned"
        deltas=[]
        for block in test_groups:
            block_rows={r["method"]:r for r in records if
                        (r["measurement_block"],r["setting_family"])==block}
            if primary in block_rows:
                classical=[r["complex_fid_rmse"] for method,r in block_rows.items()
                           if method in ("complex_TV","randomized_Hankel","fixed_pilot_complex_fit")]
                if classical and block_rows[primary]["reference_adequate"]:
                    deltas.append(min(classical)-block_rows[primary]["complex_fid_rmse"])
        ci=_bootstrap_paired_difference(deltas)
        artifact={"source":"measured exported complex FID; no vendor FFT/fits",
            "analysis_points":count,"noise_covariance_source":noise_source,
            "noise_covariance_re_im":covariance.tolist() if covariance is not None else None,
            "parameter_source":parameter_source,
            "tv_selection":tv,"hankel_selection":hankel,"neural_models":model_paths,
            "neural_training_seconds":model_training_seconds,
            "train_acquisitions":len(roles["train"]),
            "validation_acquisitions":len(roles["validation"]),
            "neural_failure":neural_failure,"test_blocks":records,
            "paired_improvement_vs_best_classical_rmse":float(np.mean(deltas)) if deltas else None,
            "paired_block_bootstrap_ci":ci,
            "reference_sem_rule":"at least four repeats and max(reference SEM/raw RMSE)<0.5"}
        atomic_json(out/"models"/"F_analysis.json",artifact)
        if covariance is not None and len(deltas)>=3 and ci is not None and ci[0]>0:
            status="SUCCESS_VALIDATED"
            reason="Complex conditioned model improved over strongest classical FID baseline on adequate independent blocks"
        elif covariance is not None and len(deltas)>=3 and ci is not None:
            status="VALID_NEGATIVE_RESULT"
            reason="Paired real FID blocks did not establish conditioned-model improvement"
        else:
            status="REFERENCE_INADEQUATE"
            reason="Insufficient independent reference blocks or neural model to establish F superiority"
        results.module("F",status,reason,reference_blocks=len(test_groups),adequate_blocks=sum(adequacy),
            neural_failure=neural_failure,paired_bootstrap_ci=ci,analysis_points=count)
        return {"status":status,"reference_blocks":len(test_groups),"adequate_blocks":sum(adequacy),
                "neural_failure":neural_failure,"paired_ci":ci}
    except (KeyError,TypeError,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        reason=f"{type(exc).__name__}: {exc}"
        status="REFERENCE_INADEQUATE" if "reference" in str(exc).lower() or "repeat" in str(exc).lower() else "METHOD_FAILED"
        results.module("F",status,reason)
        return {"status":status,"reason":reason}


def _g_readout_data(records):
    measured=[];targets=[];state_ids=[];uncertainty=[];used=[]
    for record in records:
        meta=record.metadata
        target=meta.get("g_readout_target_re_im")
        state_id=meta.get("prepared_state_id")
        prep_se=meta.get("prepared_state_uncertainty")
        if target is None or not state_id or prep_se is None:
            continue
        target=np.asarray(target,float)
        if target.ndim!=2 or target.shape[1]!=2 or not np.all(np.isfinite(target)):
            raise ValueError(f"{record.key}: malformed independently known readout target")
        signal=meta.get("g_local_observable_re_im")
        if signal is None:
            if len(target)!=1:
                raise ValueError("Multi-component readout requires local FID-fit observables")
            signal=np.array([[record.fid[0].real,record.fid[0].imag]])
            source="first exported complex FID point, fixed for all calibration states"
        else:
            if meta.get("g_local_observable_source")!="local_FID_fit":
                raise ValueError("G readout features must come from local FID fit, not vendor outputs")
            signal=np.asarray(signal,float)
            source="local FID component fit"
        if signal.shape!=target.shape or not np.all(np.isfinite(signal)):
            raise ValueError(f"{record.key}: target and measured observable dimensions differ")
        prep_se=float(prep_se)
        if not np.isfinite(prep_se) or prep_se<0:
            raise ValueError("Preparation uncertainty must be measured and finite")
        measured.append(signal[:,0]+1j*signal[:,1])
        targets.append(target[:,0]+1j*target[:,1])
        state_ids.append(str(state_id));uncertainty.append(prep_se)
        used.append({"record_id":record.key,"state_id":str(state_id),"feature_source":source})
    if not measured:return None
    if len({len(x) for x in measured})!=1:
        raise ValueError("Inconsistent G readout observable dimension")
    return np.stack(measured),np.stack(targets),state_ids,uncertainty,used


def _g_compiler_examples(roles,pilot,out):
    try:
        model,_,_,provenance=_measured_rf_model(pilot,roles)
    except (KeyError,TypeError,ValueError) as exc:
        return {"status":"DEPENDENCY_FAILED","reason":str(exc)}
    checks=[]
    for record in roles["test"]:
        pulses=record.parameters_sent.get("pulse",{}).get("hPulse",())
        if not pulses or any(float(p.get("am",0))<=0 or abs(float(p.get("freshift",0)))>1e-12
                             for p in pulses):
            continue
        gates=[fg.Gate("RXY",(0,),2*math.pi*model.rf_hz_per_percent*float(p["am"])*
                       float(p["width"])*1e-6,math.radians(float(p["phase"]))) for p in pulses]
        simplified=fg.simplify_circuit(gates,1)
        compiled=fg.compile_virtual_z(simplified,1)
        checks.append({"record_id":record.key,"input_logical_gates":len(gates),
            "simplified_logical_gates":len(simplified),
            "virtual_z_final_frames_rad":compiled["final_frames_rad"],
            "ideal_unitary_equivalent":fg.unitary_equivalent(
                fg.circuit_unitary(gates,1),fg.circuit_unitary(compiled["materialized_for_verification"],1)),
            "scope":"ideal 1q unitary inferred from measured RF rate; not a measured gate error"})
    result={"status":"MODEL_ONLY" if checks else "DEPENDENCY_FAILED",
        "rf_rate_provenance":provenance,"heldout_record_checks":checks,
        "physical_gate_comparison":"NOT_PERFORMED"}
    atomic_json(out/"models"/"G1_compiler_model_only.json",result)
    return result


def _g_model_anneal(roles,pilot,out):
    try:
        model,amplitude,duration,provenance=_measured_rf_model(pilot,roles)
        settings=pilot.get("g_model_anneal",{})
        if not isinstance(settings,Mapping):settings={}
        n=int(settings.get("segments",4))
        if n<1 or duration/n<5e-6:
            raise ValueError("G2 model-only segments outside completed single-pulse timing envelope")
        target=cde.rotation((1,0,0),math.pi/2)
        initial=np.zeros(n)
        bounds=[(-math.pi,math.pi)]*n
        def objective(phases):
            program=cde.PulseProgram((duration/n,)*n,(amplitude,)*n,tuple(phases))
            return cde.ensemble_infidelity(model,target,program,[cde.EnsemblePoint()])

        calls=int(settings.get("evaluations",24))
        anneal=fg.bounded_anneal(objective,initial,bounds,evaluations=calls,chains=3)
        baseline=fg.grid_and_nelder_mead(objective,initial,bounds,evaluations=calls)
        artifact={"scope":"simulated model-only objective; no measured gate-cost feedback",
            "rf_rate_provenance":provenance,"anneal_best_simulated_infidelity":anneal["best_cost"],
            "grid_nelder_best_simulated_infidelity":baseline["best_cost"],
            "anneal_evaluations":anneal["evaluations"],"baseline_evaluations":baseline["evaluations"],
            "anneal_trace":anneal["trace"],"baseline_trace":baseline["trace"],
            "hardware_validation":"PENDING"}
        atomic_json(out/"models"/"G2_model_only.json",artifact)
        return {"status":"MODEL_ONLY","evaluations_each":calls,
                "hardware_validation":"PENDING"}
    except (KeyError,TypeError,ValueError,RuntimeError) as exc:
        return {"status":"DEPENDENCY_FAILED","reason":f"{type(exc).__name__}: {exc}"}


def _g_dd_plan(pilot,out):
    settings=pilot.get("g_dd")
    if not isinstance(settings,Mapping):
        return {"status":"UNVERIFIED_TIMING","reason":"no verified in-sequence idle and pi pulse plan"}
    try:
        unwanted={tuple(map(int,key.split(","))):float(value)
                  for key,value in settings.get("unwanted_zz_weights",{}).items()}
        desired=[tuple(map(int,key.split(","))) for key in settings.get("desired_zz_pairs",[])]
        result=fg.design_contextual_dd(float(settings["duration_s"]),
            tuple(int(x) for x in settings["qubits"]),
            single_z_weights={int(k):float(v) for k,v in settings.get("single_z_weights",{}).items()},
            unwanted_zz_weights=unwanted,desired_zz_pairs=desired,
            timing_verified=bool(settings["timing_verified"]),
            pi_pulse_error=float(settings["pi_pulse_error"]),
            pulse_duration_s=float(settings["pi_pulse_duration_s"]))
        saved={k:v for k,v in result.items() if k!="pulses"}
        # Integral maps with tuple keys need an explicit JSON representation.
        for key in ("integrals","baseline_integrals"):
            if key in saved:
                x=dict(saved[key])
                x["zz_s"]={f"{a},{b}":v for (a,b),v in x["zz_s"].items()}
                saved[key]=x
        saved["scope"]="commuting Z/ZZ ideal pi model; pulse widths and physical effect unvalidated"
        atomic_json(out/"models"/"G3_dd_candidate.json",saved)
        return {"status":result["status"],"pulse_count":len(result.get("pulses",[])),
                "hardware_validation":"PENDING"}
    except (KeyError,TypeError,ValueError,RuntimeError) as exc:
        status="UNVERIFIED_TIMING" if "UNVERIFIED_TIMING" in str(exc) else "DEPENDENCY_FAILED"
        return {"status":status,"reason":f"{type(exc).__name__}: {exc}"}


def _g_analog_readout(roles,pilot,out,preflight,results):
    try:
        train=_g_readout_data(roles["train"])
        val=_g_readout_data(roles["validation"])
        test=_g_readout_data(roles["test"])
        if any(x is None for x in (train,val,test)):
            raise ValueError("G4 needs independently prepared train/validation/test analogue readout states")
        tm,tt,ti,tu,tr=train;vm,vt,vi,vu,vr=val;xm,xt,xi,xu,xr=test
        if min(len(tm),len(vm),len(xm))<1 or len(set([*ti,*vi,*xi]))!=len([*ti,*vi,*xi]):
            raise ValueError("G4 calibration state IDs must be distinct across all partitions")
        if tm.shape[1]!=vm.shape[1] or tm.shape[1]!=xm.shape[1]:
            raise ValueError("G4 observable vector dimensions differ across splits")
        preparation_uncertainty=float(max([*tu,*vu,*xu]))
        groups=pilot.get("g_readout_groups")
        if groups is not None:
            groups=tuple(tuple(int(j) for j in group) for group in groups)
        selected=fg.compare_readout_models(tm,tt,vm,vt,
            ridge_candidates=pilot.get("g_readout_ridge_candidates",(1e-5,1e-3,.1)),
            groups=groups,preparation_uncertainty=preparation_uncertainty)
        baseline=selected["model"]
        np.savez_compressed(out/"models"/"G4_affine_readout.npz",
            coefficients=baseline.coefficients,
            mode=np.asarray(baseline.mode),groups=np.asarray(str(baseline.groups)),
            train_state_ids=np.asarray(ti),validation_state_ids=np.asarray(vi),
            preparation_uncertainty=preparation_uncertainty)
        residual_path=None;neural_failure=None
        if preflight.get("torch",{}).get("ready") and len(tm)>=6 and len(vm)>=2:
            try:
                info=fg.train_residual_readout(tm,tt,vm,vt,baseline,
                    out/"models"/"G4_residual.pt",epochs=int(pilot.get("g_readout_epochs",60)),
                    train_state_ids=ti,validation_state_ids=vi)
                residual_path=info["model_path"]
            except (ValueError,RuntimeError) as exc:
                neural_failure=f"{type(exc).__name__}: {exc}"
        else:
            neural_failure="CPU PyTorch unavailable or too few independent calibration states"
        raw_error=np.mean(np.abs(xm-xt)**2,axis=1)
        affine=fg.apply_analog_readout(baseline,xm)
        affine_error=np.mean(np.abs(affine-xt)**2,axis=1)
        estimates={"uncorrected_analogue":raw_error,"affine_full_or_grouped":affine_error}
        if residual_path:
            learned=fg.apply_residual_readout(xm,baseline,residual_path)
            estimates["affine_plus_residual_nn"]=np.mean(np.abs(learned-xt)**2,axis=1)
        rows=[]
        test_by_key={r.key:r for r in roles["test"]}
        for i,record in enumerate(xr):
            adequate=preparation_uncertainty<.5*math.sqrt(max(float(raw_error[i]),1e-12))
            for method,errors in estimates.items():
                error=float(errors[i])
                row={"record_id":record["record_id"],"prepared_state_id":xi[i],
                    "method":method,"complex_observable_mse":error,
                    "preparation_uncertainty":preparation_uncertainty,
                    "reference_adequate":adequate,"feature_source":record["feature_source"]}
                rows.append(row)
                _row(results,"G",method,"uncorrected_analogue",
                    "heldout_analogue_readout:"+record["record_id"],
                    test_by_key[record["record_id"]].metadata.get("measurement_block",i),error,
                    acquisitions=len(tm)+len(vm)+1,
                    reason="held-out analogue observable correction, not gate improvement",
                    status="SUCCESS_VALIDATED" if adequate else "REFERENCE_INADEQUATE")
        atomic_json(out/"models"/"G4_analogue_readout.json",{
            "scope":"NMR complex observables; not bitstring confusion probabilities",
            "selection":selected["winner"],"validation_curve":selected["validation_curve"],
            "test_rows":rows,"neural_failure":neural_failure,
            "preparation_uncertainty":"maximum calibration and test state uncertainty",
            "physical_gate_improvement":"NOT_INFERRED_FROM_CORRECTED_READOUT"})
        return {"status":"SUCCESS_VALIDATED" if all(r["reference_adequate"] for r in rows)
                else "REFERENCE_INADEQUATE","test_rows":len(rows),
                "neural_failure":neural_failure}
    except (KeyError,TypeError,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        return {"status":"DEPENDENCY_FAILED","reason":f"{type(exc).__name__}: {exc}"}


def _analyze_g(roles,pilot,out,preflight,results):
    prior=dict(results.data.get("modules",{}).get("G",{}))
    prior_status=prior.get("status","PENDING")
    prior_physical=prior_status not in ("PENDING","RUNNING")
    g1=_g_compiler_examples(roles,pilot,out)
    g2=_g_model_anneal(roles,pilot,out)
    g3=_g_dd_plan(pilot,out)
    g4=_g_analog_readout(roles,pilot,out,preflight,results)
    one_qubit=g1["status"]=="MODEL_ONLY"
    two_qubit=bool(pilot.get("verified_low_level_2q_pulses",False))
    timing=bool(pilot.get("g_dd",{}).get("timing_verified",False)) if isinstance(pilot.get("g_dd"),Mapping) else False
    plan=fg.g_ablation_plan(verified_one_qubit=one_qubit,verified_two_qubit=two_qubit,
        timing_verified=timing,readout_calibrated=g4["status"]=="SUCCESS_VALIDATED")
    artifact={"scope":"offline analysis of measured FID and model-only circuit/DD/SA candidates",
        "G1":g1,"G2":g2,"G3":g3,"G4":g4,
        "physical_ablation_status":prior.get("physical_ablation_status",
            "RECORDED_BY_SEPARATE_WORKER" if prior_physical else "NOT_EXECUTED_BY_OFFLINE_ANALYSIS"),
        "ablation_plan":plan,
        "low_level_2q_3q_claim":"unverified unless physically demonstrated by separate worker"}
    atomic_json(out/"models"/"G_pipeline.json",artifact)
    offline={key:artifact[key]["status"] for key in ("G1","G2","G3","G4")}
    if prior_physical:
        details={key:value for key,value in prior.items() if key not in ("status","reason")}
        details["offline_submodules"]=offline
        details["offline_analysis_file"]="models/G_pipeline.json"
        results.module("G",prior_status,prior.get("reason",""),**details)
        final_status=prior_status
    else:
        results.module("G","DEPENDENCY_FAILED",
            "G1 compiler/model and any G4 analogue readout analyzed locally; G2/G3 physical gains and complete paired ablations need real sequences",
            offline_submodules=offline,two_qubit_verified=two_qubit,
            physical_ablation_status="NOT_EXECUTED_BY_OFFLINE_ANALYSIS")
        final_status="DEPENDENCY_FAILED"
    return {"status":final_status,"submodules":
        {key:artifact[key]["status"] for key in ("G1","G2","G3","G4")}}


def analyze_de_fg(records_by_role: Mapping[str,Sequence[RawFIDRecord]], pilot: Mapping,
                  out: Path, preflight: Mapping, results) -> dict:
    """Run D/E/F/G local analyses after serial acquisition has finished.

    `records_by_role` has disjoint pilot/train/validation/test RawFIDRecord lists.
    The caller must save records and label measurement_block/setting_family before this
    call. All model artifacts stay in `out/models`, and Results owns report rows.
    Missing physical capabilities become explicit module states. No optimizer
    here may submit a pulse; active G2 and E gate validation need the live
    worker's separate preapproved measurement plan.
    """
    out=Path(out)
    (out/"models").mkdir(parents=True,exist_ok=True)
    roles=_validated_roles(records_by_role)
    summary={}
    summary["D"]=_analyze_d(roles,pilot,out,preflight,results)
    summary["E"]=_analyze_e(roles,pilot,out,preflight,results)
    summary["F"]=_analyze_f(roles,pilot,out,preflight,results)
    summary["G"]=_analyze_g(roles,pilot,out,preflight,results)
    atomic_json(out/"models"/"DEFG_analysis_summary.json",summary)
    return summary
