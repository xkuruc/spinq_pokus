"""Finite, restartable *real hardware* benchmark for Gemini Lab / SpinQLabLink 1.0.2.

Run only on the Windows host with access to the tablet. Numerical tests on Mac
do not create measurements and are never placed in results/.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from publish_results import publish_results
from spinq_audit.common import atomic_json, utc_now
from spinq_audit.safety import HardwareLock
from spinq_benchmark.hardware import HardwareUncertain, LiveHardware, first_fid, physical_request
from spinq_benchmark.numerics import (
    ComplexGridBayes, bb1_pulses, bounded_nelder_mead, choose_echo_time,
    choose_fid_budget, complex_fit, fit_response, gaussian_process_optimize,
    hankel_denoise, optimize_grape,
)
from spinq_benchmark.output import Results, aggregate_rows, bundle, plot_comparisons


def _float(x):
    return float(x) if x is not None and math.isfinite(float(x)) else None


class Benchmark:
    def __init__(self, hardware: LiveHardware, results: Results, seed: int):
        self.hw,self.results,self.seed=hardware,results,seed
        self.calibration={}
        self.segment_verified={}
        self.tracked_hz=None

    def acquire(self,key,*,width=40.,detuning=0.,amplitude=100.,phase=90.,
                count=16000,pulses=None):
        p=physical_request(width_us=width,detuning_hz=detuning,amplitude_pct=amplitude,
                           phase_deg=phase,sample_count=count,pulses=pulses)
        row=self.hw.measure(key,p)
        self.results.data["hardware_results_present"]=True
        y,axis=first_fid(row)
        if not all(math.isfinite(float(x)) for x in axis) or any(b<=a for a,b in zip(axis,axis[1:])):
            raise ValueError("FID chart axis malformed; no local fit")
        start=time.monotonic()
        fit=complex_fit(y,p["sampleFre"],tracked_hz=self.tracked_hz)
        compute=time.monotonic()-start
        if self.tracked_hz is None and fit["target"]["amplitude"]>3*fit["residual_rms"]:
            self.tracked_hz=fit["target"]["frequency_hz"]
        row["local_fid_fit"]=fit
        row["local_fit_compute_seconds"]=compute
        atomic_json(self.hw.data/(key+".json"),row)
        self.results.save()
        return row,y,fit

    @staticmethod
    def _obs(row,fit,width,detuning,sigma=None):
        target=fit["target"]
        return {"key":row["key"],"width_us":width,"detuning_hz":detuning,
                "coefficient_re":target["coefficient_re"],"coefficient_im":target["coefficient_im"],
                "sigma":max(float(sigma or fit["residual_rms"]),1e-6),
                "wall_seconds":row["wall_seconds"]}

    def calibration_block(self,block):
        """Same shared start, balanced method order, independent reference scan."""
        prefix=f"cal_b{block:02d}"
        rng=random.Random(self.seed+block)
        shared_settings=[(40.,0.),(80.,0.),(120.,0.),(160.,0.),(200.,0.),(40.,-20.),(40.,20.)]
        shared=[]
        for i,(w,d) in enumerate(shared_settings):
            row,_,fit=self.acquire(f"{prefix}_shared_{i}",width=w,detuning=d)
            shared.append(self._obs(row,fit,w,d))
        # Estimate drift/noise across independent acquisitions, not across FID points.
        base=shared[0]
        repeat,_,rep_fit=self.acquire(f"{prefix}_shared_repeat",width=40.)
        repeated=self._obs(repeat,rep_fit,40.,0.)
        sigma=max(abs(complex(base["coefficient_re"],base["coefficient_im"])-
                      complex(repeated["coefficient_re"],repeated["coefficient_im"]))/math.sqrt(2),
                  base["sigma"],repeated["sigma"])
        shared.append(repeated)
        for x in shared:x["sigma"]=sigma
        split_pulses=[{"width":20.,"am":100.,"phase":90.,"freshift":0.},
                      {"width":20.,"am":100.,"phase":90.,"freshift":0.}]
        split_row,_,split_fit=self.acquire(f"{prefix}_segment_probe",pulses=split_pulses)
        split_coeff=complex(split_fit["target"]["coefficient_re"],split_fit["target"]["coefficient_im"])
        single_coeff=complex(repeated["coefficient_re"],repeated["coefficient_im"])
        split_difference=abs(split_coeff-single_coeff)/max(abs(single_coeff),1e-9)
        self.segment_verified[block]=bool(split_difference<.3 and abs(single_coeff)>3*sigma)
        try: preliminary=fit_response(shared)
        except Exception: preliminary=None
        methods=["fixed_fit","coarse_fine_fit","bayesian_complex"]
        rng.shuffle(methods)
        candidates={}
        # Each method starts with an independent copy of the same available data.
        for method in methods:
            own=[]
            for step in range(4):
                available=copy.deepcopy(shared+own)
                if method=="fixed_fit":
                    setting=[(80.,-20.),(120.,20.),(160.,-20.),(200.,20.)][step]
                elif method=="coarse_fine_fit":
                    if step<2: setting=[(80.,-10.),(160.,10.)][step]
                    else:
                        try:
                            estimate=fit_response(available)
                            w=float(np.clip((2*step-3)*estimate["t90_us"],40,200))
                            d=float(np.clip(estimate["resonance_detuning_hz"]+(-10 if step==2 else 10),-20,20))
                            setting=(round(w,1),round(d,1))
                        except Exception: setting=[(120.,-10.),(200.,10.)][step-2]
                else:
                    bayes=ComplexGridBayes()
                    bayes.update(available,max(5.,preliminary["linewidth_hz"] if preliminary else 20.))
                    setting=bayes.next_setting([(float(w),float(d)) for w in (40,60,80,120,160,200)
                                                 for d in (-20,-10,0,10,20)],
                                                max(5.,preliminary["linewidth_hz"] if preliminary else 20.),sigma)
                    if setting is None:break
                w,d=setting
                row,_,fit=self.acquire(f"{prefix}_{method}_{step}",width=w,detuning=d)
                own.append(self._obs(row,fit,w,d,sigma))
            try:
                estimate=fit_response(shared+own)
                if method=="bayesian_complex":
                    bayes=ComplexGridBayes();bayes.update(shared+own,max(5.,estimate["linewidth_hz"]))
                    estimate["posterior"]=bayes.summary()
                candidates[method]={"estimate":estimate,"observations":own}
            except Exception as exc:
                candidates[method]={"error":str(exc),"observations":own}
            return_row,_,return_fit=self.acquire(f"{prefix}_{method}_return_baseline",width=40.)
            candidates[method]["return_baseline"]=self._obs(return_row,return_fit,40.,0.,sigma)
        # Separate reference never participates in any method's next-point choice.
        reference=[]
        reference_settings=[(40.,-20.),(40.,0.),(40.,20.),(80.,0.),(120.,0.),(160.,0.),(200.,0.)]
        rng.shuffle(reference_settings)
        for i,(w,d) in enumerate(reference_settings):
            row,_,fit=self.acquire(f"{prefix}_reference_{i}",width=w,detuning=d)
            reference.append(self._obs(row,fit,w,d,sigma))
        try: ref=fit_response(reference)
        except Exception as exc: ref={"error":str(exc)}
        baseline_error=None
        if "t90_us" in ref:
            errors={}
            for method in methods:
                item=candidates[method]
                est=item.get("estimate")
                if est and est.get("identifiable"):
                    error=math.hypot((est["t90_us"]-ref["t90_us"])/max(ref["t90_se_us"],1.),
                                     (est["resonance_detuning_hz"]-ref["resonance_detuning_hz"])/max(ref["frequency_se_hz"],1.))
                    errors[method]=error
            baseline_error=errors.get("fixed_fit")
        for method in methods:
            item=candidates[method]
            est=item.get("estimate")
            error=errors.get(method) if "t90_us" in ref else None
            obs=item["observations"]
            own_with_return=obs+[item["return_baseline"]]
            self.results.row(topic="calibration",method=method,block=block,phase="pilot",
                measurements=len(shared)+len(own_with_return),internal_repetitions="UNKNOWN",
                requested_samples=(len(shared)+len(own_with_return))*16000,
                wall_seconds=sum(x["wall_seconds"] for x in shared+own_with_return),compute_seconds=None,
                error=error,uncertainty=math.hypot(est["t90_se_us"],est["frequency_se_hz"]) if est else None,
                improvement_vs_baseline=(baseline_error-error)/baseline_error if baseline_error and error is not None else None,
                status="PILOT" if error is not None else "NEPRESVEDČIVÉ",
                reason=item.get("error") or ("reference fit unidentifiable" if error is None else "independent reference; normalized joint parameter error"))
        valid=[c["estimate"] for c in candidates.values() if c.get("estimate",{}).get("identifiable")]
        if valid:
            self.calibration[block]=statistics.median(x["t90_us"] for x in valid)
        summary={"status":"PILOT" if valid and "t90_us" in ref else "NEPRESVEDČIVÉ",
                 "reason":"independent reference fit and three strategy estimates" if valid else "calibration not identifiable",
                 "shared_pilot_count":len(shared),"reference_count":len(reference),
                 "capability_probe_count":1,
                 "reference":ref,"methods":candidates,"method_order":methods,"sigma_complex":sigma,
                 "segment_probe_relative_difference":split_difference,
                 "segment_sequence_supported_by_probe":self.segment_verified[block],
                 "t90_for_followup_us":self.calibration.get(block)}
        self.results.topic("calibration",block,summary)
        if block==0:
            self.results.data.setdefault("frozen_plan",{})["calibration_target_joint_error"]=max(2.,baseline_error) if baseline_error is not None else None
            self.results.save()
        return summary

    def acquisition_block(self,block):
        prefix=f"acq_b{block:02d}"
        pilot=[]
        for n in (4000,8000,16000):
            for rep in range(2):
                row,_,fit=self.acquire(f"{prefix}_pilot_{n}_{rep}",count=n)
                pilot.append({"sample_count":n,"frequency_hz":fit["target"]["frequency_hz"],
                              "wall_seconds":row["wall_seconds"]})
        pilot_variability=np.std([x["frequency_hz"] for x in pilot if x["sample_count"]==16000],ddof=1)
        target=max(float(pilot_variability),1.)
        if block:
            target=float(self.results.data["frozen_plan"]["acquisition_target_se_hz"])
        plan=choose_fid_budget(pilot,target_se_hz=target)
        if not plan: raise ValueError("Adaptive sample budget cannot be estimated")
        rng=random.Random(self.seed+300+block)
        methods=["fixed_16k","adaptive_early_stop","adaptive_equal_sample_budget"]
        rng.shuffle(methods)
        readings={}
        for method in methods:
            if method=="fixed_16k": n,reps=16000,2
            elif method=="adaptive_early_stop": n,reps=plan["sample_count"],plan["repeats"]
            else: n,reps=plan["sample_count"],32000//plan["sample_count"]
            readings[method]=[]
            for r in range(reps):
                row,_,fit=self.acquire(f"{prefix}_{method}_{r}",count=n)
                readings[method].append((row,fit))
        reference=[]
        for i in range(3):
            row,_,fit=self.acquire(f"{prefix}_independent_reference_{i}")
            reference.append(fit["target"]["frequency_hz"])
        ref=float(np.mean(reference));ref_se=float(np.std(reference,ddof=1)/math.sqrt(len(reference)))
        errors={method:abs(float(np.mean([f["target"]["frequency_hz"] for _,f in rr]))-ref)
                for method,rr in readings.items()}
        baseline=errors["fixed_16k"]
        for method,rr in readings.items():
            wall=sum(row["wall_seconds"] for row,_ in rr)
            self.results.row(topic="acquisition",method=method,block=block,phase="pilot",
                measurements=len(rr),internal_repetitions="UNKNOWN",
                requested_samples=sum(row["params"]["sampleCount"] for row,_ in rr),
                wall_seconds=wall,compute_seconds=sum(row["local_fit_compute_seconds"] for row,_ in rr),
                error=errors[method],uncertainty=ref_se,
                improvement_vs_baseline=(baseline-errors[method])/baseline if baseline else None,
                status="PILOT" if errors[method]<=max(3*ref_se,target) else "NEPRESVEDČIVÉ",
                reason="independent 16k reference; equal 32k sample budget or independently checked early stop")
        summary={"status":"PILOT","reason":"real short FID acquisitions compared; T2 echo unavailable",
                 "pilot":pilot,"target_se_hz":target,"adaptive_plan":plan,"reference_hz":ref,
                 "reference_se_hz":ref_se,"order":methods,
                 "costs":{"pilot_acquisitions":len(pilot),"reference_acquisitions":len(reference),
                          "fixed_wall_seconds":sum(row["wall_seconds"] for row,_ in readings["fixed_16k"]),
                          "adaptive_early_stop_wall_seconds":sum(row["wall_seconds"] for row,_ in readings["adaptive_early_stop"]),
                          "measured_early_stop_seconds_saved":sum(row["wall_seconds"] for row,_ in readings["fixed_16k"])-
                              sum(row["wall_seconds"] for row,_ in readings["adaptive_early_stop"])},
                 "echo_t2":{"status":"SKIPPED_UNSUPPORTED","reason":"SpinQLabLink 1.0.2 T2 and SpinEcho parameters do not expose verified per-point echo delays; FID T2* is not T2"}}
        self.results.topic("acquisition",block,summary)
        if block==0:
            self.results.data.setdefault("frozen_plan",{}).update({
                "frozen_after_block":0,"acquisition_target_se_hz":target,
                "calibration_new_measurements_per_method":4,
                "pulse_candidate_evaluations_per_method":4,
                "pulse_held_out_phases_deg":[45,225],
                "robust_test_detunings_hz":[-20,0,20],
                "decision_note":"do not change these after viewing later blocks"})
            self.results.save()
        self.results.row(topic="echo_t2",method="fixed_vs_adaptive_echo",block=block,phase="pilot",
            status="SKIPPED_UNSUPPORTED",reason=summary["echo_t2"]["reason"])
        return summary

    def pulse_block(self,block):
        """Two phase-cycled readouts per candidate; held-out phases for final test."""
        t90=self.calibration.get(block)
        if not t90 or not 25<=t90<=60:
            raise ValueError("Independent H 90-degree working point unidentifiable; pulse tuning skipped")
        prefix=f"pulse_b{block:02d}"
        width=round(float(t90),1)
        # Common target from separate calibrated acquisitions, not the optimizer's measurements.
        refs={}
        for phase in (90.,270.,45.,225.):
            coeffs=[]
            for rep in range(2):
                row,_,fit=self.acquire(f"{prefix}_target_{int(phase)}_{rep}",width=width,phase=phase)
                coeffs.append(complex(fit["target"]["coefficient_re"],fit["target"]["coefficient_im"]))
            refs[phase]=sum(coeffs)/len(coeffs)
        if abs(refs[90.]+refs[270.])>max(abs(refs[90.]),abs(refs[270.]))*.7:
            raise ValueError("Phase cycling did not invert the complex response; multi-readout objective invalid")
        if self.segment_verified.get(block):
            double_pulse=[{"width":width,"am":100.,"phase":90.,"freshift":0.}]*2
            double_reference=[]
            for rep in range(2):
                row,_,fit=self.acquire(f"{prefix}_double_target_{rep}",pulses=double_pulse)
                double_reference.append(complex(fit["target"]["coefficient_re"],fit["target"]["coefficient_im"]))
            refs["double_90"]=sum(double_reference)/2
        scale=max(abs(refs[90.]),1e-9)
        rng=random.Random(self.seed+600+block)
        methods=["traditional","coordinate","bounded_nelder_mead","gaussian_process"]
        rng.shuffle(methods)
        records={}
        for method in methods:
            evaluations=[]
            def objective(x):
                amp,phase=float(x[0]),float(x[1])
                coefficients=[];seconds=0.
                for basephase in (0.,180.):
                    actual=(phase+basephase)%360
                    row,_,fit=self.acquire(f"{prefix}_{method}_eval_{len(evaluations)}_{int(basephase)}",
                                           width=width,phase=actual,amplitude=amp)
                    coefficients.append(complex(fit["target"]["coefficient_re"],fit["target"]["coefficient_im"]))
                    seconds+=row["wall_seconds"]
                loss=float(np.mean([abs(c-r)**2 for c,r in zip(coefficients,(refs[90.],refs[270.]))])/scale**2)
                evaluations.append({"amplitude_pct":amp,"phase_deg":phase,"loss":loss,"wall_seconds":seconds})
                return loss
            if method=="traditional":
                for _ in range(4): objective([100.,90.])
                best={"x":[100.,90.],"loss":float(np.mean([e["loss"] for e in evaluations]))}
            elif method=="coordinate":
                points=([100.,90.],[98.,90.],[100.,85.],[100.,95.])
                for x in points:objective(x)
                best={"x":[min(evaluations,key=lambda e:e["loss"])["amplitude_pct"],
                            min(evaluations,key=lambda e:e["loss"])["phase_deg"]],
                      "loss":min(e["loss"] for e in evaluations)}
            elif method=="bounded_nelder_mead":
                selected,_=bounded_nelder_mead(objective,[100.,90.],[(90.,100.),(80.,100.)],4)
                best={"x":selected["x"],"loss":selected["loss"]}
                while len(evaluations)<4: objective(best["x"])
            else:
                selected,_=gaussian_process_optimize(objective,[100.,90.],[(90.,100.),(80.,100.)],4,
                                                     seed=self.seed+block)
                best={"x":selected["x"],"loss":selected["loss"]}
            # Held-out phase configurations were not in the optimization objective.
            controls=[]
            for phase,target in ((45.,refs[45.]),(225.,refs[225.])):
                row,_,fit=self.acquire(f"{prefix}_{method}_control_{int(phase)}",width=width,
                    amplitude=float(best["x"][0]),phase=(float(best["x"][1])+phase-90)%360)
                coeff=complex(fit["target"]["coefficient_re"],fit["target"]["coefficient_im"])
                controls.append({"phase_deg":phase,"error":abs(coeff-target)/scale,"wall_seconds":row["wall_seconds"]})
            if self.segment_verified.get(block):
                candidate_pulse={"width":width,"am":float(best["x"][0]),
                                 "phase":float(best["x"][1]),"freshift":0.}
                row,_,fit=self.acquire(f"{prefix}_{method}_control_double",pulses=[candidate_pulse]*2)
                coeff=complex(fit["target"]["coefficient_re"],fit["target"]["coefficient_im"])
                controls.append({"sequence":"double_same_axis","error":abs(coeff-refs["double_90"])/scale,
                                 "wall_seconds":row["wall_seconds"]})
            records[method]={"best":best,"evaluations":evaluations,"controls":controls,
                             "held_out_error":float(np.mean([c["error"] for c in controls]))}
        baseline=records["traditional"]["held_out_error"]
        for method in methods:
            r=records[method];error=r["held_out_error"]
            self.results.row(topic="pulse_tuning",method=method,block=block,phase="pilot",
                measurements=2*len(r["evaluations"])+len(r["controls"])+(10 if self.segment_verified.get(block) else 8),
                internal_repetitions="UNKNOWN",requested_samples=16000*(2*len(r["evaluations"])+len(r["controls"])+(10 if self.segment_verified.get(block) else 8)),
                wall_seconds=sum(e["wall_seconds"] for e in r["evaluations"])+sum(c["wall_seconds"] for c in r["controls"]),
                compute_seconds=None,error=error,uncertainty=float(np.std([c["error"] for c in r["controls"]],ddof=1)),
                improvement_vs_baseline=(baseline-error)/baseline if baseline else None,
                status="PILOT" if self.segment_verified.get(block) else "NEPRESVEDČIVÉ",
                reason="phase-cycled complex FID proxy on held-out phases; not process fidelity")
        summary={"status":"PILOT" if self.segment_verified.get(block) else "NEPRESVEDČIVÉ",
                 "reason":"four real optimization strategies with held-out phase controls",
                 "target_coefficients":{str(k):[v.real,v.imag] for k,v in refs.items()},
                 "methods":records,"method_order":methods,"t90_us":t90,
                 "shared_reference_acquisitions":10 if self.segment_verified.get(block) else 8,
                 "double_sequence_control":"measured" if self.segment_verified.get(block) else "SKIPPED: segment probe did not match single pulse"}
        self.results.topic("pulse_tuning",block,summary)
        if block==0:
            self.results.data.setdefault("frozen_plan",{})["pulse_target_held_out_error"]=baseline
            self.results.save()
        return summary

    def robust_block(self,block):
        t90=self.calibration.get(block)
        if not t90 or not 25<=t90<=60:
            raise ValueError("Calibrated H 90-degree width unavailable")
        if not self.segment_verified.get(block):
            raise ValueError("Sequential Pulse[] effect not verified by two-half-pulse probe")
        prefix=f"robust_b{block:02d}"
        start=time.monotonic()
        design=optimize_grape(t90,100.,max_total_us=200.,seed=self.seed+block)
        design_time=time.monotonic()-start
        atomic_json(self.results.out/"models"/f"grape_b{block:02d}.json",design)
        rectangle=[{"width":float(t90),"am":100.,"phase":90.,"freshift":0.}]
        bb1=bb1_pulses(t90,90.,100.)
        shapes={"rectangle":rectangle,"bb1":bb1,"grape":design["pulses"]}
        if sum(x["width"] for x in bb1)>200:
            shapes.pop("bb1")
            self.results.row(topic="robust_pulse",method="bb1",block=block,phase="pilot",
                status="SKIPPED_UNSUPPORTED",reason="BB1 requested RF duration exceeds previously completed 200 us single-task envelope")
        methods=list(shapes)
        random.Random(self.seed+900+block).shuffle(methods)
        # Reference uses independent on-resonance rectangle measurements for each preparation.
        preparations={"none":[],"x90":[{"width":float(t90),"am":100.,"phase":0.,"freshift":0.}],
                      "y90":[{"width":float(t90),"am":100.,"phase":90.,"freshift":0.}]}
        refs={}
        for prep,pre in preparations.items():
            if sum(q["width"] for q in pre+rectangle)>200: continue
            row,_,fit=self.acquire(f"{prefix}_reference_{prep}",pulses=pre+rectangle)
            refs[prep]=complex(fit["target"]["coefficient_re"],fit["target"]["coefficient_im"])
        readings={}
        for method in methods:
            readings[method]=[]
            for detuning in (-20.,0.,20.): # held out from design grid (-10,0,10) at ±20
                for prep,pre in preparations.items():
                    if prep not in refs:continue
                    shape=copy.deepcopy(shapes[method])
                    for pulse in shape:pulse["freshift"]=detuning
                    if sum(q["width"] for q in pre+shape)>200:continue
                    row,_,fit=self.acquire(f"{prefix}_{method}_{prep}_{int(detuning)}",pulses=pre+shape)
                    z=complex(fit["target"]["coefficient_re"],fit["target"]["coefficient_im"])
                    error=abs(z-refs[prep])/max(abs(refs[prep]),1e-9)
                    readings[method].append({"detuning_hz":detuning,"prep":prep,"error":error,
                                             "wall_seconds":row["wall_seconds"],"pulse_width_us":sum(q["width"] for q in shape)})
        baseline=np.mean([x["error"] for x in readings.get("rectangle",[])]) if readings.get("rectangle") else None
        for method in methods:
            rr=readings[method]
            error=float(np.mean([x["error"] for x in rr])) if rr else None
            if error is None:status="NEPRESVEDČIVÉ"
            else:status="PILOT" if len(rr)>=6 else "NEPRESVEDČIVÉ"
            self.results.row(topic="robust_pulse",method=method,block=block,phase="pilot",
                measurements=len(rr)+len(refs),internal_repetitions="UNKNOWN",
                requested_samples=16000*(len(rr)+len(refs)),
                wall_seconds=sum(x["wall_seconds"] for x in rr),compute_seconds=design_time if method=="grape" else 0.,
                error=error,uncertainty=float(np.std([x["error"] for x in rr],ddof=1)) if len(rr)>1 else None,
                improvement_vs_baseline=(baseline-error)/baseline if baseline and error is not None else None,
                status=status,reason="complex response proxy for multiple preparation sequences; not gate fidelity")
        summary={"status":"PILOT" if readings.get("grape") else "NEPRESVEDČIVÉ",
                 "reason":"short single-H-spin robust design tested on held-out detunings; BB1 excluded if >200 us",
                 "design":design,"design_compute_seconds":design_time,"readings":readings,
                 "method_order":methods,"reference":{k:[v.real,v.imag] for k,v in refs.items()},
                 "reference_acquisitions":len(refs),
                 "h_p_coupling":"UNMEASURED; design validity limited"}
        self.results.topic("robust_pulse",block,summary)
        if block==0:
            self.results.data.setdefault("frozen_plan",{})["robust_target_error"]=float(baseline) if baseline is not None else None
            self.results.save()
        return summary

    def denoising(self):
        """Collect independent repeats, split by both acquisition block and phase family."""
        from spinq_benchmark.learning import train_complex_denoiser,apply_complex_denoiser
        start=time.monotonic()
        families={"train_0":0.,"train_90":90.,"validation_180":180.,"test_270":270.}
        blocks={"train_0":0,"train_90":0,"validation_180":1,"test_270":2}
        collected={}
        for family,phase in families.items():
            collected[family]=[]
            for rep in range(6):
                row,y,fit=self.acquire(f"denoise_{family}_{rep}",phase=phase)
                collected[family].append({"key":row["key"],"signal":y,"wall_seconds":row["wall_seconds"]})
        split={family:{"block":blocks[family],"phase_deg":families[family],
                       "inputs":[x["key"] for x in rows[:3]],"references":[x["key"] for x in rows[3:]]}
               for family,rows in collected.items()}
        atomic_json(self.results.out/"data"/"denoise_split.json",split)
        stability={}
        for family,rows in collected.items():
            early=np.mean(np.stack([x["signal"] for x in rows[:3]]),axis=0)
            late=np.mean(np.stack([x["signal"] for x in rows[3:]]),axis=0)
            stability[family]=float(np.linalg.norm(early-late)/max(np.linalg.norm(early),1e-9))
        train=[];validation=[]
        for family,rows in collected.items():
            # Independent acquisitions in source and target; references 3..5 never appear as inputs.
            pairs=[(rows[0]["signal"],rows[1]["signal"]),
                   (rows[1]["signal"],rows[2]["signal"]),
                   (rows[2]["signal"],rows[0]["signal"]),
                   (rows[0]["signal"],rows[2]["signal"])]
            if family.startswith("train_"):train+=pairs
            elif family.startswith("validation_"):validation+=pairs
        model_path=self.results.out/"models"/"complex_fid_noise2noise.pt"
        train_started=time.monotonic()
        training=train_complex_denoiser(train,validation,model_path,seed=self.seed)
        train_seconds=time.monotonic()-train_started
        test=collected["test_270"]
        source=test[0]["signal"]
        reference=np.mean(np.stack([x["signal"] for x in test[3:]]),axis=0)
        reference_fit=complex_fit(reference,10000,tracked_hz=self.tracked_hz)["target"]
        # Standard baselines share one input acquisition. Averaging uses 3 inputs and pays for them.
        inference_started=time.monotonic()
        neural_output=apply_complex_denoiser(source,model_path)
        inference_seconds=time.monotonic()-inference_started
        candidates={"window_fft_fit":source,"hankel_rank2":hankel_denoise(source),
                    "torch_noise2noise":neural_output,
                    "three_acquisition_average":np.mean(np.stack([x["signal"] for x in test[:3]]),axis=0)}
        metrics={}
        for method,signal in candidates.items():
            fit=complex_fit(signal,10000,tracked_hz=reference_fit["frequency_hz"])["target"]
            signed_amp=(fit["amplitude"]-reference_fit["amplitude"])/max(reference_fit["amplitude"],1e-9)
            signed_phase=math.atan2(math.sin(fit["phase_rad"]-reference_fit["phase_rad"]),
                                    math.cos(fit["phase_rad"]-reference_fit["phase_rad"]))
            signed_freq=fit["frequency_hz"]-reference_fit["frequency_hz"]
            amp,phase,freq=abs(signed_amp),abs(signed_phase),abs(signed_freq)
            residual=float(np.sqrt(np.mean(np.abs(signal-reference)**2)))/max(float(np.sqrt(np.mean(np.abs(reference)**2))),1e-9)
            error=float(math.sqrt(amp**2+phase**2+freq**2+residual**2))
            metrics[method]={"amplitude_relative_error":amp,"phase_error_rad":phase,
                             "frequency_error_hz":freq,"relative_fid_residual":residual,
                             "signed_amplitude_bias":signed_amp,"signed_phase_bias_rad":signed_phase,
                             "signed_frequency_bias_hz":signed_freq,
                             "combined_error":error,"input_acquisitions":3 if method=="three_acquisition_average" else 1}
        rotated=apply_complex_denoiser(source*1j,model_path)
        metrics["torch_noise2noise"]["rotation_equivariance_relative_error"]=float(
            np.linalg.norm(rotated-1j*candidates["torch_noise2noise"])/max(np.linalg.norm(candidates["torch_noise2noise"]),1e-9))
        baseline=metrics["window_fft_fit"]["combined_error"]
        for method,m in metrics.items():
            self.results.row(topic="denoising",method=method,block=2,phase="pilot",
                measurements=m["input_acquisitions"]+3,internal_repetitions="UNKNOWN",
                requested_samples=16000*(m["input_acquisitions"]+3),
                wall_seconds=sum(x["wall_seconds"] for x in test[:m["input_acquisitions"]+3]),
                compute_seconds=train_seconds if method=="torch_noise2noise" else None,
                error=m["combined_error"],uncertainty=None,
                improvement_vs_baseline=(baseline-m["combined_error"])/baseline if baseline else None,
                status="NEPRESVEDČIVÉ",reason="one held-out family/block; reference is finite independent average, not clean truth")
        summary={"status":"NEPRESVEDČIVÉ","reason":"real CPU training complete; one held-out family does not establish generalization",
                 "training":training,"training_seconds":train_seconds,"total_module_seconds":time.monotonic()-start,
                 "repeat_stability_relative_drift":stability,
                 "costs":{"training_acquisitions":12,"validation_acquisitions":6,
                          "test_acquisitions":6,"test_reference_acquisitions":3,
                          "first_use_local_seconds":train_seconds+inference_seconds,
                          "reused_model_local_seconds":inference_seconds,
                          "reused_model_input_acquisitions":1},
                 "split":split,"reference_fit":reference_fit,"metrics":metrics,
                 "noise2noise_assumption":"independence/stability tested only indirectly via repeat drift; no clean truth"}
        self.results.topic("denoising",2,summary)
        return summary


def main():
    ap=argparse.ArgumentParser(description="Real Gemini Lab five-topic pilot benchmark")
    ap.add_argument("--host",default="172.19.20.100")
    ap.add_argument("--port",type=int,default=8181)
    ap.add_argument("--blocks",type=int,default=3)
    ap.add_argument("--resume",type=Path,help="existing results/<run_id> folder; completed tasks are reused")
    ap.add_argument("--seed",type=int,default=240926)
    ap.add_argument("--no-upload",action="store_true")
    args=ap.parse_args()
    if not 3<=args.blocks<=20: ap.error("blocks must be 3..20")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    out=(args.resume or Path("results")/stamp).resolve()
    config={"host":args.host,"port":args.port,"blocks":args.blocks,"seed":args.seed,
            "sdk_required":"1.0.2","max_tasks_per_block":140,
            "max_requested_rf_us_per_block":10000.,
            "per_task_requested_rf_us":200.,"pause_seconds":2.,
            "initial_condition":"H historical physical baseline; makePps true; no persistent calibration write",
            "phase":"pilot","source_of_limits":"finite software plan, not certified hardware rating"}
    result=Results(out,config)
    result.data["environment"]={"python":sys.version.split()[0],"os":platform.platform(),
        "versions":{name:importlib.metadata.version(name) for name in
                    ("spinqlablink","numpy","scipy","scikit-learn","matplotlib","torch")}}
    result.save()
    fatal=False
    try:
        with HardwareLock(Path("~/.spinq_live_gemini.lock")),LiveHardware(out,host=args.host,port=args.port,
             max_tasks=config["max_tasks_per_block"]*args.blocks,
             max_requested_rf_us=config["max_requested_rf_us_per_block"]*args.blocks) as hw:
            bench=Benchmark(hw,result,args.seed)
            for block in range(args.blocks):
                for name,fn in (("calibration",bench.calibration_block),
                                ("acquisition",bench.acquisition_block),
                                ("pulse_tuning",bench.pulse_block),
                                ("robust_pulse",bench.robust_block)):
                    try:
                        print(f"Block {block+1}/{args.blocks}: {name}",flush=True)
                        fn(block)
                    except HardwareUncertain:
                        raise
                    except Exception as exc:
                        result.data["errors"].append(f"{name} block {block}: {type(exc).__name__}: {exc}")
                        result.topic(name,block,{"status":"NEPRESVEDČIVÉ","reason":str(exc)})
                        result.row(topic=name,method="module",block=block,phase="pilot",
                                   status="NEPRESVEDČIVÉ",reason=str(exc))
                # Independently continue across blocks; each block has fresh reference data.
            try:
                print("Denoising: collecting real FIDs and training CPU model",flush=True)
                bench.denoising()
            except HardwareUncertain:
                raise
            except Exception as exc:
                result.data["errors"].append(f"denoising: {type(exc).__name__}: {exc}")
                result.topic("denoising",2,{"status":"NEPRESVEDČIVÉ","reason":str(exc)})
    except Exception as exc:
        fatal=True
        result.data["errors"].append(f"GLOBAL STOP {type(exc).__name__}: {exc}")
        result.data["state"]="STOPPED_UNCERTAIN" if isinstance(exc,HardwareUncertain) else "STOPPED"
        result.save()
    else:
        result.data["state"]="PILOT_COMPLETED_WITH_LIMITATIONS"
    finally:
        result.data["finished_utc"]=utc_now()
        result.data["aggregate"]=aggregate_rows(result.data["rows"],result.data.get("frozen_plan"))
        result.save()
        plot_comparisons(out,result.data["rows"],result.data["topics"])
        bundle(out)
        if not args.no_upload:
            upload=publish_results(Path(__file__).resolve().parent,out/"results.zip",f"benchmark/{out.name}")
            result.data["upload"]=upload
            result.save();bundle(out)
        print(f"Report: {out/'REPORT.md'}",flush=True)
        print(f"Archive: {out/'results.zip'}",flush=True)
        print(f"Upload: {result.data['upload'].get('status')}",flush=True)
    return 2 if fatal else 0


if __name__=="__main__":
    raise SystemExit(main())
