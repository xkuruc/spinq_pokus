"""Single-owner Windows acquisition session for the eight-method benchmark.

Every acquisition passes through core.run_raw and the installed SDK. Numerical
methods consume only saved complex FIDs. This file does not run on import.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

from spinq_audit.common import atomic_json
from .abh import (CalibrationParticleFilter, CalibrationSetting,
                  calibration_particle_prior, choose_fid_acquisition_plan,
                  fit_calibration_joint, hahn_echo_timing, RabiRateMap, RFGPOptimizer,
                  rf_coarse_grid, rf_sequential_scan, iq_to_sdk)
from .core import (Capabilities, CapabilityUnavailable, RawFIDRecord,
                   Segment, SequenceIR, compile_sequence, run_raw)
from .signal import (MultipletSpec, estimate_noise, fft_local,
                     fit_complex_multiplet, validate_axis, vendor_fft_replica)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _rabi_period(width_us: list[float], coefficient: list[complex],
                 *, expected_us: tuple[float, float] = (120., 220.)) -> dict:
    """Complex signed Rabi fringe, not an unsigned peak-height fit."""
    x = np.asarray(width_us, float)
    y = np.asarray(coefficient, complex)
    if len(x) < 5 or len(np.unique(x)) < 5 or not np.all(np.isfinite(y)):
        raise ValueError("Five distinct signed Rabi observations are needed")
    total = np.r_[y.real, y.imag]
    best = None
    for period in np.linspace(expected_us[0], expected_us[1], 101):
        phi = 2*np.pi*x/period
        basis = np.column_stack((np.ones(len(x)), np.sin(phi), np.cos(phi)))
        coef = np.linalg.lstsq(basis, y, rcond=None)[0]
        error = float(np.sum(np.abs(y-basis@coef)**2))
        if best is None or error < best[0]:
            best = (error, period, coef)
    assert best is not None
    error, period, coef = best
    scatter = float(np.sum(np.abs(y-y.mean())**2))
    r2 = 1-error/max(scatter, 1e-30)
    if r2 < 0.6 or period <= expected_us[0]+1 or period >= expected_us[1]-1:
        raise ValueError(f"Rabi fringe not identified (R2={r2:.3f})")
    return {"period_us": float(period), "t90_us": float(period/4),
            "signed_complex_r2": float(r2), "fit_residual": float(error),
            "coefficient": [_jsonable(c) for c in coef]}


def _pilot_bands(record: RawFIDRecord) -> tuple[MultipletSpec,int]:
    spectrum = fft_local(record, window="hann", nfft=32768)
    frequency = spectrum["frequency_hz"]
    amplitude = np.abs(spectrum["spectrum"])
    # Exclude only the narrow DC neighborhood of a possible receiver offset.
    order = np.argsort(amplitude)[::-1]
    chosen = []
    for index in order:
        f = float(frequency[index])
        if abs(f) < 15 or abs(f) > 4700:
            continue
        if chosen and (amplitude[index] < .22*amplitude[chosen[0]] or
                       any(abs(f-frequency[previous]) < 180 for previous in chosen)):
            continue
        chosen.append(int(index))
        if len(chosen) == 2:
            break
    if not chosen:
        raise ValueError("Pilot contains no identifiable non-DC frequency peak")
    # The frequency identity is frozen before the comparison acquisitions.
    bands = sorted((max(-4900., float(frequency[i])-120.),
                    min(4900., float(frequency[i])+120.)) for i in chosen)
    if len(bands) == 2 and bands[0][1] >= bands[1][0]:
        bands = [bands[0]]
    primary_frequency=float(frequency[chosen[0]])
    primary_index=min(range(len(bands)),key=lambda i:abs(
        .5*(bands[i][0]+bands[i][1])-primary_frequency))
    return MultipletSpec(tuple(bands), starts_per_band=2, max_fit_points=1024),primary_index


class LocalSession:
    def __init__(self, hardware, results, *, blocks: int, seed: int):
        self.hw, self.results = hardware, results
        self.out = results.out
        self.blocks, self.seed = blocks, seed
        self.caps = Capabilities()
        self.pilot_records: dict[str, RawFIDRecord] = {}
        self.role_records: dict[str, list[RawFIDRecord]] = {
            role: [] for role in ("pilot", "train", "validation", "test")}
        self.noise = None
        self.spec: MultipletSpec | None = None
        self.primary_component_index=0
        self.t90_us: float | None = None
        self.reference_frequency_hz: float | None = None
        self.rf_map: RabiRateMap | None = None
        self.a_filters: dict[str, CalibrationParticleFilter] = {}
        self.a_observations: dict[str, list] = {}
        self.a_receiver_offset=0j
        self.a_receiver_gain=1+0j
        self.a_observation_cov=np.eye(2)
        self.h_gp: RFGPOptimizer | None = None
        self.h_history: dict[str, list] = {key: [] for key in
            ("coarse", "sequential", "nelder_mead", "gp")}

    def acquire(self, key: str, segments: tuple[Segment, ...], *,
                count: int = 16000, role: str = "train", block: int = -1,
                family: str = "generic", idle_probe: bool = False) -> RawFIDRecord:
        if role not in self.role_records:
            raise ValueError("Unknown dataset role")
        sequence = SequenceIR(segments, sample_count=count, label=key)
        request = compile_sequence(sequence, self.caps, idle_probe=idle_probe)
        record = run_raw(request, key=key, hardware=self.hw, output=self.out)
        record.metadata.update({"measurement_block": f"block-{block:02d}" if block >= 0 else "pilot",
                                "setting_family": family, "dataset_role": role,
                                "requested_rf_duration_us": request.rf_duration_us,
                                "sequence_duration_us": request.sequence_duration_us})
        record.save(self.out/"raw")
        self.results.data["hardware_results_present"] = True
        self.results.data["budgets"]["acquisitions_used"] = self.hw.task_count
        self.results.data["budgets"]["rf_duration_us_used"] = self.hw.rf_us
        self.results.save()
        if not any(existing.key == key for existing in self.role_records[role]):
            self.role_records[role].append(record)
        return record

    def pulse(self, key: str, width: float, *, amplitude: float = 100.,
              phase: float = 90., count: int = 16000, role: str = "train",
              block: int = -1, family: str = "rabi") -> RawFIDRecord:
        return self.acquire(key, (Segment(0., width, amplitude_pct=amplitude,
                                          phase_deg=phase),), count=count,
                            role=role, block=block, family=family)

    def coefficient(self, record: RawFIDRecord) -> complex:
        if self.spec is None:
            # Matched projection uses an actual measured pilot template. It
            # retains sign/phase and never reads vendor FFT/fit fields.
            template = self.pilot_records["pilot_40_r0"].fid
            n = min(len(template), len(record.fid), 2048)
            return complex(np.vdot(template[:n], record.fid[:n]) /
                           max(np.vdot(template[:n], template[:n]).real, 1e-12))
        started = time.monotonic_ns()
        fit = fit_complex_multiplet(record, self.spec, self.noise)
        finished = time.monotonic_ns()
        atomic_json(self.out/"models"/f"{record.key}_multiplet.json", _jsonable({
            "fit": fit, "analysis_started_monotonic_ns": started,
            "analysis_finished_monotonic_ns": finished}))
        mode = fit["modes"][self.primary_component_index]
        return complex(mode["coefficient_re"], mode["coefficient_im"])

    def frequency(self, record: RawFIDRecord) -> float:
        if self.spec is None:
            raise RuntimeError("Pilot frequency identities not frozen")
        fit = fit_complex_multiplet(record, self.spec, self.noise)
        return float(fit["modes"][self.primary_component_index]["frequency_hz"])

    def pilot(self) -> dict:
        # Repeated controls establish noise and drift at the same physical
        # setting. The remaining sweeps are short 40-200 us H pulses.
        for repeat in range(3):
            key = f"pilot_40_r{repeat}"
            self.pilot_records[key] = self.pulse(key, 40., role="pilot", family="repeat")
        for width in (80., 120., 160., 200.):
            key = f"pilot_{int(width)}"
            self.pilot_records[key] = self.pulse(key, width, role="pilot")
        pilot_failures={}
        try:
            self.noise = estimate_noise([self.pilot_records[f"pilot_40_r{i}"] for i in range(3)])
        except Exception as exc:
            pilot_failures["independent_noise"]=f"{type(exc).__name__}: {exc}"
        try:
            self.spec,self.primary_component_index = _pilot_bands(self.pilot_records["pilot_40_r0"])
        except Exception as exc:
            pilot_failures["component_identity"]=f"{type(exc).__name__}: {exc}"
        try:
            rabi = _rabi_period([40.,80.,120.,160.,200.],
                                [self.coefficient(self.pilot_records[k]) for k in
                                 ("pilot_40_r0","pilot_80","pilot_120","pilot_160","pilot_200")])
            self.t90_us = rabi["t90_us"]
            rabi["amplitude_pct"] = 100.
            components=[complex(z["re"],z["im"]) for z in rabi["coefficient"]]
            if abs(components[1])<1e-9:
                raise ValueError("Rabi signed response has no identifiable receiver scale")
            self.a_receiver_offset=components[0]
            self.a_receiver_gain=components[1]
            controls=np.asarray([self.coefficient(self.pilot_records[f"pilot_40_r{i}"])
                                 for i in range(3)])
            normal=(controls-self.a_receiver_offset)/self.a_receiver_gain
            scatter=np.cov(np.column_stack((normal.real,normal.imag)),rowvar=False)
            residual_var=float(rabi["fit_residual"])/(5*abs(self.a_receiver_gain)**2)
            self.a_observation_cov=np.asarray(scatter)+max(residual_var,1e-8)*np.eye(2)
        except Exception as exc:
            pilot_failures["rabi_t90"]=f"{type(exc).__name__}: {exc}"
            self.t90_us=None
            rabi={"status":"METHOD_FAILED","reason":pilot_failures["rabi_t90"],
                  "amplitude_pct":100.}
        try:
            self.reference_frequency_hz = self.frequency(self.pilot_records["pilot_40_r0"])
        except Exception as exc:
            pilot_failures["reference_frequency"]=f"{type(exc).__name__}: {exc}"
        axis = validate_axis(self.pilot_records["pilot_40_r0"])
        length_pilot = []
        for count in (4000,8000):
            for repeat in range(2):
                key = f"pilot_count{count}_r{repeat}"
                record = self.pulse(key, 40., count=count, role="pilot", family="fid_length")
                self.pilot_records[key] = record
                try:
                    length_pilot.append({"sample_count": count,
                                         "estimate_hz": self.frequency(record),
                                         "wall_seconds": record.metadata["wall_seconds"]})
                except Exception as exc:
                    pilot_failures[f"frequency_{key}"]=f"{type(exc).__name__}: {exc}"
        for repeat in range(3):
            record = self.pilot_records[f"pilot_40_r{repeat}"]
            try:
                length_pilot.append({"sample_count":16000,
                                     "estimate_hz":self.frequency(record),
                                     "wall_seconds":record.metadata["wall_seconds"]})
            except Exception as exc:
                pilot_failures[f"frequency_{record.key}"]=f"{type(exc).__name__}: {exc}"
        full=[r["estimate_hz"] for r in length_pilot if r["sample_count"]==16000]
        scatter = float(np.std(full, ddof=1)) if len(full)>=2 else None
        fid_target = max(3., 2*scatter) if scatter is not None else None
        try:
            if fid_target is None:raise ValueError("No independent full-length frequency reference")
            acquisition_plan = choose_fid_acquisition_plan(length_pilot,
                target_se_hz=fid_target, max_repeats=8)
        except Exception as exc:
            acquisition_plan={"status":"REFERENCE_INADEQUATE","reason":str(exc)}
        pilot = {"rabi":rabi,"noise":({"covariance":self.noise.re_im_covariance.tolist(),
                  "lag_one":self.noise.lag_one_correlation,
                  "independent_repetitions":self.noise.repetitions} if self.noise else None),
                 "multiplet_bands_hz":self.spec.bands_hz if self.spec else None,
                 "primary_component_index":self.primary_component_index,
                 "reference_frequency_hz":self.reference_frequency_hz,
                 "sample_hz":axis.sample_hz,
                 "fid_length_rows":length_pilot,"fid_acquisition_plan":acquisition_plan,
                 "failures":pilot_failures,
                 "physical_adc_clock_verified":False}
        self.results.data["pilot"] = _jsonable(pilot)
        self.results.data["frozen_plan"] = {
            "modules":"ABCDEFGH","blocks":self.blocks,"seed":self.seed,
            "frequency_bands_hz":self.spec.bands_hz if self.spec else None,"t90_us":self.t90_us,
            "frequency_tolerance_hz":fid_target,
            "rabi_t90_tolerance_us":max(5.,.1*self.t90_us) if self.t90_us else None,
            "acquisition_plan":acquisition_plan,
            "sample_count_candidates":[4000,8000,16000],
            "rf_amplitude_pct_range":[60.,100.],
            "method_order_rule":"seeded rotation per block; shared pilot charged to all methods",
            "internal_repetitions":"UNKNOWN"}
        self.results.save()
        return pilot

    def timing_probe(self) -> dict:
        """Controlled acceptance/effect check; equality is not failure.

        Idle effect is interpreted only relative to interleaved return controls.
        The terminal delay required for echo has a separate direct comparison.
        """
        single = self.pulse("timing_return_0", 40., role="pilot", family="timing")
        equal = self.acquire("timing_split_equal", (
            Segment(0.,20.),Segment(20.,20.)), role="pilot", family="timing", idle_probe=True)
        opposite = self.acquire("timing_split_opposite", (
            Segment(0.,20.),Segment(20.,20.,phase_deg=270.)),
            role="pilot", family="timing", idle_probe=True)
        reference = self.pulse("timing_return_1",40.,role="pilot",family="timing")
        first = self.acquire("timing_gap_10", (
            Segment(0.,20.),Segment(30.,20.)),role="pilot",family="timing",idle_probe=True)
        second = self.acquire("timing_gap_20", (
            Segment(0.,20.),Segment(40.,20.)),role="pilot",family="timing",idle_probe=True)
        terminal = self.acquire("timing_terminal_idle", (
            Segment(0.,40.),Segment(40.,10.,amplitude_pct=0.)),
            role="pilot",family="timing",idle_probe=True)
        last = self.pulse("timing_return_2",40.,role="pilot",family="timing")
        values = [self.coefficient(record) for record in
                  (single,equal,opposite,reference,first,second,terminal,last)]
        drift = max(abs(values[0]-values[3]),abs(values[3]-values[7]),1e-9)
        equal_consistent = abs(values[1]-values[0]) < max(3*drift,.3*abs(values[0]))
        phase_effect = abs(values[2]-values[1]) > max(3*drift,.1*abs(values[0]))
        gap_effect = abs(values[5]-values[4]) > 3*drift
        terminal_effect = abs(values[6]-values[3]) > 3*drift
        # A measured difference alone cannot certify the duration semantics;
        # require a monotone complex phase progression at two gap durations.
        progression = float(np.angle((values[5]-values[3]) /
                                     (values[4]-values[3]+1e-12)))
        self.caps = replace(self.caps,
            segment_sequence_verified=bool(equal_consistent and phase_effect),
            zero_amplitude_delay_verified=bool(gap_effect and terminal_effect and
                                               equal_consistent and phase_effect and
                                               math.isfinite(progression)))
        result = {"equal_split_consistent":bool(equal_consistent),
                  "phase_order_effect_observed":bool(phase_effect),
                  "gap_length_effect_observed":bool(gap_effect),
                  "terminal_idle_effect_observed":bool(terminal_effect),
                  "gap_phase_progression_rad":progression,
                  "return_control_drift":float(drift),
                  "sequence_verified":self.caps.segment_sequence_verified,
                  "zero_amplitude_delay_verified":self.caps.zero_amplitude_delay_verified,
                  "interpretation":"software timing feasibility only; echo identity still needs independent check"}
        self.results.data["pilot"]["timing_probe"] = result
        self.results.save()
        return result

    def vendor_fft_control(self) -> dict:
        """Freeze one vendor-spectrum replica scale, then test untouched FIDs.

        This is a separate provenance experiment: these arrays never enter
        frequency fits, RF design, training labels or stopping rules.
        """
        def vendor(record):
            path=self.out/"vendor_reference"/f"{record.key}.npz"
            if not path.is_file():raise ValueError(f"Vendor FFT absent for {record.key}")
            with np.load(path,allow_pickle=False) as source:
                real=source["fftRe"].copy();imag=source["fftIm"].copy()
            if real.ndim!=2 or imag.ndim!=2 or real.shape!=imag.shape or real.shape[1]!=2:
                raise ValueError("Vendor real/imag spectral arrays incompatible")
            if not np.array_equal(real[:,0],imag[:,0]):
                raise ValueError("Vendor FFT axes differ")
            return real[:,1]+1j*imag[:,1]
        train=self.pilot_records["pilot_40_r0"]
        target=vendor(train)
        base=vendor_fft_replica(train,{"complex_scale_re":1.,
                "axis_scale_original_units":100.,"nfft":len(target)})["spectrum"]
        if len(base)!=len(target):raise ValueError("Vendor spectrum length differs from replica")
        scale=np.vdot(base,target)/max(np.vdot(base,base).real,1e-12)
        params={"complex_scale_re":float(scale.real),"complex_scale_im":float(scale.imag),
                "axis_scale_original_units":100.,"nfft":len(target),
                "status":"frozen working hypothesis, not vendor server source"}
        rows=[]
        for key in ("pilot_40_r1","pilot_40_r2","pilot_80","pilot_120"):
            record=self.pilot_records[key]
            actual=vendor(record)
            estimated=vendor_fft_replica(record,params)["spectrum"]
            plain=fft_local(record,window="none",nfft=len(actual))["spectrum"]
            rows.append({"task":key,"relative_replica_error":float(np.linalg.norm(actual-estimated)/
                    max(np.linalg.norm(actual),1e-12)),
                "relative_plain_fft_error_after_frozen_scale":float(np.linalg.norm(actual-scale*plain)/
                    max(np.linalg.norm(actual),1e-12)),
                "source":"same acquired FID and separate vendor chart"})
        result={"frozen_parameters":params,"heldout_rows":rows,
            "server_fft_time_isolated":False,"server_processing_offload_unverified":True}
        atomic_json(self.out/"models"/"vendor_fft_replica.json",result)
        self.results.data["pilot"]["vendor_fft_replica_control"]=result
        self.results.save()
        return result

    def _init_a(self):
        assert self.t90_us is not None
        rate = 1/(4*self.t90_us*1e-6)
        for index,name in enumerate(("fixed_fit","coarse_fine_fit","posterior_fixed",
                                     "bayes_variance","bayes_thresholded")):
            particles = calibration_particle_prior((-20.,20.),(.8,1.2),(-90.,90.),
                                                    count=512,seed=self.seed+index)
            self.a_filters[name] = CalibrationParticleFilter(
                particles,rf_hz_at_100pct=rate,seed=self.seed+index)
            self.a_observations[name] = []

    def module_a_block(self, block: int) -> None:
        if not self.a_filters: self._init_a()
        assert self.t90_us is not None and self.noise is not None
        candidates = [CalibrationSetting("rabi",w*1e-6,100.,90.,expected_wall_s=6.)
                      for w in (40.,80.,120.,160.,200.)]
        candidates += [CalibrationSetting("phase",max(40.,self.t90_us)*1e-6,
                          100.,phase,expected_wall_s=6.) for phase in (0.,90.,180.,270.)]
        if self.caps.segment_sequence_verified and self.caps.zero_amplitude_delay_verified:
            for gap in (10.,20.):
                for phase in (0.,90.):
                    if 2*self.t90_us+gap<=self.caps.maximum_task_span_us:
                        candidates.append(CalibrationSetting("ramsey",self.t90_us*1e-6,
                            100.,phase,free_s=gap*1e-6,expected_wall_s=7.))
        cov=self.a_observation_cov
        tolerances=(self.results.data["frozen_plan"]["frequency_tolerance_hz"] or 20.,
                    .1,math.radians(15.))
        names=list(self.a_filters)
        names=names[block%len(names):]+names[:block%len(names)]
        # Three arms per block; the five policies rotate over ten blocks so
        # each receives six acquisitions under the same total study cap.
        for name in names[:3]:
            filt=self.a_filters[name]
            if name=="fixed_fit" or name=="posterior_fixed":
                setting=candidates[(block + (0 if name=="fixed_fit" else 2))%len(candidates)]
            elif name=="coarse_fine_fit":
                setting=candidates[(block*2)%len(candidates)] if block<self.blocks//2 else \
                        min(candidates,key=lambda s:abs(s.duration_s*1e6-self.t90_us))
            else:
                policy="variance" if name=="bayes_variance" else "thresholded_per_second"
                setting,_=filt.select(candidates,cov,tolerances,policy=policy)
            width=setting.duration_s*1e6
            if setting.family=="ramsey":
                record=self.acquire(f"A_b{block:02d}_{name}",(
                    Segment(0.,width,phase_deg=0.),
                    Segment(width+setting.free_s*1e6,width,phase_deg=setting.phase_deg)),
                    block=block,family="calibration_ramsey")
            else:
                record=self.pulse(f"A_b{block:02d}_{name}",width,
                                  phase=setting.phase_deg,block=block,family="calibration")
            observed=(self.coefficient(record)-self.a_receiver_offset)/self.a_receiver_gain
            self.a_observations[name].append((setting,observed,cov))
            # Receiver gain is the fixed pilot normalization; all arms use
            # identical scaling. A bad posterior update must not stop other
            # arms or discard the genuine FID that was just acquired.
            try:
                update=filt.update(setting,observed,cov)
            except Exception as exc:
                update={"status":"METHOD_FAILED","reason":f"{type(exc).__name__}: {exc}"}
                self.results.data["errors"].append(f"A {name} block {block}: {update['reason']}")
            atomic_json(self.out/"models"/f"A_{name}_b{block:02d}.json", _jsonable({
                "setting":setting.__dict__,"posterior":update,
                "observed":observed,"task":record.key}))
            self.results.row(module="A",method=name,baseline="fixed_fit",
                task=record.key,block=block,data_source="raw complex FID",
                acquisitions=len(self.a_observations[name]),
                wall_seconds=record.metadata["wall_seconds"],
                rf_duration_us=(2*width if setting.family=="ramsey" else width),
                status=("METHOD_FAILED" if
                    update.get("status")=="METHOD_FAILED" else "RUNNING"),
                reason=update.get("reason"))
        self.results.module("A","RUNNING","Five acquisition policies; final independent check pending")

    def module_b_block(self, block: int) -> None:
        target=self.results.data["frozen_plan"]["frequency_tolerance_hz"]
        plan=self.results.data["pilot"]["fid_acquisition_plan"]
        adaptive_count=int(plan.get("sample_count",8000)) if plan.get("status")=="PLAN_ONLY" else 8000
        adaptive_name="adaptive" if plan.get("status")=="PLAN_ONLY" else "diagnostic_8000"
        order=[(adaptive_name,adaptive_count),("fixed_16000",16000)]
        if block%2: order.reverse()
        reference=self.reference_frequency_hz
        for method,count in order:
            record=self.pulse(f"B_b{block:02d}_{method}",40.,count=count,
                              block=block,family="fid_length")
            frequency=self.frequency(record)
            error=abs(frequency-reference) if reference is not None else None
            self.results.row(module="B",method=method,baseline="fixed_16000",
                task=record.key,block=block,data_source="raw complex FID",
                acquisitions=1,wall_seconds=record.metadata["wall_seconds"],
                error=error,tolerance=target,
                status="RUNNING" if error is not None else "REFERENCE_INADEQUATE")
        self.results.module("B","RUNNING",
            "Adaptive FID length physically tested; echo T2 requires verified in-sequence timing")

    def module_b_echo_probe(self) -> dict:
        if not (self.caps.segment_sequence_verified and self.caps.zero_amplitude_delay_verified):
            raise CapabilityUnavailable("Hahn echo idle/segment timing not independently verified")
        if self.t90_us is None or self.t90_us <= 0:
            raise ValueError("Measured t90 required for Hahn echo timing")
        values=[]
        for gap_us in (5.,15.):
            w90=self.t90_us
            w180=2*w90
            tail=.5*w90+gap_us
            total=3.5*w90+2*gap_us
            if total>self.caps.maximum_task_span_us:
                raise CapabilityUnavailable("Echo probe exceeds historical 200 us study envelope")
            timing=hahn_echo_timing(width_90_s=w90*1e-6,width_180_s=w180*1e-6,
                inter_pulse_gap_s=gap_us*1e-6,acquisition_start_s=total*1e-6,
                acquisition_end_s=(total+10.)*1e-6,timing_verified=True)
            record=self.acquire(f"B_echo_gap{int(gap_us)}",(
                Segment(0.,w90,phase_deg=0.),
                Segment(w90+gap_us,w180,phase_deg=90.),
                Segment(w90+gap_us+w180,tail,amplitude_pct=0.)),
                role="pilot",family="echo_probe")
            coefficient=self.coefficient(record)
            values.append({"gap_us":gap_us,"timing":timing,
                           "coefficient":_jsonable(coefficient),"task":record.key})
        result={"echo_probe":values,"t2_estimate":None,
            "interpretation":"Two short TEs test sequence response; neither echo identity nor T2 decay model is established"}
        self.results.data["pilot"]["echo_probe"]=result
        self.results.save()
        return result

    def collect_f_block(self, block:int) -> None:
        """Independent measured FIDs for local train/validation/test splits."""
        if self.blocks==3:
            role, width, repeats = (("train",40.,3),("validation",80.,3),
                                    ("test",120.,3))[block]
        elif block in (0,1):role,width,repeats="train",40.,3
        elif block in (2,3):role,width,repeats="validation",80.,3
        elif block in (4,6,8):role,width,repeats="test",120.,4
        else:return
        for repeat in range(repeats):
            self.pulse(f"F_{role}_b{block:02d}_r{repeat}",width,
                       role=role,block=block,family=f"F_{role}_{int(width)}")

    def _fit_amplitude_rate(self, phase: float, amplitude: float,
                            reference: dict | None = None) -> tuple[float,dict]:
        widths=(40.,80.,120.,160.,200.) if amplitude==100. else (40.,80.,120.)
        observations=[]
        for width in widths:
            key=f"H_map_p{int(phase)}_a{int(amplitude)}_w{int(width)}"
            record=self.pulse(key,width,amplitude=amplitude,phase=phase,
                              role="pilot",family="rf_map")
            observations.append(self.coefficient(record))
        if amplitude==100.:
            fit=_rabi_period(list(widths),observations)
            return 1/(fit["period_us"]*1e-6),fit
        if reference is None:
            raise ValueError("Independent 100% Rabi reference required")
        # At lower amplitude, three points cannot support a free complex
        # offset plus sine and cosine coefficients. Freeze their shape from
        # the independent 100% sweep; fit only rate and a real contrast.
        refcoef=np.asarray([complex(z["re"],z["im"]) for z in reference["coefficient"]])
        upper=max(220.,4*self.t90_us*100./amplitude*1.2)
        grid=np.linspace(max(120.,4*self.t90_us*100./amplitude*.7),upper,100)
        x=np.asarray(widths)
        y=np.asarray(observations)
        scored=[]
        for period in grid:
            fringe=refcoef[1]*np.sin(2*np.pi*x/period)+refcoef[2]*np.cos(2*np.pi*x/period)
            contrast=float(np.real(np.vdot(fringe,y-refcoef[0]))/
                           max(np.vdot(fringe,fringe).real,1e-12))
            contrast=np.clip(contrast,.3,1.5)
            scored.append((float(np.sum(np.abs(y-(refcoef[0]+contrast*fringe))**2)),
                           period,contrast))
        best=min(scored)
        if best[1] in (grid[0],grid[-1]):
            raise ValueError("Low-amplitude Rabi rate hit the pilot search boundary")
        return float(1/(best[1]*1e-6)),{"period_us":float(best[1]),
            "fit_residual":best[0],"contrast_relative_to_100pct":float(best[2]),
            "model":"frozen 100% complex fringe shape; rate and real contrast fitted"}

    def h_map(self) -> dict:
        rates={"x":[],"y":[]}
        fits={"x":[],"y":[]}
        for phase,label in ((0.,"x"),(90.,"y")):
            reference_rate,reference_fit=self._fit_amplitude_rate(phase,100.)
            by_amplitude={100.:(reference_rate,reference_fit)}
            for amplitude in (60.,80.):
                by_amplitude[amplitude]=self._fit_amplitude_rate(phase,amplitude,reference_fit)
            for amplitude in (60.,80.,100.):
                rate,fit=by_amplitude[amplitude]
                rates[label].append(rate)
                fits[label].append(fit)
        self.rf_map=RabiRateMap([60.,80.,100.],rates["x"],rates["y"])
        mapping={"amplitude_pct":[60.,80.,100.],"effective_rate_x_hz":rates["x"],
                 "effective_rate_y_hz":rates["y"],
                 "fits":fits,
                 "nonlinearity":self.rf_map.non_linearity,
                 "scope":"effective amplitude/phase calibration; independent DAC branches unverified"}
        self.results.data["pilot"]["rf_map"]=mapping
        self.results.save()
        return mapping

    def module_h_block(self, block: int) -> None:
        if self.rf_map is None or self.t90_us is None:
            raise ValueError("Independent x/y Rabi map unavailable")
        # Conservative sample/relative scale bounds are inside measured 60-100%
        # command support, and no clipping is permitted. Use 80% base amplitude.
        bounds=((.88,1.12),(.88,1.12))
        if self.h_gp is None: self.h_gp=RFGPOptimizer(bounds,seed=self.seed)
        methods=list(self.h_history)
        # One common return control at 45 degrees per block anchors receiver
        # drift. This is a relative complex FID target, not a Bloch vector.
        ref_amp=math.hypot(55.,55.)
        reference=self.pulse(f"H_ref_b{block:02d}",self.t90_us,
            amplitude=ref_amp,phase=45.,block=block,family="rf_return_reference")
        target=self.coefficient(reference)
        evals_per_method=6 if self.blocks>=10 else 2
        total_evaluations=4*evals_per_method
        start=(block*total_evaluations)//self.blocks
        stop=((block+1)*total_evaluations)//self.blocks
        for ordinal in range(start,stop):
            method=methods[ordinal%4]
            index=len(self.h_history[method])
            if method=="gp":
                point,proposal=self.h_gp.propose(candidate_count=256)
            elif method=="coarse":
                point=rf_coarse_grid(bounds,3)[index]
                proposal={"method":"frozen_coarse_grid"}
            elif method=="sequential":
                point=rf_sequential_scan(bounds,5)[index]
                proposal={"method":"frozen_coordinate_scan"}
            else:
                point=self._next_h_nelder_mead(bounds)
                proposal={"method":"measured_bounded_nelder_mead"}
            sample_scale,relative_scale=map(float,point)
            pulse=iq_to_sdk([55.],[55.],sample_scale=sample_scale,
                            relative_scale=relative_scale,
                            duration_s=self.t90_us*1e-6,verified_headroom_pct=100.)[0]
            amplitude=pulse["amplitude_pct"]
            if not 60<=amplitude<=100:
                raise CapabilityUnavailable("H IQ proposal outside measured 60-100% RF map")
            record=self.pulse(f"H_b{block:02d}_{method}_{index}",self.t90_us,
                amplitude=amplitude,phase=pulse["phase_deg"],
                block=block,family="rf_compensation")
            observed=self.coefficient(record)
            cost=float(abs(observed-target)**2)
            sigma2=float(np.trace(self.a_observation_cov)*abs(self.a_receiver_gain)**2)
            variance=max(2*sigma2*cost+sigma2**2,1e-12)
            self.h_history[method].append({"point":list(point),"cost":cost,
                "cost_variance_from_independent_pilot":variance,"task":record.key,
                "command_amplitude_pct":amplitude,"command_phase_deg":pulse["phase_deg"],
                "interpolated_effective_rf_rate_hz":self.rf_map.rate_hz(amplitude,pulse["phase_deg"]),
                "proposal":proposal,"reference_task":reference.key})
            if method=="gp":self.h_gp.observe(sample_scale,relative_scale,
                                               cost=cost,cost_variance=variance)
            self.results.row(module="H",method=method,baseline="coarse",
                task=record.key,block=block,data_source="raw complex FID; IQ command mapped to SDK",
                acquisitions=index+1,wall_seconds=record.metadata["wall_seconds"],
                error=math.sqrt(cost),status="RUNNING",
                reason="relative complex FID cost; full x/y/z Bloch readout unavailable")
        self.results.module("H","RUNNING",
            "Effective RF map, matched 2D IQ command and four measured optimizers; full Bloch readout still unavailable")

    def _next_h_nelder_mead(self,bounds):
        """Replay deterministic cached scores to expose SciPy's next simplex query.

        Each new objective call is a real acquisition made by module_h_block;
        replaying cached points consumes no RF and preserves interleaving.
        """
        history=self.h_history["nelder_mead"]
        cache={tuple(round(float(v),12) for v in row["point"]):row["cost"] for row in history}
        class NewPoint(Exception):
            def __init__(self,point):self.point=tuple(map(float,point))
        def score(point):
            key=tuple(round(float(v),12) for v in point)
            if key in cache:return cache[key]
            raise NewPoint(point)
        simplex=np.asarray(((1.,1.),(1.04,1.),(1.,1.04)),float)
        try:
            minimize(score,[1.,1.],method="Nelder-Mead",bounds=bounds,
                     options={"initial_simplex":simplex,"maxfev":50,
                              "xatol":1e-5,"fatol":1e-8})
        except NewPoint as next_point:
            return next_point.point
        raise ValueError("Nelder-Mead proposal exhausted without new measured point")

    def finish_a_b_h(self) -> None:
        if self.a_filters:
            summaries={name:filt.summary() for name,filt in self.a_filters.items()}
            fits={}
            for name,rows in self.a_observations.items():
                if len(rows)>=3:
                    try:fits[name]=fit_calibration_joint(rows,
                        rf_hz_at_100pct=1/(4*self.t90_us*1e-6))
                    except Exception as exc:fits[name]={"status":"METHOD_FAILED","reason":str(exc)}
            atomic_json(self.out/"models"/"A_summary.json",_jsonable({
                "posteriors":summaries,"joint_fits":fits,
                "warning":"No independent phase/frequency reference in current H-only path"}))
            self.results.module("A","REFERENCE_INADEQUATE",
                "Physical Rabi/phase design and five estimators ran; independent joint parameter reference and Ramsey timing remain unverified",
                posteriors=summaries)
        b_rows=[r for r in self.results.data["rows"] if r["module"]=="B"]
        if b_rows:
            status="UNVERIFIED_TIMING" if not self.caps.zero_amplitude_delay_verified else "REFERENCE_INADEQUATE"
            plan=self.results.data["pilot"].get("fid_acquisition_plan",{})
            if plan.get("status")=="MODEL_MISMATCH":
                status="METHOD_FAILED"
            pairs=[]
            for block in range(self.blocks):
                own={r["method"]:r for r in b_rows if r["block"]==block}
                if "adaptive" in own and "fixed_16000" in own and all(
                    isinstance(own[name].get("error"),(int,float)) for name in
                    ("adaptive","fixed_16000")):
                    pairs.append(own["fixed_16000"]["error"]-own["adaptive"]["error"])
            comparison={"paired_blocks":len(pairs),"status":"INSUFFICIENT_BLOCKS"}
            if len(pairs)>=3:
                rng=np.random.default_rng(self.seed+42)
                sample=np.asarray(pairs)
                boot=np.mean(sample[rng.integers(0,len(sample),(1000,len(sample)))],axis=1)
                comparison={"paired_blocks":len(pairs),
                    "mean_absolute_error_reduction_hz":float(np.mean(sample)),
                    "bootstrap_ci95_hz":list(map(float,np.quantile(boot,[.025,.975]))),
                    "status":"EXPLORATORY_BLOCK_BOOTSTRAP"}
            self.results.module("B",status,
                "Physical FID-length arms measured; echo T2 requires independently validated echo/refocus and adequate TE range"
                + ("; frozen component frequency changed with FID length" if plan.get("status")=="MODEL_MISMATCH" else ""),
                fid_length_paired_comparison=comparison)
        if any(self.h_history.values()):
            atomic_json(self.out/"models"/"H_history.json",_jsonable(self.h_history))
            self.results.module("H","REFERENCE_INADEQUATE",
                "Effective x/y map and bounded 2D command search measured; independent x/y/z Bloch reconstruction and repetition cost unavailable",
                measured_evaluations={name:len(rows) for name,rows in self.h_history.items()},
                gp_posterior_used=any(row.get("proposal",{}).get("method")=="matern_gp_expected_improvement"
                                      for row in self.h_history["gp"]))
