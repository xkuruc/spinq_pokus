"""Offline regression checks for F/G; no hardware or vendor derived arrays."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spinq_local.fg import (
    DDPulse, Gate, apply_analog_readout, apply_residual_readout, apply_tvcondnet, average_fids, bounded_anneal,
    circuit_unitary, compile_virtual_z, design_contextual_dd,
    fit_analog_readout, fit_exponential_fid, g_ablation_plan,
    grouped_split, hankel_low_rank_fid, rank_mappings, simplify_circuit,
    toggling_integrals, train_residual_readout, train_tvcondnet, tv_denoise, unitary_equivalent,
)
from spinq_local.analysis_runner import analyze_de_fg, _effective_one_spin_model, _analyze_d
from spinq_local.core import RawFIDRecord
from spinq_local.report import Results
from spinq_local import cde


class DenoisingChecks(unittest.TestCase):
    def test_split_requires_whole_sessions_families_and_independent_reference(self):
        rows=[{"session_id":f"s{i}","setting_family":f"f{i}","acquisition_id":f"a{i}"}
              for i in range(6)]
        split=grouped_split(rows,seed=2)
        self.assertEqual(sorted(sum(split.values(),[])),list(range(6)))
        rows[0]["reference_ids"]=["a0"]
        with self.assertRaisesRegex(ValueError,"own reference"):
            grouped_split(rows)
        rows[0]["reference_ids"]=["a1"]
        rows[0]["setting_family"]="f1"
        # The linked pair must move together, not one train and one test.
        split=grouped_split(rows)
        self.assertTrue(any({0,1}.issubset(set(v)) for v in split.values()))

    def test_complex_tv_couples_channels_and_preserves_global_phase(self):
        rng=np.random.default_rng(4)
        clean=np.r_[np.full(64,1+0.4j),np.full(64,-0.3+0.8j)]
        noisy=clean+0.25*(rng.normal(size=128)+1j*rng.normal(size=128))
        estimate=tv_denoise(noisy,0.55,maxiter=100)
        self.assertLess(np.mean(np.abs(estimate-clean)**2),np.mean(np.abs(noisy-clean)**2))
        rotated=tv_denoise(noisy*np.exp(0.7j),0.55,maxiter=100)
        self.assertLess(np.linalg.norm(rotated-estimate*np.exp(0.7j))/np.linalg.norm(estimate),2e-4)

    def test_memory_bounded_hankel_and_variable_projection(self):
        rng=np.random.default_rng(3)
        fs=512.0;t=np.arange(512)/fs
        signal=1.4*np.exp((-9+2j*np.pi*31)*t)+0.3j*np.exp((-16-2j*np.pi*67)*t)
        noisy=signal+0.13*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
        restored=hankel_low_rank_fid(noisy,rank=2,window=60,max_elements=40000)
        self.assertLess(np.mean(np.abs(restored-signal)**2),np.mean(np.abs(noisy-signal)**2))
        fit=fit_exponential_fid(noisy,fs,[31,-67],decay_bounds_hz=(1,50))
        self.assertLess(np.mean(np.abs(fit["reconstruction"]-signal)**2),np.mean(np.abs(noisy-signal)**2))
        with self.assertRaises(ValueError):
            hankel_low_rank_fid(noisy,rank=2,window=300,max_elements=40000)

    def test_average_refuses_mismatched_acquisition_axes(self):
        a=np.ones(8,dtype=complex)
        axis=np.arange(8)*.001
        with self.assertRaisesRegex(ValueError,"axes do not match"):
            average_fids([a,a],axes=[axis,axis+.1])
        self.assertTrue(np.allclose(average_fids([a,2*a],axes=[axis,axis]),1.5*a))

    def test_real_and_complex_network_branches_predict_noise_residual(self):
        try:
            import torch  # noqa: F401
        except (ImportError,OSError,RuntimeError):
            self.skipTest("CPU PyTorch unavailable; benchmark reports DEPENDENCY_FAILED")
        rng=np.random.default_rng(7)
        t=np.arange(64)
        def example(i,group):
            clean=np.exp((-0.015+0.14j)*t)*np.exp(0.3j*i)
            noisy=clean+0.05*(rng.normal(size=64)+1j*rng.normal(size=64))
            return {"session_id":group,"setting_family":group,
                    "acquisition_id":f"{group}-{i}","reference_ids":[],
                    "reference_kind":"synthetic_corruption_of_measured_reference",
                    "noisy_fid":noisy,"target_fid":clean}
        train=[example(i,"train") for i in range(3)]
        val=[example(i,"val") for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            for variant in ("real","complex"):
                path=Path(directory)/f"{variant}.pt"
                info=train_tvcondnet(train,val,path,variant=variant,tv_lambda=0.02,
                                      epochs=2,patience=2)
                self.assertEqual(info["status"],"TRAINED")
                prediction=apply_tvcondnet(val[0]["noisy_fid"],path)
                self.assertEqual(prediction["fid"].shape,(64,))
                self.assertTrue(np.all(np.isfinite(prediction["fid"])))
                if variant=="real":
                    self.assertEqual(prediction["imaginary_channel"],"unchanged from input")


class PipelineChecks(unittest.TestCase):
    def test_virtual_z_and_simplification_preserve_unitary(self):
        gates=[Gate("RZ",(0,),0.4),Gate("RXY",(0,),math.pi/2,0.2),
               Gate("RZ",(0,),-0.2),Gate("RXY",(0,),0.3,-0.4),
               Gate("CZ",(0,1))]
        compiled=compile_virtual_z(gates,2)
        self.assertTrue(unitary_equivalent(circuit_unitary(gates,2),
                       circuit_unitary(compiled["materialized_for_verification"],2)))
        simplified=simplify_circuit([Gate("RXY",(0,),0.7,0),Gate("RXY",(0,),-0.7,0),
                                     Gate("CZ",(0,1))],2)
        self.assertEqual(simplified,[Gate("CZ",(0,1))])

    def test_anneal_uses_bounded_sequential_measurement_budget(self):
        calls=[]
        def cost(x):
            calls.append(x.copy())
            return float(np.sum((x-np.array([0.25,-0.3]))**2))
        result=bounded_anneal(cost,[0,0],[(-1,1),(-1,1)],evaluations=24,
                              chains=3,pilot_repeat=3,seed=9)
        self.assertEqual(result["evaluations"],24)
        self.assertEqual(len(calls),24)
        self.assertTrue(all(np.all(x>=-1) and np.all(x<=1) for x in calls))
        self.assertLessEqual(result["best_cost"],cost(np.array([0.,0.])))

    def test_mapping_enumerates_only_supported_physical_pairs(self):
        gates=[Gate("RXY",(0,),math.pi/2),Gate("CZ",(0,1))]
        layouts=rank_mappings(gates,2,[0,1,2],one_qubit_error={0:.01,1:.02,2:.04},
            two_qubit_error={(0,1):.03},one_qubit_duration_s={0:1,1:1,2:1},
            two_qubit_duration_s={(0,1):2})
        self.assertEqual(len(layouts),2)
        self.assertEqual(set(layouts[0]["mapping"].values()),{0,1})

    def test_toggling_knows_simultaneous_pi_does_not_cancel_zz(self):
        simultaneous=toggling_integrals(1.0,[DDPulse(.5,0),DDPulse(.5,1)],
                                         [0,1],[(0,1)])
        self.assertAlmostEqual(simultaneous["zz_s"][(0,1)],1.0)
        staggered=toggling_integrals(1.0,[DDPulse(.25,0),DDPulse(.75,0)],
                                     [0,1],[(0,1)])
        self.assertAlmostEqual(staggered["zz_s"][(0,1)],0.0)
        with self.assertRaisesRegex(RuntimeError,"UNVERIFIED_TIMING"):
            design_contextual_dd(1.0,[0],single_z_weights={0:1},
                unwanted_zz_weights={},timing_verified=False,
                pi_pulse_error=.01,pulse_duration_s=.001)

    def test_analog_full_vs_grouped_and_rank_contract(self):
        rng=np.random.default_rng(14)
        true=rng.normal(size=(80,2))+1j*rng.normal(size=(80,2))
        # The receiver mixes the two complex lines; grouped correction cannot
        # recover that correlation whereas a full affine model can.
        mixing=np.array([[1,.35],[.25,1]],complex)
        measured=true@mixing+0.03+0.02j
        full=fit_analog_readout(measured[:60],true[:60],ridge=1e-6,
                                preparation_uncertainty=.01)
        grouped=fit_analog_readout(measured[:60],true[:60],ridge=1e-6,
                                   mode="grouped",groups=((0,),(1,)))
        full_error=np.mean(np.abs(apply_analog_readout(full,measured[60:])-true[60:])**2)
        grouped_error=np.mean(np.abs(apply_analog_readout(grouped,measured[60:])-true[60:])**2)
        self.assertLess(full_error,grouped_error*.01)
        self.assertLess(full_error,1e-7)
        with self.assertRaisesRegex(ValueError,"do not span"):
            fit_analog_readout(np.ones((8,2),complex),true[:8])

    def test_residual_readout_requires_disjoint_states(self):
        try:
            import torch  # noqa: F401
        except (ImportError,OSError,RuntimeError):
            self.skipTest("CPU PyTorch unavailable")
        rng=np.random.default_rng(12)
        true=rng.normal(size=(30,1))+1j*rng.normal(size=(30,1))
        measured=true+0.04*true*np.abs(true)**2
        baseline=fit_analog_readout(measured[:20],true[:20])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"readout.pt"
            with self.assertRaisesRegex(ValueError,"Independent calibration-state IDs"):
                train_residual_readout(measured[:20],true[:20],measured[20:],true[20:],
                                       baseline,path,epochs=2)
            info=train_residual_readout(measured[:20],true[:20],measured[20:],true[20:],
                baseline,path,epochs=3,train_state_ids=[f"a{i}" for i in range(20)],
                validation_state_ids=[f"a{i}" for i in range(20,30)])
            self.assertEqual(info["status"],"TRAINED")
            estimate=apply_residual_readout(measured[20:],baseline,path)
            self.assertEqual(estimate.shape,(10,1))
            self.assertTrue(np.all(np.isfinite(estimate)))

    def test_unverified_two_qubit_benchmark_is_explicitly_blocked(self):
        plan=g_ablation_plan(verified_one_qubit=True,verified_two_qubit=False,
                             timing_verified=False,readout_calibrated=True)
        self.assertEqual(plan["status"],"PLAN_READY")
        self.assertTrue(any("2q" in row.get("reason","") for row in plan["blocked"]))


class AnalysisIntegrationChecks(unittest.TestCase):
    def test_effective_d_requires_pilot_phase_validation_and_f_disjoint_split(self):
        sample_hz=10000.
        times=np.arange(256)/sample_hz
        initial=np.diag([1.,0.]).astype(complex)
        detector=cde.spin_operator("x",0,1)+1j*cde.spin_operator("y",0,1)
        model=cde.single_spin_model(0.,1/(4*40e-6*100))
        readout=cde.ReadoutPhysics(1000*cde.spin_operator("z",0,1),1.3+.4j,
                                   .02-.01j,30.)
        def record(key,width,amp,phase,role,block,family):
            program=cde.PulseProgram((width*1e-6,),(amp,),(math.radians(phase),))
            example=cde.FIDExample(program,initial,detector,times,
                np.zeros(len(times),complex),family,block,key)
            fid=cde.predict_fid_physics(example,model,readout)
            return RawFIDRecord(key,key,"g","0","0","NMRSIG",
                np.arange(len(times))*.1,times,fid.real,fid.imag,
                {"sampleFre":sample_hz,"sampleCount":len(times),
                 "pulse":{"hPulse":[{"width":width,"am":amp,"phase":phase,
                                      "freshift":0.}]}},
                {"measurement_block":block,"setting_family":family})
        pilots=[record(f"p{w}",w,100.,90.,"pilot","pilot","repeat" if w==40 else "rabi")
                for w in (40.,80.,160.,200.)]
        pilots += [record("v_phase",40.,100.,0.,"pilot","pilot","rf_map"),
                   record("v_amp",80.,80.,90.,"pilot","pilot","rf_map")]
        train=[record("f_train",40.,100.,90.,"train","block-00","F_train_40")]
        validation=[record("f_validation",80.,100.,90.,"validation","block-02","F_validation_80")]
        test=[record("f_test",120.,100.,90.,"test","block-04","F_test_120")]
        other=[record("unrelated",40.,100.,90.,"train","block-04","G_other")]
        roles={"pilot":pilots,"train":train+other,"validation":validation,"test":test}
        pilot={"rabi":{"t90_us":40.,"amplitude_pct":100.},
               "noise":{"covariance":[[1e-8,0],[0,1e-8]]},
               "reference_frequency_hz":1000.,"multiplet_bands_hz":[[950.,1050.]]}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/"models").mkdir()
            _,_,_,_,_,effective=_effective_one_spin_model(roles,pilot,out)
            self.assertTrue(all(row["passes"] for row in effective["validation_rows"]))
            results=Results(out,{}, {"torch":{"ready":False}})
            d=_analyze_d(roles,pilot,out,{"torch":{"ready":False}},results)
            self.assertEqual(d["status"],"DEPENDENCY_FAILED")
            prediction=(out/"models"/"D_fid_prediction.json").read_text()
            self.assertIn("f_train",prediction)
            self.assertIn("f_validation",prediction)
            self.assertIn("f_test",prediction)
            self.assertNotIn("unrelated",prediction)
            self.assertIn("effective relative",prediction)

    def test_measurement_block_survives_save_and_physical_g_status_is_preserved(self):
        rng=np.random.default_rng(3)
        count=64
        times=np.arange(count)/10000
        signal=np.exp((-40+2j*np.pi*75)*times)
        def record(key,block,role):
            fid=signal+.01*(rng.normal(size=count)+1j*rng.normal(size=count))
            return RawFIDRecord(key,key,"g","0","0","NMRSIG",
                np.arange(count)*.1,times,fid.real,fid.imag,
                {"sampleFre":10000,"sampleCount":count,
                 "pulse":{"hPulse":[{"width":40.,"am":100.,"phase":0.,"freshift":0.}]}},
                {"measurement_block":block,"setting_family":role})
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            pilot=record("pilot","pilot","pilot")
            pilot.save(out/"raw")
            self.assertEqual(RawFIDRecord.load(out/"raw","pilot").metadata["measurement_block"],"pilot")
            results=Results(out,{}, {"torch":{"ready":False}})
            results.module("G","SUCCESS_VALIDATED","independent physical paired control",
                           physical_ablation_status="COMPLETE",physical_acquisitions=12)
            heldout=[record(f"test_{i}","block-04","F_test_120") for i in range(4)]
            summary=analyze_de_fg({"pilot":[pilot],"train":[],"validation":[],"test":heldout},
                {"rabi":{"t90_us":40.}},out,{"torch":{"ready":False}},results)
            self.assertEqual(summary["G"]["status"],"SUCCESS_VALIDATED")
            self.assertEqual(results.data["modules"]["G"]["physical_ablation_status"],"COMPLETE")
            self.assertTrue((out/"models"/"F_analysis.json").exists())
            self.assertTrue(any(row["module"]=="F" for row in results.data["rows"]))


if __name__=="__main__":
    unittest.main()
