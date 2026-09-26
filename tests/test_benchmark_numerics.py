"""Numerical regression checks only; no simulated data are reported as hardware results."""

import math
import unittest

try:
    import numpy as np
    from spinq_benchmark.hardware import physical_request, validate_request
    from spinq_benchmark.numerics import (
        ComplexGridBayes, bb1_pulses, complex_fit, fit_response,
        hankel_denoise, response_model,
    )
    DEPS=True
except ImportError:
    DEPS=False


@unittest.skipUnless(DEPS,"benchmark numerical dependencies not installed")
class NumericalTests(unittest.TestCase):
    def test_complex_fid_tracks_frequency_and_phase(self):
        t=np.arange(4000)/10000
        y=(2+1j)*np.exp((-5+2j*np.pi*143)*t)
        fit=complex_fit(y,10000)
        self.assertAlmostEqual(fit["target"]["frequency_hz"],143,places=2)
        self.assertAlmostEqual(fit["target"]["amplitude"],math.sqrt(5),places=2)
        self.assertLess(fit["residual_rms"],1e-4)

    def test_rabi_fit_and_bayesian_update(self):
        obs=[]
        for w,d in ((40,0),(80,0),(120,0),(160,0),(200,0),(40,-20),(40,20),(80,-10),(160,10)):
            z=response_model(w,d,40,3,15,.2j,1+2j)
            obs.append({"width_us":w,"detuning_hz":d,"coefficient_re":z.real,
                        "coefficient_im":z.imag,"sigma":.1})
        fit=fit_response(obs)
        self.assertTrue(fit["identifiable"])
        self.assertAlmostEqual(fit["t90_us"],40,places=2)
        self.assertAlmostEqual(fit["resonance_detuning_hz"],3,places=2)
        bayes=ComplexGridBayes()
        prior=bayes.summary()
        bayes.update(obs,15)
        self.assertLess(bayes.summary()["t90_sd_us"],prior["t90_sd_us"])
        next_point=bayes.next_setting([(40,0),(60,10),(120,-20)],15,.1)
        self.assertNotEqual(next_point,(40,0))

    def test_hankel_and_bb1_cost(self):
        t=np.arange(255)/255
        y=np.exp((-2+100j)*t)
        clean=hankel_denoise(y,rank=1,max_matrix=64)
        self.assertLess(np.linalg.norm(y-clean)/np.linalg.norm(y),.02)
        pulses=bb1_pulses(40,90,100)
        self.assertAlmostEqual(sum(p["width"] for p in pulses),360)
        self.assertAlmostEqual(pulses[1]["phase"],90+math.degrees(math.acos(-1/8)))

    def test_payload_guard_preserves_historical_fields(self):
        p=physical_request(width_us=40,detuning_hz=10,sample_count=4000)
        self.assertEqual(validate_request(p,0),40)
        p["gradient"]=[{"path":0,"value":.1}]
        with self.assertRaises(ValueError):validate_request(p,0)


if __name__=="__main__":unittest.main()
