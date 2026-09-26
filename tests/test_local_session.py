"""Small numerical regressions for the real Windows session planner."""

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spinq_local.core import RawFIDRecord
from spinq_local.session import LocalSession, _jsonable, _rabi_period
from spinq_local.signal import MultipletSpec


class SessionNumericalTests(unittest.TestCase):
    def test_frequency_rejects_frozen_band_edge_or_unchecked_fit(self):
        session=object.__new__(LocalSession)
        session.spec=MultipletSpec(((-1810.,-1570.),))
        session.noise=None
        session.primary_component_index=0
        record=SimpleNamespace(key="pilot_40_r0")
        def fit(status, at_edge):
            return {"status":status,"modes":[{"component_id":0,
                "frequency_hz":-1570.,"frequency_at_band_edge":at_edge}]}
        with patch("spinq_local.session.fit_complex_multiplet",
                   return_value=fit("FIT_COMPLETED",True)):
            with self.assertRaisesRegex(ValueError,"band_edge_components=\\[0\\]"):
                session.frequency(record)
        with patch("spinq_local.session.fit_complex_multiplet",
                   return_value=fit("MODEL_CHECK_REQUIRED",False)):
            with self.assertRaisesRegex(ValueError,"status=MODEL_CHECK_REQUIRED"):
                session.frequency(record)
        with patch("spinq_local.session.fit_complex_multiplet",
                   return_value=fit("FIT_COMPLETED",False)):
            self.assertEqual(session.frequency(record),-1570.)

    def test_bad_full_length_pilot_reference_skips_short_fid_acquisitions(self):
        session=object.__new__(LocalSession)
        session.blocks=10
        session.seed=1
        session.pilot_records={}
        session.noise=None
        session.spec=None
        session.primary_component_index=0
        session.t90_us=None
        session.reference_frequency_hz=None
        session.a_receiver_offset=0j
        session.a_receiver_gain=1+0j
        session.a_observation_cov=np.eye(2)
        session.results=SimpleNamespace(data={"pilot":{},"frozen_plan":{}},save=lambda:None)
        acquired=[]
        def pulse(key,width,**_):
            acquired.append(key)
            return SimpleNamespace(key=key,width=width,metadata={"wall_seconds":1.})
        session.pulse=pulse
        session.coefficient=lambda record: complex(np.sin(2*np.pi*record.width/152.))
        noise=SimpleNamespace(re_im_covariance=np.eye(2),lag_one_correlation=0.,repetitions=3)
        rejected={"status":"MODEL_CHECK_REQUIRED","modes":[{
            "component_id":0,"frequency_hz":-1570.,"frequency_at_band_edge":True}]}
        with patch("spinq_local.session.estimate_noise",return_value=noise), \
             patch("spinq_local.session._pilot_bands",return_value=(MultipletSpec(((-1810.,-1570.),)),0)), \
             patch("spinq_local.session.validate_axis",return_value=SimpleNamespace(sample_hz=10000.)), \
             patch("spinq_local.session.fit_complex_multiplet",return_value=rejected):
            pilot=session.pilot()
        self.assertEqual(len(acquired),7)
        self.assertFalse(any("count" in key for key in acquired))
        self.assertIsNone(session.reference_frequency_hz)
        self.assertEqual(pilot["fid_acquisition_plan"]["status"],"REFERENCE_INADEQUATE")
        self.assertIn("reference_frequency",pilot["failures"])

    def test_invalid_short_fid_fits_cannot_create_trivial_full_length_plan(self):
        session=object.__new__(LocalSession)
        session.blocks=10
        session.seed=1
        session.pilot_records={}
        session.noise=None
        session.spec=None
        session.primary_component_index=0
        session.t90_us=None
        session.reference_frequency_hz=None
        session.a_receiver_offset=0j
        session.a_receiver_gain=1+0j
        session.a_observation_cov=np.eye(2)
        session.results=SimpleNamespace(data={"pilot":{},"frozen_plan":{}},save=lambda:None)
        acquired=[]
        def pulse(key,width,**_):
            acquired.append(key)
            return SimpleNamespace(key=key,width=width,metadata={"wall_seconds":1.})
        session.pulse=pulse
        session.coefficient=lambda record: complex(np.sin(2*np.pi*record.width/152.))
        def frequency(record):
            if "count" in record.key:
                raise ValueError("frozen fit hit band edge")
            return -1690.+{"pilot_40_r0":0.,"pilot_40_r1":1.,
                           "pilot_40_r2":-1.}.get(record.key,0.)
        session.frequency=frequency
        noise=SimpleNamespace(re_im_covariance=np.eye(2),lag_one_correlation=0.,repetitions=3)
        with patch("spinq_local.session.estimate_noise",return_value=noise), \
             patch("spinq_local.session._pilot_bands",return_value=(MultipletSpec(((-1810.,-1570.),)),0)), \
             patch("spinq_local.session.validate_axis",return_value=SimpleNamespace(sample_hz=10000.)):
            pilot=session.pilot()
        self.assertEqual(len(acquired),11)
        self.assertIsNone(session.reference_frequency_hz)
        self.assertEqual(pilot["fid_acquisition_plan"]["status"],"REFERENCE_INADEQUATE")
        self.assertIn("No shorter FID length",pilot["fid_acquisition_plan"]["reason"])

    def test_h_rf_map_consumes_serialized_complex_rabi_coefficients(self):
        session=object.__new__(LocalSession)
        session.t90_us=39.
        def measured_response(key,width,*,amplitude=100.,phase=90.,**_):
            period=156.*100./amplitude
            angle=2*np.pi*width/period
            return (0.15+0.1j)+(1.1+0.2j)*np.sin(angle)+(0.3-0.1j)*np.cos(angle)
        session.pulse=measured_response
        session.coefficient=lambda value:value
        reference_rate,reference=session._fit_amplitude_rate(90.,100.)
        lower_rate,lower=session._fit_amplitude_rate(90.,80.,reference)
        self.assertGreater(reference_rate,lower_rate)
        self.assertAlmostEqual(reference["period_us"],156.,delta=2.)
        self.assertAlmostEqual(lower["period_us"],195.,delta=5.)

    def test_signed_projection_uses_fixed_measured_fid_not_multiplet_or_vendor(self):
        fs=10000
        samples=1024
        t=np.arange(samples)/fs
        envelope=np.exp((-5+2j*np.pi*170)*t)

        def record(width):
            scale=np.sin(2*np.pi*width/156)
            signal=scale*envelope
            return RawFIDRecord(
                key=f"pulse_{int(width)}",task_id=f"task_{int(width)}",
                group="g",path="0",qubit="0",step="NMRSIG",
                axis_original=np.asarray(np.arange(samples)*.1,dtype=np.float32).astype(float),
                time_seconds=t,re=signal.real,im=signal.imag,
                parameters_sent={"sampleFre":fs,"sampleCount":samples},
                metadata={"untrusted_vendor_fft": 1e9})

        with tempfile.TemporaryDirectory() as temporary:
            session=object.__new__(LocalSession)
            session.out=Path(temporary)
            session.pilot_records={"pilot_40_r0":record(40)}
            session.spec=MultipletSpec(((150,190),))
            session.noise=None
            widths=[40,80,120,160,200]
            with patch("spinq_local.session.fit_complex_multiplet",
                       side_effect=AssertionError("frequency fit must stay separate")):
                coefficients=[session.coefficient(record(width)) for width in widths]
            self.assertGreater(coefficients[0].real,0)
            self.assertLess(coefficients[2].real,0)
            self.assertGreater(coefficients[4].real,0)
            fit=_rabi_period(widths,coefficients)
            self.assertAlmostEqual(fit["period_us"],156,delta=1)
            self.assertGreater(fit["signed_complex_r2"],.99)
            artifact=json.loads((Path(temporary)/"models"/"pulse_120_projection.json").read_text())
            self.assertEqual(artifact["template_key"],"pulse_40")
            self.assertEqual(artifact["window_points"],512)
            self.assertFalse(artifact["vendor_fft_or_fit_used"])

    def test_signed_historical_rabi_fringe(self):
        fit=_rabi_period([40,80,120,160,200],
                         [2134.5205,-237.0326,-2158.9366,411.2401,2070.5585])
        self.assertAlmostEqual(fit["period_us"],155,delta=2)
        self.assertAlmostEqual(fit["t90_us"],38.75,delta=.5)
        self.assertGreater(fit["signed_complex_r2"],.99)
        self.assertEqual(len(fit["coefficient"]),3)
        self.assertTrue(all(set(item)=={"re","im"} for item in fit["coefficient"]))
        json.dumps(fit)

    def test_jsonable_recurses_through_numpy_complex_values(self):
        converted=_jsonable(np.asarray([1+2j,3+4j]))
        self.assertEqual(converted,[{"re":1.,"im":2.},{"re":3.,"im":4.}])
        json.dumps(converted)

    def test_measured_nelder_mead_exposes_new_bounded_points(self):
        session=object.__new__(LocalSession)
        session.h_history={"nelder_mead":[]}
        bounds=((.88,1.12),(.88,1.12))
        for _ in range(6):
            point=session._next_h_nelder_mead(bounds)
            self.assertTrue(.88<=point[0]<=1.12 and .88<=point[1]<=1.12)
            session.h_history["nelder_mead"].append({
                "point":list(point),"cost":(point[0]-.97)**2+2*(point[1]-1.03)**2})
        points={tuple(item["point"]) for item in session.h_history["nelder_mead"]}
        self.assertEqual(len(points),6)


if __name__=="__main__":
    unittest.main()
