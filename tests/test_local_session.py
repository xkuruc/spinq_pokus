"""Small numerical regressions for the real Windows session planner."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spinq_local.core import RawFIDRecord
from spinq_local.session import LocalSession, _jsonable, _rabi_period
from spinq_local.signal import MultipletSpec


class SessionNumericalTests(unittest.TestCase):
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
