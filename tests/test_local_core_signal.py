"""Small synthetic contract tests; these are never reported as device data."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spinq_local.core import (Capabilities, CapabilityUnavailable, IncompleteFID,
                              RawFIDRecord, Segment, SequenceIR, assemble_fid,
                              compile_sequence)
from spinq_local.signal import (MultipletSpec, estimate_noise, fft_local,
                                fit_complex_multiplet, validate_axis)


def fake_events(signal, *, task="task-one", truncate=False, vendor=1.0):
    n=len(signal)-(1 if truncate else 0)
    x=np.arange(n)*.1
    common={"taskId":task,"group":"exp_layer_physical","path":"0",
            "qubit":"0","step":"NMRSIG"}
    def chart(name, values):
        return {"kind":"s_post_exp_chart_updated","payload":{"chart_data":{
            **common,"chart_name":name,"points":np.column_stack((x,values[:n])).tolist()}}}
    return [chart("fftFit",np.full(len(signal),vendor)),
            chart("fidRe",signal.real),chart("fidIm",signal.imag),
            {"kind":"s_post_exp_chart_updated_finished","payload":{"json_data":common}},
            {"kind":"s_post_exp_finished","payload":{"json_data":common}}]


class LocalCoreSignalTests(unittest.TestCase):
    def test_sequence_timing_fails_closed_until_verified(self):
        seq=SequenceIR((Segment(0,40),Segment(60,20)))
        with self.assertRaises(CapabilityUnavailable): compile_sequence(seq,Capabilities())
        spec=compile_sequence(seq,Capabilities(zero_amplitude_delay_verified=True,
                                                segment_sequence_verified=True))
        self.assertEqual([p["am"] for p in spec.payload["pulse"]["hPulse"]],[100.,0.,100.])
        self.assertEqual(spec.sequence_duration_us,80)
        self.assertTrue(spec.idle_probe)
        echo_tail=SequenceIR((Segment(0,40),Segment(40,10,amplitude_pct=0)))
        with self.assertRaises(CapabilityUnavailable):compile_sequence(echo_tail,Capabilities())
        compiled=compile_sequence(echo_tail,Capabilities(zero_amplitude_delay_verified=True,
                                                          segment_sequence_verified=True))
        self.assertEqual(compiled.rf_duration_us,40)
        self.assertTrue(compiled.idle_probe)

    def test_event_pairing_one_point_short_and_vendor_independence(self):
        t=np.arange(4000)/10000
        signal=2*np.exp((-4+2j*np.pi*120)*t)
        params={"sampleFre":10000,"sampleCount":4000}
        a=assemble_fid(fake_events(signal,truncate=True,vendor=1),"task-one",
                       key="a",parameters_sent=params)
        b=assemble_fid(fake_events(signal,truncate=True,vendor=10000),"task-one",
                       key="a",parameters_sent=params)
        self.assertEqual(len(a.re),3999)
        self.assertTrue(np.array_equal(a.fid,b.fid))
        self.assertAlmostEqual(validate_axis(a).sample_hz,10000)
        self.assertTrue(np.array_equal(fft_local(a)["spectrum"],fft_local(b)["spectrum"]))
        with tempfile.TemporaryDirectory() as d:
            a.save(Path(d));loaded=RawFIDRecord.load(Path(d),"a")
            self.assertTrue(np.array_equal(a.fid,loaded.fid))
        events=fake_events(signal)
        events[2]["payload"]["chart_data"]["points"][10][0]+=.01
        with self.assertRaises(IncompleteFID):
            assemble_fid(events,"task-one",parameters_sent=params)

    def test_fixed_identity_multiplet_and_independent_noise(self):
        rng=np.random.default_rng(22)
        t=np.arange(4000)/10000
        clean=(3+1j)*np.exp((-5+2j*np.pi*120)*t)+(1-2j)*np.exp((-8+2j*np.pi*430)*t)
        records=[]
        for k in range(4):
            y=clean+.1*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
            records.append(assemble_fid(fake_events(y,task=f"t{k}"),f"t{k}",
                                        key=f"r{k}",parameters_sent={"sampleFre":10000,"sampleCount":4000}))
        noise=estimate_noise(records)
        self.assertGreater(np.linalg.eigvalsh(noise.re_im_covariance).min(),0)
        spec=MultipletSpec(((110,130),(420,440)),max_fit_points=512)
        full=fit_complex_multiplet(records[0],spec,noise)
        self.assertLess(abs(full["modes"][0]["frequency_hz"]-120),1)
        self.assertLess(abs(full["modes"][1]["frequency_hz"]-430),1)
        short=RawFIDRecord(**{**records[0].__dict__,
            "axis_original":records[0].axis_original[:1000],
            "time_seconds":records[0].time_seconds[:1000],
            "re":records[0].re[:1000],"im":records[0].im[:1000],
            "parameters_sent":{"sampleFre":10000,"sampleCount":1000}})
        fit=fit_complex_multiplet(short,spec,noise)
        self.assertEqual([m["component_id"] for m in fit["modes"]],[0,1])


if __name__=="__main__":unittest.main()
