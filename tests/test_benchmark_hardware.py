"""Regression for a completed SpinQ chart with one fewer point than requested."""

import unittest

from spinq_benchmark.hardware import paired_fid_graphs


class FidPairTests(unittest.TestCase):
    def test_one_missing_final_point_is_preserved_but_axis_mismatch_is_rejected(self):
        real=[[i/10000, i] for i in range(15999)]
        imag=[[i/10000, -i] for i in range(15999)]
        good=paired_fid_graphs([{"fidRe":real,"fidIm":imag}],16000)
        self.assertEqual(len(good),1)
        self.assertEqual(good[0]["actual_sample_count"],15999)
        self.assertEqual(good[0]["requested_sample_count"],16000)
        shifted=[[t+0.0001,v] for t,v in imag]
        self.assertEqual(paired_fid_graphs([{"fidRe":real,"fidIm":shifted}],16000),[])


if __name__=="__main__":unittest.main()
