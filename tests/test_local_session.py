"""Small numerical regressions for the real Windows session planner."""

import unittest

from spinq_local.session import LocalSession, _rabi_period


class SessionNumericalTests(unittest.TestCase):
    def test_signed_historical_rabi_fringe(self):
        fit=_rabi_period([40,80,120,160,200],
                         [2134.5205,-237.0326,-2158.9366,411.2401,2070.5585])
        self.assertAlmostEqual(fit["period_us"],155,delta=2)
        self.assertAlmostEqual(fit["t90_us"],38.75,delta=.5)
        self.assertGreater(fit["signed_complex_r2"],.99)

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
