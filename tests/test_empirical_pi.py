import unittest
import mpmath
from empirical_pi import PhysicalPiInference

class TestEmpiricalPi(unittest.TestCase):
    def setUp(self):
        self.inference = PhysicalPiInference()
        mpmath.mp.dps = 50

    def test_stefan_boltzmann_inference(self):
        pi_sb, _ = self.inference.infer_from_stefan_boltzmann()
        # Should be very close to pi because sigma is defined to match h, c, k
        self.assertAlmostEqual(float(pi_sb), float(mpmath.pi), places=10)

    def test_qed_inference(self):
        pi_qed, _ = self.inference.infer_from_qed(order=2)
        # Should be close to pi but limited by 2nd order expansion accuracy
        diff = abs(pi_qed - mpmath.pi)
        # 2nd order QED gives ~4-5 decimals of pi
        self.assertLess(diff, 1e-4)

if __name__ == '__main__':
    unittest.main()
