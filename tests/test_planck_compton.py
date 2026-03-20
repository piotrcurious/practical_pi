import unittest
import numpy as np
from from_frequencies.pi_compton import infer_pi as infer_pi_1
from from_frequencies.pi_compton2 import infer_pi as infer_pi_2
from from_frequencies.pi_from_planck import infer_pi as infer_pi_planck

class TestPlanckComptonPi(unittest.TestCase):
    def test_pi_compton_1(self):
        inferred_pi = infer_pi_1()
        self.assertAlmostEqual(inferred_pi, np.pi, places=7)

    def test_pi_compton_2(self):
        inferred_pi = infer_pi_2()
        self.assertAlmostEqual(inferred_pi, np.pi, places=7)

    def test_pi_from_planck(self):
        inferred_pi = infer_pi_planck()
        self.assertAlmostEqual(inferred_pi, np.pi, places=5)

if __name__ == '__main__':
    unittest.main()
