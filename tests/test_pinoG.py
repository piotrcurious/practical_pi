import unittest
import numpy as np
from from_frequencies.pinoG import PiEstimator

class TestPinoG(unittest.TestCase):
    def test_infer_pi(self):
        # Infer pi using the optimization logic
        estimator = PiEstimator()
        inferred_value = estimator.infer_pi(precision=15)
        # Should be reasonably close to pi given the precision of constants
        self.assertAlmostEqual(inferred_value, np.pi, places=5)

if __name__ == '__main__':
    unittest.main()
