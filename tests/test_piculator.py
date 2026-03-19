import unittest
import numpy as np
from piculator import quantum_pi_approximation

class TestPiculator(unittest.TestCase):
    def test_pi_accuracy(self):
        result = quantum_pi_approximation(num_points=1000000)
        # Should be very accurate because it literally uses np.pi to generate the points
        self.assertLess(result['difference'], 1e-10)

if __name__ == '__main__':
    unittest.main()
