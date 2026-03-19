import unittest
import mpmath
from pisplider import QuantumPiCalculator

class TestPisplider(unittest.TestCase):
    def setUp(self):
        self.precision = 100
        self.calculator = QuantumPiCalculator(precision=self.precision)
        mpmath.mp.dps = self.precision

    def test_quantum_circle_method(self):
        # This is just the Leibniz series, should converge very slowly
        pi_calc = self.calculator.quantum_circle_method(terms=1000)
        diff = abs(mpmath.mpf(str(pi_calc)) - mpmath.pi)
        # 1000 terms gives ~3 decimals of accuracy
        print(f"Quantum circle method diff: {diff}")

    def test_ramanujan_quantum_series(self):
        # Like physical_pi, this should be accurate if not for the "correction"
        pi_calc = self.calculator.ramanujan_quantum_series(terms=5)
        diff = abs(mpmath.mpf(str(pi_calc)) - mpmath.pi)
        print(f"Pisplider Ramanujan diff: {diff}")

    def test_chudnovsky_quantum_algorithm(self):
        # Like physical_pi, this should be accurate if not for the "correction"
        pi_calc = self.calculator.chudnovsky_quantum_algorithm(terms=5)
        diff = abs(mpmath.mpf(str(pi_calc)) - mpmath.pi)
        print(f"Pisplider Chudnovsky diff: {diff}")

if __name__ == '__main__':
    unittest.main()
