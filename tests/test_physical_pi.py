import unittest
import mpmath
from physical_pi import UltraPreciseQuantumPiCalculator

class TestPhysicalPi(unittest.TestCase):
    def setUp(self):
        self.precision = 50
        self.calculator = UltraPreciseQuantumPiCalculator(precision=self.precision)
        mpmath.mp.dps = self.precision

    def test_chudnovsky_precision(self):
        # The Chudnovsky algorithm should be very accurate if implemented correctly
        results = self.calculator.calculate_pi(method='chudnovsky', terms=5)
        pi_calc = results['chudnovsky']
        diff = abs(pi_calc - mpmath.pi)
        # Should be accurate to many places
        self.assertLess(diff, mpmath.mpf('1e-50'))

    def test_ramanujan_precision(self):
        # Ramanujan is very accurate
        results = self.calculator.calculate_pi(method='ramanujan', terms=5)
        pi_calc = results['ramanujan']
        diff = abs(pi_calc - mpmath.pi)
        # 5 terms of Ramanujan is very precise
        self.assertLess(diff, mpmath.mpf('1e-30'))

    def test_quantum_geometric_series(self):
        # Leibniz series converges slowly but should eventually get close
        results = self.calculator.calculate_pi(method='quantum', terms=1000)
        pi_calc = results['quantum']
        diff = abs(pi_calc - mpmath.pi)
        # 1000 terms should give ~3 decimals
        self.assertLess(diff, mpmath.mpf('1e-2'))

if __name__ == '__main__':
    unittest.main()
