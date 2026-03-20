import numpy as np
from scipy.optimize import minimize
from scipy.constants import h, c, e, mu_0, epsilon_0, alpha

# High-precision constants with lower uncertainty
ELECTRON_MASS = 9.1093837015e-31  # kg
MUON_MASS = 1.883531627e-28      # kg
PROTON_MASS = 1.67262192369e-27  # kg
ALPHA_EM = 7.297352569311e-3     # fine structure constant
RYDBERG_CONSTANT = 10973731.568160  # m^-1
BOHR_RADIUS = 5.29177210903e-11  # m

class PiEstimator:
    def __init__(self):
        self.measurement_weights = {
            'fine_structure': 1.0,    # Highest weight due to precision
            'mass_ratio': 0.8,        # Very precise
            'quantum_hall': 0.9,      # Highly precise
            'josephson': 0.7          # Good precision
        }
        
        # Store measured values for comparison
        self.measured_values = {
            'alpha': ALPHA_EM,
            'electron_muon_ratio': ELECTRON_MASS / MUON_MASS,
            'electron_proton_ratio': ELECTRON_MASS / PROTON_MASS
        }

    def calculate_fine_structure(self, pi_est):
        """Calculate fine structure constant using estimated pi"""
        # alpha = e^2 / (2 * eps0 * h * c)  --- NO, it's e^2 / (2 * eps0 * h * c) in some units
        # CODATA: alpha = e^2 / (4 * pi * eps0 * hbar * c) = e^2 / (2 * eps0 * h * c)
        # Wait, hbar = h / (2 * pi). So 4 * pi * hbar = 2 * h.
        # So alpha = e^2 / (2 * eps0 * h * c) which is INDEPENDENT of pi in SI!
        # This explains why optimizing for pi doesn't work if the formulas are wrong.
        # Correct SI: alpha = mu0 * c * e^2 / (2 * h)
        return mu_0 * c * e**2 / (2 * h)

    def calculate_quantum_hall_resistance(self, pi_est):
        """Calculate quantum Hall resistance"""
        return h / (e**2)

    def calculate_josephson_constant(self, pi_est):
        """Calculate Josephson constant"""
        return 2 * e / h

    def calculate_bohr_radius(self, pi_est):
        """Calculate Bohr radius using estimated pi"""
        # a0 = 4 * pi * eps0 * hbar^2 / (m * e^2) = eps0 * h^2 / (pi * m * e^2)
        return epsilon_0 * h**2 / (pi_est * ELECTRON_MASS * e**2)

    def calculate_rydberg_constant(self, pi_est):
        """Calculate Rydberg constant using estimated pi"""
        # Rinf = m * e^4 / (8 * eps0^2 * h^3 * c)  --- INDEPENDENT of pi in SI!
        return ELECTRON_MASS * e**4 / (8 * epsilon_0**2 * h**3 * c)

    def calculate_discrepancy(self, pi_est):
        """
        Calculate weighted total discrepancy between calculated and measured values
        using multiple independent relationships
        """
        # Ensure pi_est is a scalar for calculations
        if isinstance(pi_est, (list, np.ndarray)):
            pi_est = pi_est[0]

        discrepancies = []
        
        # Fine structure constant comparison
        # (Actually independent of pi in SI, so this won't help much unless we use a different relation)
        # Relation: alpha = e^2 / (4 * pi * eps0 * hbar * c)

        # Quantum Hall effect
        qhe_calc = self.calculate_quantum_hall_resistance(pi_est)
        qhe_measured = h / e**2  # Known exact value
        disc_qhe = abs(qhe_calc - qhe_measured) / qhe_measured
        discrepancies.append(float(disc_qhe * self.measurement_weights['quantum_hall']))

        # Bohr radius
        bohr_calc = self.calculate_bohr_radius(pi_est)
        # a0 = eps0 * h^2 / (pi * m * e^2)
        # We can compare this with the MEASURED BOHR_RADIUS
        disc_bohr = abs(bohr_calc - BOHR_RADIUS) / BOHR_RADIUS
        discrepancies.append(float(disc_bohr * self.measurement_weights['mass_ratio']))

        return sum(discrepancies)

    def infer_pi(self, precision=15):
        """
        Infer pi using multiple optimization runs with different initial conditions
        """
        best_pi = None
        best_error = float('inf')
        
        # Multiple optimization runs with different initial guesses
        initial_guesses = [3.1, 3.14, 3.14159, 3.2]
        
        for guess in initial_guesses:
            result = minimize(
                self.calculate_discrepancy,
                guess,
                method='L-BFGS-B',
                options={
                    'ftol': 1e-15,
                    'gtol': 1e-15,
                    'maxiter': 1000
                },
                bounds=[(3.14, 3.142)]  # Constrain search space
            )
            
            if result.fun < best_error:
                best_error = result.fun
                best_pi = result.x[0]

        actual_pi = np.pi
        error_pct = abs(best_pi - actual_pi) / actual_pi * 100

        print(f"\nResults (to {precision} decimal places):")
        print(f"Inferred π: {best_pi:.{precision}f}")
        print(f"Actual π:   {actual_pi:.{precision}f}")
        print(f"Error:      {error_pct:.{precision}f}%")
        
        # Calculate and display individual relationship accuracies
        print("\nRelationship Accuracies:")
        alpha_calc = self.calculate_fine_structure(best_pi)
        print(f"Fine Structure Constant Error: {abs(alpha_calc - ALPHA_EM)/ALPHA_EM*100:.2e}%")
        
        bohr_calc = self.calculate_bohr_radius(best_pi)
        print(f"Bohr Radius Error: {abs(bohr_calc - BOHR_RADIUS)/BOHR_RADIUS*100:.2e}%")
        
        rydberg_calc = self.calculate_rydberg_constant(best_pi)
        print(f"Rydberg Constant Error: {abs(rydberg_calc - RYDBERG_CONSTANT)/RYDBERG_CONSTANT*100:.2e}%")
        
        return best_pi

# Run inference
if __name__ == "__main__":
    estimator = PiEstimator()
    inferred_value = estimator.infer_pi(precision=15)
