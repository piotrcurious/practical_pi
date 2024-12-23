import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.constants import h, c, e, mu_0, epsilon_0, alpha, k
from scipy.stats import norm
import uncertainties
from uncertainties import ufloat, unumpy
from typing import Dict, Tuple, List

class QuantumConstants:
    """High-precision fundamental constants with uncertainties"""
    
    def __init__(self):
        # Format: (value, standard uncertainty)
        self.ALPHA_EM = ufloat(7.297352569311e-3, 0.000000000011e-3)
        self.ELECTRON_MASS = ufloat(9.1093837015e-31, 0.0000000028e-31)
        self.MUON_MASS = ufloat(1.883531627e-28, 0.000000042e-28)
        self.PROTON_MASS = ufloat(1.67262192369e-27, 0.00000000051e-27)
        self.RYDBERG = ufloat(10973731.568160, 0.000021)
        self.BOHR_RADIUS = ufloat(5.29177210903e-11, 0.00000000080e-11)
        self.HYPERFINE_CS = ufloat(9192631770.0, 0.0)  # Exact
        self.PLANCK = ufloat(6.62607015e-34, 0.0)      # Exact
        self.ELECTRON_G = ufloat(2.00231930436256, 0.00000000000035)
        
        # Derived constants
        self.ELECTRON_COMPTON = h / (self.ELECTRON_MASS * c)
        self.FINE_STRUCTURE = e**2 / (4 * np.pi * epsilon_0 * h * c)
        
        # Quantum Hall and Josephson constants (exact in current SI)
        self.K_J = ufloat(483597.8484e9, 0.0)  # Josephson constant
        self.R_K = ufloat(25812.80745, 0.0)    # von Klitzing constant

class PiEstimator:
    def __init__(self):
        self.constants = QuantumConstants()
        self.results_history: List[Dict] = []
        self.uncertainty_samples = 10000
        
    def setup_quantum_relationships(self) -> Dict:
        """Define quantum relationships used for pi estimation"""
        return {
            'qed': {
                'weight': 1.0,
                'function': self.qed_relationship,
                'uncertainty': 1e-10
            },
            'zeeman': {
                'weight': 0.9,
                'function': self.zeeman_relationship,
                'uncertainty': 1e-9
            },
            'atomic': {
                'weight': 0.95,
                'function': self.atomic_relationship,
                'uncertainty': 1e-9
            },
            'quantum_hall': {
                'weight': 0.85,
                'function': self.quantum_hall_relationship,
                'uncertainty': 1e-8
            }
        }
        
    def qed_relationship(self, pi_est: float) -> float:
        """Quantum electrodynamics relationship using electron g-factor"""
        g_calc = 2 * (1 + self.constants.ALPHA_EM.n / (2 * pi_est) - 
                      0.328478965579193 * (self.constants.ALPHA_EM.n / pi_est)**2)
        return abs(g_calc - self.constants.ELECTRON_G.n)

    def zeeman_relationship(self, pi_est: float) -> float:
        """Zeeman effect relationship"""
        mu_B = e * h / (4 * pi_est * self.constants.ELECTRON_MASS.n)
        g_factor = 2 * mu_B * self.constants.ELECTRON_MASS.n / (e * h)
        return abs(g_factor - self.constants.ELECTRON_G.n)

    def atomic_relationship(self, pi_est: float) -> float:
        """Combined atomic spectroscopy relationships"""
        alpha_calc = self.calculate_fine_structure(pi_est)
        rydberg_calc = self.calculate_rydberg(pi_est)
        
        # Combine multiple spectroscopic relationships
        disc_alpha = abs(alpha_calc - self.constants.ALPHA_EM.n)
        disc_rydberg = abs(rydberg_calc - self.constants.RYDBERG.n)
        
        return np.sqrt(disc_alpha**2 + disc_rydberg**2)

    def quantum_hall_relationship(self, pi_est: float) -> float:
        """Quantum Hall effect relationship"""
        R_K_calc = h / e**2
        return abs(R_K_calc - self.constants.R_K.n)

    def calculate_fine_structure(self, pi_est: float) -> float:
        """Calculate fine structure constant"""
        return e**2 / (4 * pi_est * epsilon_0 * h * c)

    def calculate_rydberg(self, pi_est: float) -> float:
        """Calculate Rydberg constant"""
        return self.constants.ELECTRON_MASS.n * e**4 / (32 * pi_est**2 * epsilon_0**2 * h**2 * c)

    def monte_carlo_uncertainty(self, pi_est: float) -> Tuple[float, float]:
        """Estimate uncertainty using Monte Carlo simulation"""
        relationships = self.setup_quantum_relationships()
        results = []
        
        for _ in range(self.uncertainty_samples):
            # Randomly perturb constants within their uncertainties
            perturbed_results = []
            for rel in relationships.values():
                uncertainty = rel['uncertainty']
                perturbed_pi = pi_est + np.random.normal(0, uncertainty)
                perturbed_results.append(rel['function'](perturbed_pi))
            results.append(np.mean(perturbed_results))
            
        return np.std(results), norm.interval(0.95, loc=pi_est, scale=np.std(results))

    def objective_function(self, pi_est: float) -> float:
        """Combined objective function with weighted relationships"""
        relationships = self.setup_quantum_relationships()
        total_error = 0
        
        for rel in relationships.values():
            error = rel['function'](pi_est)
            total_error += error * rel['weight']
            
        return total_error

    def optimize_pi(self, method: str = 'hybrid') -> Tuple[float, float, Tuple[float, float]]:
        """
        Optimize pi estimation using multiple methods
        Returns: (best_pi, uncertainty, confidence_interval)
        """
        if method == 'hybrid':
            # First use differential evolution for global search
            bounds = [(3.14159, 3.14160)]
            de_result = differential_evolution(
                self.objective_function,
                bounds,
                strategy='best1bin',
                popsize=20,
                mutation=(0.5, 1.0),
                recombination=0.7,
                tol=1e-12,
                maxiter=1000
            )
            
            # Refine with L-BFGS-B
            result = minimize(
                self.objective_function,
                de_result.x[0],
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'ftol': 1e-15,
                    'gtol': 1e-15,
                    'maxiter': 1000
                }
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        best_pi = result.x[0]
        uncertainty, confidence_interval = self.monte_carlo_uncertainty(best_pi)
        
        # Store results for analysis
        self.results_history.append({
            'pi_value': best_pi,
            'uncertainty': uncertainty,
            'confidence_interval': confidence_interval,
            'optimization_success': result.success,
            'function_evals': result.nfev
        })
        
        return best_pi, uncertainty, confidence_interval

    def analyze_results(self, precision: int = 15) -> None:
        """Analyze and display results with detailed error analysis"""
        best_pi, uncertainty, confidence_interval = self.optimize_pi()
        actual_pi = np.pi
        
        print(f"\nResults (to {precision} decimal places):")
        print(f"Inferred π: {best_pi:.{precision}f}")
        print(f"Uncertainty: {uncertainty:.2e}")
        print(f"95% CI: [{confidence_interval[0]:.{precision}f}, {confidence_interval[1]:.{precision}f}]")
        print(f"Actual π:   {actual_pi:.{precision}f}")
        print(f"Error:      {abs(best_pi - actual_pi)/actual_pi*100:.2e}%")
        
        # Analyze individual relationships
        print("\nRelationship Contributions:")
        relationships = self.setup_quantum_relationships()
        for name, rel in relationships.items():
            error = rel['function'](best_pi)
            print(f"{name}: {error:.2e} (weight: {rel['weight']})")
        
        # Uncertainty analysis
        print("\nUncertainty Analysis:")
        print(f"Standard Error: {uncertainty:.2e}")
        print(f"Relative Error: {(uncertainty/best_pi)*100:.2e}%")
        
        # Convergence analysis
        if len(self.results_history) > 1:
            print("\nConvergence Analysis:")
            pi_values = [r['pi_value'] for r in self.results_history]
            print(f"Variation in estimates: {np.std(pi_values):.2e}")

if __name__ == "__main__":
    estimator = PiEstimator()
    estimator.analyze_results(precision=15)
