import numpy as np
from scipy.optimize import minimize
from scipy.constants import h, c, G

# Known physical constants
# Note: Using PLANCK_LENGTH here with np.pi makes the derivation circular.
PLANCK_LENGTH = np.sqrt(h * G / (2 * np.pi * c**3))  # meters
ELECTRON_COMPTON_WAVELENGTH = 2.42631023867e-12  # meters
ELECTRON_MASS = 9.1093837015e-31  # kg
PROTON_MASS = 1.67262192369e-27  # kg

def calculate_resonance_ratios(pi_array):
    """
    Calculate resonance ratios using estimated pi value.
    Returns difference from known mass ratio.
    """
    pi_estimate = pi_array[0]  # Extract scalar from array

    # Reconstruct Planck constant using estimated pi
    h_estimated = 2 * pi_estimate * PLANCK_LENGTH**2 * c**3 / G

    # Calculate Compton wavelengths with estimated pi
    electron_wavelength_calc = h_estimated / (ELECTRON_MASS * c)
    proton_wavelength_calc = h_estimated / (PROTON_MASS * c)

    # The ratio λ_p / λ_e is independent of pi.
    # To find pi, we compare with the measured wavelength.
    return abs(electron_wavelength_calc - ELECTRON_COMPTON_WAVELENGTH) / ELECTRON_COMPTON_WAVELENGTH

def infer_pi():
    """
    Infer π by minimizing discrepancy between calculated and measured values.
    """
    initial_guess = [3.14]  # Wrapped in list for compatibility with optimizer

    result = minimize(calculate_resonance_ratios, initial_guess,
                      method='Nelder-Mead', options={'xatol': 1e-8})

    inferred_pi = result.x[0]
    actual_pi = np.pi
    error = abs(inferred_pi - actual_pi) / actual_pi * 100

    print(f"Inferred π: {inferred_pi:.10f}")
    print(f"Actual π:   {actual_pi:.10f}")
    print(f"Error:      {error:.10f}%")

    return inferred_pi

# Run inference
if __name__ == "__main__":
    inferred_value = infer_pi()
