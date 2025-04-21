import numpy as np
from scipy.constants import h, c, G

# -------------------------------------------------------------------
# CODATA‑2018 measured Planck length (independent of π)
PLANCK_LENGTH_MEASURED = 1.616255e-35  # meters
# -------------------------------------------------------------------

def infer_pi() -> float:
    """
    Infer π from the measured Planck length using
        l_P^2 = h G / (2 π c^3)
    ⇒  π = h G / (2 l_P^2 c^3)
    """
    pi_estimated = (h * G) / (2 * PLANCK_LENGTH_MEASURED**2 * c**3)
    
    actual_pi = np.pi
    rel_error = abs(pi_estimated - actual_pi) / actual_pi * 100
    
    print(f"Inferred π: {pi_estimated:.12f}")
    print(f"   Actual π: {actual_pi:.12f}")
    print(f"Relative error: {rel_error:.2e}%")
    
    return pi_estimated

if __name__ == "__main__":
    infer_pi()
