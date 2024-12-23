from mpmath import mp, sqrt

# Set precision to 50 decimal places
mp.dps = 50

# Constants with high precision
hbar = mp.mpf('1.0545718176461565e-34')  # Reduced Planck constant (J·s)
c = mp.mpf('299792458')                  # Speed of light in vacuum (m/s)
L_P_empirical = mp.mpf('1.616255e-35')   # Planck length (m)
E_P_empirical = mp.mpf('1.956e9')        # Planck energy (J)

# Step 1: Numerically calculate π
def calculate_pi():
    a = mp.mpf(1)
    b = 1 / sqrt(mp.mpf(2))
    t = mp.mpf(0.25)
    p = mp.mpf(1)

    for _ in range(10):  # Iterations for high precision
        a_next = (a + b) / 2
        b = sqrt(a * b)
        t -= p * (a - a_next)**2
        a = a_next
        p *= 2

    return (a + b)**2 / (4 * t)

pi_numerical = calculate_pi()

# Step 2: Infer G using Planck relations and empirical constants
G_inferred = hbar * c / (L_P_empirical**2 * E_P_empirical)

# Step 3: Refine empirical π using G_inferred
# π = E * E_P / (2 * hbar * c^2 * f)
E_photon = mp.mpf('3.2e-19')  # Photon energy (J)
f_photon = mp.mpf('4.8e14')   # Photon frequency (Hz)
pi_empirical_refined = (E_photon * E_P_empirical) / (2 * hbar * c**2 * f_photon)

# Combine results: Average numerical and empirical π for refinement
pi_combined = (pi_numerical + pi_empirical_refined) / 2

# Output results
print(f"Numerical π: {pi_numerical}")
print(f"Empirical π (refined): {pi_empirical_refined}")
print(f"Combined π: {pi_combined}")
print(f"Inferred G: {G_inferred}")
