from mpmath import mp

# Set precision to 50 decimal places
mp.dps = 50

# Constants with high-precision values
hbar = mp.mpf('1.0545718176461565e-34')  # Reduced Planck constant (J·s)
c = mp.mpf('299792458')                  # Speed of light in vacuum (m/s)

# Empirical measurements
E = mp.mpf('3.2e-19')                    # Photon energy (J)
f = mp.mpf('4.8e14')                     # Photon frequency (Hz)
E_P = mp.mpf('1.956e9')                  # Planck energy (J)

# Calculate pi using the derived formula
pi_inferred = (E * E_P) / (2 * hbar * c**2 * f)

# Output the inferred value of pi
print(f"Inferred value of π: {pi_inferred}")
