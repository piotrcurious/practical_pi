from mpmath import mp, sqrt

# Set precision to 50 decimal places
mp.dps = 50

# Constants with high precision
hbar = mp.mpf('1.0545718176461565e-34')  # Reduced Planck constant (J·s)
c = mp.mpf('299792458')                  # Speed of light in vacuum (m/s)

# Numerically calculate π using the Gauss-Legendre algorithm
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

# Calculate π
pi_calculated = calculate_pi()

# Empirical measurements
E_P = mp.mpf('1.956e9')  # Planck energy (J)
L_P = mp.mpf('1.616255e-35')  # Planck length (m)

# Infer G using the relationship: G = hbar * c / (L_P^2 * E_P)
G_inferred = hbar * c / (L_P**2 * E_P)

# Output results
print(f"Numerically calculated π: {pi_calculated}")
print(f"Inferred value of G: {G_inferred}")
