import mpmath
from dataclasses import dataclass
from typing import Tuple

# Set high precision for calculations
mpmath.mp.dps = 100

@dataclass
class MeasuredValue:
    value: mpmath.mpf
    uncertainty: mpmath.mpf
    name: str
    unit: str

class PhysicalPiInference:
    """
    Infers π from fundamental physical constants and state-of-the-art
    measured values, noting the theoretical foundations and limitations.
    """

    def __init__(self):
        # CODATA 2022 Exact Constants
        self.h = mpmath.mpf('6.62607015e-34')    # Planck constant (J·s)
        self.c = mpmath.mpf('299792458')        # Speed of light (m/s)
        self.k = mpmath.mpf('1.380649e-23')     # Boltzmann constant (J/K)

        # CODATA 2022 Measured Values (with uncertainties)
        # Stefan-Boltzmann constant (W·m⁻²·K⁻⁴)
        # Note: In the new SI, sigma is exactly derived from h, c, k.
        # But we can treat it as an 'empirical' value from the perspective of blackbody physics.
        self.sigma = MeasuredValue(
            value=mpmath.mpf('5.670374419e-8'),
            uncertainty=mpmath.mpf('0'), # Exact in 2022 CODATA due to definitions
            name="Stefan-Boltzmann constant",
            unit="W·m⁻²·K⁻⁴"
        )

        # Fine-structure constant (dimensionless)
        self.alpha = MeasuredValue(
            value=mpmath.mpf('7.2973525643e-3'),
            uncertainty=mpmath.mpf('0.0000000011e-3'),
            name="Fine-structure constant",
            unit=""
        )

        # Electron anomalous magnetic moment (g-2)/2
        self.a_e = MeasuredValue(
            value=mpmath.mpf('1.15965218128e-3'),
            uncertainty=mpmath.mpf('0.00000000018e-3'),
            name="Electron anomalous magnetic moment",
            unit=""
        )

    def infer_from_stefan_boltzmann(self) -> Tuple[mpmath.mpf, mpmath.mpf]:
        """
        Infers π from the Stefan-Boltzmann law: σ = (2 * π⁵ * k⁴) / (15 * h³ * c²)
        => π = ((15 * h³ * c² * σ) / (2 * k⁴))^(1/5)
        """
        sigma = self.sigma.value
        pi_inferred = mpmath.power((15 * self.h**3 * self.c**2 * sigma) / (2 * self.k**4), 0.2)

        # Uncertainty propagation (sigma is exact in new SI, so uncertainty is 0 if we use CODATA)
        # If sigma were measured with uncertainty Delta_sigma:
        # Delta_pi = (1/5) * pi * (Delta_sigma / sigma)
        uncertainty = (1/5) * pi_inferred * (self.sigma.uncertainty / sigma) if sigma != 0 else 0

        return pi_inferred, uncertainty

    def infer_from_qed(self, order: int = 2) -> Tuple[mpmath.mpf, mpmath.mpf]:
        """
        Infers π from Quantum Electrodynamics (QED) expansion of electron g-factor:
        a_e = (α / (2 * π)) - 0.328478965... * (α / π)² + ...

        We solve the quadratic equation for (α/π):
        0.328478965 * (α/π)² - 0.5 * (α/π) + a_e = 0
        """
        alpha = self.alpha.value
        ae = self.a_e.value

        # Schwinger limit (1st order): a_e ≈ α / (2π) => π ≈ α / (2 * a_e)
        if order == 1:
            pi_inferred = alpha / (2 * ae)
            # Delta_pi = pi * sqrt((Delta_alpha/alpha)^2 + (Delta_ae/ae)^2)
            rel_uncert = mpmath.sqrt((self.alpha.uncertainty/alpha)**2 + (self.a_e.uncertainty/ae)**2)
            return pi_inferred, pi_inferred * rel_uncert

        # 2nd order correction (Sommerfield/Petermann/Schwinger)
        C2 = mpmath.mpf('-0.32847896557919378')
        # Solving C2 * x^2 + 0.5 * x - ae = 0  where x = alpha/pi
        # x = (-0.5 + sqrt(0.5^2 - 4 * C2 * (-ae))) / (2 * C2)
        # x = (-0.5 + sqrt(0.25 + 4 * C2 * ae)) / (2 * C2)

        x = (-0.5 + mpmath.sqrt(0.25 + 4 * C2 * ae)) / (2 * C2)
        pi_inferred = alpha / x

        # Simplified uncertainty propagation
        rel_uncert = mpmath.sqrt((self.alpha.uncertainty/alpha)**2 + (self.a_e.uncertainty/ae)**2)
        return pi_inferred, pi_inferred * rel_uncert

def main():
    inference = PhysicalPiInference()

    print("Inference of π from Physical Constants (CODATA 2022)")
    print("=" * 50)

    pi_sb, uncert_sb = inference.infer_from_stefan_boltzmann()
    print(f"\n1. From Stefan-Boltzmann Law (Blackbody Radiation):")
    print(f"   Inferred π: {pi_sb}")
    print(f"   Uncertainty: {uncert_sb}")
    print(f"   Difference:  {pi_sb - mpmath.pi}")

    pi_qed, uncert_qed = inference.infer_from_qed(order=2)
    print(f"\n2. From QED (Electron g-2 anomalous magnetic moment):")
    print(f"   Inferred π: {pi_qed}")
    print(f"   Uncertainty: {uncert_qed}")
    print(f"   Difference:  {pi_qed - mpmath.pi}")

    print("\nLimitations & Discussion:")
    print("-" * 30)
    print("a. Circularity: In the post-2019 SI system, many 'measured' constants are now defined")
    print("   using fixed values of h, e, k, and c. Thus, CODATA values for sigma are exact")
    print("   by definition, making the inference a check of the system's consistency.")
    print("b. QED Precision: The QED derivation assumes the validity of the expansion.")
    print("   The 2nd order approximation is limited by the exclusion of higher-order terms.")
    print("c. Measurement Uncertainty: Even if the theory is perfect, we are limited by the")
    print("   relative standard uncertainty of alpha (~0.15 ppb) and a_e (~0.16 ppb).")

if __name__ == "__main__":
    main()
