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
        self.e = mpmath.mpf('1.602176634e-19')   # Elementary charge (C)

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

        # Bohr radius (m)
        self.a0 = MeasuredValue(
            value=mpmath.mpf('5.29177210544e-11'),
            uncertainty=mpmath.mpf('0.00000000082e-11'),
            name="Bohr radius",
            unit="m"
        )

        # Electron mass (kg)
        self.m_e = MeasuredValue(
            value=mpmath.mpf('9.1093837139e-31'),
            uncertainty=mpmath.mpf('0.0000000028e-31'),
            name="Electron mass",
            unit="kg"
        )

        # Vacuum electric permittivity (F/m)
        self.eps0 = MeasuredValue(
            value=mpmath.mpf('8.8541878188e-12'),
            uncertainty=mpmath.mpf('0.0000000014e-12'),
            name="Vacuum electric permittivity",
            unit="F/m"
        )

        # Proton gyromagnetic ratio (s⁻¹·T⁻¹)
        self.gamma_p = MeasuredValue(
            value=mpmath.mpf('2.6752218708e8'),
            uncertainty=mpmath.mpf('0.0000000011e8'),
            name="Proton gyromagnetic ratio",
            unit="s⁻¹·T⁻¹"
        )

        # Proton magnetic moment (J/T)
        self.mu_p = MeasuredValue(
            value=mpmath.mpf('1.41060679545e-26'),
            uncertainty=mpmath.mpf('0.00000000060e-26'),
            name="Proton magnetic moment",
            unit="J/T"
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

    def infer_from_bohr_radius(self) -> Tuple[mpmath.mpf, mpmath.mpf]:
        """
        Infers π from Bohr radius formula: a₀ = (ε₀ * h²) / (π * mₑ * e²)
        => π = (ε₀ * h²) / (a₀ * mₑ * e²)
        """
        eps0 = self.eps0.value
        a0 = self.a0.value
        me = self.m_e.value
        e = self.e
        h = self.h

        pi_inferred = (eps0 * h**2) / (a0 * me * e**2)

        # Delta_pi = pi * sqrt((Delta_eps0/eps0)^2 + (Delta_a0/a0)^2 + (Delta_me/me)^2)
        rel_uncert = mpmath.sqrt(
            (self.eps0.uncertainty/eps0)**2 +
            (self.a0.uncertainty/a0)**2 +
            (self.m_e.uncertainty/me)**2
        )
        return pi_inferred, pi_inferred * rel_uncert

    def infer_from_proton_gyromagnetic_ratio(self) -> Tuple[mpmath.mpf, mpmath.mpf]:
        """
        Infers π from proton gyromagnetic ratio: γₚ = (2 * μₚ) / ħ = (4 * π * μₚ) / h
        => π = (γₚ * h) / (4 * μₚ)
        """
        gamma_p = self.gamma_p.value
        mu_p = self.mu_p.value
        h = self.h

        pi_inferred = (gamma_p * h) / (4 * mu_p)

        # Delta_pi = pi * sqrt((Delta_gamma_p/gamma_p)^2 + (Delta_mu_p/mu_p)^2)
        rel_uncert = mpmath.sqrt(
            (self.gamma_p.uncertainty/gamma_p)**2 +
            (self.mu_p.uncertainty/mu_p)**2
        )
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

    pi_bohr, uncert_bohr = inference.infer_from_bohr_radius()
    print(f"\n3. From Bohr Radius (Atomic Physics):")
    print(f"   Inferred π: {pi_bohr}")
    print(f"   Uncertainty: {uncert_bohr}")
    print(f"   Difference:  {pi_bohr - mpmath.pi}")

    pi_gamma, uncert_gamma = inference.infer_from_proton_gyromagnetic_ratio()
    print(f"\n4. From Proton Gyromagnetic Ratio (Nuclear Physics):")
    print(f"   Inferred π: {pi_gamma}")
    print(f"   Uncertainty: {uncert_gamma}")
    print(f"   Difference:  {pi_gamma - mpmath.pi}")

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
