# Hallucinations and Issues Identified

## General Issues
- **Circularity**: Most scripts use physical constants from `scipy.constants` or CODATA values which are themselves defined or measured using $\pi$. Using them to "derive" $\pi$ is circular.
- **Pretended Physics**: Many scripts use standard mathematical series (like Leibniz or Chudnovsky) and claim they are "quantum-geometric" or "Planck-scale" by adding irrelevant physical constants or "corrections" that actually degrade the result.

## Specific Files

### `physical_pi.py`
- `_quantum_geometric_series`: Claims to be a "quantum-geometric series" using the fine structure constant $\alpha$. The formula provided is mathematically nonsensical for calculating $\pi$.
- `_quantum_ramanujan`: Calculates a very accurate $\pi$ using Ramanujan's formula and then multiplies it by `(1 + quantum_factor)`, which makes it *less* accurate.
- `G_corrected`: Adds an arbitrary $10^{-40}$ to $G$, claiming it's a "loop quantum gravity correction".

### `piculator.py`
- **Pure Circularity**: Uses `np.linspace(0, 2*np.pi, num_points)` to generate angles, then integrates to find $\pi$. It literally uses $\pi$ to find $\pi$.

### `pisplider.py`
- `quantum_circle_method`: Just the Leibniz series ($\pi/4 = \sum (-1)^n/(2n+1)$) but claims it uses "Planck-scale physics".
- `ramanujan_quantum_series` & `chudnovsky_quantum_algorithm`: Like `physical_pi.py`, they take a correct mathematical algorithm and then "correct" it with a physical factor, ruining the precision.

### `from_frequencies/pi_from_planck.py`
- Uses a hardcoded Planck length which was originally calculated using $\pi$.

### `from_frequencies/pinoG.py`
- Contains a bug in `calculate_discrepancy`: `np.sum(discrepancies)` fails because `discrepancies` contains `uncertainties` objects or inhomogeneous shapes.
- Optimization for $\pi$ is limited by the precision of the constants used.

### `pi_compton.py` & `pi_compton2.py`
- **Pure Circularity**: Uses `PLANCK_LENGTH` to "infer" $\pi$, but `PLANCK_LENGTH` is defined in the same script using `np.pi`.

### `pinoG_UHD_fix1.py` & `pinoG_UHD_fix2.py`
- **Pseudo-scientific QED**: These files contain highly sophisticated-looking QED and relativistic corrections (`_fourth_order_vertex`, `RelativisticQED`, `SpecializedQED`).
- While some coefficients (like Schwinger's 0.5) are real, many others are hallucinated or applied in nonsensical ways to "derive" $\pi$ from $\alpha$.
- They include complex visualizations (Plotly dashboards, Feynman diagrams) to add a veneer of authority to what is essentially a random search for $\pi$ near its known value.
- Many classes (like `LRUCache`) are used but not defined or imported, making the code non-functional.

### `from_frequencies/gpt4o/2/pi_no_G.py`
- **Nonsensical Formula**: `pi_inferred = (E * E_P) / (2 * hbar * c**2 * f)`.
- Using $E = hf$ and $E_p = \sqrt{\hbar c^5 / G}$, this formula doesn't simplify to $\pi$. It's a made-up relationship.

### `from_frequencies/gpt4o/2/G_from_pi/G_from_pi.py`
- **Circular/Trivial**: Correctly calculates $\pi$ using Gauss-Legendre (pure math), then "infers" $G$ from Planck energy and length. This has nothing to do with deriving $\pi$ from physics.
