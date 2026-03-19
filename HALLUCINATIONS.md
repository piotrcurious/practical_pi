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
