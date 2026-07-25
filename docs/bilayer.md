# Bilayer Interlayer Coupling

## Overview

The bilayer stage fits interlayer hopping parameters between WSe₂ and WS₂ layers to reproduce the top valence bands from bilayer ARPES data along the Γ–K path. The fit uses a 44×44 Hamiltonian (22 orbitals per layer × 2 layers, spin-degenerate) with `n_shells=0` (no moiré supercell expansion, i.e. a single mini-Brillouin zone).

## Interlayer coupling form

The interlayer coupling matrix is a 22×22 block that connects WSe₂ and WS₂ orbitals at the same k-point. Only two orbitals per spin block participate in interlayer hopping:

- **p_z^e** (index 8, even parity p_z orbital)
- **d_z²** (index 5, even parity d orbital)

For each orbital type, the coupling has the form:

```
t(k) = w1 + w2 · Σ_{i=1}^{6} exp(i k · e_i)
```

where `e_i` are the 6 nearest-neighbor vectors in the moiré lattice, obtained by rotating the base vector `[a·√3, 0]` by multiples of π/3:

```python
e_i = a · √3 · R_z(i·π/3) @ [1, 0]    for i = 0, ..., 5
```

The four fitted parameters are:

| Parameter | Orbital | Role |
|---|---|---|
| `w1p` | p_z^e | On-site interlayer hopping (k-independent) |
| `w2p` | p_z^e | k-dependent modulation via 6 NN vectors |
| `w1d` | d_z² | On-site interlayer hopping (k-independent) |
| `w2d` | d_z² | k-dependent modulation via 6 NN vectors |

At Γ (k=0), the sum `Σ exp(i k · e_i) = 6`, so the total coupling is `w1 + 6·w2`. Away from Γ, the phase factors interfere and reduce the coupling strength.

## Minimization

The fit minimizes a chi-squared objective comparing computed band energies to symmetrized bilayer ARPES data along the Γ–K path:

```
χ² = (1/N) Σ_{b=1}^{3} Σ_{i} w(k_i) · [E_TB[b, k_i] - E_ARPES[b, k_i]]²
```

where:
- **3 bands**: the top 3 valence bands (indices 27, 26, 25 out of 44)
- **Gamma weighting**: `w(k) = 1 + γ_weight · exp(-k² / (2σ²))` gives higher weight to points near Γ. Default: `γ_weight = 5.0`, `σ = 0.15 Å⁻¹`
- **Energy offset**: the S11 sample offset of −0.47 eV is applied to all computed energies

The optimization uses `scipy.optimize.minimize` with the Nelder-Mead method, launched from multiple random starting points (default: 10) within bounds of [−5, 5] eV for all four parameters. The best result across all starts is selected.

## Quick start

```bash
# Run interlayer coupling fit (default: 10 starts, gamma_weight=5.0)
python scripts/fit_bilayer_coupling.py

# With more starting points and custom gamma weighting
python scripts/fit_bilayer_coupling.py --n-starts 20 --gamma-weight 10.0 --gamma-sigma 0.1

# Verbose output
python scripts/fit_bilayer_coupling.py --verbose

# Save debug plots during optimization
python scripts/fit_bilayer_coupling.py --debug --debug-max 20
```

## Export script

The export script computes the top 8 valence bands along two Brillouin zone cuts using the fitted interlayer parameters and saves both data and plots:

```bash
python scripts/export_bilayer_bands.py
```

**Output** (all in `Data/interlayer_fit/`):

| File | Description |
|---|---|
| `bilayer_bands_KpGK.txt` | Top 8 valence bands along K′–Γ–K (201 k-points), tab-separated |
| `bilayer_bands_KpGK.png` | Plot of the above with ARPES data overlaid |
| `bilayer_bands_KpMK.txt` | Top 8 valence bands along K′–M–K (101 k-points), tab-separated |
| `bilayer_bands_KpMK.png` | Plot of the above (TB only, no ARPES data for this path) |

The `.txt` files contain 9 columns: cumulative |k| distance (Å⁻¹) followed by 8 band energies (eV) for bands 27 down to 20, with the S11 energy offset (−0.47 eV) applied.

## Output

Fitted interlayer parameters are saved to:

| File | Content |
|---|---|
| `Inputs/bilayer_fitting/interlayer_params.npy` | NumPy array `[w1p, w1d, w2p, w2d]` |
| `Inputs/bilayer_fitting/interlayer_params_metadata.json` | Metadata: parameter values, chi², nfev, success flag, timestamp |
| `Figures/bilayer_fit.png` | Final fit plot with parameter values and ARPES comparison |

