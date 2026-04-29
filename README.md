# TMD heterobilayer WSe₂/WS₂

Tight-binding model of WSe₂/WS₂ heterobilayer moiré superlattices. Two-stage computational workflow:

1. **Monolayer** — Fit 43 tight-binding parameters per TMD (WSe₂, WS₂) to ARPES data
2. **Bilayer** — Build moiré supercell Hamiltonian, compute EDCs, extract moiré potential & interlayer coupling

## Monolayer Fitting

### Overview

The monolayer stage fits a 22×22 tight-binding Hamiltonian (11 orbitals × 2 spins) to reproduce ARPES-measured band dispersions along high-symmetry paths K′–Γ–K and K–M–K′. The fit optimizes 43 parameters against experimental data using Nelder-Mead minimization with multiple physical constraints.

### Experimental data processing

ARPES band dispersion data is stored as tab-delimited text files in `Inputs/` and processed through a three-step pipeline before fitting:

#### 1. Raw loading

Files follow the naming convention `{path}_{TMD}_band{N}.txt` (e.g. `KpGK_WSe2_band1.txt`). Each file contains two columns: momentum (Å⁻¹) and energy (eV). Missing energy values are encoded as `NAN` or empty lines and stored as `np.nan`. The number of bands per path and material is defined in `Inputs/manifest.json`.

| File pattern | Path | Bands | Points (typical) |
|---|---|---|---|
| `KpGK_*_band{1,2}.txt` | K′ → Γ → K | 2 (top valence) | ~770 |
| `KpGK_*_band{3,4}.txt` | K′ → Γ → K | 2 (deeper valence) | ~70 |
| `KpGK_*_band{5,6}.txt` | K′ → Γ → K | 2 (deep valence) | ~25 |
| `KMKp_*_band{1-4}.txt` | K → M → K′ | 4 | ~425 / ~120 |

#### 2. Symmetrization

The raw data spans both sides of the high-symmetry points (Γ for KpGK, M for KMKp). Symmetrization averages equivalent segments:

- **General case**: The data is split at the symmetry point. The left segment is reversed and averaged with the right segment. Where only one side has valid data, that value is kept. Points where both sides are NaN are discarded.
- **KpGK bands 3–6**: These bands have sparse data with only negative momenta. They are converted to positive `|k|` and reversed so momentum increases from Γ outward. No averaging is performed.
- **WS2 KMKp bands 3–4**: Only the left segment (K→M) is used because the right segment (M→K′) has poorer experimental quality. The left side is mirrored to positive momentum.

The symmetrized data is cached in `Data/sym_{TMD}.npz` to avoid re-processing on subsequent runs. The cache is invalidated when any raw input file is modified.

#### 3. Interpolation

Symmetrized data is interpolated onto `pts` equidistant points along the combined Γ–K–M path. The output array has shape `(pts, 9)`:

| Column | Content |
|---|---|
| 0 | `|k|` — cumulative distance along the path |
| 1 | `kx` — x-component of momentum |
| 2 | `ky` — y-component of momentum |
| 3–8 | `E_band1` through `E_band6` — band energies (NaN where no data) |

**Energy offset**: The KMKp segment receives a material-specific energy shift to align it with the KpGK segment, correcting for experimental calibration differences between the two measurement paths:

| Material | Offset (eV) |
|---|---|
| WSe₂ | −0.052 |
| WS₂ | +0.010 |

### Hamiltonian basis

| Index | Orbital | Parity |
|---|---|---|
| 0–1 | d_xz, d_yz | odd |
| 2 | p_z^o | odd |
| 3–4 | p_x^o, p_y^o | odd |
| 5 | d_z² | even |
| 6–7 | d_xy, d_x²-y² | even |
| 8 | p_z^e | even |
| 9–10 | p_x^e, p_y^e | even |

Indices 11–21 are the spin-down counterparts.

### 43 fitted parameters

| Range | Type | Count | Description |
|---|---|---|---|
| 0–6 | ε | 7 | On-site energies |
| 7–27 | t₁ | 21 | Nearest-neighbor hoppings |
| 28–35 | t₅ | 8 | M–X coupling hoppings |
| 36–39 | t₆ | 4 | Second-nearest-neighbor hoppings |
| 40 | offset | 1 | Global energy shift |
| 41–42 | L_W, L_S | 2 | Spin-orbit coupling strengths |

### Chi-squared objective

The minimization optimizes a weighted sum of six terms:

| Term | Weight | Description |
|---|---|---|
| Band distance | 1.0 | Σ(TB band − ARPES band)² |
| K₁ | variable | Parameter distance from DFT values |
| K₂ | variable | Orbital band content at M point |
| K₃ | variable | Orbital occupation at Γ and K vs DFT |
| K₄ | variable | Conduction band minimum position at K |
| K₅ | variable | Band gap at K vs DFT gap |
| K₆ | variable | Weight multiplier for high-symmetry points (Γ, K, M) |

### Quick start

```bash
# Fit WSe₂ with parameter set index 0
python scripts/fit_monolayer.py WSe2 0

# Fit WS₂ with parameter set index 5
python scripts/fit_monolayer.py WS2 5
```

The index selects a combination of constraint weights (K₁–K₆) from a parameter grid. There are 2×10×10×2×2×2 = 1600 combinations total.

### Programmatic usage

```python
from tmdmoire import TMDMaterial, ARPESData, ParameterFitter

# Create material with DFT initial parameters
material = TMDMaterial("WSe2")

# Load experimental ARPES data (symmetrized data is cached automatically)
arpes = ARPESData("WSe2", master_folder="/path/to/repo/", pts=91)

# Configure the fitter
config = {
    "pts": 91,
    "Ks": (1e-5, 0.5, 1.0, 1.0, 0.5, 5.0),  # K1-K6 weights
    "boundType": "absolute",
    "Bs": (5, 2, 4, 1, 0),  # bounds for eps, t1, t5, t6, SOC
}

fitter = ParameterFitter(material, arpes, config)
result = fitter.run(material.dft_params, max_eval=int(5e6))

print(f"Final chi²: {result['fun']}")
print(f"Optimized parameters: {result['x']}")
```

### Output

Fitted parameters are saved as `.npy` files. Intermediate results during minimization are saved as `.npz` files in `Data/`. Symmetrized ARPES data is cached as `Data/sym_{TMD}.npz`.

### Lattice constants

| Material | a (Å) |
|---|---|
| WS₂ | 3.18 |
| WSe₂ | 3.32 |

## Bilayer Moiré Bands

*(Documentation forthcoming)*

### Overview

The bilayer stage constructs a moiré supercell Hamiltonian for a twisted WSe₂/WS₂ heterobilayer and computes Energy Distribution Curves (EDCs) at high-symmetry points (Γ and K) to extract the moiré potential amplitude/phase and interlayer coupling strengths.

### Quick start

```bash
# EDC analysis at Gamma, chunk 0
python scripts/analyze_edc.py G 0

# EDC analysis at K, chunk 5
python scripts/analyze_edc.py K 5
```

### Parameters

| Parameter | Description | Typical range |
|---|---|---|
| Vg, Vk | Moiré potential amplitude at Γ and K | 0.001–0.040 eV |
| φG, φK | Moiré potential phase | 0–360° |
| w1p | Interlayer p-orbital coupling | −1.58 to −1.53 eV |
| w1d | Interlayer d-orbital coupling | 1.12 to 1.17 eV |

### Output

Results are saved as chunked HDF5 files in `Data/`.

## References

*(Documentation forthcoming)*
