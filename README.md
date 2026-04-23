# TMD heterobilayer WSe2/WS2

Tight-binding model of WSe₂/WS₂ heterobilayer moiré superlattices. Two-stage computational workflow:

1. **Monolayer** — Fit 43 tight-binding parameters per TMD (WSe₂, WS₂) to ARPES data
2. **Bilayer** — Build moiré supercell Hamiltonian, compute EDCs, extract moiré potential & interlayer coupling

## Monolayer Fitting

### Overview

The monolayer stage fits a 22×22 tight-binding Hamiltonian (11 orbitals × 2 spins) to reproduce ARPES-measured band dispersions along high-symmetry paths K′–Γ–K and K–M–K′. The fit optimizes 43 parameters against experimental data using Nelder-Mead minimization with multiple physical constraints.

### Hamiltonian basis

| Index | Orbital | Parity |
|-------|---------|--------|
| 0–1   | d_xz, d_yz | odd |
| 2     | p_z^o | odd |
| 3–4   | p_x^o, p_y^o | odd |
| 5     | d_z² | even |
| 6–7   | d_xy, d_x²-y² | even |
| 8     | p_z^e | even |
| 9–10  | p_x^e, p_y^e | even |

Indices 11–21 are the spin-down counterparts.

### 43 fitted parameters

| Range | Type | Count | Description |
|-------|------|-------|-------------|
| 0–6   | ε    | 7     | On-site energies |
| 7–27  | t₁   | 21    | Nearest-neighbor hoppings |
| 28–35 | t₅   | 8     | M–X coupling hoppings |
| 36–39 | t₆   | 4     | Second-nearest-neighbor hoppings |
| 40    | offset | 1   | Global energy shift |
| 41–42 | L_W, L_S | 2 | Spin-orbit coupling strengths |

### Chi-squared objective

The minimization optimizes a weighted sum of six terms:

| Term | Weight | Description |
|------|--------|-------------|
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

# Load experimental ARPES data
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

### Input data

ARPES band data lives in `Inputs/`:

```
Inputs/
├── KpGK_WSe2_band{1-6}.txt    # Band dispersions along K′–Γ–K
├── KpGK_WS2_band{1-6}.txt
├── KMKp_WSe2_band{1-4}.txt    # Band dispersions along K–M–K′
└── KMKp_WS2_band{1-4}.txt
```

Each file contains tab-delimited `momentum  energy` pairs.

### Output

Fitted parameters are saved as `.npy` files. Intermediate results during minimization are saved as `.npz` files in `Data/`.

### Lattice constants

| Material | a (Å) |
|----------|-------|
| WS₂      | 3.18  |
| WSe₂     | 3.32  |

## Bilayer

*(Documentation forthcoming)*

## References

*(Documentation forthcoming)*
