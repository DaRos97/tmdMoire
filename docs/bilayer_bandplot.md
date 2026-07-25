# Bilayer Moiré Band Plotting

## Overview

The `plot_moire_bands.py` script produces ARPES-like intensity heatmaps of the full moiré superlattice band structure. It diagonalizes the (44·N)×(44·N) supercell Hamiltonian along the Γ→K and K→M paths, mirrors them to produce the full K′→Γ→K and K→M→K′ cuts, computes orbital weights from eigenvectors, and spreads intensity using Gaussian or Lorentzian kernels.

## Input parameters

All parameters are loaded from `Inputs/plot_bilayer/`:

| File | Content |
|---|---|
| `tb_WSe2.npy` | 43 monolayer TB parameters for WSe₂ |
| `tb_WS2.npy` | 43 monolayer TB parameters for WS₂ |
| `interlayer_G.npy` | Dict: w1p, w1d, w2p, w2d (eV), Vg (eV), phiG (rad) |
| `interlayer_K.npy` | Dict: Vk (eV), phiK (rad) |

## Band selection

The script computes the full Hamiltonian eigenvalues and eigenvectors, then slices to the top valence bands. For each mini-BZ cell there are 44 bands; the top valence band (TVB) is at index 28 (0-based: 27). The script keeps bands 18–27 per cell (10 bands below and including the TVB), giving `10 × n_cells` bands total.

## Output figures

Three figures are generated in the cache directory:

| Figure | Description |
|---|---|
| `moire_bands_simulated.png` | Simulated intensity for both K′→Γ→K and K→M→K′ cuts |
| `arpes_data.png` | Experimental ARPES intensity for both cuts |
| `moire_bands_half_arpes.png` | Half ARPES / half simulated: ARPES on k < 0 side, simulated on k > 0 side |

All figures use a Greys colormap, centered momentum axes (k = 0 at Γ or M), and fixed xlim: ±1.4 Å⁻¹ for K′→Γ→K, ±1.2 Å⁻¹ for K→M→K′.

## CLI options

```bash
python scripts/plot_moire_bands.py [options]
```

| Option | Default | Description |
|---|---|---|
| `--k-pts` | 300 | Number of points along Γ→K→M path |
| `--n-shells` | 2 | Number of moiré shells (n_cells = 1 + 3n(n+1)) |
| `--spread-type` | Gauss | Spreading kernel: Gauss or Lorentz |
| `--spread-k` | 0.005 | k-space spreading width (Å⁻¹) |
| `--spread-e` | 0.015 | Energy spreading width (eV) |
| `--pow-factor` | 2.0 | Exponent on eigenvector amplitudes |
| `--shade-ws2` | 0.1 | Weight multiplier for WS₂ orbitals |
| `--shade-e-factor` | 3.0 | Linear energy shading multiplier at E_max (E_min fixed at 0.1) |
| `--e-min` | -3.5 | Minimum energy (eV) |
| `--e-max` | 0.0 | Maximum energy (eV) |
| `--delta-e` | 0.01 | Energy grid spacing (eV) |
| `--no-cache` | — | Ignore cache and recompute |
| `--sample` | S11 | Sample name for energy offset |
| `--theta` | — | Twist angle in degrees (overrides sample) |
| `--Vg` | — | Override moiré potential at Γ (eV) |
| `--Vk` | — | Override moiré potential at K (eV) |
| `--Vg` | — | Override moiré potential at Γ (eV) |
| `--Vk` | — | Override moiré potential at K (eV) |

## Energy shading

A linear multiplicative gradient is applied along the energy axis to mimic ARPES intensity falloff at deeper binding energies. The shading factor starts at **0.1 at E_min** and increases linearly to **`shade_e_factor` at E_max**. With the default `--shade-e-factor 3.0`, bands near the Fermi level are amplified 30× relative to the deepest bands.

## Two-level caching

To support rapid exploration of different spreading and shading parameters without re-diagonalizing, the script uses a two-level cache in `Data/plot_bilayer_moire/`:

```
Data/plot_bilayer_moire/
  diag_k<k-pts>_n<n-shells>_<w1p>_<w1d>_<w2p>_<w2d>_<Vg>_<phiG>_<Vk>_<phiK>/
    diag.npz                                        ← eigenvalues, eigenvectors, k-points, norms
    intensity_<type>_<sk>_<se>_<pow>_<shade_ws2>_<shade_e>/
      spread.npz                                    ← spread intensity maps
      moire_bands_simulated.png
      arpes_data.png
      moire_bands_half_arpes.png
    intensity_<type2>_<sk2>_<se2>_<pow2>_<shade_ws2_2>_<shade_e_2>/
      ...
  diag_k<k-pts>_n<n-shells>_.../
    ...
```

**Diagonalization cache** (expensive, minutes to hours):
- Hashed by: n_shells, k_pts, interlayer params (w1p, w1d, w2p, w2d), moire params (Vg, phiG, Vk, phiK)
- Stores: evals, evecs, norm, k_list for both paths
- Saved as: `diag_<params>/diag.npz`

**Spread intensity cache** (cheap, seconds):
- Hashed by: spread_type, spread_k, spread_e, pow_factor, shade_ws2, shade_e_factor
- Stores: spread_kgk, spread_kmkp intensity maps
- Saved as: `diag_<params>/intensity_<params>/spread.npz`
- Figures are saved directly in the intensity directory

Changing spreading or shading parameters (e.g. `--spread-e`, `--shade-e-factor`, `--pow-factor`) reuses the cached diagonalization and only recomputes the intensity spreading (~seconds). Changing `--n-shells`, `--k-pts`, or any interlayer/moire parameter triggers a new diagonalization.

Old cache directories accumulate on disk. Clean manually when needed:

```bash
# Remove all cached data
rm -rf Data/plot_bilayer_moire/

# Remove a specific diagonalization cache
rm -rf Data/plot_bilayer_moire/diag_k300_n2_*/
```

