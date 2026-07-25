# Monolayer Fitting

## Overview

The monolayer stage fits a 22×22 tight-binding Hamiltonian (11 orbitals × 2 spins) to reproduce ARPES-measured band dispersions along high-symmetry paths K′–Γ–K and K–M–K′. The fit optimizes 43 parameters against experimental data using **Nelder-Mead minimization** starting from DFT-derived initial values, with multiple physical constraints.

## Experimental data processing

ARPES band dispersion data is stored as tab-delimited text files in `Inputs/` and processed through a three-step pipeline before fitting:

### 1. Raw loading

Files follow the naming convention `{path}_{TMD}_band{N}.txt` (e.g. `KpGK_WSe2_band1.txt`). Each file contains two columns: momentum (Å⁻¹) and energy (eV). Missing energy values are encoded as `NAN` or empty lines and stored as `np.nan`. The number of bands per path and material is defined in `Inputs/manifest.json`.

| File pattern | Path | Bands | Points (typical) |
|---|---|---|---|
| `KpGK_*_band{1,2}.txt` | K′ → Γ → K | 2 (top valence) | ~770 |
| `KpGK_*_band{3,4}.txt` | K′ → Γ → K | 2 (deeper valence) | ~70 |
| `KpGK_*_band{5,6}.txt` | K′ → Γ → K | 2 (deep valence) | ~25 |
| `KMKp_*_band{1-4}.txt` | K → M → K′ | 4 | ~425 / ~120 |

### 2. Symmetrization

The raw data spans both sides of the high-symmetry points (Γ for KpGK, M for KMKp). Symmetrization averages equivalent segments:

- **General case**: The data is split at the symmetry point. The left segment is reversed and averaged with the right segment. Where only one side has valid data, that value is kept. Points where both sides are NaN are discarded.
- **KpGK bands 3–6**: These bands have sparse data with only negative momenta. They are converted to positive `|k|` and reversed so momentum increases from Γ outward. No averaging is performed.
- **WS2 KMKp bands 3–4**: Only the left segment (K→M) is used because the right segment (M→K′) has poorer experimental quality. The left side is mirrored to positive momentum.

The symmetrized data is cached in `Data/sym_{TMD}.npz` to avoid re-processing on subsequent runs. The cache is invalidated when any raw input file is modified.

### 3. Interpolation

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

## Hamiltonian basis

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

## 43 fitted parameters

| Range | Type | Count | Description |
|---|---|---|---|
| 0–6 | ε | 7 | On-site energies |
| 7–27 | t₁ | 21 | Nearest-neighbor hoppings |
| 28–35 | t₅ | 8 | M–X coupling hoppings |
| 36–39 | t₆ | 4 | Second-nearest-neighbor hoppings |
| 40 | offset | 1 | Global energy shift |
| 41–42 | L_W, L_S | 2 | Spin-orbit coupling strengths |

## Chi-squared objective

The minimization optimizes a weighted sum:

```
χ² = χ²_band_weighted + K₁·C₁ + K₂·C₂ + K₃·C₃ + K₄·C₄ + K₅·C₅
```

where `χ²_band_weighted` is the K₆-weighted band distance and `C₁`–`C₅` are the five physical constraint terms. The weights `K₁`–`K₆` are scanned over a grid of 1,600 combinations to find the best trade-off between band accuracy and physical constraints.

### Band distance

**What it does**: Measures how well the TB band energies match the experimental ARPES data across all 6 bands and all k-points.

**Implementation**: For each band `b`, compute the squared residual `(E_TB - E_ARPES)²` over all valid (non-NaN) k-points, divide by that band's valid-point count, then sum across bands. This **per-band normalization** gives equal weight to each band regardless of how many data points it has. Four special k-points (Γ, top of band 1, minimum of band 2, and M) receive an additional weight multiplier `K₆`:

```python
χ²_band = Σ_b [ Σ_i w_i · (E_TB[b,i] − E_ARPES[b,i])²  /  N_valid[b] ]
```

where `w_i = K₆` at the four special points and `w_i = 1` elsewhere. `N_valid[b]` is the number of valid ARPES data points for band `b`.

Two variants of the band distance are stored:
- **`band_K6`** (K₆-weighted, used in the objective function)
- **`band_dist`** (unweighted, i.e. `w_i = 1` everywhere — used for cross-comparison between results with different K₆)

### K₁ — parameter distance from DFT

**What it does**: Penalizes parameters that deviate far from their DFT-derived initial values, preventing unphysical results.

**Implementation**: Mean absolute relative deviation of all parameters (except the global offset) from their DFT values. Excludes the offset (index 40) since it is a fitting artifact, not a physical parameter:

```python
C₁ = [ Σ_{i≠40} |p_i - p_DFT,i| / |p_DFT,i| ] / (N_params - 1)
```

Typical range: 0 (at DFT) to ~2 (large deviations).

### K₂ — orbital band content at M

**What it does**: Minimizes the weight of interlayer-coupling orbitals (p_z^o, d_z², p_z^e) in the top valence bands at the M point. These are the orbitals that participate in interlayer hopping in the bilayer model. Since ARPES shows no noticeable change in the band structure at M between monolayer and bilayer, the interlayer-coupling orbital character at M should remain small — the fit penalizes any mixing of these orbitals into the valence bands at M.

**Implementation**: Sum of squared eigenvector components `|c|²` for the 6 interlayer-coupling orbitals (p_z^o, d_z², p_z^e, both spin blocks; `IND_ILC = [2, 5, 8, 13, 16, 19]`), summed across the top valence bands at M:

```python
C₂ = Σ_{orb ∈ ILC} Σ_{band ∈ TVB} |⟨orb|ψ_band(M)⟩|²
```

For WS₂ the result is multiplied by 2 to give the term the same order of magnitude as for WSe₂ (since WS₂ has 2 top valence bands vs 4 for WSe₂). There is no normalization by the number of orbitals or bands — this is a raw sum.

### K₃ — orbital occupation at Γ and K

**What it does**: Enforces the DFT-derived orbital occupations of the top valence bands at the high-symmetry points Γ and K. These occupations are well-defined from symmetry and serve as strong physical anchors.

**Implementation**: Sum of eight absolute differences between target DFT occupations and computed occupations:

- **At Γ** (4 terms): p_z^e and d_z² content in each of the two degenerate TVB states
- **At K** (4 terms): p₋₁^e and d₋₂ content in each of the two TVB states (p₋₁^e = (p_x^e − i·p_y^e)/√2, d₋₂ = (d_x²−y² − i·d_xy)/√2)

```python
C₃ = Σ |occ_DFT − occ_TB|
```

The raw sum of 8 absolute differences (not mean, no division by 8).

| Material | Γ p_z^e | Γ d_z² | K p₋₁^e (TVB1) | K p₋₁^e (TVB2) | K d₋₂ (TVB1) | K d₋₂ (TVB2) |
|---|---|---|---|---|---|---|
| WSe₂ | 0.2740 | 0.6606 | 0.1856 | 0.2116 | 0.8144 | 0.7763 |
| WS₂ | 0.3205 | 0.6571 | 0.1960 | 0.2366 | 0.8040 | 0.7575 |

### K₄ — conduction band minimum at K

**What it does**: Forces the conduction band minimum (CBM) to sit at the K point, as required by the physics of TMD monolayers.

**Implementation**: Binary penalty: 0 if the k-point of the CBM is within 10⁻³ of |K|, 1 otherwise.

```python
C₄ = 0   if | |k_CBM| − |K| | < 1×10⁻³
C₄ = 1   otherwise
```

### K₅ — band gap at K

**What it does**: Keeps the band gap at K close to the DFT-predicted value.

**Implementation**: Absolute difference between the current gap and the DFT gap at K:

```python
C₅ = |gap_DFT − gap_TB|
```

The DFT gap is precomputed once from the DFT-derived parameters and stored as `_gap_DFT`.

### K₆ — high-symmetry point weight

**What it does**: Increases the importance of four special k-points in the band distance term: Γ (index 0), the top of band 1, the minimum of band 2, and the M point (last index). These points are physically significant and should be fitted accurately.

**Implementation**: Multiplies the residual weight `w_i` by `K₆` at these four k-points. With `K₆ = 5`, each residual at a special point contributes 5× as much to χ²_band as a regular point.

## Quick start

```bash
# Fit WSe₂ with parameter set index 0
python scripts/fit_monolayer.py WSe2 0

# Fit WS₂ with parameter set index 5
python scripts/fit_monolayer.py WS2 5
```

The index selects a combination of constraint weights (K₁–K₆) from the grid defined in `Inputs/monolayer_fitting/fit_config.json`.

## Grid search

Instead of running individual fits, you can sweep over all combinations of constraint weights (K₁–K₆) defined in `Inputs/monolayer_fitting/fit_config.json`:

```bash
# Run all combinations for WSe₂
python scripts/run_monolayer_grid.py WSe2

# Run a subset (for chunking on HPC)
python scripts/run_monolayer_grid.py WSe2 --start 0 --end 100

# Score and rank existing results (v3.0-style ranking)
python scripts/run_monolayer_grid.py WSe2 --score

# Show top results
python scripts/run_monolayer_grid.py WSe2 --score --top 20

# Export best params to Inputs/bilayer_fitting/
python scripts/run_monolayer_grid.py WSe2 --score --export
```

The default grid has 2×10×10×2×2×2 = **1,600 combinations**. Each fit uses Nelder-Mead minimization (maxiter=1,000,000, fatol=1×10⁻⁴) starting from DFT parameters.

### Scoring and Ranking

Results are scored with a v3.0-style procedure:

1. **K-value range mask**: filters results to keep only physically relevant weight ranges (K₂ between −2⁻⁸ and 10, K₃ > −0.012, etc.)
2. **Bounds-saturation filter** (WSe₂ only): excludes results where any parameter group saturated its bounds (i.e. parameters hit ±B within 1% tolerance)
3. **Primary ranking**: sort by `band_K6` (K₆-weighted band distance, which is the `χ²_band` term from the objective function) — index 1 (2nd best) for WSe₂, index 0 (best) for WS₂
4. **Secondary ranking**: sort by `band_K6 + K₂_val` (band distance + M orbital content) with the same `ind_chosen` convention

Both rankings are presented side by side. The export step saves the best result from the primary (`band_K6`) ranking.

### Visualizing results (`sort_monolayer_results.py`)

The script `scripts/sort_monolayer_results.py` loads merged HDF5 files (from v3.0) or individual `.npz` results, plots 2D heatmaps of min(chi2) and min(chi2+K₂_M) versus K₂ and K₃, and offers interactive inspection and export:

```bash
# From v3.0 merged HDF5
python scripts/sort_monolayer_results.py --tmd WSe2 --input Data/WSe2_run1/merged_WSe2_absolute.h5
python scripts/sort_monolayer_results.py --tmd WS2 --input Data/WS2_run1/merged_WS2_absolute.h5

# From .npz directory (auto-detected)
python scripts/sort_monolayer_results.py --tmd WSe2 --input-dir Data/WSe2_default

# Custom cutoff for heatmap (default: 0.3)
python scripts/sort_monolayer_results.py --tmd WSe2 --input data.h5 --cutoff 0.1
```

After the heatmaps close, an interactive prompt lets you inspect individual results (bands, parameters, orbital content) and optionally save the best parameters as `Data/result_{TMD}.npy`.


## Initial point control

The optimizer always starts from the DFT-derived parameters as the initial point (`x0`). This is the standard v3.0 approach: Nelder-Mead is a local optimizer, so starting from the physically motivated DFT values ensures convergence to a physically meaningful minimum.

## HPC workflow

For the full grid search on the HPC cluster (SGE/rademaker queue), use the scripts in `HPC/`:

```bash
# Submit 128 parallel tasks for WSe₂ (default run ID)
./HPC/mono_job.sh WSe2

# Submit with a named run ID
./HPC/mono_job.sh WSe2 001

# Submit for WS₂
./HPC/mono_job.sh WS2 002

# Submit with custom number of tasks
./HPC/mono_job.sh WSe2 001 256
```

Each job array submission creates N SGE tasks. Chunk boundaries are computed automatically from `fit_config.json`. Output goes to `Scratch/grid_<material>_<run_id>_task<N>.out`.

After all tasks complete, score the results:

```bash
python scripts/run_monolayer_grid.py WSe2 --score --id 001
```

## Run management

Each run is stored in its own subdirectory under `Data/<TMD>_<id>/`. When you start a run, `Inputs/monolayer_fitting/fit_config.json` is copied into the run directory as a snapshot, making each run fully self-contained and reproducible.

```
Data/
  WSe2_001/
    fit_config.json          ← snapshot of config used for this run
    fit_idx0.npz
    fit_idx1.npz
    ...
  WS2_002/
    fit_config.json          ← different config (e.g. finer grid)
    fit_WS2_idx0.npz
    ...
```

**Iterative workflow:**

1. Run the initial grid search: `./HPC/mono_job.sh WSe2 001`
2. Score results and inspect the best fits
3. Edit `Inputs/monolayer_fitting/fit_config.json` to refine the grid (e.g. narrower ranges, finer spacing)
4. Run again with a new ID: `./HPC/mono_job.sh WSe2 002`
5. Compare runs: `python scripts/run_monolayer_grid.py WSe2 --score --id 001` and `--id 002`

The `--id` flag works with all scripts:

```bash
python scripts/run_monolayer_grid.py WSe2 --start 0 --end 100 --id 002
python scripts/run_monolayer_grid.py WSe2 --score --id 002 --top 20
python scripts/fit_monolayer.py WSe2 42 --id 002
```

## Programmatic usage

```python
from tmdmoire.material import TMDMaterial
from tmdmoire.monolayer.data import MonolayerData
from tmdmoire.monolayer.fitter import ParameterFitter
from tmdmoire.utils.paths import get_repo_root

# Create material with DFT initial parameters
material = TMDMaterial("WSe2")

# Load experimental ARPES data (symmetrized data is cached automatically)
data = MonolayerData("WSe2", master_folder=get_repo_root(), pts=91)

# Configure the fitter
config = {
    "Ks": (0.0, 0.125, 0.01, 1.0, 0.1, 5),   # K1-K6 weights
    "boundType": "absolute",
    "Bs": (8, 4, 5, 2, 0),  # bounds for eps, t1, t5, t6, SOC (±eV)
    "optimizer": {"nm_maxiter": 1000000, "nm_fatol": 1e-4},
}

fitter = ParameterFitter(material, data, config)
result = fitter.run(seed=42, output_dir="Data")

print(f"Final chi²: {result['fun']}")
print(f"Optimized parameters: {result['x']}")
```

The fitter starts from DFT parameters and minimizes via Nelder-Mead. Intermediate best results are saved to `temp_best_{idx}.npz` during the run.

## Output

Fitted parameters from grid searches are saved as `.npz` files in `Data/<TMD>_<id>/`. Each file contains the optimized parameters, chi-squared values, individual constraint values, and the computed band energies. Symmetrized ARPES data is cached as `Data/sym_{TMD}.npz`.

## Lattice constants

| Material | a (Å) |
|---|---|
| WS₂ | 3.18 |
| WSe₂ | 3.32 |

