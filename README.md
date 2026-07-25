# TMD heterobilayer WSe₂/WS₂

Tight-binding model of WSe₂/WS₂ heterobilayer moiré superlattices. Three-stage computational workflow:

1. **Monolayer fitting** — Fit 43 tight-binding parameters per TMD (WSe₂, WS₂) to ARPES band dispersion data
2. **Bilayer interlayer coupling** — Fit 4 interlayer hopping parameters (w1p, w1d, w2p, w2d) to bilayer ARPES data
3. **Bilayer moiré potential** — Sweep moiré potential parameters (Vg, Vk, φG, φK) to match experimental EDC peaks

## Table of Contents

- [Monolayer Fitting](#monolayer-fitting)
  - [Overview](#overview)
  - [Experimental data processing](#experimental-data-processing)
    - [1. Raw loading](#1-raw-loading)
    - [2. Symmetrization](#2-symmetrization)
    - [3. Interpolation](#3-interpolation)
  - [Hamiltonian basis](#hamiltonian-basis)
  - [43 fitted parameters](#43-fitted-parameters)
  - [Chi-squared objective](#chi-squared-objective)
    - [Band distance](#band-distance)
    - [K₁ — parameter distance from DFT](#k1--parameter-distance-from-dft)
    - [K₂ — orbital band content at M](#k2--orbital-band-content-at-m)
    - [K₃ — orbital occupation at Γ and K](#k3--orbital-occupation-at-%CE%93-and-k)
    - [K₄ — conduction band minimum at K](#k4--conduction-band-minimum-at-k)
    - [K₅ — band gap at K](#k5--band-gap-at-k)
    - [K₆ — high-symmetry point weight](#k6--high-symmetry-point-weight)
  - [Quick start](#quick-start)
  - [Grid search](#grid-search)
  - [HPC workflow](#hpc-workflow)
  - [Run management](#run-management)
  - [Programmatic usage](#programmatic-usage)
  - [Output](#output)
  - [Lattice constants](#lattice-constants)
- [Bilayer Moiré Bands](#bilayer-moir%C3%A9-bands)
  - [Overview](#bilayer-overview)
  - [Interlayer coupling form](#interlayer-coupling-form)
  - [Minimization](#minimization)
  - [Quick start](#bilayer-quick-start)
  - [Export script](#export-script)
  - [Output](#bilayer-output)
- [Bilayer Moiré Potential (EDC Analysis)](#bilayer-moir%C3%A9-potential-edc-analysis)
  - [Overview](#edc-overview)
  - [EDC Intensity Profile](#edc-intensity-profile)
  - [Gamma-Point Sweep](#gamma-point-sweep)
  - [K-Point Sweep](#k-point-sweep)
  - [Configuration](#edc-configuration)
  - [Quick Start](#edc-quick-start)
  - [HPC Workflow](#edc-hpc-workflow)
  - [Run Management](#edc-run-management)
  - [Output Format](#edc-output-format)
- [Bilayer Moiré Band Plotting](#bilayer-moir%C3%A9-band-plotting)
  - [Overview](#plotting-overview)
  - [Input parameters](#input-parameters)
  - [Band selection](#band-selection)
  - [Output figures](#output-figures)
  - [CLI options](#cli-options)
  - [Two-level caching](#two-level-caching)
- [Real-Space LDOS](#real-space-ldos)
  - [Overview](#ldos-overview)
  - [Physical formula](#physical-formula)
  - [Wavefunction reconstruction](#wavefunction-reconstruction)
  - [CLI options](#ldos-cli-options)
  - [Two-level caching](#ldos-caching)
  - [Output](#ldos-output)
- [References](#references)

## Monolayer Fitting

### Overview

The monolayer stage fits a 22×22 tight-binding Hamiltonian (11 orbitals × 2 spins) to reproduce ARPES-measured band dispersions along high-symmetry paths K′–Γ–K and K–M–K′. The fit optimizes 43 parameters against experimental data using **Nelder-Mead minimization** starting from DFT-derived initial values, with multiple physical constraints.

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

The minimization optimizes a weighted sum:

```
χ² = χ²_band_weighted + K₁·C₁ + K₂·C₂ + K₃·C₃ + K₄·C₄ + K₅·C₅
```

where `χ²_band_weighted` is the K₆-weighted band distance and `C₁`–`C₅` are the five physical constraint terms. The weights `K₁`–`K₆` are scanned over a grid of 1,600 combinations to find the best trade-off between band accuracy and physical constraints.

#### Band distance

**What it does**: Measures how well the TB band energies match the experimental ARPES data across all 6 bands and all k-points.

**Implementation**: For each band `b`, compute the squared residual `(E_TB - E_ARPES)²` over all valid (non-NaN) k-points, divide by that band's valid-point count, then sum across bands. This **per-band normalization** gives equal weight to each band regardless of how many data points it has. Four special k-points (Γ, top of band 1, minimum of band 2, and M) receive an additional weight multiplier `K₆`:

```python
χ²_band = Σ_b [ Σ_i w_i · (E_TB[b,i] − E_ARPES[b,i])²  /  N_valid[b] ]
```

where `w_i = K₆` at the four special points and `w_i = 1` elsewhere. `N_valid[b]` is the number of valid ARPES data points for band `b`.

Two variants of the band distance are stored:
- **`band_K6`** (K₆-weighted, used in the objective function)
- **`band_dist`** (unweighted, i.e. `w_i = 1` everywhere — used for cross-comparison between results with different K₆)

#### K₁ — parameter distance from DFT

**What it does**: Penalizes parameters that deviate far from their DFT-derived initial values, preventing unphysical results.

**Implementation**: Mean absolute relative deviation of all parameters (except the global offset) from their DFT values. Excludes the offset (index 40) since it is a fitting artifact, not a physical parameter:

```python
C₁ = [ Σ_{i≠40} |p_i - p_DFT,i| / |p_DFT,i| ] / (N_params - 1)
```

Typical range: 0 (at DFT) to ~2 (large deviations).

#### K₂ — orbital band content at M

**What it does**: Minimizes the weight of interlayer-coupling orbitals (p_z^o, d_z², p_z^e) in the top valence bands at the M point. These are the orbitals that participate in interlayer hopping in the bilayer model. Since ARPES shows no noticeable change in the band structure at M between monolayer and bilayer, the interlayer-coupling orbital character at M should remain small — the fit penalizes any mixing of these orbitals into the valence bands at M.

**Implementation**: Sum of squared eigenvector components `|c|²` for the 6 interlayer-coupling orbitals (p_z^o, d_z², p_z^e, both spin blocks; `IND_ILC = [2, 5, 8, 13, 16, 19]`), summed across the top valence bands at M:

```python
C₂ = Σ_{orb ∈ ILC} Σ_{band ∈ TVB} |⟨orb|ψ_band(M)⟩|²
```

For WS₂ the result is multiplied by 2 to give the term the same order of magnitude as for WSe₂ (since WS₂ has 2 top valence bands vs 4 for WSe₂). There is no normalization by the number of orbitals or bands — this is a raw sum.

#### K₃ — orbital occupation at Γ and K

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

#### K₄ — conduction band minimum at K

**What it does**: Forces the conduction band minimum (CBM) to sit at the K point, as required by the physics of TMD monolayers.

**Implementation**: Binary penalty: 0 if the k-point of the CBM is within 10⁻³ of |K|, 1 otherwise.

```python
C₄ = 0   if | |k_CBM| − |K| | < 1×10⁻³
C₄ = 1   otherwise
```

#### K₅ — band gap at K

**What it does**: Keeps the band gap at K close to the DFT-predicted value.

**Implementation**: Absolute difference between the current gap and the DFT gap at K:

```python
C₅ = |gap_DFT − gap_TB|
```

The DFT gap is precomputed once from the DFT-derived parameters and stored as `_gap_DFT`.

#### K₆ — high-symmetry point weight

**What it does**: Increases the importance of four special k-points in the band distance term: Γ (index 0), the top of band 1, the minimum of band 2, and the M point (last index). These points are physically significant and should be fitted accurately.

**Implementation**: Multiplies the residual weight `w_i` by `K₆` at these four k-points. With `K₆ = 5`, each residual at a special point contributes 5× as much to χ²_band as a regular point.

### Quick start

```bash
# Fit WSe₂ with parameter set index 0
python scripts/fit_monolayer.py WSe2 0

# Fit WS₂ with parameter set index 5
python scripts/fit_monolayer.py WS2 5
```

The index selects a combination of constraint weights (K₁–K₆) from the grid defined in `Inputs/monolayer_fitting/fit_config.json`.

### Grid search

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

#### Scoring and Ranking

Results are scored with a v3.0-style procedure:

1. **K-value range mask**: filters results to keep only physically relevant weight ranges (K₂ between −2⁻⁸ and 10, K₃ > −0.012, etc.)
2. **Bounds-saturation filter** (WSe₂ only): excludes results where any parameter group saturated its bounds (i.e. parameters hit ±B within 1% tolerance)
3. **Primary ranking**: sort by `band_K6` (K₆-weighted band distance, which is the `χ²_band` term from the objective function) — index 1 (2nd best) for WSe₂, index 0 (best) for WS₂
4. **Secondary ranking**: sort by `band_K6 + K₂_val` (band distance + M orbital content) with the same `ind_chosen` convention

Both rankings are presented side by side. The export step saves the best result from the primary (`band_K6`) ranking.

#### Visualizing results (`sort_monolayer_results.py`)

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

### Results

Pre-computed results from the v3.0 grid search are available in `Data/WSe2_run1/` and `Data/WS2_run1/` as merged HDF5 files. Both runs used the same physical bounds **Bs = (8, 4, 5, 2, 0)** (±8 eV on-site, ±4 eV t₁, ±5 eV t₅, ±2 eV t₆, SOC fixed). The weights K₄=1, K₅=0.01, and K₆=5 were held fixed in both runs.

#### Selection convention

Following the v3.0 procedure, results are ranked by the band-distance component (K₆-weighted band distance, `elements[:, 0]` in the HDF5). To avoid artifacts from parameter-bound saturation, the **2nd best** result (`ind_chosen = 1`) is selected for **WSe₂**, while the **1st best** (`ind_chosen = 0`) is selected for **WS₂**. The values reported below are the results that `sort_monolayer_results.py` selects and displays.

#### WSe₂ (`Data/WSe2_run1/`)

| Aspect | Value |
|---|---|
| Total runs (after filtering) | 263 |
| K₁ swept | {1×10⁻⁶, 1×10⁻⁵, 1×10⁻⁴} |
| K₂ swept | 66 values in [0.01, 1] |
| K₃ swept | {0.005, 0.010, 0.015, 0.020, 0.025} |
| K₄, K₅, K₆ fixed | K₄=1, K₅=0.01, K₆=5 |

Top 3 results ranked by band distance (`χ²_band`):

| Rank | K₁ | K₂ | K₃ | χ²_band | Selected |
|---|---|---|---|---|---|
| 0 | 1×10⁻⁵ | 0.205 | 0.010 | 0.004035 | |
| **1** | **1×10⁻⁴** | **0.13** | **0.005** | **0.004206** | **←** |
| 2 | 1×10⁻⁴ | 0.43 | 0.010 | 0.004253 | |

**Selected result** (rank 1, `ind_chosen = 1`):

| Constraint | Weight K | Component value C | K·C |
|---|---|---|---|
| Band distance | — | 0.004206 | — |
| K₁ (DFT distance) | 1×10⁻⁴ | 3.988 | 3.99×10⁻⁴ |
| K₂ (M orbital content) | 0.13 | 0.00260 | 3.38×10⁻⁴ |
| K₃ (Γ/K occupation) | 0.005 | 0.0160 | 8.0×10⁻⁵ |
| K₄ (CBM at K) | 1 | 0 (at K) | 0 |
| K₅ (band gap) | 0.01 | ~0 (matches DFT) | ~0 |

Rank 0 has slightly better band distance (0.004035) but was rejected because for WSe₂ the `ind_chosen = 1` convention avoids results where parameters may have saturated their bounds.

#### WS₂ (`Data/WS2_run1/`)

| Aspect | Value |
|---|---|
| Total runs (after filtering) | 100 |
| K₁ swept | {0, 1×10⁻⁶, 1×10⁻⁵, 1×10⁻⁴} |
| K₂ swept | {0.0625, 0.125, 0.1875, 0.25} |
| K₃ swept | {0.00781, 0.01105, 0.01563, 0.02344, 0.03125} |
| K₄, K₅, K₆ fixed | K₄=1, K₅=0.01, K₆=5 |

Top 3 results ranked by band distance:

| Rank | K₁ | K₂ | K₃ | χ²_band | Selected |
|---|---|---|---|---|---|
| **0** | **0** | **0.125** | **0.01105** | **0.000995** | **←** |
| 1 | 1×10⁻⁶ | 0.0625 | 0.00781 | 0.001901 | |
| 2 | 0 | 0.0625 | 0.00781 | 0.001932 | |

**Selected result** (rank 0, `ind_chosen = 0`):

| Constraint | Weight K | Component value C | K·C |
|---|---|---|---|
| Band distance | — | 0.000995 | — |
| K₁ (DFT distance) | 0 | 5.326 | 0 |
| K₂ (M orbital content) | 0.125 | 5.6×10⁻⁵ | 7.0×10⁻⁶ |
| K₃ (Γ/K occupation) | 0.01105 | 0.0557 | 6.15×10⁻⁴ |
| K₄ (CBM at K) | 1 | 0 (at K) | 0 |
| K₅ (band gap) | 0.01 | ~0 (matches DFT) | ~0 |

### Initial point control

The optimizer always starts from the DFT-derived parameters as the initial point (`x0`). This is the standard v3.0 approach: Nelder-Mead is a local optimizer, so starting from the physically motivated DFT values ensures convergence to a physically meaningful minimum.

### HPC workflow

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

### Run management

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

### Programmatic usage

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

### Output

Fitted parameters from grid searches are saved as `.npz` files in `Data/<TMD>_<id>/`. Each file contains the optimized parameters, chi-squared values, individual constraint values, and the computed band energies. Symmetrized ARPES data is cached as `Data/sym_{TMD}.npz`.

### Lattice constants

| Material | a (Å) |
|---|---|
| WS₂ | 3.18 |
| WSe₂ | 3.32 |

## Bilayer Moiré Bands

### Overview

The bilayer stage fits interlayer hopping parameters between WSe₂ and WS₂ layers to reproduce the top valence bands from bilayer ARPES data along the Γ–K path. The fit uses a 44×44 Hamiltonian (22 orbitals per layer × 2 layers, spin-degenerate) with `n_shells=0` (no moiré supercell expansion, i.e. a single mini-Brillouin zone).

### Interlayer coupling form

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

### Minimization

The fit minimizes a chi-squared objective comparing computed band energies to symmetrized bilayer ARPES data along the Γ–K path:

```
χ² = (1/N) Σ_{b=1}^{3} Σ_{i} w(k_i) · [E_TB[b, k_i] - E_ARPES[b, k_i]]²
```

where:
- **3 bands**: the top 3 valence bands (indices 27, 26, 25 out of 44)
- **Gamma weighting**: `w(k) = 1 + γ_weight · exp(-k² / (2σ²))` gives higher weight to points near Γ. Default: `γ_weight = 5.0`, `σ = 0.15 Å⁻¹`
- **Energy offset**: the S11 sample offset of −0.47 eV is applied to all computed energies

The optimization uses `scipy.optimize.minimize` with the Nelder-Mead method, launched from multiple random starting points (default: 10) within bounds of [−5, 5] eV for all four parameters. The best result across all starts is selected.

### Quick start

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

### Export script

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

### Output

Fitted interlayer parameters are saved to:

| File | Content |
|---|---|
| `Inputs/bilayer_fitting/interlayer_params.npy` | NumPy array `[w1p, w1d, w2p, w2d]` |
| `Inputs/bilayer_fitting/interlayer_params_metadata.json` | Metadata: parameter values, chi², nfev, success flag, timestamp |
| `Figures/bilayer_fit.png` | Final fit plot with parameter values and ARPES comparison |

## Bilayer Moiré Potential (EDC Analysis)

### EDC Overview

With monolayer parameters and interlayer couplings fixed (Steps 1–2), this stage sweeps the moiré potential parameters to match experimental Energy Distribution Curve (EDC) peak positions at Γ and K. The workflow runs in two sequential stages:

1. **Gamma-point sweep** — 4D grid over Vg, φG, w1p, w1d (w2p/w2d fixed to Step 2 values; Vk and φK fixed). Fits 4 Lorentzians to the EDC intensity profile and saves the top 3 (TVB main, TVB side, WS2 LVB).
2. **K-point sweep** — 2D grid over Vk, φK with all other parameters fixed to the Gamma best fit. Fits 2 Lorentzians (TVB + moiré side band).

### Moiré Supercell Hamiltonian

The EDC is computed from the supercell Hamiltonian built by `MoireHamiltonian` (see [`hamiltonian.py`](tmdmoire/bilayer/hamiltonian.py)). For `n_shells=2`, the supercell contains **19 mini-BZ cells** and the Hamiltonian is **836×836** (44 orbitals per cell × 2 layers × 19 cells).

**Basis ordering** (per cell):
- **Indices 0–21**: WSe₂ layer (11 orbitals × 2 spins)
- **Indices 22–43**: WS₂ layer (11 orbitals × 2 spins)

The Hamiltonian has three components:

**1. Intralayer blocks** — Each mini-BZ cell has two diagonal 22×22 monolayer Hamiltonians (WSe₂ and WS₂) shifted by their respective lattice constants:
```
K_n = k_point + G_moiré[1] · lu[n][0] + G_moiré[2] · lu[n][1]
```
where `lu` is the lookup table mapping cell index `n` to reciprocal lattice coordinates.

**2. Interlayer coupling** — Connects WSe₂ to WS₂ *within the same cell* (`n_shells=0` blocking). Only two orbitals per spin participate:
```
t_pz(k) = w1p + w2p · Σ_{j=1}^{6} exp(i k · δ_j)    (p_z^e, orbital 8)
t_dz(k) = w1d + w2d · Σ_{j=1}^{6} exp(i k · δ_j)    (d_z²,  orbital 5)
```
where `δ_j` are the six nearest-neighbor vectors within a monolayer. The same couplings apply to spin-down (indices +11). At Γ (k=0), `Σ exp(i k·δ_j) = 6` so the total coupling is `w1 + 6·w2`. Away from Γ, phase interference reduces the coupling.

**3. Moiré potential** — A 22×22 diagonal matrix couples *neighboring* mini-BZ cells through the six moiré reciprocal lattice directions. The complex amplitude is:
```
out-of-plane orbitals (dxz, dyz, pz^o, dz2, pz^e):  V_G · exp(i·ψ_G)
in-plane  orbitals  (px^o, py^o, dxy, dx2-y2, px^e, py^e):  V_K · exp(i·ψ_K)
```
The matrix alternates between `V_moire` and `V_moire*` depending on neighbor parity to ensure Hermiticity. The potential is applied identically to both layers.

**Diagonalization** uses `scipy.linalg.eigh` (Hermitian solver) with `overwrite_a=True` for speed. Per-point cost is ~0.6 s, dominated by the 836×836 diagonalization.

### EDC Intensity Computation

For a given set of parameters, the EDC intensity profile is built in four steps:

**1. Band window selection** — Only a subset of eigenstates is used for the EDC, centered around the top valence band:

| BZ point | TVB index | Window bounds | Description |
|---|---|---|---|
| **Γ** | `28·N − 1` (=531 for N=19) | `index_lvb − 2N + 1` to `index_tvb` (≈ 455–531) | Covers TVB + lower valence bands |
| **K** | `28·N − 1` | `index_tvb − N + argmax(weights[...])` to `index_tvb` | Adaptive: finds the peak weight within N cells below TVB |

where `index_lvb = 26·N − 1` and `N = n_cells`.

**2. Spectral weights (central-cell projection)** — For each eigenstate `n`, the weight is the sum of squared eigenvector amplitudes over the **central cell only** in both layers:
```python
weights[n] = Σ_{i=0}^{21} |evecs[i, n]|² + Σ_{i=22N}^{22(N+1)−1} |evecs[i, n]|²
    #        ──────────── WSe₂ central cell ──   ───────── WS₂ central cell ───
```
This projects onto the local density of states at the central moiré site — the observable in ARPES EDC measurements.

**3. Lorentzian broadening** — Each eigenvalue is convolved with a Lorentzian kernel (no k-spreading, single k-point):
```
I(E) = Σ_{i} w_i · (σ/π) / [(E - E_i)² + σ²]
```
where `σ = 0.03 eV` (fixed). The energy grid spacing is **0.005 eV** (5 meV) with ±50% padding around the min/max eigenvalues in the window.

**4. Peak seeding** — Before fitting, seed centers are determined from the intensity profile:

- **At Γ** (`find_peak_seeds_gamma`): Uses `scipy.signal.find_peaks` with `height = max·0.005` and `distance = 0.01 eV`. Peaks are classified into TVB region (E > −1.5 eV) and LVB region (E < −1.5 eV). For each region, the main peak (highest intensity) and side peak (next highest with separation > 0.01 eV) are identified. Returns 4 seeds sorted by energy descending: `[TVB_main, TVB_side, LVB_main, LVB_side]`.

- **At K**: Simply takes the top 2 eigenstates by weight, with minimum separation > 0.01 eV.

**5. Lorentzian fitting** — Uses `lmfit.Model` with a multi-Lorentzian model:

| | Gamma | K |
|---|---|---|
| **Model** | `_four_lorentzian(x, a1,c1,g1, a2,c2,g2, a3,c3,g3, a4,c4,g4)` | `_two_lorentzian(x, a1,c1,g1, a2,c2,g2)` |
| **Lorentzian form** | `amplitude · γ² / ((x − center)² + γ²)` | Same |
| **Amplitude bounds** | `a_i ≥ 0` | `a_i ≥ 0` |
| **Width bounds** | `γ_i ∈ [1e−4, 0.2] eV` | `γ_i ∈ [1e−4, 0.2] eV` |
| **Center bounds** | `seed ± 0.05 eV` | `seed ± 0.05 eV` |
| **Saved peaks** | Top 3 (c1–c3, a1–a3, g1–g3) | All 2 (c1,c2, a1,a2, g1,g2) |

The 4th Gamma peak (LVB side) is fitted to improve overall fit quality but is **not saved**, since only 3 peaks are compared against experiment.

Failed fits (where `lmfit` does not converge) leave `NaN` values in the HDF5 output.

### Gamma-Point Sweep (`edc_grid_gamma.py`)

**Script**: [`scripts/edc_grid_gamma.py`](scripts/edc_grid_gamma.py)

Sweeps a 4D parameter grid at the Γ-point. The diagonalization is computed once per grid point and reused for all Lorentzian fits at that point.

**Parameters varied** (4D grid):

| Parameter | Description | Range | Step | # values |
|---|---|---|---|---|
| **Vg** | Out-of-plane moiré potential | 1–20 meV | 0.5 meV | 39 |
| **φG** | Out-of-plane moiré phase | 160°–180° | 1° | 21 |
| **w1p** | p_z interlayer coupling (k-independent) | −1.500 to −1.300 eV | 5 meV | 41 |
| **w1d** | d_z² interlayer coupling (k-independent) | 0.300 to 0.500 eV | 5 meV | 41 |

**Total**: `39 × 21 × 41 × 41 ≈ 1.38M` grid points.

**Parameters fixed to Step 2 values**:

| Parameter | Fixed value | Rationale |
|---|---|---|
| **Vk** | 7.7 meV | In-plane potential has minimal effect at Γ |
| **φK** | 106° | Same |
| **w2p** | from `interlayer_params.npy` | Controls k-dependence of p_z coupling; pinned by Step 2 band dispersion fit |
| **w2d** | from `interlayer_params.npy` | Controls k-dependence of d_z² coupling; pinned by Step 2 band dispersion fit |

> **Why w1p/w1d are re-swept here**: The Step 2 fit used `n_shells=0` (single BZ cell, no moiré potential). With a moiré potential present, the momentum-averaged interlayer coupling can shift. The k-independent terms w1p/w1d are the most sensitive to this, while w2p/w2d are already well-constrained by the dispersion shape fit in Step 2.

**Experimental targets** (`EDC_G_POSITIONS["S11"]`):

| Peak | Energy (eV) | Label |
|---|---|---|
| Peak 1 | −1.1599 | TVB main |
| Peak 2 | −1.2531 | TVB side band |
| Peak 3 | −1.8200 | WS₂ LVB |

**HDF5 output columns** (per dataset, float64, NaN-filled for failures):

| Column | Description |
|---|---|
| `Vg`, `phiG`, `w1p`, `w1d` | Input parameters (grid coordinates) |
| `c1`, `a1`, `g1` | Lorentzian 1: center (eV), amplitude, width (eV) |
| `c2`, `a2`, `g2` | Lorentzian 2: center, amplitude, width |
| `c3`, `a3`, `g3` | Lorentzian 3: center, amplitude, width |
| `redchi` | Reduced chi-squared of the 4-Lorentzian fit |

### K-Point Sweep (`edc_grid_k.py`)

**Script**: [`scripts/edc_grid_k.py`](scripts/edc_grid_k.py)

After the Gamma best fit is known, sweeps the in-plane moiré potential at the K-point. The k-point is set to the monolayer K-point: `k_K = [4π/(3·a_WSe2), 0]`.

**Parameters varied** (2D grid):

| Parameter | Description | Range | Step | # values |
|---|---|---|---|---|
| **Vk** | In-plane moiré potential | 1–40 meV | 2 meV | 20 |
| **φK** | In-plane moiré phase | 0°–359° | 10° | 36 |

**Total**: `20 × 36 = 720` grid points.

**Parameters fixed to Gamma best fit**:

| Parameter | Typical value |
|---|---|
| Vg | 15 meV |
| φG | 180° |
| w1p | −1.3232 eV |
| w1d | 0.5012 eV |
| w2p | −0.1774 eV |
| w2d | 0.0295 eV |

**Experimental targets** (`EDC_K_POSITIONS["S11"]`):

| Peak | Energy (eV) | Label |
|---|---|---|
| Peak 1 | −0.8990 | TVB |
| Peak 2 | −1.0696 | Moiré band |

**HDF5 output columns**:

| Column | Description |
|---|---|
| `Vk`, `phiK` | Input parameters (grid coordinates) |
| `c1`, `a1`, `g1` | Lorentzian 1: center (eV), amplitude, width |
| `c2`, `a2`, `g2` | Lorentzian 2: center, amplitude, width |
| `redchi` | Reduced chi-squared of the 2-Lorentzian fit |

### Grid Configuration

Each BZ point has its own config file in `Inputs/bilayer_fitting/`:

| File | Content |
|---|---|
| [`grid_config_gamma.json`](Inputs/bilayer_fitting/grid_config_gamma.json) | Vg, φG, w1p, w1d grid; fixed Vk=7.7 meV, φK=106° |
| [`grid_config_k.json`](Inputs/bilayer_fitting/grid_config_k.json) | Vk, φK grid; fixed Vg=15 meV, φG=180°, all interlayer params |

**Grid spec format**:

```jsonc
// Absolute range:
{"min_ev": -1.500, "max_ev": -1.300, "step_ev": 0.005}

// Relative range (± around fitted Step 2 value):
{"range_ev": 0.010, "step_ev": 0.002}

// Angular:
{"min_deg": 160, "max_deg": 180, "step_deg": 1}
```

The Gamma config uses absolute ranges for interlayer params; the K config lists relative ranges for reference but the K script uses fixed values from the `"fixed"` block.

### Chunking and Combining

The full grid is divided into chunks for HPC parallelization:

**`edc_grid_gamma.py`**:
```bash
python scripts/edc_grid_gamma.py --chunk <id>/<total> --id <run_id>
```
- The total grid is computed via `itertools.product`, then sliced with `itertools.islice`
- Chunk boundaries: `chunk_size = total // n_chunks` with remainders distributed to early chunks
- Each chunk writes to `Data/edc_gamma_<id>/chunk_<id>_<total>.h5`
- Metadata (grid sizes, fixed params, config snapshot) is saved only by the first chunk

**`combine_edc_chunks.py`**:
```bash
python scripts/combine_edc_chunks.py --bz-point gamma --id <run_id>
```
- Scans for all `chunk_*.h5` files in the run directory
- Reads column names from the first file
- Concatenates all datasets via `np.concatenate`
- Skips corrupt files with a warning
- Saves to `Data/edc_gamma_<id>/combined.h5`

### Gamma Analysis (`analyze_edc_gamma.py`)

**Script**: [`scripts/analyze_edc_gamma.py`](scripts/analyze_edc_gamma.py)

Loads `combined.h5` and computes two complementary distance metrics:

**L1 distance** (absolute position error):
```
dist_L1 = |c1 − E_TVB| + |c2 − E_side| + |c3 − E_LVB|
```

**L2 distance** (peak separation error, independent of absolute energy shift):
```
dist_L2 = | |c1−c2| − |E_TVB−E_side| | + | |c1−c3| − |E_TVB−E_LVB| |
```
where the experimental separations are:
- `|E_TVB − E_side| = |−1.1599 − (−1.2531)| = 0.0932 eV`
- `|E_TVB − E_LVB|  = |−1.1599 − (−1.82)|    = 0.6601 eV`

**Three-stage filtering** (applied in order):

| Stage | Criterion | Default | What it removes |
|---|---|---|---|
| 1. L1 cutoff | `dist_L1 > --l1-cutoff` | 26 meV | Points where peaks are too far from experimental positions |
| 2. L2 cutoff | `dist_L2 > --cutoff` | 10 meV | Points where peak separations are wrong |
| 3. Ratio cutoff | `a2/a1 < --ratio-cutoff` | 0.1 | Points where the TVB side peak is too weak |

The **L2 distance** is the primary ranking metric used to find the global minimum.

**Aggregation and heatmaps**: For each (Vg, φG) cell, the minimum L2 distance across all (w1p, w1d) values is taken. Two heatmap pairs are produced:

1. **`analysis.png`** — L1 and L2 distance over φG (x) × Vg (y), with:
   - Red dashed reference lines at φG = 60°, 180°, 300°
   - Horizontal guide lines every 4 meV
   - φG range clipped to [160°, 200°]

2. **`analysis_wpw_d.png`** — L1 and L2 distance over w1p (x) × w1d (y), minimizing over Vg and φG

**Selection mode** (`--vg <V> --phig <deg>`):
- Highlights the chosen (Vg, φG) cell with a cyan diamond on a zoomed heatmap (`analysis_zoom.png`)
- Prints the best interlayer params (w1p, w1d) and peak positions at that cell
- Produces an EDC intensity profile plot with the 3-Lorentzian fit overlaid (`edc_profile_*.png`)
- Produces a full-range EDC profile with the 4-Lorentzian fit and experimental position markers (`edc_profile_4L_*.png`)
- Exports all parameters as JSON (`Vg<X>meV_phiG<Y>deg.json`)

**Boundary exclusion** (`--exclude-boundary`):
- Detects cells where w1p or w1d hit their grid boundaries
- Masks these out to ensure the best-fit parameters are within the sweep range

### K Analysis (`analyze_edc_k.py`)

**Not yet implemented.** Planned features:
- Load `combined.h5` from `edc_grid_k_<id>/`
- Compute L1/L2 distance from `EDC_K_POSITIONS` (`[−0.8990, −1.0696]` eV)
- 2D heatmap: φK (x) × Vk (y), color = distance
- Band gap heatmap

### EDC Quick Start

```bash
# Gamma sweep: single chunk (for testing)
python scripts/edc_grid_gamma.py --chunk 0/1000000 --id test

# Gamma sweep: combine chunks
python scripts/combine_edc_chunks.py --bz-point gamma --id test

# Gamma sweep: analyze results (global best)
python scripts/analyze_edc_gamma.py --id test

# Gamma sweep: select specific (Vg, phiG) cell
python scripts/analyze_edc_gamma.py --id test --vg 0.012 --phig 177

# With custom distance cutoff and ratio cutoff
python scripts/analyze_edc_gamma.py --id test --cutoff 0.030 --ratio-cutoff 0.15

# K sweep (after Gamma best fit is known)
python scripts/edc_grid_k.py --chunk 0/10000 --id test_k

# K sweep: combine and analyze
python scripts/combine_edc_chunks.py --bz-point k --id test_k
python scripts/analyze_edc_k.py --id test_k
```

### EDC HPC Workflow

```bash
# Submit Gamma sweep (default run ID, 128 tasks)
./HPC/edc_gamma_job.sh

# Submit with run ID (default 128 tasks)
./HPC/edc_gamma_job.sh 001

# Submit with run ID and custom task count
./HPC/edc_gamma_job.sh 001 256

# Submit K sweep
./HPC/edc_k_job.sh 001 128
```

### EDC Run Management

Each run is stored in a self-contained directory:

```
Data/
  edc_gamma_run_001/
    metadata.json                           ← run info + grid config snapshot + fitted interlayer params
    interlayer_params.npy                   ← snapshot of fitted interlayer params from Step 2
    chunk_0_128.h5
    chunk_1_128.h5
    ...
    combined.h5                             ← all chunks concatenated
    analysis.png                            ← L1 + L2 heatmaps over (Vg, φG)
    analysis_wpw_d.png                      ← L1 + L2 heatmaps over (w1p, w1d)
    analysis_zoom.png                       ← zoomed heatmap with selection marker (if --vg/--phig)
    edc_profile_Vg<N>meV_phiG<N>deg.png     ← EDC intensity + 3-Lorentzian fit
    edc_profile_4L_Vg<N>meV_phiG<N>deg.png  ← full-range EDC + 4-Lorentzian fit + exp markers
    Vg<N>meV_phiG<N>deg.json                ← exported best-fit parameters for that (Vg, φG) cell
```

### EDC Output Format

Each `.h5` chunk file contains float64 columns pre-allocated with `fillvalue=np.nan`. Failed fits leave NaN.

**Gamma HDF5 columns** (14 columns):

| Column | Description |
|---|---|
| `Vg`, `phiG`, `w1p`, `w1d` | Input parameters (grid coordinates) |
| `c1`, `a1`, `g1` | Lorentzian 1: center (eV), amplitude, width (eV) |
| `c2`, `a2`, `g2` | Lorentzian 2: center, amplitude, width |
| `c3`, `a3`, `g3` | Lorentzian 3: center, amplitude, width |
| `redchi` | Reduced chi-squared |

**K HDF5 columns** (9 columns):

| Column | Description |
|---|---|
| `Vk`, `phiK` | Input parameters (grid coordinates) |
| `c1`, `a1`, `g1` | Lorentzian 1: center (eV), amplitude, width (eV) |
| `c2`, `a2`, `g2` | Lorentzian 2: center, amplitude, width (eV) |
| `redchi` | Reduced chi-squared |

## Bilayer Moiré Band Plotting

### Overview

The `plot_moire_bands.py` script produces ARPES-like intensity heatmaps of the full moiré superlattice band structure. It diagonalizes the (44·N)×(44·N) supercell Hamiltonian along the Γ→K and K→M paths, mirrors them to produce the full K′→Γ→K and K→M→K′ cuts, computes orbital weights from eigenvectors, and spreads intensity using Gaussian or Lorentzian kernels.

### Input parameters

All parameters are loaded from `Inputs/plot_bilayer/`:

| File | Content |
|---|---|
| `tb_WSe2.npy` | 43 monolayer TB parameters for WSe₂ |
| `tb_WS2.npy` | 43 monolayer TB parameters for WS₂ |
| `interlayer_G.npy` | Dict: w1p, w1d, w2p, w2d (eV), Vg (eV), phiG (rad) |
| `interlayer_K.npy` | Dict: Vk (eV), phiK (rad) |

### Band selection

The script computes the full Hamiltonian eigenvalues and eigenvectors, then slices to the top valence bands. For each mini-BZ cell there are 44 bands; the top valence band (TVB) is at index 28 (0-based: 27). The script keeps bands 18–27 per cell (10 bands below and including the TVB), giving `10 × n_cells` bands total.

### Output figures

Three figures are generated in the cache directory:

| Figure | Description |
|---|---|
| `moire_bands_simulated.png` | Simulated intensity for both K′→Γ→K and K→M→K′ cuts |
| `arpes_data.png` | Experimental ARPES intensity for both cuts |
| `moire_bands_half_arpes.png` | Half ARPES / half simulated: ARPES on k < 0 side, simulated on k > 0 side |

All figures use a Greys colormap, centered momentum axes (k = 0 at Γ or M), and fixed xlim: ±1.4 Å⁻¹ for K′→Γ→K, ±1.2 Å⁻¹ for K→M→K′.

### CLI options

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

### Energy shading

A linear multiplicative gradient is applied along the energy axis to mimic ARPES intensity falloff at deeper binding energies. The shading factor starts at **0.1 at E_min** and increases linearly to **`shade_e_factor` at E_max**. With the default `--shade-e-factor 3.0`, bands near the Fermi level are amplified 30× relative to the deepest bands.

### Two-level caching

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

## Real-Space LDOS

### LDOS Overview

The `compute_ldos.py` script computes the **local density of states** in real space along a 1D cut through the moiré unit cell. The LDOS reveals which stacking sites (W/W, Se/W, W/S) host the flat moiré bands at a given energy, and how strongly the electronic states localize at different positions in the moiré pattern.

### Physical formula

The LDOS is the **k-integrated spectral function** evaluated in real space:

$$LDOS(\mathbf{r}, E) = \frac{1}{N_k} \sum_{\mathbf{k}} \sum_{n} \left|\psi_{n\mathbf{k}}(\mathbf{r})\right|^2 \cdot \frac{\eta}{\pi\Big((E - E_{n\mathbf{k}})^2 + \eta^2\Big)}$$

where:
- $\mathbf{k}$ — k-points spanning the mini-Brillouin zone (uniform $k_\text{pts} \times k_\text{pts}$ grid)
- $n$ — eigenstate index ($836$ states for $n_\text{shells}=2$, $19$ cells)
- $E_{n\mathbf{k}}$ — eigenvalue of eigenstate $n$ at k-point $\mathbf{k}$
- $\eta$ — Lorentzian broadening (energy spreading, in eV)
- $N_k$ — total number of k-points

The spatial probability density $|\psi_{n\mathbf{k}}(\mathbf{r})|^2$ is summed over all 44 orbitals (both layers):

$$\left|\psi_{n\mathbf{k}}(\mathbf{r})\right|^2 = \sum_{\alpha=1}^{44} \left|\psi_{n\mathbf{k}}^\alpha(\mathbf{r})\right|^2$$

### Wavefunction reconstruction

Each orbital's real-space wavefunction is reconstructed from the supercell eigenvectors $c_{\alpha n}^{i_c}(\mathbf{k})$ by Fourier transform over the moiré reciprocal lattice:

$$\psi_{n\mathbf{k}}^\alpha(\mathbf{r}) = \sum_{i_c=1}^{n_\text{cells}} e^{i(\mathbf{k} + \mathbf{G}_{i_c})\cdot\mathbf{r}} \cdot c_{\alpha n}^{i_c}(\mathbf{k})$$

where:
- $\alpha$ — orbital index (0–43: WSe₂ indices 0–21, WS₂ indices 22–43)
- $i_c$ — mini-BZ cell index (0 to $n_\text{cells}-1$)
- $\mathbf{G}_{i_c}$ — reciprocal lattice vector offset of cell $i_c$:
  $$\mathbf{G}_{i_c} = \mathbf{G}_1 \cdot l_{i_c}[0] + \mathbf{G}_2 \cdot l_{i_c}[1]$$
  where $\mathbf{G}_1, \mathbf{G}_2$ are the two basis moiré reciprocal vectors and $l_{i_c}$ is the lookup table entry for cell $i_c$

The implementation is vectorized: all $n_\text{cells}$ and 44 orbitals are handled simultaneously via a precomputed index array of shape $(44, n_\text{cells})$ that maps each (orbital, cell) pair to its position in the $44 \cdot n_\text{cells}$ eigenvector.

### Real-space path

The spatial cut follows the direction $\mathbf{a}_1 + \mathbf{a}_2$ (diagonal of the moiré unit cell), starting at one W/W high-symmetry point and passing through the other stacking configurations:

```
W/W ──→ Se/W ──→ W/S ──→ W/W
```

| Position | r | Stacking | Description |
|---|---|---|---|
| W/W | 0 | AA-like | W atoms of both layers aligned |
| Se/W | $\|\mathbf{a}_1+\mathbf{a}_2\| / 3$ | AB/BA | Se atom over W site |
| W/S | $2\|\mathbf{a}_1+\mathbf{a}_2\| / 3$ | AB/BA | W atom over S site |
| W/W | $\|\mathbf{a}_1+\mathbf{a}_2\|$ | AA-like | Back to start |

The total length of this path is $|\mathbf{a}_1 + \mathbf{a}_2| = a_M \cdot \sqrt{3}$, where $a_M$ is the moiré lattice constant (~50 Å for S11 at 2.8°).

### Real-space sharpness

The spatial resolution of the LDOS is controlled by **`--n-shells`**, not `--k-pts`. Each shell adds higher-order moiré reciprocal lattice vectors $\mathbf{G}_{i_c}$ to the Fourier sum, analogous to adding higher harmonics:

| n_shells | Cells | G vectors | Fourier resolution |
|---|---|---|---|
| 1 | 7 | 7 | Coarse |
| 2 | 19 | 19 | Good (sweet spot) |
| 3 | 37 | 37 | Marginal gain |

Increasing `--k-pts` refines the **k-integral convergence** but does not add new Fourier components — once the integral is converged (≥10 k-points is usually sufficient), further increases have negligible effect on spatial sharpness.

### LDOS CLI options

All monolayer and interlayer parameters are loaded from `Inputs/plot_bilayer/`.

```bash
python scripts/compute_ldos.py [options]
```

| Option | Default | Description |
|---|---|---|
| `--k-pts` | 12 | Number of k-points per mini-BZ side ($k_\text{pts}^2$ total) |
| `--r-pts` | 300 | Number of real-space grid points along the cut |
| `--n-shells` | 2 | Number of moiré shells ($n_\text{cells} = 1 + 3n(n+1)$) |
| `--e-min` | −1.0 | Minimum energy (eV) |
| `--e-max` | 0.0 | Maximum energy (eV) |
| `--delta-e` | 0.005 | Energy grid spacing (eV) |
| `--eta` | 0.01 | Lorentzian broadening width (eV) |
| `--r-extra` | 0.0 | Extra fraction of the moiré period to extend past the W/W point (e.g. 0.166 for padding) |
| `--center` | G | BZ point to center the k-grid on: `G` (Γ) or `K` |
| `--sample` | S11 | Sample name for energy offset and twist angle |
| `--theta` | — | Twist angle in degrees (overrides sample) |
| `--Vg` | — | Override moiré potential at Γ (eV) |
| `--Vk` | — | Override moiré potential at K (eV) |
| `--phiG` | — | Override moiré phase at Γ (degrees) |
| `--phiK` | — | Override moiré phase at K (degrees) |
| `--no-cache` | — | Ignore cache and recompute |

### LDOS Caching

The script uses a two-level cache under `Data/ldos/`:

```
Data/ldos/
  diag_<mono_hash>_<w1p>_<w1d>_<w2p>_<w2d>_<Vg>_<phiG>_<Vk>_<phiK>_t<theta>_n<n_shells>_k<k_pts>_<center>/
    metadata.json                      ← run parameters snapshot
    diag.npz                           ← evals, evecs, k_flat (diagonalization)
    ldos_<r_pts>_<e_min>_<e_max>_<delta_e>_<eta>_re<r_extra>/
      ldos.npz                         ← LDOS array [n_r, n_e], r_list, e_list, rL
      ldos_meta.json                   ← LDOS computation parameters
      ldos.png                         ← pcolormesh plot
    ldos_<other_params>/
      ...
  diag_<other_hamiltonian_params>/
    ...
```

**Diagonalization cache** (expensive, minutes): hashed by monolayer param hash + interlayer + moiré + theta + n_shells + k_pts + center. Changing any of these triggers a new diagonalization.

**LDOS cache** (cheap, seconds): hashed by r_pts + energy range + broadening + r_extra. Exploring different energy windows, resolutions, or broadening values reuses the cached diagonalization.

### LDOS Output

**Data file** (`ldos.npz`):
- `ldos`: array of shape `(n_r, n_e)` — LDOS values
- `r_list`: array of shape `(n_r, 2)` — real-space positions (Å)
- `e_list`: array of shape `(n_e,)` — energy grid (eV)
- `rL`: float — moiré period $|\mathbf{a}_1 + \mathbf{a}_2|$ (Å)

**Plot** (`ldos.png`):
- X-axis: Energy (eV)
- Y-axis: Position along $\mathbf{a}_1 + \mathbf{a}_2$, with stacking site labels (W/W, Se/W, W/S, W/W)
- Colormap: `hot` (yellow–red–black)
- Colorbar: "low" / "high" labels, no ticks
- Y-axis inverted (r = 0 at top)

### LDOS Quick start

```bash
# Basic run at Gamma
python scripts/compute_ldos.py --k-pts 10 --r-pts 80 --n-shells 2 \
    --e-min -1.28 --e-max -1.11 --delta-e 0.002 --eta 0.005 --r-extra 0.166

# With different moiré phase
python scripts/compute_ldos.py --k-pts 10 --r-pts 80 --n-shells 2 \
    --e-min -1.28 --e-max -1.11 --delta-e 0.002 --eta 0.005 \
    --phiG 170 --r-extra 0.166

# K-centered LDOS
python scripts/compute_ldos.py --k-pts 10 --r-pts 80 --n-shells 2 \
    --e-min -1.28 --e-max -1.11 --delta-e 0.002 --center K

# Force recompute (ignore cache)
python scripts/compute_ldos.py --k-pts 10 --r-pts 80 --n-shells 2 --no-cache
```

## References

*(Documentation forthcoming)*
