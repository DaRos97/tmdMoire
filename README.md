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
- [References](#references)

## Monolayer Fitting

### Overview

The monolayer stage fits a 22×22 tight-binding Hamiltonian (11 orbitals × 2 spins) to reproduce ARPES-measured band dispersions along high-symmetry paths K′–Γ–K and K–M–K′. The fit optimizes 43 parameters against experimental data using dual annealing (global search) followed by Nelder-Mead (local refinement), with multiple physical constraints.

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
χ² = χ²_band + K₁·C₁ + K₂·C₂ + K₃·C₃ + K₄·C₄ + K₅·C₅
```

where `χ²_band` is the band distance term and `C₁`–`C₅` are the five physical constraints. All constraint terms are normalized to O(0–1) so that the weights `K₁`–`K₅` directly encode physical importance.

#### Band distance

**What it does**: Measures how well the TB band energies match the experimental ARPES data across all 6 bands and all k-points.

**Implementation**: For each band `b` and k-point `i`, compute the squared residual `(E_TB - E_ARPES)²`. Sum over all valid (non-NaN) data points across all bands, then divide by the total number of valid points. Four special k-points (Γ, top of band 1, minimum of band 2, and M) receive an additional weight multiplier `K₆`:

```python
χ²_band = Σ_b Σ_i [w_i · (E_TB[b,i] - E_ARPES[b,i])²] / N_total_valid
```

where `w_i = K₆` at the four special points and `w_i = 1` elsewhere.

#### K₁ — parameter distance from DFT

**What it does**: Penalizes parameters that deviate far from their DFT-derived initial values, preventing unphysical results.

**Implementation**: Mean absolute relative deviation of all parameters (except the global offset) from their DFT values. Excludes the offset (index 40) since it is a fitting artifact, not a physical parameter:

```python
C₁ = [ Σ_{i≠40} |p_i - p_DFT,i| / |p_DFT,i| ] / (N_params - 1)
```

Typical range: 0 (at DFT) to ~2 (large deviations).

#### K₂ — orbital band content at M

**What it does**: Minimizes the weight of interlayer-coupling orbitals (d_z², p_z^e) in the top valence bands at the M point. These are the only orbitals that participate in interlayer hopping in the bilayer model. Since ARPES shows no noticeable change in the band structure at M between monolayer and bilayer, the interlayer-coupling orbital character at M should remain small — the fit penalizes any mixing of d_z² and p_z^e into the valence bands at M.

**Implementation**: Sum the squared eigenvector components `|c|²` for the 4 interlayer-coupling orbitals (d_z² and p_z^e, both spin blocks; IND_ILC) across the top valence bands (4 for WSe₂, 2 for WS₂) at the M point, then normalize by the number of terms:

```python
C₂ = Σ_{orb ∈ ILC} Σ_{band ∈ TVB} |⟨orb|ψ_band(M)⟩|² / (|ILC| × |TVB|)
```

The p_z^o orbital is excluded because it does not enter the bilayer interlayer coupling Hamiltonian. Typical range: 0.01–0.2 (DFT values are small, ~0.05 for WSe₂, ~0.11 for WS₂).

#### K₃ — orbital occupation at Γ and K

**What it does**: Enforces the DFT-derived orbital occupations of the top valence bands at the high-symmetry points Γ and K. These occupations are well-defined from symmetry and serve as strong physical anchors.

**Implementation**: Eight absolute differences between target DFT occupations and the computed occupations:

- **At Γ** (4 terms): p_z^e and d_z² content in each of the two degenerate TVB states
- **At K** (4 terms): p₋₁^e and d₋₂ content in each of the two TVB states (p₋₁^e = (p_x^e - i·p_y^e)/√2, d₋₂ = (d_x²-y² - i·d_xy)/√2)

The sum is divided by 8 to give a mean occupation error:

```python
C₃ = [ Σ |occ_DFT - occ_TB| ] / 8
```

Typical range: 0 (perfect match) to ~0.5 (poor match).

#### K₄ — conduction band minimum at K

**What it does**: Forces the conduction band minimum (CBM) to sit at the K point, as required by the physics of TMD monolayers.

**Implementation**: Squared relative distance between the k-point where the CBM occurs and the K point magnitude:

```python
C₄ = [(|k_CBM| - |K|) / |K|]²
```

Value is 0 when the CBM is exactly at K, ~0.34 when at M, and ~1 when at Γ. This provides a smooth gradient that the optimizer can follow.

#### K₅ — band gap at K

**What it does**: Keeps the band gap at K close to the DFT-predicted value. The absolute gap size is less certain than the band dispersion shape, so this acts as a soft constraint.

**Implementation**: Relative difference between the current gap and the DFT gap at K:

```python
C₅ = |gap_DFT - gap_TB| / gap_DFT
```

Typical range: 0 (matches DFT) to ~0.5 (50% deviation).

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

# Score existing results
python scripts/run_monolayer_grid.py WSe2 --score

# Show top 20 results
python scripts/run_monolayer_grid.py WSe2 --score --top 20

# Adjust the K4 hard filter threshold (default: 0.05)
python scripts/run_monolayer_grid.py WSe2 --score --k4-threshold 0.1
```

The default grid has 3×4×4×4×4×2 = **1,536 combinations**. Each fit uses dual annealing (maxiter=100) followed by Nelder-Mead refinement (fatol=1e-3, maxiter=50).

### Initial point control

The `use_dft_x0` option in `fit_config.json` controls whether the DFT-derived parameters are used as the initial point (`x0`) for the optimizer:

- **`true`** (default): One individual in the DE population (or the starting point for dual annealing) is seeded with the DFT parameters. This biases the search toward the DFT basin.
- **`false`**: The entire population is randomly initialized within the bounds. This maximizes exploration and avoids any gravitational pull toward the DFT parameters, at the cost of potentially slower convergence.

Set `use_dft_x0: false` when you suspect the best fit lies far from the DFT starting point, or when running with `K1 = 0` (no DFT penalty).

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
    fit_WSe2_idx0.npz
    fit_WSe2_idx1.npz
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
result = fitter.run(maxiter=3000, seed=42)

print(f"Final chi²: {result['fun']}")
print(f"Optimized parameters: {result['x']}")
```

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

With monolayer parameters and interlayer couplings fixed, this stage sweeps the moiré potential parameters to match experimental Energy Distribution Curve (EDC) peak positions. The workflow runs in two stages:

1. **Gamma-point sweep** — 4D grid over Vg, φG, w1p, w1d (w2p/w2d fixed to Step 2 values; Vk and φK fixed). Fits 3 Lorentzians to the EDC intensity profile (TVB main/side + LVB main).
2. **K-point sweep** — 2D grid over Vk, φK with all other parameters fixed to the Gamma best fit. Fits 2 Lorentzians (TVB + moire side band) and computes the band gap near K.

### EDC Intensity Profile

The EDC intensity at a given k-point is computed from the supercell Hamiltonian eigenvalues and eigenvectors:

1. **Diagonalization**: Build and diagonalize the (44·N)×(44·N) supercell Hamiltonian at the target k-point (N = 19 cells for n_shells=2 → 836×836 matrix).
2. **Spectral weights**: For each eigenstate, compute the weight as the sum of |eigenvector|² over the WSe₂ block (indices 0–21) and WS₂ block (indices 22N–22(N+1)−1).
3. **Lorentzian broadening**: Convolve each eigenvalue with a Lorentzian of width `spreadE = 0.03 eV`:
   ```
   I(E) = Σ_i w_i · (spreadE/π) / [(E - E_i)² + spreadE²]
   ```
4. **Peak fitting**: Fit a sum of Lorentzians to the resulting intensity profile using `lmfit`.

### Gamma-Point Sweep

**Grid**: 4 dimensions — Vg, φG, w1p, w1d. Fixed parameters: Vk = 7.7 meV, φK = 106°. The w2p and w2d hopping parameters are also fixed to the Step 2 interlayer coupling fit, as they control the momentum dependence of the interlayer coupling; since the EDC analysis focuses on small k-variations around Gamma, their values are already well constrained by the main band dispersion fit.

**Peak structure**: 4 Lorentzians are fitted (TVB main, TVB side, LVB main, LVB side), but only the first three peak details (c1–c3, a1–a3, g1–g3) are saved. The first three peaks are the ones compared against ARPES experimental positions; the fourth peak (LVB side band) is fitted to improve the overall fit quality but is not used in the distance metric.

**Distance metric**:
```
dist = |c1 - E_TVB| + |c2 - E_side| + |c3 - E_LVB|
```
where experimental values are `EDC_G_POSITIONS = [-1.1599, -1.2531, -1.82]` eV for sample S11.

**Analysis**: 2D heatmap of minimum distance over (Vg, φG), minimizing over all interlayer parameter combinations. The global best-fit point is marked with a red star and its parameters shown in the legend. Supports a **cell-selection mode** (`--vg`/`--phig`) to drill into a specific (Vg, φG) cell: highlights the chosen cell on the heatmap with a cyan diamond marker, prints its interlayer params and peak positions to stdout, exports parameters to a JSON file, and produces a separate EDC intensity profile plot with the 3-Lorentzian fit overlaid:

### K-Point Sweep

**Grid**: 2 dimensions — Vk, φK. Fixed parameters: Vg, φG, w1p, w1d from the Gamma best fit; w2p, w2d from Step 2.

**Peak structure**: 2 Lorentzians — TVB and moire side band.

**Band gap**: Computed by diagonalizing the Hamiltonian along a 51-point path near K and taking the minimum gap between the top valence band and the next band.

### EDC Configuration

Each BZ point has its own config file in `Inputs/bilayer_fitting/`:

| File | Purpose |
|---|---|
| `grid_config_gamma.json` | Gamma sweep: interlayer ranges/steps, Vg/φG grid, fixed Vk/φK |
| `grid_config_k.json` | K sweep: Vk/φK grid, fixed Vg/φG/w1p/w1d (w2p/w2d from Step 2) |

Interlayer parameter ranges are specified as `range_ev` (± around fitted value) and `step_ev`. Moiré parameters use `min_ev`/`max_ev`/`step_ev` or `min_deg`/`max_deg`/`step_deg`.

### EDC Quick Start

```bash
# Gamma sweep: single chunk (for testing)
python scripts/edc_grid_gamma.py --chunk 0/1000000 --id test

# Gamma sweep: combine chunks
python scripts/combine_edc_chunks.py --bz-point gamma --id test

# Gamma sweep: analyze results (global best)
python scripts/analyze_edc_gamma.py --id test

# Gamma sweep: select specific (Vg, phiG) cell
#   - highlights cell on 2D heatmap (cyan diamond vs red star global best)
#   - prints interlayer params + peak positions to stdout
#   - exports params to <run_dir>/Vg<X>meV_phiG<Y>deg.json
#   - produces EDC intensity profile plot with 3-Lorentzian fit
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
    metadata.json              ← run info + grid config + fitted interlayer params
    interlayer_params.npy      ← snapshot of fitted interlayer params
    chunk_0_128.h5
    chunk_1_128.h5
    ...
    combined.h5                    ← all chunks concatenated
    analysis.png                   ← 2D distance heatmap (with selection marker if --vg/--phig used)
    analysis_params.png            ← per-parameter heatmaps
    analysis_ratio.png             ← intensity ratio heatmap
    edc_profile_Vg<N>meV_phiG<N>deg.png  ← EDC intensity + fit (selection mode)
    Vg<N>meV_phiG<N>deg.json            ← exported params (selection mode)
```

### EDC Output Format

Each `.h5` file contains 16 datasets. Note: 4 Lorentzians are fitted per EDC, but only the first 3 peak details are saved since those are the ones compared against ARPES experimental positions.

| Column | Description |
|---|---|
| `Vg`, `phiG`, `w1p`, `w1d` | Input parameters (w2p/w2d are fixed, stored in metadata) |
| `c1`–`c3` | Fitted peak centers (eV) — TVB main, TVB side, LVB main |
| `a1`–`a3` | Fitted peak amplitudes |
| `g1`–`g3` | Fitted peak widths (gamma) |
| `redchi` | Reduced chi-squared of the fit |

The K-point sweep additionally includes `gap` (band gap in eV).

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

## References

*(Documentation forthcoming)*
