# TMD heterobilayer WSe₂/WS₂

Tight-binding model of WSe₂/WS₂ heterobilayer moiré superlattices. Two-stage computational workflow:

1. **Monolayer** — Fit 43 tight-binding parameters per TMD (WSe₂, WS₂) to ARPES data
2. **Bilayer** — Build moiré supercell Hamiltonian, compute EDCs, extract moiré potential & interlayer coupling

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

**What it does**: Ensures the top valence bands at the M point have the correct interlayer-coupling orbital character (p_z^o, d_z², p_z^e). DFT predicts these orbitals should have low weight at M for the valence bands.

**Implementation**: Sum the squared eigenvector components `|c|²` for the 6 interlayer-coupling orbitals (IND_ILC) across the top valence bands (4 for WSe₂, 2 for WS₂) at the M point, then normalize by the number of terms:

```python
C₂ = Σ_{orb ∈ ILC} Σ_{band ∈ TVB} |⟨orb|ψ_band(M)⟩|² / (|ILC| × |TVB|)
```

Typical range: 0.01–0.2 (DFT values are small, ~0.05 for WSe₂, ~0.11 for WS₂).

#### K₃ — orbital occupation at Γ and K

**What it does**: Enforces the DFT-derived orbital occupations of the top valence bands at the high-symmetry points Γ and K. These occupations are well-defined from symmetry and serve as strong physical anchors.

**Implementation**: Eight absolute differences between target DFT occupations and the computed occupations:

- **At Γ** (4 terms): p_z^e and d_z² content in each of the two degenerate TVB states
- **At K** (4 terms): p₋₁^e and d₂ content in each of the two TVB states (p₋₁^e = (p_x^e - i·p_y^e)/√2, d₂ = (d_x²-y² - i·d_xy)/√2)

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

The index selects a combination of constraint weights (K₁–K₆) from the grid defined in `Inputs/grid_config.json`.

### Grid search

Instead of running individual fits, you can sweep over all combinations of constraint weights (K₁–K₆) defined in `Inputs/grid_config.json`:

```bash
# Run all combinations for WSe₂
python scripts/run_grid.py WSe2

# Run a subset (for chunking on HPC)
python scripts/run_grid.py WSe2 --start 0 --end 100

# Score existing results
python scripts/run_grid.py WSe2 --score

# Show top 20 results
python scripts/run_grid.py WSe2 --score --top 20

# Adjust the K4 hard filter threshold (default: 0.05)
python scripts/run_grid.py WSe2 --score --k4-threshold 0.1
```

The default grid has 3×4×4×4×4×2 = **1,536 combinations**. Each fit uses dual annealing (maxiter=100) followed by Nelder-Mead refinement (fatol=1e-3, maxiter=50).

### HPC workflow

For the full grid search on the HPC cluster (SGE/rademaker queue), use the scripts in `HPC/`:

```bash
# Submit 128 parallel tasks for WSe₂ (default run ID)
./HPC/job.sh WSe2

# Submit with a named run ID
./HPC/job.sh WSe2 001

# Submit for WS₂
./HPC/job.sh WS2 002
```

Each job array submission creates 128 SGE tasks (one per CPU), with ~28 fits per task. Output goes to `Scratch/grid_<material>_<run_id>_task<N>.out`.

After all tasks complete, score the results:

```bash
python scripts/run_grid.py WSe2 --score --run-id 001
```

### Run management

Each run is stored in its own subdirectory under `Data/run_<id>/`. When you start a run, `Inputs/grid_config.json` is copied into the run directory as a snapshot, making each run fully self-contained and reproducible.

```
Data/
  run_001/
    grid_config.json          ← snapshot of config used for this run
    fit_WSe2_idx0.npz
    fit_WSe2_idx1.npz
    ...
  run_002/
    grid_config.json          ← different config (e.g. finer grid)
    fit_WSe2_idx0.npz
    ...
```

**Iterative workflow:**

1. Run the initial grid search: `./HPC/job.sh WSe2 001`
2. Score results and inspect the best fits
3. Edit `Inputs/grid_config.json` to refine the grid (e.g. narrower ranges, finer spacing)
4. Run again with a new ID: `./HPC/job.sh WSe2 002`
5. Compare runs: `python scripts/run_grid.py WSe2 --score --run-id 001` and `--run-id 002`

The `--run-id` flag works with all scripts:

```bash
python scripts/run_grid.py WSe2 --start 0 --end 100 --run-id 002
python scripts/run_grid.py WSe2 --score --run-id 002 --top 20
python scripts/fit_monolayer.py WSe2 42 --run-id 002
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

Fitted parameters from grid searches are saved as `.npz` files in `Data/run_<id>/`. Each file contains the optimized parameters, chi-squared values, individual constraint values, and the computed band energies. Symmetrized ARPES data is cached as `Data/sym_{TMD}.npz`.

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
