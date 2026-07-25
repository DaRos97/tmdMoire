# Bilayer Moiré Potential (EDC Analysis)

## EDC Overview

With monolayer parameters and interlayer couplings fixed (Steps 1–2), this stage sweeps the moiré potential parameters to match experimental Energy Distribution Curve (EDC) peak positions at Γ and K. The workflow runs in two sequential stages:

1. **Gamma-point sweep** — 4D grid over Vg, φG, w1p, w1d (w2p/w2d fixed to Step 2 values; Vk and φK fixed). Fits 4 Lorentzians to the EDC intensity profile and saves the top 3 (TVB main, TVB side, WS2 LVB).
2. **K-point sweep** — 2D grid over Vk, φK with all other parameters fixed to the Gamma best fit. Fits 2 Lorentzians (TVB + moiré side band).

## Moiré Supercell Hamiltonian

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

## EDC Intensity Computation

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

## Gamma-Point Sweep (`edc_grid_gamma.py`)

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

## K-Point Sweep (`edc_grid_k.py`)

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

## Grid Configuration

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

## Chunking and Combining

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

## Gamma Analysis (`analyze_edc_gamma.py`)

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

## K Analysis (`analyze_edc_k.py`)

**Not yet implemented.** Planned features:
- Load `combined.h5` from `edc_grid_k_<id>/`
- Compute L1/L2 distance from `EDC_K_POSITIONS` (`[−0.8990, −1.0696]` eV)
- 2D heatmap: φK (x) × Vk (y), color = distance
- Band gap heatmap

## EDC Quick Start

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

## EDC HPC Workflow

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

## EDC Run Management

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

## EDC Output Format

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

