# AGENTS.md

## Project

Tight-binding model of WSe2/WS2 heterobilayer moire superlattices. Three-stage workflow:

1. **Monolayer fitting** — fit 43 TB parameters per TMD (WSe2, WS2) to ARPES band dispersion data via differential evolution + Nelder-Mead. Results saved as `.npz` files; best params exported to `Inputs/bilayer_fitting/tb_{TMD}.npy` via `run_grid.py --score --export`.
2. **Bilayer interlayer coupling** — fit 4 interlayer hopping parameters (w1p, w2p, w1d, w2d) to reproduce the 3 top valence bands from bilayer ARPES data (`WSe2WS2_Band*.txt`). Uses `scipy.optimize.minimize` (Nelder-Mead) from multiple starting points. Output saved to `Inputs/bilayer_fitting/interlayer_params.npy`.
3. **Bilayer moire potential** — with interlayer params fixed from Step 2, sweep moire potential parameters (Vg, Vk, phiG, phiK) and match experimental EDC peak positions via Voigt profile fitting.

## Environment

Use `../PyEnv` as the Python virtual environment. Activate with:
```
source ../PyEnv/bin/activate
```
Or use `../PyEnv/bin/python` directly.

## Commands

```
# Monolayer: single fit with a specific constraint weight combination
python scripts/fit_monolayer.py <WSe2|WS2> <index>

# Monolayer: grid search over all K1-K6 weight combinations (chunkable for HPC)
python scripts/run_grid.py WSe2                          # all combinations
python scripts/run_grid.py WSe2 --start 0 --end 400      # chunk
python scripts/run_grid.py WSe2 --score --top 20         # score existing results
python scripts/run_grid.py WSe2 --score --export          # export best params for bilayer

# Bilayer: plot ARPES data pipeline (raw → sym → interp)
python scripts/plot_bilayer_data.py

# Bilayer: interlayer coupling fit
python scripts/fit_bilayer_coupling.py

# Bilayer: moire potential EDC sweep
python scripts/analyze_edc.py <G|K> <index>
```

No build/test/lint tooling exists. Verification = does it run and produce `.npz`/`.npy`/`.h5` output.

## Architecture

### Top-level modules

| Module | Role |
|---|---|
| `constants.py` | Single source of truth: lattice constants, DFT initial params (43/material), orbital indices, symmetry relations, sample configs |
| `material.py` | `TMDMaterial` class: holds 43 params, builds hopping matrices, on-site energies, SOC Hamiltonian, parameter bounds |

### Package: `tmdmoire/monolayer/`

| Module | Role |
|---|---|
| `data.py` | `MonolayerData`: loads raw monolayer ARPES txt files, symmetrizes K→Γ/Γ→K segments, interpolates onto uniform grid. Caches in `Data/sym_{TMD}.npz` |
| `hamiltonian.py` | `MonolayerHamiltonian`: builds 22×22 H(k) at arbitrary k-points (batch) |
| `fitter.py` | `ParameterFitter`: chi-squared objective (band distance + K1-K6 constraints), dual annealing + Nelder-Mead optimization |
| `scoring.py` | `GridScorer`: loads all fit results, applies K4 hard filter, ranks by chi2_band_unweighted |

### Package: `tmdmoire/bilayer/`

| Module | Role |
|---|---|
| `data.py` | `BilayerData`: loads 3 bilayer bands from `WSe2WS2_Band*.txt`, symmetrizes K'↔K by averaging, interpolates onto uniform |k| grid |
| `geometry.py` | `MoireGeometry`: computes moire lattice constant, mini-BZ rotation, reciprocal vectors, n_cells formula, lu_table |
| `hamiltonian.py` | `MoireHamiltonian`: builds (44·N)×(44·N) supercell Hamiltonian with interlayer coupling + moire potential |
| `fitter.py` | `BilayerFitter`: fits 4 interlayer hopping params (w1p, w1d, w2p, w2d) via Nelder-Mead |
| `edc_analyzer.py` | `EDCAnalyzer`: computes EDCs from supercell eigenvalues, fits Voigt profiles, computes band gap, LDOS |

### Package: `tmdmoire/plotting/`

| Module | Role |
|---|---|
| `monolayer.py` | `plot_data_pipeline`, `plot_bands`, `plot_parameters_absolute`, `plot_orbital_content`, `plot_top_results` |
| `bilayer.py` | `plot_bilayer_data`, `plot_bilayer_fit` |

### Package: `tmdmoire/utils/`

| Module | Role |
|---|---|
| `paths.py` | `get_repo_root` (git-based), `prepare_run_dir`, `get_filename` |
| `kpoints.py` | `R_z` (2D rotation matrix), `get_k_list` (k-point path generation) |

### Entry points: `scripts/`

| Script | Role |
|---|---|
| `fit_monolayer.py` | Single fit: reads `Inputs/monolayer_fitting/grid_config.json`, creates `ParameterFitter`, runs optimization |
| `run_grid.py` | Grid search: iterates over all K1-K6 combinations, supports `--start/--end` chunking, `--score` mode, `--export` |
| `plot_bilayer_data.py` | Loads and plots bilayer ARPES data pipeline (raw → symmetrized → interpolated) |
| `fit_bilayer_coupling.py` | Fits interlayer coupling parameters to bilayer ARPES data |
| `analyze_edc.py` | Bilayer EDC sweep: loads pre-fitted monolayer params, sweeps Vg/Vk/phi/w1p/w1d, fits peaks, saves to `.h5` |

## Key facts

- **Hamiltonian basis**: 11 orbitals × 2 spins = 22×22 monolayer. Orbitals: `[d_xz, d_yz, p_z^o, p_x^o, p_y^o, d_z2, d_xy, d_x2-y2, p_z^e, p_x^e, p_y^e]`
- **43 parameters**: 7 on-site (indices 0-6) + 21 t1 (7-27) + 8 t5 (28-35) + 4 t6 (36-39) + 1 offset (40) + 2 SOC L_W/L_S (41-42)
- **Supercell**: `(44 × n_cells) × (44 × n_cells)` where `n_cells = 1 + 3×n_shells×(n_shells+1)`. nShells=2 → 19 cells → 836×836 matrix
- **Samples**: S3 (theta=1.8°), S11 (theta=2.8°, energy offset=-0.47 eV)
- **Lattice constants**: WS2=3.18 Å, WSe2=3.32 Å
- **Chi-squared terms**: band distance (always, K6-weighted in objective) + K1 (param distance from DFT) + K2 (orbital content at M) + K3 (orbital occupation at G/K) + K4 (CBM position at K) + K5 (band gap at K) + K6 (weight multiplier for high-symmetry points)
- **Stored chi2_band**: K6-weighted band distance (matches objective function); `chi2_band_unweighted` stores pure band distance for cross-comparison across grid points
- **Inputs/monolayer_fitting/**: ARPES band data (`KpGK_*.txt`, `KMKp_*.txt`), pre-fitted TB params (`tb_*.npy`), grid config (`grid_config.json`), manifest (`manifest.json`)
- **Inputs/bilayer_fitting/**: Bilayer ARPES bands (`WSe2WS2_Band*.txt`), exported monolayer params (`tb_WSe2.npy`, `tb_WS2.npy`), interlayer params (Step 2 output), moire params (Step 3 output)
- **Outputs**: `Data/run_<id>/fit_{TMD}_idx{N}.npz` (fitting results), `Data/run_<id>/Figures/` (plots for top results), `Data/sym_{TMD}.npz` (symmetrized ARPES cache), bilayer `.h5` files (EDC sweep results)
- **Default grid**: 3×3×3×3×3×1 = 243 combinations. Differential evolution maxiter=500 (popsize=20, strategy=best1exp), Nelder-Mead fatol=1e-4, maxiter=500.

## Workflow

```
Monolayer stage:
  ARPES txt files ──→ MonolayerData (symmetrize + interpolate) ──┐
  DFT initial params ─→ TMDMaterial ────────────────────────────┤
                                                                  ↓
                                                    ParameterFitter.chi2()
                                                    (band distance + K1-K6 constraints)
                                                                  ↓
                                                    dual_annealing + Nelder-Mead
                                                                  ↓
                                                    Data/fit_{TMD}_idx{N}.npz
                                                                  ↓
                                                    GridScorer → best params (.npy)
                                                    run_grid.py --score --export
                                                                  ↓
                                              Inputs/bilayer_fitting/tb_{TMD}.npy

Bilayer stage:
  tb_{TMD}.npy ───────────────────────────────────────────────────┐
  WSe2WS2_Band*.txt ─→ BilayerData (symmetrize + interp) ────────┤
                                                                  ↓
                                                    Step 1: Interlayer coupling fit
                                                    Fit w1p, w2p, w1d, w2d (4 params)
                                                    scipy.optimize.minimize (Nelder-Mead)
                                                    44×44 Hamiltonian (n_shells=0, V=0)
                                                                  ↓
                                                    Inputs/bilayer_fitting/interlayer_params.npy
                                                                  ↓
                                                    Step 2: Moire potential fit
                                                    Fix interlayer params from Step 1
                                                    Sweep Vg, Vk, phiG, phiK
                                                    EDCAnalyzer → Voigt peak fitting
                                                    Match experimental EDC peaks
                                                                  ↓
                                                    .h5 files with peak positions
```

## Conventions

- SOC bounds set to 0 in config → fit only TB params (HSO fixed from DFT)
- Bound types: "relative" (fraction of DFT value) or "absolute" (fixed eV range)
- Grid config defines K1-K6 value lists; Cartesian product gives all combinations
- Scoring: hard filter K4 < 0.05 (CBM must be at K), then rank by chi2_band_unweighted ascending
- Seed is read from grid_config.json; CLI --seed overrides in run_grid.py

## Naming conventions

- **All Python identifiers use snake_case** (PEP 8). No camelCase variables, parameters, or local names.
- **Private module-level helpers** prefixed with underscore: `_find_t`, `_find_e`, `_find_HSO` in `material.py`.
- **`R_z`** (2D rotation matrix) is defined **once** in `utils/kpoints.py`. All other modules import it from there — never redefine locally.
- **Dict keys saved to `.npz` files** (e.g. `boundType`) are preserved for backward compatibility with existing results, even if they don't match snake_case.
- **Domain abbreviations** are acceptable where unambiguous: `k_pts`, `n_cells`, `n_shells`, `mod_k`, `pts_gk`, `bz_point`, `chi2`.
- **Config dict keys** use camelCase when inherited from external files (`grid_config.json`): `boundType`, `Ks`, `Bs`.
