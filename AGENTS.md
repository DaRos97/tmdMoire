# AGENTS.md

## Project

Tight-binding model of WSe2/WS2 heterobilayer moire superlattices. Three-stage workflow:

1. **Monolayer fitting** — fit 43 TB parameters per TMD (WSe2, WS2) to ARPES band dispersion data via differential evolution + Nelder-Mead. Results saved as `.npz` files; best params exported to `Inputs/bilayer_fitting/tb_{TMD}.npy` via `run_monolayer_grid.py --score --export`.
2. **Bilayer interlayer coupling** — fit 4 interlayer hopping parameters (w1p, w2p, w1d, w2d) to reproduce the 3 top valence bands from bilayer ARPES data (`WSe2WS2_Band*.txt`). Uses `scipy.optimize.minimize` (Nelder-Mead) from multiple starting points. Output saved to `Inputs/bilayer_fitting/interlayer_params.npy`.
3. **Bilayer moiré potential** — with interlayer params fixed, sweep moiré potential parameters at Gamma (6D: Vg, Vk, phiG, phiK, w1p, w1d, w2p, w2d) and K (2D: Vk, phiK) to match experimental EDC peak positions via Lorentzian profile fitting. Gamma sweep finds best interlayer + moire params; K sweep refines Vk/phiK and computes band gap.

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
python scripts/run_monolayer_grid.py WSe2                          # all combinations
python scripts/run_monolayer_grid.py WSe2 --start 0 --end 400      # chunk
python scripts/run_monolayer_grid.py WSe2 --score --top 20         # score existing results
python scripts/run_monolayer_grid.py WSe2 --score --export          # export best params for bilayer

# Bilayer: plot ARPES data pipeline (raw → sym → interp)
python scripts/plot_bilayer_data.py

# Bilayer: interlayer coupling fit
python scripts/fit_bilayer_coupling.py

# Bilayer: moire potential EDC sweep (Gamma, 6D grid)
python scripts/edc_grid_gamma.py --chunk <id>/<total> --run-id <id>
python scripts/combine_edc_chunks.py --bz-point gamma --run-id <id>
python scripts/analyze_edc_gamma.py --run-id <id>

# Bilayer: moire potential EDC sweep (K, 2D grid)
python scripts/edc_grid_k.py --chunk <id>/<total> --run-id <id>
python scripts/combine_edc_chunks.py --bz-point k --run-id <id>
python scripts/analyze_edc_k.py --run-id <id>
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
| `scoring.py` | `GridScorer`: loads all fit results, applies K4 hard filter, ranks by band_dist |

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
| `fit_monolayer.py` | Single fit: reads `Inputs/monolayer_fitting/fit_config.json`, creates `ParameterFitter`, runs optimization |
| `run_monolayer_grid.py` | Grid search: iterates over all K1-K6 combinations, supports `--start/--end` chunking, `--score` mode, `--export` |
| `plot_bilayer_data.py` | Loads and plots bilayer ARPES data pipeline (raw → symmetrized → interpolated) |
| `fit_bilayer_coupling.py` | Fits interlayer coupling parameters to bilayer ARPES data |
| `edc_grid_gamma.py` | Gamma-point EDC sweep: 6D grid over Vg, phiG, w1p, w1d, w2p, w2d; fits 4 Lorentzians; saves to `.h5` |
| `edc_grid_k.py` | K-point EDC sweep: 2D grid over Vk, phiK; fits 2 Lorentzians + band gap; saves to `.h5` |
| `combine_edc_chunks.py` | Combines chunked `.h5` files into single `combined.h5` |
| `analyze_edc_gamma.py` | Analyzes Gamma results: computes distance from experiment, plots 2D heatmap |
| `analyze_edc_k.py` | Analyzes K results: computes distance + band gap, plots 2D heatmap |

### Development scripts: `scripts/dev/`

Scripts in `scripts/dev/` are for testing, debugging, and inspecting intermediate steps of the main pipeline. They are **not part of the production workflow**. See `scripts/dev/README.md` for details. Key scripts:

| Script | Role |
|---|---|
| `test_edc_gamma.py` | Single-point EDC at Gamma: computes intensity profile, fits 4 Lorentzians, plots result |
| `benchmark_edc.py` | Measures wall-clock time per EDC point (diagonalization + spreading + fitting) |
| `check_k3_d2.py` | Verifies K3 orbital occupation constraint at K against DFT targets |
| `check_k5_detail.py` | Detailed K5 analysis: DFT vs fitted band shifts at K |
| `check_exported_params.py` | Computes full constraint breakdown for exported `tb_*.npy` files |
| `check_config_flow.py` | Traces all `fit_config.json` keys through the pipeline, reports unread keys |
| `test_plot_params_mono.py` | Tests parameter bar plotting with random params within bounds |

## Key facts

- **Hamiltonian basis**: 11 orbitals × 2 spins = 22×22 monolayer. Orbitals: `[d_xz, d_yz, p_z^o, p_x^o, p_y^o, d_z2, d_xy, d_x2-y2, p_z^e, p_x^e, p_y^e]`
- **43 parameters**: 7 on-site (indices 0-6) + 21 t1 (7-27) + 8 t5 (28-35) + 4 t6 (36-39) + 1 offset (40) + 2 SOC L_W/L_S (41-42)
- **Supercell**: `(44 × n_cells) × (44 × n_cells)` where `n_cells = 1 + 3×n_shells×(n_shells+1)`. nShells=2 → 19 cells → 836×836 matrix
- **Samples**: S3 (theta=1.8°), S11 (theta=2.8°, energy offset=-0.47 eV)
- **Lattice constants**: WS2=3.18 Å, WSe2=3.32 Å
- **Chi-squared terms**: band distance (always, K6-weighted in objective) + K1 (param distance from DFT) + K2 (orbital content at M) + K3 (orbital occupation at G/K) + K4 (CBM position at K) + K5 (band gap at K) + K6 (weight multiplier for high-symmetry points)
- **Stored band_K6**: K6-weighted band distance (matches objective function); `band_dist` stores pure band distance for cross-comparison across grid points
- **Inputs/monolayer_fitting/**: ARPES band data (`KpGK_*.txt`, `KMKp_*.txt`), pre-fitted TB params (`tb_*.npy`), fit config (`fit_config.json`), manifest (`manifest.json`)
- **Inputs/bilayer_fitting/**: Bilayer ARPES bands (`WSe2WS2_Band*.txt`), exported monolayer params (`tb_WSe2.npy`, `tb_WS2.npy`), interlayer params (Step 2 output), grid configs (`grid_config_gamma.json`, `grid_config_k.json`)
- **Outputs**: `Data/run_<id>/fit_{TMD}_idx{N}.npz` (fitting results), `Data/run_<id>/Figures/` (plots for top results), `Data/sym_{TMD}.npz` (symmetrized ARPES cache), `Data/edc_grid_gamma_run_<id>/` and `Data/edc_grid_k_run_<id>/` (EDC sweep results as `.h5` files)
- **Default Gamma grid**: 11×37×11×11×11×11 = ~6M combinations. ~0.6s per point (836×836 diagonalization dominates).
- **Default K grid**: 20×37 = 740 combinations (configurable).
- **EDC fitting**: Lorentzian broadening (spreadE=0.03 eV) + 4-Lorentzian fit at Gamma, 2-Lorentzian fit at K.

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
                                                    run_monolayer_grid.py --score --export
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
                                                     Step 2a: Moire potential at Gamma
                                                     Fix interlayer params from Step 1
                                                     Sweep Vg, phiG, w1p, w1d, w2p, w2d (6D)
                                                     Fixed: Vk=7.7 meV, phiK=106 deg
                                                     836×836 diagonalization + 4-Lorentzian fit
                                                     Match EDC_G_POSITIONS
                                                                   ↓
                                                     Data/edc_grid_gamma_run_<id>/
                                                     combined.h5 + analysis.png
                                                                   ↓
                                                     Step 2b: Moire potential at K
                                                     Fix all params from Gamma best fit
                                                     Sweep Vk, phiK (2D)
                                                     2-Lorentzian fit + band gap
                                                     Match EDC_K_POSITIONS
                                                                   ↓
                                                     Data/edc_grid_k_run_<id>/
                                                     combined.h5 + analysis.png
```

## Conventions

- **Never save files outside the project repository.** All output (figures, data files, debug plots) must stay within the project directory tree (e.g. `Data/`, `Figures/`, `Inputs/`).
- SOC bounds set to 0 in config → fit only TB params (HSO fixed from DFT)
- Bound types: "relative" (fraction of DFT value) or "absolute" (fixed eV range)
- Grid config defines K1-K6 value lists; Cartesian product gives all combinations
- `use_dft_x0` in config controls whether DFT params seed the optimizer (true) or population is fully random (false)
- Scoring: hard filter K4 < 0.05 (CBM must be at K), then rank by band_dist ascending
- Seed is read from fit_config.json; CLI --seed overrides in run_monolayer_grid.py

## Naming conventions

- **All Python identifiers use snake_case** (PEP 8). No camelCase variables, parameters, or local names.
- **Private module-level helpers** prefixed with underscore: `_find_t`, `_find_e`, `_find_HSO` in `material.py`.
- **`R_z`** (2D rotation matrix) is defined **once** in `utils/kpoints.py`. All other modules import it from there — never redefine locally.
- **Dict keys saved to `.npz` files** (e.g. `boundType`) are preserved for backward compatibility with existing results, even if they don't match snake_case.
- **Domain abbreviations** are acceptable where unambiguous: `k_pts`, `n_cells`, `n_shells`, `mod_k`, `pts_gk`, `bz_point`, `chi2`.
- **Config dict keys** use camelCase when inherited from external files (`fit_config.json`): `boundType`, `Ks`, `Bs`.
