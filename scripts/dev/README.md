# scripts/dev/ — Development and debugging scripts

This directory contains scripts for testing, debugging, and inspecting intermediate steps of the main pipeline. They are **not part of the production workflow** and use hardcoded paths or parameters for quick iteration.

## EDC development

| Script | Purpose |
|---|---|
| `test_edc_gamma.py` | Computes the EDC intensity profile at Gamma for a single parameter set (Vg=15 meV, phiG=180°), fits 4 Lorentzians, and plots the intensity profile with individual peak contributions. Saves to `Data/edc_intensity_test/`. |
| `benchmark_edc.py` | Measures the wall-clock time for a single EDC computation (diagonalization + spreading + fitting). Reports average time per point over N runs. |

## Monolayer constraint debugging

| Script | Purpose |
|---|---|
| `check_k3_d2.py` | Computes d₋₂ and p₋₁^e orbital occupations at K from DFT-derived Hamiltonian and compares them to the `ORBITAL_CHARACTER` targets in `constants.py`. Used to verify the K3 constraint implementation. |
| `check_k5_detail.py` | Detailed K5 analysis: compares band energies at K between DFT and fitted parameters, checks if all bands shift uniformly (pure offset effect), and reports the shift range. |
| `check_exported_params.py` | Loads exported `tb_{TMD}.npy` files from `Inputs/bilayer_fitting/`, computes the full constraint breakdown (band_dist, K1–K5), and prints a summary for both WSe2 and WS2. |
| `check_config_flow.py` | Traces all keys from `fit_config.json` through the pipeline: loads the raw config, simulates `build_grid()`, lists what `ParameterFitter.run()` reads, and reports any unread keys. |
| `test_plot_params_mono.py` | Generates a parameter bar plot with random parameters within bounds to verify the `plot_parameters_absolute` function works correctly. |

## Usage

Scripts in this directory are run directly from the repository root:

```bash
python scripts/dev/test_edc_gamma.py
python scripts/dev/benchmark_edc.py
python scripts/dev/check_k5_detail.py
```

They are intended for one-off inspection and should not be used for production grid sweeps.
