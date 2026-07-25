# Monolayer Fitting Results

## Results

Pre-computed results from the v3.0 grid search are available in `Data/WSe2_run1/` and `Data/WS2_run1/` as merged HDF5 files. Both runs used the same physical bounds **Bs = (8, 4, 5, 2, 0)** (±8 eV on-site, ±4 eV t₁, ±5 eV t₅, ±2 eV t₆, SOC fixed). The weights K₄=1, K₅=0.01, and K₆=5 were held fixed in both runs.

### Selection convention

Following the v3.0 procedure, results are ranked by the band-distance component (K₆-weighted band distance, `elements[:, 0]` in the HDF5). To avoid artifacts from parameter-bound saturation, the **2nd best** result (`ind_chosen = 1`) is selected for **WSe₂**, while the **1st best** (`ind_chosen = 0`) is selected for **WS₂**. The values reported below are the results that `sort_monolayer_results.py` selects and displays.

### WSe₂ (`Data/WSe2_run1/`)

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

### WS₂ (`Data/WS2_run1/`)

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

