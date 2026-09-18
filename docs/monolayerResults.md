# Monolayer Fitting — Best Results

This document summarises the best tight-binding (TB) parameter fit for the
monolayer stage of the workflow, one per material (WSe₂ and WS₂). It is the
companion to `docs/monolayer.md`, which describes the fitting procedure in
full; this file reports the *outcomes* of the canonical v3.0 grid search and
the values of the objective function at the selected minimum.

The objective that Nelder-Mead minimises is

```
chi2 = band_K6 + K1*K1_val + K2*K2_val + K3*K3_val + K4*K4_val + K5*K5_val
```

where `band_K6` is the K₆-weighted band distance. Because the public question
"value of chi²" and "value of the objective function at the minimum" refer to
the same quantity in this codebase, both are reported together below.

## Source data

The canonical results are stored in v3.0 merged HDF5 files produced by the
grid-search step (see `docs/monolayer.md` for definitions of the constraint
terms):

| Material | Source file | Total entries | After K-range mask | After WSe₂ bounds-saturation mask |
|---|---|---|---|---|
| WSe₂ | `Data/WSe2_run1/merged_WSe2_absolute.h5` | 1 005 | 1 005 | 263 |
| WS₂  | `Data/WS2_run1/merged_WS2_absolute.h5`   |   100 |   100 | (mask not applied) |

Each HDF5 file stores one row per (K₁ … K₆) combination, with:

- `Ks` — the constraint weights (K₁ … K₆),
- `elements` — six columns: `[band_K6, K1_val, K2_val, K3_val, K4_val, K5_val]`,
- `pars` — 43 fitted tight-binding parameters,
- `Bs` — bounds tuple `(8, 4, 5, 2, 0)`, i.e. ±8 eV (on-site), ±4 eV (t₁),
  ±5 eV (t₅), ±2 eV (t₆), SOC fixed at the DFT value.

The k-point grid used for the 6 top-valence bands is the **default 91-point
Γ–K–M path** (see `scripts/run_monolayer_grid.py`, field `pts`). K₄ = 1,
K₅ = 0.01 and K₆ = 5 were held fixed in both grids; K₁, K₂, K₃ were swept:

- **WSe₂** swept K₁ ∈ {1×10⁻⁶, 1×10⁻⁵, 1×10⁻⁴}, K₂ ∈ 66 values in
  [0.01, 1.0], K₃ ∈ {0.005, 0.010, 0.015, 0.020, 0.025}.
- **WS₂** swept K₁ ∈ {0, 1×10⁻⁶, 1×10⁻⁵, 1×10⁻⁴}, K₂ ∈ {0.0625, 0.0884,
  0.125, 0.1768, 0.25}, K₃ ∈ {0.0078, 0.0111, 0.0156, 0.0221, 0.0313}.

## Selection procedure (v3.0)

1. **K-range mask** — keep entries with K₁ > −1×10⁻⁷, K₂ ∈ (−2⁻⁸, 10),
   K₃ > −0.012, K₆ > −1.
2. **Bounds-saturation filter** (WSe₂ only) — drop entries where any
   parameter group (eps / t₁ / t₅ / t₆) sat at ±B within tolerance
   tol = 1×10⁻².
3. **Primary ranking** — sort survivors by `band_K6` ascending.
4. **Secondary ranking** — sort survivors by `band_K6 + K2_val` ascending.
5. **`ind_chosen`** — for WSe₂ the **2nd best** result is selected
   (`ind_chosen = 1`, to avoid the most aggressive DFT-deviating fits),
   while for WS₂ the **1st best** is selected (`ind_chosen = 0`).

The parameter vector of the primary-ranked fit is the one exported to
`Inputs/plot_bilayer/tb_{TMD}.npy` and to the versioned copies in
`Inputs/bilayer_fitting/` and `Inputs/monolayer_fitting/`.

## WSe₂ — best fit

**Grid entry idx 710**, after the masks above (1005 → 1005 → 263 entries,
primary ranking → take position 1).

### Constraint weights

| K₁ | K₂ | K₃ | K₄ | K₅ | K₆ |
|---|---|---|---|---|---|
| 0.0001 | 0.13 | 0.005 | 1 | 0.01 | 5 |

### Objective, band distance and constraint breakdown

| Quantity | Value | Notes |
|---|---|---|
| **chi2 (objective at minimum)** | **0.005023** | `band_K6 + Σ Kᵢ·Kᵢ_val` |
| **band_K6** | **0.004206** | K₆-weighted mean-squared band distance (term in the objective) |
| band_dist (pure, derived = band_K6 / K₆) | 0.000841 | unweighted band distance, for cross-comparison across K₆ |
| K₁_val (parameter distance from DFT) | 3.9879 | mean abs relative deviation |
| K₂_val (M orbital content) | 0.00260 | ILC weight in TVBs at M |
| K₃_val (Γ / K orbital occupation) | 0.01597 | sum of 8 |occ_DFT − occ_TB| |
| K₄_val (CBM at K) | 0 | CBM is at K — satisfied |
| K₅_val (band gap at K) | 5.6 × 10⁻¹⁰ | DFT gap reproduced |

Contributions to chi² from each term: `band_K6 = 0.004206`,
K₁·K₁_val = 3.99 × 10⁻⁴, K₂·K₂_val = 3.38 × 10⁻⁴, K₃·K₃_val = 8.0 × 10⁻⁵,
K₄·K₄_val = 0, K₅·K₅_val = 5.6 × 10⁻¹².

### Top 5 by band_K6

| Rank | K₁ | K₂ | K₃ | band_K6 | chi2 | Selected |
|---|---|---|---|---|---|---|
| 0 | 1 × 10⁻⁵ | 0.205 | 0.010 | 0.004035 | 0.004453 | |
| **1** | **1 × 10⁻⁴** | **0.13** | **0.005** | **0.004206** | **0.005023** | **←** |
| 2 | 1 × 10⁻⁴ | 0.43  | 0.010 | 0.004253 | 0.005457 | |
| 3 | 1 × 10⁻⁶ | 0.175 | 0.005 | 0.004312 | 0.004737 | |
| 4 | 1 × 10⁻⁴ | 0.955 | 0.020 | 0.004359 | 0.005681 | |

Rank 0 has the smallest band_K6 (0.004035) but is rejected for export
because the WSe₂ convention (`ind_chosen = 1`) prefers the fit that is one
step further from the bound-saturation edge.

### Secondary ranking (band_K6 + K2_val)

Position 1 in the secondary ranking is **idx 988** with
K = (1 × 10⁻⁴, 0.955, 0.020, 1, 0.01, 5), band_K6 = 0.004359 and
`band_K6 + K2_val` = 0.004997. This fit has the lowest K₂_val (0.00064 vs
0.00260 of the primary pick) at the cost of a slightly larger band_K6.

## WS₂ — best fit

**Grid entry idx 86**, after the masks above (100 → 100 entries, primary
ranking → take position 0).

### Constraint weights

| K₁ | K₂ | K₃ | K₄ | K₅ | K₆ |
|---|---|---|---|---|---|
| 0 | 0.125 | 0.01105 | 1 | 0.01 | 5 |

### Objective, band distance and constraint breakdown

| Quantity | Value | Notes |
|---|---|---|
| **chi2 (objective at minimum)** | **0.001617** | `band_K6 + Σ Kᵢ·Kᵢ_val` |
| **band_K6** | **0.000995** | K₆-weighted mean-squared band distance |
| band_dist (pure, derived = band_K6 / K₆) | 0.000199 | unweighted band distance |
| K₁_val (parameter distance from DFT) | 5.3261 | mean abs relative deviation |
| K₂_val (M orbital content) | 5.6 × 10⁻⁵ | ILC weight in TVBs at M |
| K₃_val (Γ / K orbital occupation) | 0.05566 | sum of 8 \|occ_DFT − occ_TB\| |
| K₄_val (CBM at K) | 0 | CBM is at K — satisfied |
| K₅_val (band gap at K) | 4.3 × 10⁻⁸ | DFT gap reproduced |

Contributions to chi² from each term: `band_K6 = 0.000995`,
K₁·K₁_val = 0 (K₁ = 0), K₂·K₂_val = 7.0 × 10⁻⁶, K₃·K₃_val = 6.15 × 10⁻⁴,
K₄·K₄_val = 0, K₅·K₅_val = 4.3 × 10⁻¹⁰.

### Top 5 by band_K6

| Rank | K₁ | K₂ | K₃ | band_K6 | chi2 | Selected |
|---|---|---|---|---|---|---|
| **0** | **0** | **0.125** | **0.01105** | **0.000995** | **0.001617** | **←** |
| 1 | 1 × 10⁻⁶ | 0.0625 | 0.0078  | 0.001901 | 0.002026 | |
| 2 | 0         | 0.0625 | 0.0078  | 0.001932 | 0.002166 | |
| 3 | 1 × 10⁻⁶ | 0.0884 | 0.01105 | 0.002019 | 0.002404 | |
| 4 | 1 × 10⁻⁵ | 0.125  | 0.0156  | 0.002063 | 0.002324 | |

### Secondary ranking (band_K6 + K2_val)

The primary and secondary rankings coincide for WS₂: the K₂_val of the best
fit (5.6 × 10⁻⁵) is so small that adding it does not reorder the top of the
list. The exported params therefore come from idx 86 in both rankings.

## Side-by-side summary

| | WSe₂ | WS₂ |
|---|---|---|
| Grid source | `Data/WSe2_run1/merged_WSe2_absolute.h5` | `Data/WS2_run1/merged_WS2_absolute.h5` |
| Total entries (raw → after masks) | 1005 → 263 | 100 → 100 |
| `ind_chosen` | 1 (2nd best by band_K6) | 0 (1st best by band_K6) |
| **K₁, K₂, K₃, K₄, K₅, K₆** | **0.0001, 0.13, 0.005, 1, 0.01, 5** | **0, 0.125, 0.01105, 1, 0.01, 5** |
| **chi2 (objective minimum)** | **0.005023** | **0.001617** |
| **band_K6** | **0.004206** | **0.000995** |
| band_dist (derived) | 0.000841 | 0.000199 |
| K₁_val / K₂_val / K₃_val / K₄_val / K₅_val | 3.988 / 0.00260 / 0.01597 / 0 / 5.6 × 10⁻¹⁰ | 5.326 / 5.6 × 10⁻⁵ / 0.0557 / 0 / 4.3 × 10⁻⁸ |
| Exported as | `Inputs/plot_bilayer/tb_WSe2.npy`, versioned copies in `Inputs/bilayer_fitting/` and `Inputs/monolayer_fitting/` | `Inputs/plot_bilayer/tb_WS2.npy`, versioned copies in `Inputs/bilayer_fitting/` and `Inputs/monolayer_fitting/` |

## Reproducibility

To re-derive the best fits from the raw HDF5 files (read-only, no writes),
run the helper script that ships with the repository. It applies the same K-range
mask and bounds-saturation filter and then ranks by `band_K6` /
`band_K6 + K2_val`:

```bash
# Inspect the WSe2 grid and dump the best fits
python scripts/sort_monolayer_results.py --tmd WSe2 \
    --input Data/WSe2_run1/merged_WSe2_absolute.h5

# Same for WS2
python scripts/sort_monolayer_results.py --tmd WS2 \
    --input Data/WS2_run1/merged_WS2_absolute.h5
```

To regenerate the constraint breakdown for the currently exported parameters
(K₁_val … K₅_val, raw band_K6 and K5-weighted band distance), see
`scripts/dev/check_exported_params.py`, which loads the file matching
`tb_{TMD}*.npy` from `Inputs/bilayer_fitting/`, constructs the constraint
breakdown via `ParameterFitter._compute_constraint_breakdown`, and prints the
same numbers reported above.

## Cross-reference

- Definitions of K₁…K₆ and the chi² formulation: `docs/monolayer.md`.
- The same summary in an older, more compact format:
  `docs/results.md` (WSe₂/WS₂ best fits, without the explicit chi²).
- Source code of the chi² function and per-result `.npz` schema:
  `tmdmoire/monolayer/fitter.py:84` (chi²), `fitter.py:197` (save schema).
- Grid-scorer that selects the exports:
  `tmdmoire/monolayer/scoring.py:100` (score) and
  `scripts/run_monolayer_grid.py:150` (do_score with --export).
