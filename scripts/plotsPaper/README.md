# Plots for paper

Self-contained system for producing paper-quality plots that can be shared with others. Covers EDC Gamma analysis, EDC vs V_G, and moiré band structure around Γ.

## Files

```
scripts/
├── export_edc_gamma_data.py              # Extracts shareable .npz from an EDC run
├── export_edc_vs_V.py                    # Computes EDC TVB–side distance vs V_G, exports .npz
├── export_moire_bands.py                 # Computes moiré band structure and exports .npz
└── plotsPaper/
    ├── README.md                          # This file
    ├── data/                              # Exported .npz data files
    ├── plot_distance_heatmaps.py          # 2D distance heatmaps (full + zoom over Vg/phiG)
    ├── plot_distance_w_heatmap.py         # 2D distance heatmap over w1p/w1d
    ├── plot_edc_profile.py                # EDC intensity profile + 4-Lorentzian fit
    ├── plot_edc_vs_V.py                   # EDC TVB–side distance vs V_G
    └── plot_moire_bands.py                # Moiré bands around Γ (V_G = 0 vs 12 meV)
```

## Workflow

### EDC Gamma

#### 1. Export data from an EDC run

```bash
# Heatmap data only (any run with combined.h5)
python scripts/export_edc_gamma_data.py --id 001
# -> scripts/plotsPaper/data/edc_gamma_001.npz

# Heatmap + EDC profile for a specific (Vg, phiG) cell
python scripts/export_edc_gamma_data.py --id 001 --vg 0.012 --phig 176
# -> scripts/plotsPaper/data/edc_gamma_001_Vg12meV_phiG176deg.npz

# Custom output path
python scripts/export_edc_gamma_data.py --id 001 --vg 0.012 --phig 176 --output my_data.npz
```

This step requires the tmdmoire environment. It reads `Data/edc_gamma_<id>/combined.h5` and `metadata.json`, aggregates the 2D distance grid, and optionally recomputes the full EDC intensity profile at the selected cell via Hamiltonian diagonalization + 4-Lorentzian fitting.

#### 2. Share

Send the `.npz` file together with the plotter scripts to anyone.

#### 3. Plot (standalone, no tmdmoire needed)

```bash
# Distance heatmaps over Vg/phiG (full-range + zoom)
python plot_distance_heatmaps.py data/edc_gamma_001.npz
python plot_distance_heatmaps.py data.npz --output-dir ./figures

# Distance heatmap over w1p/w1d
python plot_distance_w_heatmap.py data/edc_gamma_001.npz
python plot_distance_w_heatmap.py data.npz --output-dir ./figures

# EDC intensity profile (requires .npz exported with --vg/--phig)
python plot_edc_profile.py data/edc_gamma_001_Vg_12meV_phiG_176deg.npz
python plot_edc_profile.py data.npz --output-dir ./figures
```

### Moiré bands around Γ

#### 1. Export data

```bash
python scripts/export_moire_bands.py
# -> scripts/plotsPaper/data/moire_bands_k251_n1_Vg0_12.npz
```

This step requires the tmdmoire environment. Computes the supercell band structure along a G→K line through Γ (±0.4 Å⁻¹) for V_G = 0 and 12 meV at n_shells=1 (7 cells, 308×308 Hamiltonian).

#### 2. Share

Send the `.npz` file together with `plot_moire_bands.py` to anyone.

#### 3. Plot (standalone, no tmdmoire needed)

```bash
python plot_moire_bands.py data/moire_bands_k251_n1_Vg0_12.npz
python plot_moire_bands.py data.npz --output-dir ./figures
```

### EDC vs V_G at Γ

#### 1. Export data

```bash
python scripts/export_edc_vs_V.py
# -> scripts/plotsPaper/data/edc_vs_V_n20_Vg1-20.npz
```

This step requires the tmdmoire environment. Computes EDC intensity profiles at Gamma for V_G = 1–20 meV (20 points), fits 4 Lorentzians, and extracts the distance between the TVB main peak and the side-band peak.

#### 2. Share

Send the `.npz` file together with `plot_edc_vs_V.py` to anyone.

#### 3. Plot (standalone, no tmdmoire needed)

```bash
python plot_edc_vs_V.py data/edc_vs_V_n20_Vg1-20.npz
python plot_edc_vs_V.py data.npz --output-dir ./figures
```

### Plots produced

**`plot_distance_heatmaps.py`:**
| Plot | Content |
|---|---|
| `distance_heatmap.png` | 2D pcolormesh of min distance over (Vg, phiG), global best marked with red star, vertical ref lines at 60/180/300 deg |
| `distance_heatmap_zoom.png` | Same, zoomed to phiG in [160, 200] |

**`plot_distance_w_heatmap.py`:**
| Plot | Content |
|---|---|
| `distance_w_heatmap.png` | 2D pcolormesh of min distance over (w1p, w1d), global best marked with red star |

**`plot_edc_profile.py`:**
| Plot | Content |
|---|---|
| `edc_profile_4L.png` | EDC intensity curve + 4-Lorentzian total fit + experimental ARPES positions |

**`plot_moire_bands.py`:**
| Plot | Content |
|---|---|
| `moire_bands_gamma.png` | 1×2 panels: TVB bands along k ∈ [−0.4, 0.4] Å⁻¹ for V_G = 0 and 12 meV, thin gray lines + weight-proportional blue circles |

**`plot_edc_vs_V.py`:**
| Plot | Content |
|---|---|
| `edc_vs_V.png` | TVB–side band distance (meV) vs V_G (meV), black markers + red dashed line at ARPES distance |

## .npz data format

### EDC Gamma — always present

| Key | Description |
|---|---|
| `run_id` | Run identifier string |
| `Vg_vals_meV` | Vg grid centers in meV (shape: n_Vg) |
| `phiG_vals_deg` | phiG grid centers in degrees (shape: n_phi) |
| `dist_2d_meV` | Minimum L1 distance per (Vg, phiG) cell (shape: n_Vg, n_phi) |
| `dist_sep_2d_meV` | Minimum separation distance per (Vg, phiG) cell |
| `phi_edges` | pcolormesh edge coordinates for phiG axis |
| `Vg_edges_meV` | pcolormesh edge coordinates for Vg axis |
| `best_Vg_meV` | Global best Vg value |
| `best_phiG_deg` | Global best phiG value |
| `best_dist_meV` | Distance at global best |
| `best_w1p_ev` | w1p at global best |
| `best_w1d_ev` | w1d at global best |
| `w1p_vals_meV` | w1p grid centers in meV (shape: n_w1p) |
| `w1d_vals_meV` | w1d grid centers in meV (shape: n_w1d) |
| `dist_w_2d_meV` | Min L1 distance per (w1p, w1d) cell (shape: n_w1d, n_w1p) |
| `dist_sep_w_2d_meV` | Min separation distance per (w1p, w1d) cell |
| `w1p_edges_meV` | pcolormesh edge coordinates for w1p axis |
| `w1d_edges_meV` | pcolormesh edge coordinates for w1d axis |

### EDC Gamma — present when --vg/--phig given

| Key | Description |
|---|---|
| `energy_list` | Energy axis in eV (shape: n_e) |
| `weight_list` | EDC intensity (shape: n_e) |
| `fit_4L_curve` | 4-Lorentzian total fit curve (shape: n_e) |
| `fit_4L_centers` | Fitted peak centers in eV (shape: 4) |
| `fit_4L_redchi` | Reduced chi-squared of 4-Lorentzian fit |
| `exp_positions_ev` | Experimental ARPES peak positions (shape: 3) |
| `selected_Vg_meV` | Selected cell Vg |
| `selected_phiG_deg` | Selected cell phiG |
| `selected_w1p_ev` | Selected cell w1p |
| `selected_w1d_ev` | Selected cell w1d |
| `selected_w2p_ev` | Fixed w2p from Step 2 |
| `selected_w2d_ev` | Fixed w2d from Step 2 |

### Moiré bands around Γ

| Key | Description |
|---|---|
| `k_vals` | k-axis values in Å⁻¹ (shape: n_kpts) |
| `evals_0` | Eigenvalues for V_G = 0 meV (shape: n_kpts, n_bands) |
| `weights_0` | Central-cell weights for V_G = 0 (shape: n_kpts, n_bands) |
| `evals_1` | Eigenvalues for V_G = 12 meV (shape: n_kpts, n_bands) |
| `weights_1` | Central-cell weights for V_G = 12 meV (shape: n_kpts, n_bands) |
| `Vg_values_meV` | Array of V_G values: `[0, 12]` |
| `Vg_labels` | Label strings: `["0 meV", "12 meV"]` |
| `n_shells` | Number of moiré shells |
| `n_cells` | Number of mini-BZ cells |
| `n_kpts` | Number of k-points |
| `k_range` | k-axis range in Å⁻¹ (symmetric: [−k_range, k_range]) |
| `phiG_deg` | Moiré potential phase at Γ (degrees) |
| `interlayer_w1p` | Interlayer coupling w1p (eV) |
| `interlayer_w1d` | Interlayer coupling w1d (eV) |
| `interlayer_w2p` | Interlayer coupling w2p (eV) |
| `interlayer_w2d` | Interlayer coupling w2d (eV) |

### EDC vs V_G at Γ

| Key | Description |
|---|---|
| `Vg_vals_meV` | V_G values in meV (shape: n_Vg) |
| `distances_meV` | TVB–side band distances in meV (shape: n_Vg) |
| `arpes_distance_meV` | Experimental ARPES TVB–side distance (scalar) |
| `interlayer_w1p` | Interlayer coupling w1p (eV) |
| `interlayer_w1d` | Interlayer coupling w1d (eV) |
| `interlayer_w2p` | Interlayer coupling w2p (eV) |
| `interlayer_w2d` | Interlayer coupling w2d (eV) |
| `phiG_deg` | Moiré potential phase at Γ (degrees) |
| `n_shells` | Number of moiré shells |

## Distance metrics

Two metrics are computed for each fitted EDC point:

**L1 distance** — sum of absolute differences between fitted peak positions and experimental values:
```
dist = |c1 - E_TVB| + |c2 - E_side| + |c3 - E_LVB|
```

**Separation distance** — sum of absolute differences between peak separations (insensitive to global energy shifts):
```
dist_sep = ||c1 - c2| - |E_TVB - E_side|| + ||c1 - c3| - |E_TVB - E_LVB||
```

where experimental values for S11 are `[-1.1599, -1.2531, -1.82]` eV. Both heatmaps show the minimum over the marginalized dimensions at each cell.
