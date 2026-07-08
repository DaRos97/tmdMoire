# Plots for paper

Self-contained system for producing paper-quality EDC Gamma plots that can be shared with others.

## Files

```
scripts/
├── export_edc_gamma_data.py              # Extracts shareable .npz from an EDC run
└── plotsPaper/
    ├── README.md                          # This file
    ├── data/                              # Exported .npz data files
├── plot_distance_heatmaps.py          # 2D distance heatmaps (full + zoom over Vg/phiG)
├── plot_distance_w_heatmap.py         # 2D distance heatmap over w1p/w1d
└── plot_edc_profile.py                # EDC intensity profile + 4-Lorentzian fit
```

## Workflow

### 1. Export data from an EDC run

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

### 2. Share

Send the `.npz` file together with the two plotter scripts to anyone.

### 3. Plot (standalone, no tmdmoire needed)

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

#### Plots produced

**`plot_distance_heatmaps.py`:**
| Plot | Content |
|---|---|
| `distance_heatmap.png` | 2D pcolormesh of min distance over (Vg, phiG), global best marked with red star, vertical ref lines at 60/180/300 deg |
| `distance_heatmap_zoom.png` | Same, zoomed to phiG in [150, 210] |

**`plot_distance_w_heatmap.py`:**
| Plot | Content |
|---|---|
| `distance_w_heatmap.png` | 2D pcolormesh of min distance over (w1p, w1d), global best marked with red star |

**`plot_edc_profile.py`:**
| Plot | Content |
|---|---|
| `edc_profile_4L.png` | EDC intensity curve + 4-Lorentzian total fit + experimental ARPES positions |

## .npz data format

### Always present

| Key | Description |
|---|---|
| `run_id` | Run identifier string |
| `Vg_vals_meV` | Vg grid centers in meV (shape: n_Vg) |
| `phiG_vals_deg` | phiG grid centers in degrees (shape: n_phi) |
| `dist_2d_meV` | Minimum L1 distance per cell (shape: n_Vg, n_phi) |
| `phi_edges` | pcolormesh edge coordinates for phiG axis |
| `Vg_edges_meV` | pcolormesh edge coordinates for Vg axis |
| `best_Vg_meV` | Global best Vg value |
| `best_phiG_deg` | Global best phiG value |
| `best_dist_meV` | Distance at global best |
| `best_w1p_ev` | w1p at global best |
| `best_w1d_ev` | w1d at global best |
| `w1p_vals_meV` | w1p grid centers in meV (shape: n_w1p) |
| `w1d_vals_meV` | w1d grid centers in meV (shape: n_w1d) |
| `dist_w_2d_meV` | Min distance per (w1p, w1d) cell, minimizing over (Vg, phiG) (shape: n_w1d, n_w1p) |
| `w1p_edges_meV` | pcolormesh edge coordinates for w1p axis |
| `w1d_edges_meV` | pcolormesh edge coordinates for w1d axis |

### Present when --vg/--phig given

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

## Distance metric

```
dist = |c1 - E_TVB| + |c2 - E_side| + |c3 - E_LVB|
```

where experimental values for S11 are `[-1.1599, -1.2531, -1.82]` eV. The 2D grid shows the minimum distance over all (w1p, w1d) combinations at each (Vg, phiG) cell.
