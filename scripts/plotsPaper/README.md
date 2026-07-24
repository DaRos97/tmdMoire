# Plots for paper

Standalone plotting scripts for EDC Gamma analysis, EDC vs V_G, and moire band structure. No tmdmoire environment needed — only numpy + matplotlib.

Run all commands from `scripts/plotsPaper/`. Output goes to `figures/`.

## Dataset reference

| ID | Sample | Twist | Best Vg | Best phiG | w1p | w1d | L2 |
|----|--------|:-----:|:-------:|:---------:|------|------|----:|
| 4Dsm8a | S11 | 2.8° | 10.5 meV | 176° | -1.220 | +0.460 | 0.12 meV |
| 4Dsm8b | S11_m25 | 2.5° | 12.5 meV | 176° | -1.190 | +0.445 | 0.68 meV |
| 4Dsm8c | S11_p31 | 3.1° | 8.5 meV | 176° | -1.185 | +0.445 | 0.67 meV |
| S3_2a | S3 | 1.8° | 11.5 meV | 175° | -1.200 | +0.455 | — |

Fixed interlayer: w2p = -0.1694 eV, w2d = +0.0215 eV (from Step 2).

S11 experimental EDC positions: -1.1599, -1.2531, -1.8200 eV.
S3 experimental EDC positions: -0.69484, -0.77307, -1.35 eV.

Filters applied: L1 < 26 meV, L2 < 10 meV, a2/a1 >= 0.

## Exporting data (requires tmdmoire + PyEnv)

The data `.npz` files in `data/` are produced by the `scripts/export_*.py` scripts. Each script now accepts `--sample`, `--w1p/--w1d/--w2p/--w2d`, `--phiG`, and `--Vg` CLI arguments. Defaults are the S11 values, so existing usage is unchanged.

```bash
# ── In the repo root ──────────────────────────────────────────────────────────

source ../PyEnv/bin/activate

# Step 1: Export EDC Gamma data (reads Data/edc_gamma_{id}/combined.h5)
python scripts/export_edc_gamma_data.py --id S3_2a --sample S3 --vg 0.0115 --phig 175

# Step 2: Export moire bands
python scripts/export_moire_bands.py --sample S3 --Vg 11.5 --w1p -1.2 --w1d 0.455 --phiG 175

# Step 3: Export EDC vs V_G
python scripts/export_edc_vs_V.py --sample S3 --w1p -1.2 --w1d 0.455 --phiG 175

# Step 4: Export LDOS
python scripts/export_ldos.py --sample S3 --Vg 0.0115 --phiG 175 --w1p -1.2 --w1d 0.455 \
    --e-min -0.85 --e-max -0.55

# Step 5: Export bilayer bands (n_shells=0, no moire)
python scripts/export_bilayer_bands.py --sample S3
```

## Quick start — plotting

```bash
# S11 — 4Dsm8a (S11, 2.8 deg)
python plot_distance_heatmaps.py  data/edc_gamma_4Dsm8a_Vg_10.5meV_phiG_176deg.npz
python plot_distance_w_heatmap.py data/edc_gamma_4Dsm8a_Vg_10.5meV_phiG_176deg.npz
python plot_edc_profile.py        data/edc_gamma_4Dsm8a_Vg_10.5meV_phiG_176deg.npz

# S11 — 4Dsm8b (S11_m25, 2.5 deg)
python plot_distance_heatmaps.py  data/edc_gamma_4Dsm8b_Vg_12.5meV_phiG_176deg.npz
python plot_distance_w_heatmap.py data/edc_gamma_4Dsm8b_Vg_12.5meV_phiG_176deg.npz
python plot_edc_profile.py        data/edc_gamma_4Dsm8b_Vg_12.5meV_phiG_176deg.npz

# S11 — 4Dsm8c (S11_p31, 3.1 deg)
python plot_distance_heatmaps.py  data/edc_gamma_4Dsm8c_Vg_8.5meV_phiG_176deg.npz
python plot_distance_w_heatmap.py data/edc_gamma_4Dsm8c_Vg_8.5meV_phiG_176deg.npz
python plot_edc_profile.py        data/edc_gamma_4Dsm8c_Vg_8.5meV_phiG_176deg.npz

# S3 — S3_2a (S3, 1.8 deg)
python plot_distance_heatmaps.py  data/edc_gamma_S3_2a_S3_Vg_11.5meV_phiG_175deg.npz
python plot_distance_w_heatmap.py data/edc_gamma_S3_2a_S3_Vg_11.5meV_phiG_175deg.npz
python plot_edc_profile.py        data/edc_gamma_S3_2a_S3_Vg_11.5meV_phiG_175deg.npz

# EDC vs V and moire bands
python plot_edc_vs_V.py data/edc_vs_V_n20_Vg1-20.npz       # S11
python plot_edc_vs_V.py data/edc_vs_V_S3_n20_Vg1-20.npz    # S3
python plot_moire_bands.py data/moire_bands_k301_n2_Vg0_10.5.npz       # S11
python plot_moire_bands.py data/moire_bands_S3_k301_n2_Vg0_11.5.npz    # S3

# LDOS
python plot_ldos.py data/ldos_S11_n2_k10_10.5meV_170deg.npz
python plot_ldos.py data/ldos_S3_n2_k10_11.5meV_175deg.npz
```

## Distance metrics in heatmaps

Each grid point in the EDC sweep produces a 3-Lorentzian fit with peak centers c1, c2, c3 (in eV). The heatmaps show, for each (Vg, phiG) cell, the minimum distance (over w1p, w1d) among points passing the cutoffs.

**L1 distance** — absolute position error of fitted peaks vs experiment:

```
dist = |c1 - E1_exp| + |c2 - E2_exp| + |c3 - E3_exp|
```

Example for 4Dsm8a at Vg=10.5 meV, phiG=176°:
```
|c1 - (-1.1599)| + |c2 - (-1.2531)| + |c3 - (-1.8200)|
= |-1.1594 + 1.1599| + |-1.2527 + 1.2531| + |-1.8195 + 1.8200|
= 0.0005 + 0.0004 + 0.0005 = 1.35 meV
```

S3 example (Vg=11.5 meV, phiG=175°):
```
|c1 - (-0.69484)| + |c2 - (-0.77307)| + |c3 - (-1.35)|
= |-0.6903 + 0.69484| + |-0.7678 + 0.77307| + |-1.3453 + 1.35|
= 0.0045 + 0.0053 + 0.0047 = 14.5 meV
```

**L2 distance** — error in peak separations (insensitive to global energy shifts):

```
dist_sep = ||c1 - c2| - |E1_exp - E2_exp|| + ||c1 - c3| - |E1_exp - E3_exp||
```

4Dsm8a example:
```
|0.0933 - 0.0932| + |0.6601 - 0.6601|
= 0.0001 + 0.0000 = 0.12 meV
```

Experimental separations:
- S11: |E1 - E2| = 93.2 meV, |E1 - E3| = 660.1 meV
- S3:  |E1 - E2| = 78.23 meV, |E1 - E3| = 655.16 meV

## Plots

### Distance heatmaps over Vg/phiG

```bash
python plot_distance_heatmaps.py <data.npz> [--output-dir ./figures]
# -> figures/distance_heatmap_{run_id}.png
```

1x2 panels: L1 and L2 distance over (Vg, phiG). Vg range 0-20 meV with dashed lines every 2 meV, vertical refs at 60/180/300 deg. Red star at global best.

### Distance heatmap over w1p/w1d

```bash
python plot_distance_w_heatmap.py <data.npz> [--output-dir ./figures]
# -> figures/distance_w_heatmap_{run_id}.png
```

1x2 panels: L1 and L2 distance over (w1p, w1d). Red star at global best.

### EDC intensity profile

```bash
python plot_edc_profile.py <data.npz> [--output-dir ./figures]
# -> figures/edc_profile_4L_{run_id}.png
```

EDC intensity curve + 4-Lorentzian total fit + experimental ARPES reference lines. Requires `.npz` with EDC profile data.

### EDC TVB-side band distance vs V_G

```bash
python plot_edc_vs_V.py <data.npz> [--output-dir ./figures]
# -> figures/edc_vs_V.png
```

TVB-side band distance (meV) vs V_G (meV). Black markers + red dashed line at ARPES distance.

### Moire bands around Gamma

```bash
python plot_moire_bands.py <data.npz> [--output-dir ./figures]
# -> figures/moire_bands_gamma.png
```

1x2 panels: TVB bands along k in [-0.4, 0.4] A^-1 for two V_G values. Thin gray lines + weight-proportional blue circles. Y-axis auto-scales from the data (works for both S11 and S3).

### LDOS in real space

```bash
python plot_ldos.py <data.npz> [--output-dir ./figures]
# -> figures/ldos.png
```

pcolormesh of LDOS(r, E) along the a1+a2 moire diagonal. Energy on x-axis, position on y-axis (inverted) with stacking site labels (W/W, Se/W, W/S, W/W). Hot colormap, "low"/"high" colorbar, title with all Hamiltonian parameters.
