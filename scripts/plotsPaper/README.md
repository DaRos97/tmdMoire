# Plots for paper

Standalone plotting scripts for EDC Gamma analysis, EDC vs V_G, and moire band structure. No tmdmoire environment needed — only numpy + matplotlib.

Run all commands from `scripts/plotsPaper/`. Output goes to `figures/`.

## Dataset reference

| ID | Sample | Twist | Best Vg | Best phiG | w1p | w1d | L2 |
|----|--------|:-----:|:-------:|:---------:|------|------|----:|
| 4Dsm8a | S11 | 2.8° | 10.5 meV | 176° | -1.220 | +0.460 | 0.12 meV |
| 4Dsm8b | S11_m25 | 2.5° | 12.5 meV | 176° | -1.190 | +0.445 | 0.68 meV |
| 4Dsm8c | S11_p31 | 3.1° | 8.5 meV | 176° | -1.185 | +0.445 | 0.67 meV |

Fixed interlayer: w2p = -0.1694 eV, w2d = +0.0215 eV (from Step 2).  
Experimental EDC positions: -1.1599, -1.2531, -1.8200 eV.  
Filters applied: L1 < 26 meV, L2 < 10 meV, a2/a1 >= 0.

## Quick start

```bash
# 4Dsm8a (S11, 2.8 deg)
python plot_distance_heatmaps.py  data/edc_gamma_4Dsm8a_Vg_10.5meV_phiG_176deg.npz
python plot_distance_w_heatmap.py data/edc_gamma_4Dsm8a_Vg_10.5meV_phiG_176deg.npz
python plot_edc_profile.py        data/edc_gamma_4Dsm8a_Vg_10.5meV_phiG_176deg.npz

# 4Dsm8b (S11_m25, 2.5 deg)
python plot_distance_heatmaps.py  data/edc_gamma_4Dsm8b_Vg_12.5meV_phiG_176deg.npz
python plot_distance_w_heatmap.py data/edc_gamma_4Dsm8b_Vg_12.5meV_phiG_176deg.npz
python plot_edc_profile.py        data/edc_gamma_4Dsm8b_Vg_12.5meV_phiG_176deg.npz

# 4Dsm8c (S11_p31, 3.1 deg)
python plot_distance_heatmaps.py  data/edc_gamma_4Dsm8c_Vg_8.5meV_phiG_176deg.npz
python plot_distance_w_heatmap.py data/edc_gamma_4Dsm8c_Vg_8.5meV_phiG_176deg.npz
python plot_edc_profile.py        data/edc_gamma_4Dsm8c_Vg_8.5meV_phiG_176deg.npz

# EDC vs V and moire bands
python plot_edc_vs_V.py data/edc_vs_V_n20_Vg1-20.npz
python plot_moire_bands.py data/moire_bands_k301_n2_Vg0_10.5.npz
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

**L2 distance** — error in peak separations (insensitive to global energy shifts):

```
dist_sep = ||c1 - c2| - |E1_exp - E2_exp|| + ||c1 - c3| - |E1_exp - E3_exp||
```

Example for 4Dsm8a at Vg=10.5 meV, phiG=176°:
```
|0.0933 - 0.0932| + |0.6601 - 0.6601|
= 0.0001 + 0.0000 = 0.12 meV
```

(Experimental separations: |E1 - E2| = 93.2 meV, |E1 - E3| = 660.1 meV)

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

1x2 panels: TVB bands along k in [-0.4, 0.4] A^-1 for two V_G values. Thin gray lines + weight-proportional blue circles.
