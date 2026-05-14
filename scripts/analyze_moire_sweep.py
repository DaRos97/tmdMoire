"""Analyze moire parameter sweep results.

Aggregates all chunk results, computes max absolute error vs ARPES peaks,
and plots 2D heatmap of accuracy as function of V and phi.

Usage: python scripts/analyze_moire_sweep.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tmdmoire import (
    get_repo_root, get_filename,
    TWIST_ANGLES, EDC_G_POSITIONS,
)

master_folder = get_repo_root()
n_chunks = 128

BILAYER_DATA_DIR = "/home/users/r/rossid/bilayer_v2.0/Data/"

sample = "S11"
theta_deviation = 0
n_shells = 2
spreadE = 0.03

arpes_peaks = EDC_G_POSITIONS[sample]
print(f"ARPES reference peaks (eV): {arpes_peaks}")

interlayer_params_path = master_folder + "/Inputs/bilayer_fitting/interlayer_params.npy"
interlayer_vals = np.load(interlayer_params_path)
w1p_start = interlayer_vals[0]
w1d_start = interlayer_vals[1]
w2p_start = interlayer_vals[2]
w2d_start = interlayer_vals[3]

listV = np.linspace(0.001, 0.040, 20)
listPhi = np.linspace(160, 180, 11) / 180 * np.pi
listW1p = np.linspace(w1p_start - 0.050, w1p_start + 0.050, 11)
listW1d = np.linspace(w1d_start - 0.050, w1d_start + 0.050, 11)
listW2p = np.linspace(w2p_start - 0.050, w2p_start + 0.050, 11)
listW2d = np.linspace(w2d_start - 0.050, w2d_start + 0.050, 11)

filename = get_filename(
    (listV[0], listV[-1], len(listV),
     int(listPhi[0] / np.pi * 180), int(listPhi[-1] / np.pi * 180), len(listPhi),
     listW1p[0], listW1p[-1], len(listW1p),
     listW1d[0], listW1d[-1], len(listW1d),
     listW2p[0], listW2p[-1], len(listW2p),
     listW2d[0], listW2d[-1], len(listW2d))
)

dirname = get_filename(
    ("edcGSweep", theta_deviation, n_shells, spreadE),
    dirname=BILAYER_DATA_DIR,
    float_precision=4,
) + "_" + filename + "/"

print(f"Loading results from: {dirname}")

all_results = []
loaded_chunks = 0
for i in range(n_chunks):
    fpath = dirname + f"chunk_{i}_{n_chunks}.h5"
    if Path(fpath).exists():
        df = pd.read_hdf(fpath, key="results")
        all_results.append(df)
        loaded_chunks += 1
        print(f"  Loaded chunk {i}: {len(df)} points")
    else:
        print(f"  Missing chunk {i}")

if not all_results:
    print("No results found. Run sweep_moire_params.py first.")
    sys.exit(1)

df_all = pd.concat(all_results, ignore_index=True)
print(f"\nTotal points loaded: {len(df_all)}")

valid_mask = df_all[["p1", "p2", "p3"]].notna().all(axis=1)
df_valid = df_all[valid_mask].copy()
print(f"Valid fits: {len(df_valid)} / {len(df_all)} ({100*len(df_valid)/len(df_all):.1f}%)")

errors = np.abs(df_valid[["p1", "p2", "p3"]].values - arpes_peaks[np.newaxis, :])
max_errors = np.max(errors, axis=1)
df_valid["max_error"] = max_errors

n_v = len(listV)
n_phi = len(listPhi)

error_map = np.full((n_v, n_phi), np.nan)
best_w1p_map = np.full((n_v, n_phi), np.nan)
best_w1d_map = np.full((n_v, n_phi), np.nan)
best_w2p_map = np.full((n_v, n_phi), np.nan)
best_w2d_map = np.full((n_v, n_phi), np.nan)
best_p1_map = np.full((n_v, n_phi), np.nan)
best_p2_map = np.full((n_v, n_phi), np.nan)
best_p3_map = np.full((n_v, n_phi), np.nan)

for iv, v in enumerate(listV):
    for ip, phi in enumerate(listPhi):
        mask = (np.abs(df_valid["V"].values - v) < 1e-6) & (np.abs(df_valid["phi"].values - phi) < 1e-6)
        subset = df_valid[mask]
        if len(subset) > 0:
            idx_min = subset["max_error"].idxmin()
            best = subset.loc[idx_min]
            error_map[iv, ip] = best["max_error"]
            best_w1p_map[iv, ip] = best["w1p"]
            best_w1d_map[iv, ip] = best["w1d"]
            best_w2p_map[iv, ip] = best["w2p"]
            best_w2d_map[iv, ip] = best["w2d"]
            best_p1_map[iv, ip] = best["p1"]
            best_p2_map[iv, ip] = best["p2"]
            best_p3_map[iv, ip] = best["p3"]

V_mesh, phi_mesh = np.meshgrid(listPhi * 180 / np.pi, listV)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

im0 = axes[0, 0].pcolormesh(phi_mesh, V_mesh * 1000, error_map, shading="auto", cmap="viridis_r")
fig.colorbar(im0, ax=axes[0, 0], label="Max absolute error (eV)")
axes[0, 0].set_xlabel("phi (deg)")
axes[0, 0].set_ylabel("V (meV)")
axes[0, 0].set_title("Max absolute error vs ARPES")

for map_2d, title, unit, ax_idx in [
    (best_w1p_map, "Best w1p", "eV", 0),
    (best_w1d_map, "Best w1d", "eV", 1),
    (best_w2p_map, "Best w2p", "eV", 2),
    (best_w2d_map, "Best w2d", "eV", 3),
    (best_p1_map, "Best p1 (TVB)", "eV", 4),
]:
    row = (ax_idx + 1) // 3
    col = (ax_idx + 1) % 3
    if ax_idx == 4:
        continue
    im = axes[row, col].pcolormesh(phi_mesh, V_mesh * 1000, map_2d, shading="auto", cmap="plasma")
    fig.colorbar(im, ax=axes[row, col], label=unit)
    axes[row, col].set_xlabel("phi (deg)")
    axes[row, col].set_ylabel("V (meV)")
    axes[row, col].set_title(title)

plt.tight_layout()

output_dir = master_folder + "/Data/Figures/"
os.makedirs(output_dir, exist_ok=True)
fig_path = output_dir + "moire_sweep_accuracy.png"
fig.savefig(fig_path, dpi=200)
print(f"\nSaved figure: {fig_path}")

output_data = {
    "listV": listV,
    "listPhi": listPhi,
    "error_map": error_map,
    "best_w1p_map": best_w1p_map,
    "best_w1d_map": best_w1d_map,
    "best_w2p_map": best_w2p_map,
    "best_w2d_map": best_w2d_map,
    "best_p1_map": best_p1_map,
    "best_p2_map": best_p2_map,
    "best_p3_map": best_p3_map,
    "arpes_peaks": arpes_peaks,
    "n_valid": len(df_valid),
    "n_total": len(df_all),
}
output_path = master_folder + "/Data/moire_sweep_results.npz"
np.savez(output_path, **output_data)
print(f"Saved data: {output_path}")

min_error = np.nanmin(error_map)
min_idx = np.unravel_index(np.nanargmin(error_map), error_map.shape)
best_V = listV[min_idx[0]] * 1000
best_phi = listPhi[min_idx[1]] * 180 / np.pi
print(f"\nBest fit: V={best_V:.1f} meV, phi={best_phi:.1f} deg, max_error={min_error:.4f} eV")
print(f"  Best couplings: w1p={best_w1p_map[min_idx]:.4f}, w1d={best_w1d_map[min_idx]:.4f}, w2p={best_w2p_map[min_idx]:.4f}, w2d={best_w2d_map[min_idx]:.4f}")
print(f"  Best peaks: p1={best_p1_map[min_idx]:.4f}, p2={best_p2_map[min_idx]:.4f}, p3={best_p3_map[min_idx]:.4f}")
