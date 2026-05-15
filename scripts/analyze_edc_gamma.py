"""Analyze EDC Gamma grid results.

Loads combined.h5 from a run directory, computes distance from experimental
peak positions, and produces a 2D heatmap of minimum distance over (Vg, phiG)
with the global best-fit point marked and interlayer parameters shown.

Usage:
    python scripts/analyze_edc_gamma.py --run-id 001
    python scripts/analyze_edc_gamma.py --run-id 001 --output Figures/edc_gamma_analysis.png
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tmdmoire import EDC_G_POSITIONS

# ─── Parse arguments ─────────────────────────────────────────────────────────

run_id = "default"
output = None

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--run-id" and i + 1 < len(args):
        run_id = args[i + 1]
        i += 2
    elif args[i] == "--output" and i + 1 < len(args):
        output = Path(args[i + 1])
        i += 2
    else:
        i += 1

# ─── Load combined data ──────────────────────────────────────────────────────

run_dir = Path("Data") / f"edc_grid_gamma_run_{run_id}"
combined_fn = run_dir / "combined.h5"

if not combined_fn.exists():
    print(f"Combined file not found: {combined_fn}")
    print("Run: python scripts/combine_edc_chunks.py --run-id {run_id}")
    sys.exit(1)

with h5py.File(combined_fn, "r") as f:
    Vg = f["Vg"][:]
    phiG = f["phiG"][:]
    w1p = f["w1p"][:]
    w1d = f["w1d"][:]
    w2p = f["w2p"][:]
    w2d = f["w2d"][:]
    c1 = f["c1"][:]
    c2 = f["c2"][:]
    c3 = f["c3"][:]
    c4 = f["c4"][:]

n_points = len(Vg)
print(f"Loaded {n_points} points from {combined_fn}")

# ─── Compute distance ────────────────────────────────────────────────────────

exp = EDC_G_POSITIONS["S11"]  # [-1.1599, -1.2531, -1.82]

# Only use points where all 3 peaks were fitted
valid = ~np.isnan(c1) & ~np.isnan(c2) & ~np.isnan(c3)
n_valid = valid.sum()
print(f"Valid fits: {n_valid} / {n_points}")

dist = np.full(n_points, np.nan)
dist[valid] = np.sqrt(
    (c1[valid] - exp[0]) ** 2
    + (c2[valid] - exp[1]) ** 2
    + (c3[valid] - exp[2]) ** 2
) / 3.0

# ─── Find global minimum ─────────────────────────────────────────────────────

idx_best = np.nanargmin(dist)
print(f"\nGlobal minimum distance: {dist[idx_best]:.4f} eV")
print(f"  Vg   = {Vg[idx_best]:.4f} eV")
print(f"  phiG = {phiG[idx_best]:.1f} deg")
print(f"  w1p  = {w1p[idx_best]:.4f} eV")
print(f"  w1d  = {w1d[idx_best]:.4f} eV")
print(f"  w2p  = {w2p[idx_best]:.4f} eV")
print(f"  w2d  = {w2d[idx_best]:.4f} eV")
print(f"  c1   = {c1[idx_best]:.4f} eV (exp: {exp[0]:.4f})")
print(f"  c2   = {c2[idx_best]:.4f} eV (exp: {exp[1]:.4f})")
print(f"  c3   = {c3[idx_best]:.4f} eV (exp: {exp[2]:.4f})")
print(f"  c4   = {c4[idx_best]:.4f} eV")

# ─── Aggregate: min distance over (Vg, phiG) grid ───────────────────────────

# Get unique sorted values
phiG_vals = np.unique(phiG)
Vg_vals = np.unique(Vg)
n_phi = len(phiG_vals)
n_Vg = len(Vg_vals)

print(f"\nGrid: {n_Vg} Vg values x {n_phi} phiG values")

# Build 2D array: for each (Vg, phiG) cell, take min over all interlayer combos
dist_2d = np.full((n_Vg, n_phi), np.nan)

for iv, vg in enumerate(Vg_vals):
    for ip, pg in enumerate(phiG_vals):
        mask = (Vg == vg) & (phiG == pg) & valid
        if mask.any():
            dist_2d[iv, ip] = np.min(dist[mask])

# ─── Plot ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# 2D heatmap
im = ax.pcolormesh(
    phiG_vals, Vg_vals * 1000, dist_2d,
    cmap="viridis_r", shading="auto",
    norm=LogNorm(vmin=np.nanmin(dist_2d) * 0.9, vmax=np.nanmax(dist_2d) * 1.1),
)

# Mark global minimum
ax.scatter(
    phiG[idx_best], Vg[idx_best] * 1000,
    marker="*", s=200, c="red", edgecolors="white", linewidths=1.5,
    zorder=5, label="Best fit",
)

# Legend with interlayer params
legend_text = (
    f"Best fit (dist = {dist[idx_best]:.4f} eV)\n"
    f"w1p = {w1p[idx_best]:+.4f} eV\n"
    f"w1d = {w1d[idx_best]:+.4f} eV\n"
    f"w2p = {w2p[idx_best]:+.4f} eV\n"
    f"w2d = {w2d[idx_best]:+.4f} eV"
)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="none", edgecolor="none"),
]
ax.legend(
    handles=[Patch(facecolor="none", edgecolor="none", label=legend_text)],
    loc="upper right", fontsize=9, framealpha=0.9,
    handlelength=0, handletextpad=0,
)

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Min distance (eV)", fontsize=11)

ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
ax.set_ylabel(r"$V_G$ (meV)", fontsize=12)
ax.set_title(
    f"EDC Gamma: min distance over interlayer params\n"
    f"Run: {run_id}  |  {n_valid} valid fits",
    fontsize=12,
)

# Save
if output is None:
    out_dir = run_dir
    output = out_dir / "analysis.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nFigure saved to {output}")
