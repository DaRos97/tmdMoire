"""Analyze EDC Gamma grid results.

Loads combined.h5 from a run directory, computes distance from experimental
peak positions, and produces a 2D heatmap of minimum distance over (Vg, phiG)
with the global best-fit point marked and interlayer parameters shown.

Usage:
    python scripts/analyze_edc_gamma.py --id 001
    python scripts/analyze_edc_gamma.py --id 001 --cutoff 0.030
    python scripts/analyze_edc_gamma.py --id 001 --output Figures/edc_gamma_analysis.png
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tmdmoire import EDC_G_POSITIONS

# ─── Parse arguments ─────────────────────────────────────────────────────────

run_id = "default"
output = None
cutoff_ev = 0.026  # 26 meV default

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--id" and i + 1 < len(args):
        run_id = args[i + 1]
        i += 2
    elif args[i] == "--output" and i + 1 < len(args):
        output = Path(args[i + 1])
        i += 2
    elif args[i] == "--cutoff" and i + 1 < len(args):
        cutoff_ev = float(args[i + 1])
        i += 2
    else:
        i += 1

# ─── Load combined data ──────────────────────────────────────────────────────

run_dir = Path("Data") / f"edc_gamma_{run_id}"
combined_fn = run_dir / "combined.h5"

if not combined_fn.exists():
    print(f"Combined file not found: {combined_fn}")
    print("Run: python scripts/combine_edc_chunks.py --id {run_id}")
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

# ─── Load metadata ───────────────────────────────────────────────────────────

meta_fn = run_dir / "metadata.json"
with open(meta_fn) as f:
    meta = json.load(f)

grid_cfg = meta["grid_config"]

# ─── Determine actual parameter bounds from data ─────────────────────────────
# (metadata.json range_ev may be inaccurate; use observed min/max)

param_arrays = {"w1p": w1p, "w1d": w1d, "w2p": w2p, "w2d": w2d}
bounds = {}
for pname, parr in param_arrays.items():
    lo = float(np.nanmin(parr))
    hi = float(np.nanmax(parr))
    bounds[pname] = (lo, hi)
    center = (lo + hi) / 2
    half_range = (hi - lo) / 2
    print(f"  {pname}: [{lo:+.4f}, {hi:+.4f}] (center {center:+.4f}, ±{half_range:.4f})")

tol = 1e-4  # tolerance for "at bound" comparison

n_points = len(Vg)
print(f"Loaded {n_points} points from {combined_fn}")

# ─── Compute distance ────────────────────────────────────────────────────────

exp = EDC_G_POSITIONS["S11"]  # [-1.1599, -1.2531, -1.82]

# Only use points where all 3 peaks were fitted
valid = ~np.isnan(c1) & ~np.isnan(c2) & ~np.isnan(c3)
n_valid = valid.sum()
print(f"Valid fits: {n_valid} / {n_points}")

dist = np.full(n_points, np.nan)
dist[valid] = (
    np.abs(c1[valid] - exp[0])
    + np.abs(c2[valid] - exp[1])
    + np.abs(c3[valid] - exp[2])
)

# ─── Apply cutoff ────────────────────────────────────────────────────────────

above_cutoff = dist > cutoff_ev
dist[above_cutoff] = np.nan
n_cutoff = above_cutoff.sum()
n_within_cutoff = np.sum(~np.isnan(dist))
print(f"Points within cutoff: {n_within_cutoff} / {n_points}")

if n_within_cutoff == 0:
    print("No points within cutoff. Exiting.")
    sys.exit(0)

# ─── Find global minimum ─────────────────────────────────────────────────────

idx_best = np.nanargmin(dist)
print(f"\nGlobal minimum distance: {dist[idx_best]*1000:.2f} meV")
print(f"  Vg   = {Vg[idx_best]*1000:.1f} meV")
print(f"  phiG = {phiG[idx_best]:.1f} deg")
print(f"  w1p  = {w1p[idx_best]:+.4f} eV")
print(f"  w1d  = {w1d[idx_best]:+.4f} eV")
print(f"  w2p  = {w2p[idx_best]:+.4f} eV")
print(f"  w2d  = {w2d[idx_best]:+.4f} eV")
print(f"  c1   = {c1[idx_best]:.4f} eV (exp: {exp[0]:.4f} eV)")
print(f"  c2   = {c2[idx_best]:.4f} eV (exp: {exp[1]:.4f} eV)")
print(f"  c3   = {c3[idx_best]:.4f} eV (exp: {exp[2]:.4f} eV)")

# ─── Aggregate: min distance over (Vg, phiG) grid ───────────────────────────

# Get unique sorted values
phiG_vals = np.unique(phiG)
Vg_vals = np.unique(Vg)
n_phi = len(phiG_vals)
n_Vg = len(Vg_vals)

print(f"\nGrid: {n_Vg} Vg values x {n_phi} phiG values")

# Build 2D array: for each (Vg, phiG) cell, take min over all interlayer combos
dist_2d = np.full((n_Vg, n_phi), np.nan)
bounds_2d = [[[] for _ in range(n_phi)] for _ in range(n_Vg)]
param_2d = {pname: np.full((n_Vg, n_phi), np.nan) for pname in param_arrays}

for iv, vg in enumerate(Vg_vals):
    for ip, pg in enumerate(phiG_vals):
        mask = (Vg == vg) & (phiG == pg) & ~np.isnan(dist)
        if mask.any():
            idx_min = np.nanargmin(dist[mask])
            idx_global = np.where(mask)[0][idx_min]
            dist_2d[iv, ip] = dist[idx_global]
            for pname, parr in param_arrays.items():
                val = parr[idx_global]
                param_2d[pname][iv, ip] = val
                lo, hi = bounds[pname]
                if abs(val - lo) < tol:
                    bounds_2d[iv][ip].append((pname, "lo"))
                elif abs(val - hi) < tol:
                    bounds_2d[iv][ip].append((pname, "hi"))

# ─── Plot ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# 2D heatmap
im = ax.pcolormesh(
    phiG_vals, Vg_vals * 1000, dist_2d * 1000,
    cmap="viridis_r", shading="auto",
)

# Mark global minimum
ax.scatter(
    phiG[idx_best], Vg[idx_best] * 1000,
    marker="*", s=200, c="red", edgecolors="white", linewidths=1.5,
    zorder=5, label="Best fit",
)

# Legend with interlayer params
legend_text = (
    f"Best fit (dist = {dist[idx_best]*1000:.1f} meV)\n"
    f"w1p = {w1p[idx_best]:+.4f} eV\n"
    f"w1d = {w1d[idx_best]:+.4f} eV\n"
    f"w2p = {w2p[idx_best]:+.4f} eV\n"
    f"w2d = {w2d[idx_best]:+.4f} eV"
)
from matplotlib.patches import Patch

bound_colors = {"w1p": "red", "w1d": "blue", "w2p": "green", "w2d": "orange"}
bound_styles = {"w1p": "-", "w1d": "--", "w2p": "-.", "w2d": ":"}
bound_widths = {"lo": 2, "hi": 4}

all_legend_handles = [
    Patch(facecolor="none", edgecolor="none", label=legend_text),
]
for p in ["w1p", "w1d", "w2p", "w2d"]:
    all_legend_handles.append(
        Patch(facecolor="none", edgecolor=bound_colors[p], linewidth=bound_widths["lo"],
              linestyle=bound_styles[p], label=f"{p} at lower bound")
    )
    all_legend_handles.append(
        Patch(facecolor="none", edgecolor=bound_colors[p], linewidth=bound_widths["hi"],
              linestyle=bound_styles[p], label=f"{p} at upper bound")
    )
ax.legend(
    handles=all_legend_handles,
    loc="upper left", fontsize=9, framealpha=0.9,
)

# ─── Draw borders for cells at parameter bounds ─────────────────────────────

dVg = (Vg_vals[1] - Vg_vals[0]) * 1000 if len(Vg_vals) > 1 else 1
dphiG = (phiG_vals[1] - phiG_vals[0]) if len(phiG_vals) > 1 else 1

for iv in range(n_Vg):
    for ip in range(n_phi):
        if not bounds_2d[iv][ip]:
            continue
        for pname, side in bounds_2d[iv][ip]:
            rect = Rectangle(
                (phiG_vals[ip] - dphiG / 2, Vg_vals[iv] * 1000 - dVg / 2),
                dphiG, dVg,
                fill=False, edgecolor=bound_colors[pname],
                linewidth=bound_widths[side], linestyle=bound_styles[pname],
                zorder=4,
            )
            ax.add_patch(rect)

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Min distance (meV)", fontsize=11)

ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
ax.set_ylabel(r"$V_G$ (meV)", fontsize=12)
ax.set_title(
    f"EDC Gamma: min distance over interlayer params\n"
    f"Run: {run_id}  |  {n_within_cutoff}/{n_points} within {cutoff_ev*1000:.0f} meV cutoff",
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

# ─── Plot interlayer parameters as function of (Vg, phiG) ───────────────────

param_names = ["w1p", "w1d", "w2p", "w2d"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.flatten()

fit_values = {p: (bounds[p][0] + bounds[p][1]) / 2 for p in param_names}

for idx, pname in enumerate(param_names):
    ax = axes[idx]
    im = ax.pcolormesh(
        phiG_vals, Vg_vals * 1000, param_2d[pname] * 1000,
        cmap="viridis", shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"{pname} (meV)", fontsize=10)

    # Mark global minimum
    ax.scatter(
        phiG[idx_best], Vg[idx_best] * 1000,
        marker="*", s=150, c="red", edgecolors="white", linewidths=1.5,
        zorder=5,
    )

    # Draw bound rectangles
    for iv in range(n_Vg):
        for ip in range(n_phi):
            cell_params = [p for p, _ in bounds_2d[iv][ip]]
            if pname not in cell_params:
                continue
            side = [s for p, s in bounds_2d[iv][ip] if p == pname][0]
            rect = Rectangle(
                (phiG_vals[ip] - dphiG / 2, Vg_vals[iv] * 1000 - dVg / 2),
                dphiG, dVg,
                fill=False, edgecolor=bound_colors[pname],
                linewidth=bound_widths[side], linestyle=bound_styles[pname],
                zorder=4,
            )
            ax.add_patch(rect)

    ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=11)
    ax.set_ylabel(r"$V_G$ (meV)", fontsize=11)
    ax.set_title(f"{pname}  (center = {fit_values[pname]*1000:.1f} meV)", fontsize=12)

fig.suptitle(
    f"Interlayer parameters at min distance\n"
    f"Run: {run_id}",
    fontsize=13, y=1.02,
)

param_output = run_dir / "analysis_params.png"
fig.savefig(param_output, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Parameter figure saved to {param_output}")
