"""Analyze EDC Gamma grid results.

Loads combined.h5 from a run directory, computes distance from experimental
peak positions, and produces a 2D heatmap of minimum distance over (Vg, phiG)
with the global best-fit point marked and interlayer parameters shown.

Selection mode (--vg/--phig): highlights the chosen cell on the heatmap,
prints its details to stdout, and produces an EDC intensity profile plot
with the Lorentzian fit overlaid.

Usage:
    python scripts/analyze_edc_gamma.py --id 001
    python scripts/analyze_edc_gamma.py --id 001 --cutoff 0.030
    python scripts/analyze_edc_gamma.py --id 001 --ratio-cutoff 0.15
    python scripts/analyze_edc_gamma.py --id 001 --output Figures/edc_gamma_analysis.png
    python scripts/analyze_edc_gamma.py --id sm03 --vg 0.012 --phig 176
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
from matplotlib.patches import Rectangle, Patch

from tmdmoire import EDC_G_POSITIONS, TMDMaterial, MoireGeometry, MoireHamiltonian
from tmdmoire import TWIST_ANGLES, ENERGY_OFFSETS
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

# ─── Parse arguments ─────────────────────────────────────────────────────────

run_id = "default"
output = None
cutoff_ev = 0.026  # 26 meV default
ratio_cutoff = 0.1  # 10% default
vg_selected = None
phig_selected = None

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
    elif args[i] == "--ratio-cutoff" and i + 1 < len(args):
        ratio_cutoff = float(args[i + 1])
        i += 2
    elif args[i] == "--vg" and i + 1 < len(args):
        vg_selected = float(args[i + 1])
        i += 2
    elif args[i] == "--phig" and i + 1 < len(args):
        phig_selected = float(args[i + 1])
        i += 2
    else:
        i += 1

have_selection = vg_selected is not None and phig_selected is not None
if have_selection:
    print(f"Selection mode: Vg = {vg_selected*1000:.0f} meV, phiG = {phig_selected:.0f} deg")

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
    a1 = f["a1"][:]
    a2 = f["a2"][:]
    a3 = f["a3"][:]
    g1 = f["g1"][:]
    g2 = f["g2"][:]
    g3 = f["g3"][:]
    redchi = f["redchi"][:]

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

# ─── Apply cutoffs ───────────────────────────────────────────────────────────

above_cutoff = dist > cutoff_ev
dist[above_cutoff] = np.nan
n_cutoff = above_cutoff.sum()
n_within_cutoff = np.sum(~np.isnan(dist))
print(f"Points within distance cutoff: {n_within_cutoff} / {n_points}")

ratio = np.full(n_points, np.nan)
ratio[valid & ~np.isnan(dist)] = a2[valid & ~np.isnan(dist)] / a1[valid & ~np.isnan(dist)]

below_ratio_cutoff = ~np.isnan(ratio) & (ratio < ratio_cutoff)
dist[below_ratio_cutoff] = np.nan
n_below_ratio = below_ratio_cutoff.sum()
n_after_ratio = np.sum(~np.isnan(dist))
print(f"Points above ratio cutoff a2/a1 >= {ratio_cutoff}: {n_after_ratio} / {n_points}")

if n_after_ratio == 0:
    print("No points pass both cutoffs. Exiting.")
    sys.exit(0)

# ─── Selection mode: find specific (Vg, phiG) cell ───────────────────────────

idx_selected = None
if have_selection:
    tol_vg = 1e-6
    tol_phig = 1e-6
    mask_sel = (
        np.abs(Vg - vg_selected) < tol_vg
    ) & (
        np.abs(phiG - phig_selected) < tol_phig
    ) & ~np.isnan(dist)

    if not mask_sel.any():
        vg_vals = sorted(set(Vg))
        pg_vals = sorted(set(phiG))
        print(f"No valid fits at Vg={vg_selected*1000:.0f} meV, phiG={phig_selected:.0f} deg")
        print(f"Available Vg [meV]: {[v*1000 for v in vg_vals]}")
        print(f"Available phiG [deg]: {pg_vals}")
        if have_selection:
            print("Exiting due to invalid selection.")
            sys.exit(1)

    idx_sel_local = np.nanargmin(dist[mask_sel])
    idx_selected = np.where(mask_sel)[0][idx_sel_local]

    print(f"\n{'─'*60}")
    print(f"Selected cell: Vg = {Vg[idx_selected]*1000:.1f} meV, phiG = {phiG[idx_selected]:.1f} deg")
    print(f"  w1p  = {w1p[idx_selected]:+.4f} eV")
    print(f"  w1d  = {w1d[idx_selected]:+.4f} eV")
    print(f"  w2p  = {w2p[idx_selected]:+.4f} eV")
    print(f"  w2d  = {w2d[idx_selected]:+.4f} eV")
    print(f"  c1   = {c1[idx_selected]:.4f} eV (exp: {exp[0]:.4f} eV)")
    print(f"  c2   = {c2[idx_selected]:.4f} eV (exp: {exp[1]:.4f} eV)")
    print(f"  c3   = {c3[idx_selected]:.4f} eV (exp: {exp[2]:.4f} eV)")
    print(f"  a1   = {a1[idx_selected]:.4f}")
    print(f"  a2   = {a2[idx_selected]:.4f}")
    print(f"  a3   = {a3[idx_selected]:.4f}")
    print(f"  redchi = {redchi[idx_selected]:.6f}")
    print(f"  a2/a1 = {ratio[idx_selected]:.4f}")
    print(f"  distance = {dist[idx_selected]*1000:.2f} meV")
    print(f"{'─'*60}")

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
print(f"  a2/a1 = {ratio[idx_best]:.4f}")

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

# Mark selected cell
if idx_selected is not None:
    ax.scatter(
        phiG[idx_selected], Vg[idx_selected] * 1000,
        marker="D", s=120, c="cyan", edgecolors="black", linewidths=2.0,
        zorder=6, label="Selected",
    )

# Legend with interlayer params
legend_text = (
    f"Best fit (dist = {dist[idx_best]*1000:.1f} meV)\n"
    f"w1p = {w1p[idx_best]:+.4f} eV\n"
    f"w1d = {w1d[idx_best]:+.4f} eV\n"
    f"w2p = {w2p[idx_best]:+.4f} eV\n"
    f"w2d = {w2d[idx_best]:+.4f} eV"
)
bound_colors = {"w1p": "red", "w1d": "blue", "w2p": "green", "w2d": "orange"}
bound_styles = {"w1p": "-", "w1d": "--", "w2p": "-.", "w2d": ":"}
bound_widths = {"lo": 2, "hi": 4}

all_legend_handles = [False]

if idx_selected is not None:
    sel_legend_text = (
        f"Selected (Vg={Vg[idx_selected]*1000:.1f} meV, phiG={phiG[idx_selected]:.0f} deg)\n"
        f"dist = {dist[idx_selected]*1000:.1f} meV, redchi = {redchi[idx_selected]:.4f}\n"
        f"w1p = {w1p[idx_selected]:+.4f}  w1d = {w1d[idx_selected]:+.4f}\n"
        f"w2p = {w2p[idx_selected]:+.4f}  w2d = {w2d[idx_selected]:+.4f}\n"
        f"c1 = {c1[idx_selected]:.4f}  c2 = {c2[idx_selected]:.4f}  c3 = {c3[idx_selected]:.4f} eV\n"
        f"a2/a1 = {ratio[idx_selected]:.4f}"
    )
    all_legend_handles.append(
        Patch(facecolor="none", edgecolor="none", label=sel_legend_text)
    )

all_legend_handles[0] = Patch(facecolor="none", edgecolor="none", label=legend_text)
cutoff_handle = Patch(
    facecolor="none", edgecolor="none",
    label=f"Cutoffs: dist <= {cutoff_ev*1000:.0f} meV, a2/a1 >= {ratio_cutoff:.2f}"
)
all_legend_handles.append(cutoff_handle)

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
    loc="lower left", fontsize=9, framealpha=0.9,
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
    f"Run: {run_id}  |  {n_after_ratio}/{n_points} pass both cutoffs",
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

    # Mark selected cell
    if idx_selected is not None:
        ax.scatter(
            phiG[idx_selected], Vg[idx_selected] * 1000,
            marker="D", s=90, c="cyan", edgecolors="black", linewidths=1.8,
            zorder=6,
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

# ─── Aggregate: intensity ratio a2/a1 over (Vg, phiG) grid ──────────────────

ratio_2d = np.full((n_Vg, n_phi), np.nan)

for iv, vg in enumerate(Vg_vals):
    for ip, pg in enumerate(phiG_vals):
        mask = (Vg == vg) & (phiG == pg) & ~np.isnan(dist)
        if mask.any():
            idx_min = np.nanargmin(dist[mask])
            idx_global = np.where(mask)[0][idx_min]
            if a1[idx_global] > 0:
                ratio_2d[iv, ip] = a2[idx_global] / a1[idx_global]

# ─── Plot intensity ratio ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

im = ax.pcolormesh(
    phiG_vals, Vg_vals * 1000, ratio_2d,
    cmap="viridis", shading="auto",
)

ax.scatter(
    phiG[idx_best], Vg[idx_best] * 1000,
    marker="*", s=200, c="red", edgecolors="white", linewidths=1.5,
    zorder=5, label="Best fit",
)

# Mark selected cell
if idx_selected is not None:
    ax.scatter(
        phiG[idx_selected], Vg[idx_selected] * 1000,
        marker="D", s=120, c="cyan", edgecolors="black", linewidths=2.0,
        zorder=6, label="Selected",
    )

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label(r"$a_2 / a_1$ (adjacent band / TVB)", fontsize=11)

ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
ax.set_ylabel(r"$V_G$ (meV)", fontsize=12)
ax.set_title(
    f"EDC Gamma: adjacent band intensity relative to TVB\n"
    f"Run: {run_id}  |  at min-distance interlayer params",
    fontsize=12,
)

ratio_output = run_dir / "analysis_ratio.png"
fig.savefig(ratio_output, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Ratio figure saved to {ratio_output}")

# ─── EDC intensity profile plot for selected (Vg, phiG) ─────────────────────

if idx_selected is None:
    sys.exit(0)

print(f"\nProducing EDC intensity profile for selected cell...")

_vg = Vg[idx_selected]
_phig_deg = phiG[idx_selected]
_w1p = w1p[idx_selected]
_w1d = w1d[idx_selected]
_w2p = w2p[idx_selected]
_w2d = w2d[idx_selected]
_c1 = c1[idx_selected]
_c2 = c2[idx_selected]
_c3 = c3[idx_selected]
_a1 = a1[idx_selected]
_a2 = a2[idx_selected]
_a3 = a3[idx_selected]
_g1 = g1[idx_selected]
_g2 = g2[idx_selected]
_g3 = g3[idx_selected]

sample = "S11"
n_shells = 2
theta = TWIST_ANGLES[sample]
spreadE = 0.03
n_cells_geom = MoireGeometry.n_cells(n_shells)

Vk = meta["fixed_params"].get("Vk_ev", 0.0077)
phiK_deg = meta["fixed_params"].get("phiK_deg", 106)
phiG_rad = _phig_deg / 180 * np.pi
phiK_rad = phiK_deg / 180 * np.pi
pars_V = (_vg, Vk, phiG_rad, phiK_rad)

monolayer_fns = {
    "WSe2": master_folder + "/Inputs/monolayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "/Inputs/monolayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

_wse2 = TMDMaterial("WSe2")
_wse2.load_fitted(monolayer_fns["WSe2"])
_ws2 = TMDMaterial("WS2")
_ws2.load_fitted(monolayer_fns["WS2"])

pars_interlayer = {"stacking": "P", "w1p": _w1p, "w2p": _w2p, "w1d": _w1d, "w2d": _w2d}
geometry = MoireGeometry(theta)

moire_ham = MoireHamiltonian(_wse2, _ws2, geometry)
evals_raw, evecs_raw = moire_ham.diagonalize(
    np.array([np.zeros(2)]), n_shells, pars_interlayer, pars_V
)
evals_raw = evals_raw[0] + ENERGY_OFFSETS.get(sample, 0.0)
evecs_raw = evecs_raw[0]

ab = np.absolute(evecs_raw) ** 2
weights = np.sum(ab[:22, :], axis=0) + np.sum(ab[22 * n_cells_geom:22 * (1 + n_cells_geom), :], axis=0)

index_tvb = 28 * n_cells_geom - 1
index_lvb = 26 * n_cells_geom - 1
index_l = index_lvb - 2 * n_cells_geom + 1

full_energy_values = evals_raw[index_l:index_tvb + 1]
full_weight_values = weights[index_l:index_tvb + 1]

min_e = full_energy_values[0]
max_e = full_energy_values[-1]
delta = max_e - min_e
min_e -= delta / 2
max_e += delta / 2
n_e = int((max_e - min_e) / 0.005)
energy_list = np.linspace(min_e, max_e, n_e)
weight_list = np.zeros(len(energy_list))

for j in range(len(full_energy_values)):
    weight_list += spreadE / np.pi * full_weight_values[j] / (
        (energy_list - full_energy_values[j]) ** 2 + spreadE ** 2
    )


def _lorentz_peak(x, amp, cen, gam):
    return amp * gam**2 / ((x - cen)**2 + gam**2)


fit_total = (
    _lorentz_peak(energy_list, _a1, _c1, _g1)
    + _lorentz_peak(energy_list, _a2, _c2, _g2)
    + _lorentz_peak(energy_list, _a3, _c3, _g3)
)

fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

_redchi = redchi[idx_selected]
ax.plot(energy_list, weight_list, "k-", lw=1.5, label="EDC intensity")
ax.plot(energy_list, fit_total, "r--", lw=2,
        label=rf"3-Lorentzian fit ($\chi^2_\nu$ = {_redchi:.4f})")

colors_pk = ["C0", "C1", "C2"]
c_vals = [_c1, _c2, _c3]
amps = [_a1, _a2, _a3]
gams = [_g1, _g2, _g3]
for k, (amp, cen, gam, col) in enumerate(zip(amps, c_vals, gams, colors_pk)):
    pk_curve = _lorentz_peak(energy_list, amp, cen, gam)
    ax.plot(energy_list, pk_curve, color=col, ls="-.", lw=1.2, alpha=0.7,
            label=rf"peak {k+1}: $E$={cen:.3f} eV")

ax.set_xlabel("Energy (eV)", fontsize=12)
ax.set_ylabel("Intensity (a.u.)", fontsize=12)
ax.set_title(
    f"EDC at Gamma: Vg={_vg*1000:.0f} meV, phiG={_phig_deg:.0f} deg\n"
    f"w1p={_w1p:.3f}, w1d={_w1d:.3f}, w2p={_w2p:.3f}, w2d={_w2d:.3f}",
    fontsize=11,
)
ax.set_xlim(-1.4, -1.0)
ax.legend(fontsize=9, loc="upper left")

edc_output = run_dir / f"edc_profile_Vg{_vg*1000:.0f}meV_phiG{_phig_deg:.0f}deg.png"
fig.savefig(edc_output, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"EDC profile saved to {edc_output}")

params_output = run_dir / f"Vg{_vg*1000:.0f}meV_phiG{_phig_deg:.0f}deg.json"
exported = {
    "Vg_ev": float(_vg),
    "Vg_meV": float(_vg * 1000),
    "phiG_deg": float(_phig_deg),
    "phiG_rad": float(_phig_deg / 180 * np.pi),
    "w1p": float(_w1p),
    "w1d": float(_w1d),
    "w2p": float(_w2p),
    "w2d": float(_w2d),
    "c1": float(_c1),
    "c2": float(_c2),
    "c3": float(_c3),
    "a1": float(_a1),
    "a2": float(_a2),
    "a3": float(_a3),
    "g1": float(_g1),
    "g2": float(_g2),
    "g3": float(_g3),
    "redchi": float(_redchi),
    "distance_meV": float(dist[idx_selected] * 1000),
}
with open(params_output, "w") as f:
    json.dump(exported, f, indent=2)
print(f"Params exported to {params_output}")
