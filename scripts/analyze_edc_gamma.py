"""Analyze EDC Gamma grid results.

Loads combined.h5 from a run directory, computes distance from experimental
peak positions, and produces a 2D heatmap of minimum distance over (Vg, phiG)
with the global best-fit point marked and interlayer parameters shown.

Selection mode (--vg/--phig): highlights the chosen cell on the heatmap,
prints its details to stdout, and produces an EDC intensity profile plot
with the Lorentzian fit overlaid.

By default, cells where interlayer parameters hit their bounds are included.
Use --exclude-boundary to mask them out.

Usage:
    python scripts/analyze_edc_gamma.py --id 001
    python scripts/analyze_edc_gamma.py --id 001 --cutoff 0.030
    python scripts/analyze_edc_gamma.py --id 001 --ratio-cutoff 0.15
    python scripts/analyze_edc_gamma.py --id 001 --output Figures/edc_gamma_analysis.png
    python scripts/analyze_edc_gamma.py --id sm03 --vg 0.012 --phig 176
    python scripts/analyze_edc_gamma.py --id bg02 --cutoff 0.010 --exclude-boundary
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

from tmdmoire import EDC_G_POSITIONS, TMDMaterial, MoireGeometry, MoireHamiltonian
from tmdmoire import TWIST_ANGLES, ENERGY_OFFSETS, EDC_G_SEED_BOUNDARY
from tmdmoire.bilayer.edc_analyzer import find_peak_seeds_gamma
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

# ─── Parse arguments ─────────────────────────────────────────────────────────

run_id = "default"
sample = "S11"
output = None
l1_cutoff_ev = 0.026  # 26 meV default, applied to L1 (peak position) distance
l2_cutoff_ev = 0.010  # 10 meV default, applied to L2 (separation) distance
ratio_cutoff = 0.1  # 10% default
vg_selected = None
phig_selected = None
vg_max_mev = None
exclude_boundary = False

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--id" and i + 1 < len(args):
        run_id = args[i + 1]
        i += 2
    elif args[i] == "--output" and i + 1 < len(args):
        output = Path(args[i + 1])
        i += 2
    elif args[i] == "--l1-cutoff" and i + 1 < len(args):
        l1_cutoff_ev = float(args[i + 1])
        i += 2
    elif args[i] == "--cutoff" and i + 1 < len(args):
        l2_cutoff_ev = float(args[i + 1])
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
    elif args[i] == "--vg-max" and i + 1 < len(args):
        vg_max_mev = float(args[i + 1])
        i += 2
    elif args[i] == "--exclude-boundary":
        exclude_boundary = True
        i += 1
    elif args[i] == "--sample" and i + 1 < len(args):
        sample = args[i + 1]
        i += 2
    else:
        i += 1

have_selection = vg_selected is not None and phig_selected is not None
if have_selection:
    print(f"Selection mode: Vg = {vg_selected*1000:.1f} meV, phiG = {phig_selected:.0f} deg")

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

# Read fixed w2p/w2d from metadata (not in HDF5 after 6D→4D change)
fixed_params = meta.get("fixed_params", {})
w2p_fixed = fixed_params.get("w2p_ev", None)
w2d_fixed = fixed_params.get("w2d_ev", None)
if w2p_fixed is None or w2d_fixed is None:
    fitted_il = meta.get("fitted_interlayer", {})
    w2p_fixed = fitted_il.get("w2p", None)
    w2d_fixed = fitted_il.get("w2d", None)

# ─── Determine actual parameter bounds from data ─────────────────────────────
# (metadata.json range_ev may be inaccurate; use observed min/max)

param_arrays = {"w1p": w1p, "w1d": w1d}
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

# ─── Apply Vg limit ───────────────────────────────────────────────────────────

if vg_max_mev is not None:
    mask_vg_high = Vg * 1000 >= vg_max_mev
    n_excluded = mask_vg_high.sum()
    for arr in [c1, c2, c3, a1, a2, a3, g1, g2, g3, redchi]:
        arr[mask_vg_high] = np.nan
    print(f"Excluded {n_excluded} points with Vg >= {vg_max_mev:.0f} meV")

# ─── Compute distance ────────────────────────────────────────────────────────

exp = EDC_G_POSITIONS[sample]  # [-1.1599, -1.2531, -1.82]
exp_sep_TVB_side = np.abs(exp[0] - exp[1])
exp_sep_TVB_LVB = np.abs(exp[0] - exp[2])

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

dist_sep = np.full(n_points, np.nan)
dist_sep[valid] = (
    np.abs(np.abs(c1[valid] - c2[valid]) - exp_sep_TVB_side)
    + np.abs(np.abs(c1[valid] - c3[valid]) - exp_sep_TVB_LVB)
)

# ─── Apply cutoffs ───────────────────────────────────────────────────────────

above_l1_cutoff = dist > l1_cutoff_ev
dist[above_l1_cutoff] = np.nan
dist_sep[above_l1_cutoff] = np.nan
n_l1 = (~np.isnan(dist_sep)).sum()
print(f"Points within L1 cutoff ({l1_cutoff_ev*1000:.0f} meV): {n_l1} / {n_points}")

above_l2_cutoff = dist_sep > l2_cutoff_ev
dist[above_l2_cutoff] = np.nan
dist_sep[above_l2_cutoff] = np.nan
n_l2 = (~np.isnan(dist_sep)).sum()
print(f"Points within L2 cutoff ({l2_cutoff_ev*1000:.0f} meV): {n_l2} / {n_points}")

ratio = np.full(n_points, np.nan)
ratio[valid & ~np.isnan(dist)] = a2[valid & ~np.isnan(dist)] / a1[valid & ~np.isnan(dist)]

below_ratio_cutoff = ~np.isnan(ratio) & (ratio < ratio_cutoff)
dist[below_ratio_cutoff] = np.nan
dist_sep[below_ratio_cutoff] = np.nan
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
    ) & ~np.isnan(dist_sep)

    if not mask_sel.any():
        vg_vals = sorted(set(Vg))
        pg_vals = sorted(set(phiG))
        print(f"No valid fits at Vg={vg_selected*1000:.1f} meV, phiG={phig_selected:.0f} deg")
        print(f"Available Vg [meV]: {[v*1000 for v in vg_vals]}")
        print(f"Available phiG [deg]: {pg_vals}")
        if have_selection:
            print("Exiting due to invalid selection.")
            sys.exit(1)

    idx_sel_local = np.nanargmin(dist_sep[mask_sel])
    idx_selected = np.where(mask_sel)[0][idx_sel_local]

    print(f"\n{'─'*60}")
    print(f"Selected cell: Vg = {Vg[idx_selected]*1000:.1f} meV, phiG = {phiG[idx_selected]:.1f} deg")
    print(f"  w1p  = {w1p[idx_selected]:+.4f} eV")
    print(f"  w1d  = {w1d[idx_selected]:+.4f} eV")
    if w2p_fixed is not None:
        print(f"  w2p  = {w2p_fixed:+.4f} eV  (fixed)")
    if w2d_fixed is not None:
        print(f"  w2d  = {w2d_fixed:+.4f} eV  (fixed)")
    print(f"  c1   = {c1[idx_selected]:.4f} eV (exp: {exp[0]:.4f} eV)")
    print(f"  c2   = {c2[idx_selected]:.4f} eV (exp: {exp[1]:.4f} eV)")
    print(f"  c3   = {c3[idx_selected]:.4f} eV (exp: {exp[2]:.4f} eV)")
    print(f"  a1   = {a1[idx_selected]:.4f}")
    print(f"  a2   = {a2[idx_selected]:.4f}")
    print(f"  a3   = {a3[idx_selected]:.4f}")
    print(f"  redchi = {redchi[idx_selected]:.6f}")
    print(f"  a2/a1 = {ratio[idx_selected]:.4f}")
    print(f"  L2    = {dist_sep[idx_selected]*1000:.2f} meV")
    if not np.isnan(dist[idx_selected]):
        print(f"  L1    = {dist[idx_selected]*1000:.2f} meV")
    print(f"{'─'*60}")

# ─── Find global minimum ─────────────────────────────────────────────────────

idx_best = np.nanargmin(dist_sep)
print(f"\nGlobal minimum L2 distance: {dist_sep[idx_best]*1000:.2f} meV")
print(f"  Vg   = {Vg[idx_best]*1000:.1f} meV")
print(f"  phiG = {phiG[idx_best]:.1f} deg")
print(f"  w1p  = {w1p[idx_best]:+.4f} eV")
print(f"  w1d  = {w1d[idx_best]:+.4f} eV")
print(f"  c1   = {c1[idx_best]:.4f} eV (exp: {exp[0]:.4f} eV)")
print(f"  c2   = {c2[idx_best]:.4f} eV (exp: {exp[1]:.4f} eV)")
print(f"  c3   = {c3[idx_best]:.4f} eV (exp: {exp[2]:.4f} eV)")
print(f"  L1   = {dist[idx_best]*1000:.2f} meV")
print(f"  a2/a1 = {ratio[idx_best]:.4f}")

# ─── Aggregate: min distance over (Vg, phiG) grid ───────────────────────────

# Get unique sorted values
phiG_vals = np.unique(phiG)
Vg_vals = np.unique(Vg)
n_phi = len(phiG_vals)
n_Vg = len(Vg_vals)

print(f"\nGrid: {n_Vg} Vg values x {n_phi} phiG values")

# Build 2D array: single pass O(n) over all points, track min dist per (Vg, phiG) cell
dist_2d = np.full((n_Vg, n_phi), np.nan)
dist_sep_2d = np.full((n_Vg, n_phi), np.nan)
bounds_2d = [[[] for _ in range(n_phi)] for _ in range(n_Vg)]
param_2d = {pname: np.full((n_Vg, n_phi), np.nan) for pname in param_arrays}

vg_to_iv = {vg: iv for iv, vg in enumerate(Vg_vals)}
pg_to_ip = {pg: ip for ip, pg in enumerate(phiG_vals)}

# dict: (vg, pg) -> index in original arrays of the min-L2 point
best_per_cell = {}
for i in range(n_points):
    if np.isnan(dist_sep[i]):
        continue
    key = (Vg[i], phiG[i])
    if key not in best_per_cell or dist_sep[i] < dist_sep[best_per_cell[key]]:
        best_per_cell[key] = i

if vg_max_mev is not None:
    vg_max_ev = vg_max_mev / 1000
    Vg_vals = Vg_vals[Vg_vals <= vg_max_ev]
    n_Vg = len(Vg_vals)
    best_per_cell = {k: v for k, v in best_per_cell.items() if k[0] <= vg_max_ev}
    vg_to_iv = {vg: iv for iv, vg in enumerate(Vg_vals)}
    dist_2d = np.full((n_Vg, n_phi), np.nan)
    dist_sep_2d = np.full((n_Vg, n_phi), np.nan)
    bounds_2d = [[[] for _ in range(n_phi)] for _ in range(n_Vg)]
    param_2d = {pname: np.full((n_Vg, n_phi), np.nan) for pname in param_arrays}
    print(f"Vg max = {vg_max_mev:.0f} meV → {n_Vg} Vg values")

for (vg, pg), idx in best_per_cell.items():
    iv = vg_to_iv[vg]
    ip = pg_to_ip[pg]
    dist_2d[iv, ip] = dist[idx]
    dist_sep_2d[iv, ip] = dist_sep[idx]
    for pname, parr in param_arrays.items():
        val = parr[idx]
        param_2d[pname][iv, ip] = val
        lo, hi = bounds[pname]
        if abs(val - lo) < tol:
            bounds_2d[iv][ip].append((pname, "lo"))
        elif abs(val - hi) < tol:
            bounds_2d[iv][ip].append((pname, "hi"))

n_at_bounds = 0
for iv in range(n_Vg):
    for ip in range(n_phi):
        if bounds_2d[iv][ip]:
            n_at_bounds += 1

if exclude_boundary:
    for iv in range(n_Vg):
        for ip in range(n_phi):
            if bounds_2d[iv][ip]:
                dist_2d[iv, ip] = np.nan
                dist_sep_2d[iv, ip] = np.nan
                for pname in param_arrays:
                    param_2d[pname][iv, ip] = np.nan
    print(f"Cells at interlayer parameter bounds (excluded from heatmap): {n_at_bounds} / {n_Vg * n_phi}")
else:
    print(f"Cells at interlayer parameter bounds (included in heatmap): {n_at_bounds} / {n_Vg * n_phi}")

phi_step = phiG_vals[1] - phiG_vals[0]
phi_edges = np.append(phiG_vals - phi_step / 2, phiG_vals[-1] + phi_step / 2)
vg_step = Vg_vals[1] - Vg_vals[0]
vg_half_step_mev = vg_step * 1000 / 2
vg_edges_mev = np.append(Vg_vals * 1000 - vg_half_step_mev, Vg_vals[-1] * 1000 + vg_half_step_mev)

# Horizontal reference lines every 4 meV
vg_line_vals = np.arange(4, 23, 4)

# ─── Plot Vg/phiG distance heatmaps (L1 + L2) ─────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

for ax, d2d, title, cmap_name in [
    (ax1, dist_2d * 1000, r"L1 distance: $\Sigma\,|c_i - E_i^{\mathrm{exp}}|$", "viridis_r"),
    (ax2, dist_sep_2d * 1000, r"L2 distance: $\Sigma\,|\Delta E - \Delta E^{\mathrm{exp}}|$", "plasma_r"),
]:
    im = ax.pcolormesh(
        phi_edges, vg_edges_mev, d2d,
        cmap=cmap_name, shading="flat",
    )
    for deg in [60, 180, 300]:
        ax.axvline(x=deg, color="red", ls="--", lw=1, alpha=0.6)
    for v in vg_line_vals:
        ax.axhline(y=v, color="gray", ls="--", lw=0.5, alpha=0.6)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Min distance (meV)", fontsize=11)
    ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
    ax.set_ylabel(r"$V_G$ (meV)", fontsize=12)
    ax.set_xticks([160, 170, 180, 190, 200])
    ax.set_yticks(np.arange(0, 23, 2))
    ax.set_ylim(0, 22)
    ax.set_xlim(160, 200)
    ax.set_title(title, fontsize=11)

fig.suptitle(
    f"EDC Gamma: {run_id}  |  {n_after_ratio}/{n_points} pass both cutoffs",
    fontsize=13, y=1.02,
)

# Save
if output is None:
    out_dir = run_dir
    output = out_dir / "analysis.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nFigure saved to {output}")

# ─── Aggregate: min distance over (w1p, w1d) grid ───────────────────────────

w1p_vals = np.unique(w1p)
w1d_vals = np.unique(w1d)
n_w1p = len(w1p_vals)
n_w1d = len(w1d_vals)

dist_w_2d = np.full((n_w1d, n_w1p), np.nan)
dist_sep_w_2d = np.full((n_w1d, n_w1p), np.nan)
best_per_cell_w = {}

for i in range(n_points):
    if np.isnan(dist_sep[i]):
        continue
    key = (w1p[i], w1d[i])
    if key not in best_per_cell_w or dist_sep[i] < dist_sep[best_per_cell_w[key]]:
        best_per_cell_w[key] = i

w1p_to_iw = {wp: iw for iw, wp in enumerate(w1p_vals)}
w1d_to_iw = {wd: iw for iw, wd in enumerate(w1d_vals)}

for (wp, wd), idx in best_per_cell_w.items():
    iw1p = w1p_to_iw[wp]
    iw1d = w1d_to_iw[wd]
    dist_w_2d[iw1d, iw1p] = dist[idx]
    dist_sep_w_2d[iw1d, iw1p] = dist_sep[idx]

w1p_step = w1p_vals[1] - w1p_vals[0] if n_w1p > 1 else 0.002
w1p_edges = np.append(w1p_vals - w1p_step / 2, w1p_vals[-1] + w1p_step / 2)
w1d_step = w1d_vals[1] - w1d_vals[0] if n_w1d > 1 else 0.002
w1d_edges = np.append(w1d_vals - w1d_step / 2, w1d_vals[-1] + w1d_step / 2)

print(f"w1p/w1d distance grid: {n_w1p} x {n_w1d}")

# ─── Plot w1p/w1d distance heatmaps (L1 + L2) ──────────────────────

fig_w, (ax_w1, ax_w2) = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

for ax, d2d, title, cmap_name in [
    (ax_w1, dist_w_2d * 1000, r"L1 distance: $\Sigma\,|c_i - E_i^{\mathrm{exp}}|$", "viridis_r"),
    (ax_w2, dist_sep_w_2d * 1000, r"L2 distance: $\Sigma\,|\Delta E - \Delta E^{\mathrm{exp}}|$", "plasma_r"),
]:
    im = ax.pcolormesh(
        w1p_edges * 1000, w1d_edges * 1000, d2d,
        cmap=cmap_name, shading="flat",
    )
    cbar = fig_w.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Min distance (meV)", fontsize=11)
    ax.set_xlabel(r"$w_{1p}$ (meV)", fontsize=12)
    ax.set_ylabel(r"$w_{1d}$ (meV)", fontsize=12)
    ax.set_title(title, fontsize=11)

fig_w.suptitle(
    f"EDC Gamma: min distance over Vg, phiG  |  Run: {run_id}",
    fontsize=13, y=1.02,
)

pw_output = run_dir / "analysis_wpw_d.png"
fig_w.savefig(pw_output, dpi=200, bbox_inches="tight")
plt.close(fig_w)

print(f"w1p/w1d distance figure saved to {pw_output}")

# ─── Zoomed plot (phiG 150–210) with selected cell marker ──────────────────────

if idx_selected is not None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    im = ax.pcolormesh(
        phi_edges, vg_edges_mev, dist_sep_2d * 1000,
        cmap="plasma_r", shading="flat",
    )

    ax.scatter(
        phiG[idx_selected], Vg[idx_selected] * 1000,
        marker="D", s=80, c="cyan", edgecolors="black", linewidths=1.5,
        zorder=6,
    )

    for deg in [60, 180, 300]:
        ax.axvline(x=deg, color="red", ls="--", lw=1, alpha=0.6)
    for v in vg_line_vals:
        ax.axhline(y=v, color="gray", ls="--", lw=0.3, alpha=0.6, zorder=0)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("L2 distance (meV)", fontsize=11)

    ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
    ax.set_ylabel(r"$V_G$ (meV)", fontsize=12)
    ax.set_yticks(np.arange(0, 23, 2))
    ax.set_ylim(0, 22)
    ax.set_xlim(160, 200)
    ax.set_title(
        f"EDC Gamma: L2 distance zoom phiG [160, 200]\n"
        f"Run: {run_id}  |  Vg={vg_selected*1000:.1f} meV, phiG={phig_selected:.0f} deg",
        fontsize=12,
    )

    zoom_output = run_dir / "analysis_zoom.png"
    fig.savefig(zoom_output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Zoomed figure saved to {zoom_output}")

# ─── EDC intensity profile plot for selected (Vg, phiG) ─────────────────────

if idx_selected is None:
    sys.exit(0)

print(f"\nProducing EDC intensity profile for selected cell...")

_vg = Vg[idx_selected]
_phig_deg = phiG[idx_selected]
_w1p = w1p[idx_selected]
_w1d = w1d[idx_selected]
_w2p = w2p_fixed
_w2d = w2d_fixed
_c1 = c1[idx_selected]
_c2 = c2[idx_selected]
_c3 = c3[idx_selected]
_a1 = a1[idx_selected]
_a2 = a2[idx_selected]
_a3 = a3[idx_selected]
_g1 = g1[idx_selected]
_g2 = g2[idx_selected]
_g3 = g3[idx_selected]

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
    f"EDC at Gamma: Vg={_vg*1000:.1f} meV, phiG={_phig_deg:.0f} deg\n"
    f"w1p={_w1p:.3f}, w1d={_w1d:.3f}, w2p={_w2p:.3f}, w2d={_w2d:.3f}",
    fontsize=11,
)
ax.set_xlim(-1.4, -1.0)
ax.legend(fontsize=9, loc="upper left")

edc_output = run_dir / f"edc_profile_Vg{_vg*1000:.1f}meV_phiG{_phig_deg:.0f}deg.png"
fig.savefig(edc_output, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"EDC profile saved to {edc_output}")

# ─── Full-range EDC profile with 4-Lorentzian fit ─────────────────────────────
print(f"\nProducing full-range EDC with 4-Lorentzian fit...")

def _four_lorentzians(x, a1, c1, g1, a2, c2, g2, a3, c3, g3, a4, c4, g4):
    return (_lorentz_peak(x, a1, c1, g1) +
            _lorentz_peak(x, a2, c2, g2) +
            _lorentz_peak(x, a3, c3, g3) +
            _lorentz_peak(x, a4, c4, g4))

seed_boundary = EDC_G_SEED_BOUNDARY.get(sample, -1.5)
peak_states_4 = find_peak_seeds_gamma(weight_list, energy_list, full_energy_values, full_weight_values,
                                      boundary_ev=seed_boundary)
if len(peak_states_4) < 4:
    peak_states_4 = [(float(_c1), float(_a1)), (float(_c2), float(_a2)),
                     (float(_c3), float(_a3)), (float(_c3) - 0.05, float(_a3) / 2)]

import lmfit as lmfit_mod_4L
model_4L = lmfit_mod_4L.Model(_four_lorentzians)
params_fit_4L = model_4L.make_params(
    a1=peak_states_4[0][1], c1=peak_states_4[0][0], g1=spreadE,
    a2=peak_states_4[1][1], c2=peak_states_4[1][0], g2=spreadE,
    a3=peak_states_4[2][1], c3=peak_states_4[2][0], g3=spreadE,
    a4=peak_states_4[3][1], c4=peak_states_4[3][0], g4=spreadE,
)
for p in ["a1", "a2", "a3", "a4"]:
    params_fit_4L[p].set(min=0)
for p in ["g1", "g2", "g3", "g4"]:
    params_fit_4L[p].set(min=1e-4, max=0.2)
for i, p in enumerate(["c1", "c2", "c3", "c4"]):
    seed = peak_states_4[i][0]
    params_fit_4L[p].set(min=seed - 0.05, max=seed + 0.05)

try:
    result_4L = model_4L.fit(weight_list, params_fit_4L, x=energy_list)
    if result_4L.success:
        fits = [
            (result_4L.best_values["a1"], result_4L.best_values["c1"], result_4L.best_values["g1"]),
            (result_4L.best_values["a2"], result_4L.best_values["c2"], result_4L.best_values["g2"]),
            (result_4L.best_values["a3"], result_4L.best_values["c3"], result_4L.best_values["g3"]),
            (result_4L.best_values["a4"], result_4L.best_values["c4"], result_4L.best_values["g4"]),
        ]
        fits.sort(key=lambda x: x[1], reverse=True)
        popt_4 = [
            fits[0][0], fits[0][1], fits[0][2],
            fits[1][0], fits[1][1], fits[1][2],
            fits[2][0], fits[2][1], fits[2][2],
            fits[3][0], fits[3][1], fits[3][2],
        ]
        centers_4 = [fits[0][1], fits[1][1], fits[2][1], fits[3][1]]
        fit_total_4 = _four_lorentzians(energy_list, *popt_4)
        redchi_4 = result_4L.redchi
        print(f"  4-Lorentzian fit centers: {[f'{c:.4f}' for c in centers_4]} eV")
        print(f"  Reduced chi-squared: {redchi_4:.6f}")
    else:
        raise RuntimeError("lmfit did not converge")
except Exception as exc:
    print(f"  4-Lorentzian fit failed: {exc}")
    popt_4 = None
    centers_4 = []

fig4, ax4 = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax4.plot(energy_list, weight_list, "k-", lw=1.5, label="EDC intensity")

if popt_4 is not None:
    ax4.plot(energy_list, fit_total_4, "r--", lw=2,
             label=rf"4-Lorentzian fit ($\chi^2_\nu$ = {redchi_4:.4f})")

exp_colors = ["#2ecc71", "#2ecc71", "#2ecc71"]
for i, (e_val, ec) in enumerate(zip(exp, exp_colors)):
    lbl = r"ARPES EDC position" if i == 0 else ""
    ax4.axvline(x=e_val, color=ec, ls="--", lw=1.5, alpha=0.8, label=lbl)

center_colors = ["#e74c3c", "#e74c3c", "#e74c3c", "#e74c3c"]
for i, c_val in enumerate(centers_4):
    lbl = r"4-Lor. center" if i == 0 else ""
    ax4.axvline(x=c_val, color=center_colors[i], ls=":", lw=1.5, alpha=0.8, label=lbl)

ax4.set_xlabel("Energy (eV)", fontsize=12)
ax4.set_ylabel("Intensity (a.u.)", fontsize=12)
ax4.set_title(
    f"EDC at Gamma (full range): Vg={_vg*1000:.1f} meV, phiG={_phig_deg:.0f} deg\n"
    f"w1p={_w1p:.3f}, w1d={_w1d:.3f}, w2p={_w2p:.3f}, w2d={_w2d:.3f}",
    fontsize=11,
)
ax4.legend(fontsize=9, loc="upper left")

edc4_output = run_dir / f"edc_profile_4L_Vg{_vg*1000:.1f}meV_phiG{_phig_deg:.0f}deg.png"
fig4.savefig(edc4_output, dpi=200, bbox_inches="tight")
plt.close(fig4)
print(f"Full-range EDC profile saved to {edc4_output}")

params_output = run_dir / f"Vg{_vg*1000:.1f}meV_phiG{_phig_deg:.0f}deg.json"
exported = {
    "Vg_ev": float(_vg),
    "Vg_meV": float(_vg * 1000),
    "phiG_deg": float(_phig_deg),
    "phiG_rad": float(_phig_deg / 180 * np.pi),
    "w1p": float(_w1p),
    "w1d": float(_w1d),
    "w2p": float(w2p_fixed) if w2p_fixed is not None else None,
    "w2d": float(w2d_fixed) if w2d_fixed is not None else None,
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
    "L2_meV": float(dist_sep[idx_selected] * 1000),
    "L1_meV": float(dist[idx_selected] * 1000) if not np.isnan(dist[idx_selected]) else None,
}
with open(params_output, "w") as f:
    json.dump(exported, f, indent=2)
print(f"Params exported to {params_output}")
