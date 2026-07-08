"""Export shareable data from a Gamma EDC run for standalone plotting.

Loads combined.h5 + metadata.json from an EDC Gamma run directory,
aggregates distance-2D data, finds the global best fit, and optionally
recomputes the EDC intensity profile at a selected (Vg, phiG) cell.

Output: a self-contained .npz file with everything needed by
plot_edc_gamma_standalone.py (numpy + matplotlib only, no tmdmoire).

Usage:
    python scripts/export_edc_gamma_data.py --id 001
    python scripts/export_edc_gamma_data.py --id 001 --vg 0.012 --phig 176
    python scripts/export_edc_gamma_data.py --id 001 --vg 0.012 --phig 176 --output my_data.npz
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import h5py

from tmdmoire import EDC_G_POSITIONS, TMDMaterial, MoireGeometry, MoireHamiltonian
from tmdmoire import TWIST_ANGLES, ENERGY_OFFSETS
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

# ─── Parse arguments ─────────────────────────────────────────────────────────

run_id = "default"
output = None
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
    elif args[i] == "--vg" and i + 1 < len(args):
        vg_selected = float(args[i + 1])
        i += 2
    elif args[i] == "--phig" and i + 1 < len(args):
        phig_selected = float(args[i + 1])
        i += 2
    else:
        i += 1

have_selection = vg_selected is not None and phig_selected is not None

# ─── Load combined data ──────────────────────────────────────────────────────

run_dir = Path("Data") / f"edc_gamma_{run_id}"
combined_fn = run_dir / "combined.h5"

if not combined_fn.exists():
    print(f"Combined file not found: {combined_fn}")
    print("Run: python scripts/combine_edc_chunks.py --bz-point gamma --id " + run_id)
    sys.exit(1)

print(f"Loading {combined_fn}...")
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

n_points = len(Vg)
print(f"Loaded {n_points} points")

# ─── Load metadata ───────────────────────────────────────────────────────────

meta_fn = run_dir / "metadata.json"
with open(meta_fn) as f_meta:
    meta = json.load(f_meta)

fixed_params = meta.get("fixed_params", {})
w2p_fixed = fixed_params.get("w2p_ev")
w2d_fixed = fixed_params.get("w2d_ev")
if w2p_fixed is None or w2d_fixed is None:
    fitted_il = meta.get("fitted_interlayer", {})
    w2p_fixed = fitted_il.get("w2p")
    w2d_fixed = fitted_il.get("w2d")

# ─── Compute distance ────────────────────────────────────────────────────────

exp = EDC_G_POSITIONS["S11"]

valid = ~np.isnan(c1) & ~np.isnan(c2) & ~np.isnan(c3)

dist = np.full(n_points, np.nan)
dist[valid] = (
    np.abs(c1[valid] - exp[0])
    + np.abs(c2[valid] - exp[1])
    + np.abs(c3[valid] - exp[2])
)

# ─── Aggregate: min distance per (Vg, phiG) cell ────────────────────────────

Vg_vals = np.unique(Vg)
phiG_vals = np.unique(phiG)
n_Vg = len(Vg_vals)
n_phi = len(phiG_vals)

vg_to_iv = {vg: iv for iv, vg in enumerate(Vg_vals)}
pg_to_ip = {pg: ip for ip, pg in enumerate(phiG_vals)}

dist_2d = np.full((n_Vg, n_phi), np.nan)
w1p_best_2d = np.full((n_Vg, n_phi), np.nan)
w1d_best_2d = np.full((n_Vg, n_phi), np.nan)

best_per_cell = {}
for idx in range(n_points):
    if np.isnan(dist[idx]):
        continue
    key = (Vg[idx], phiG[idx])
    if key not in best_per_cell or dist[idx] < dist[best_per_cell[key]]:
        best_per_cell[key] = idx

for (vg, pg), idx in best_per_cell.items():
    iv = vg_to_iv[vg]
    ip = pg_to_ip[pg]
    dist_2d[iv, ip] = dist[idx]
    w1p_best_2d[iv, ip] = w1p[idx]
    w1d_best_2d[iv, ip] = w1d[idx]

dist_2d_meV = dist_2d * 1000.0

# ─── Aggregate: min distance per (w1p, w1d) cell ──────────────────────────

w1p_vals = np.unique(w1p)
w1d_vals = np.unique(w1d)
n_w1p = len(w1p_vals)
n_w1d = len(w1d_vals)

dist_w_2d = np.full((n_w1d, n_w1p), np.nan)
w1p_to_iw = {wp: iw for iw, wp in enumerate(w1p_vals)}
w1d_to_iw = {wd: iw for iw, wd in enumerate(w1d_vals)}

best_per_cell_w = {}
for idx in range(n_points):
    if np.isnan(dist[idx]):
        continue
    key = (w1p[idx], w1d[idx])
    if key not in best_per_cell_w or dist[idx] < dist[best_per_cell_w[key]]:
        best_per_cell_w[key] = idx

for (wp, wd), idx in best_per_cell_w.items():
    iw1p = w1p_to_iw[wp]
    iw1d = w1d_to_iw[wd]
    dist_w_2d[iw1d, iw1p] = dist[idx]

dist_w_2d_meV = dist_w_2d * 1000.0

w1p_step = w1p_vals[1] - w1p_vals[0] if n_w1p > 1 else 0.002
w1p_edges = np.append(w1p_vals - w1p_step / 2, w1p_vals[-1] + w1p_step / 2)
w1d_step = w1d_vals[1] - w1d_vals[0] if n_w1d > 1 else 0.002
w1d_edges = np.append(w1d_vals - w1d_step / 2, w1d_vals[-1] + w1d_step / 2)

print(f"w1p/w1d distance grid: {n_w1p} x {n_w1d}")

# Global best fit
idx_best = np.nanargmin(dist)
best_Vg_meV = float(Vg[idx_best] * 1000)
best_phiG_deg = float(phiG[idx_best])
best_dist_meV = float(dist[idx_best] * 1000)
best_w1p_ev = float(w1p[idx_best])
best_w1d_ev = float(w1d[idx_best])

print(f"Global best: Vg={best_Vg_meV:.1f} meV, phiG={best_phiG_deg:.1f} deg, "
      f"dist={best_dist_meV:.2f} meV, w1p={best_w1p_ev:.4f}, w1d={best_w1d_ev:.4f}")

# ─── Edge arrays for pcolormesh ──────────────────────────────────────────────

phi_step = phiG_vals[1] - phiG_vals[0]
phi_edges = np.append(phiG_vals - phi_step / 2, phiG_vals[-1] + phi_step / 2)

Vg_step_ev = Vg_vals[1] - Vg_vals[0]
Vg_edges_meV = np.append(Vg_vals * 1000 - Vg_step_ev * 500, Vg_vals[-1] * 1000 + Vg_step_ev * 500)

print(f"Distance grid: {n_Vg} x {n_phi}")

# ─── Build export dict ───────────────────────────────────────────────────────

export = {
    "run_id": run_id,
    "Vg_vals_meV": Vg_vals * 1000,
    "phiG_vals_deg": phiG_vals,
    "dist_2d_meV": dist_2d_meV,
    "phi_edges": phi_edges,
    "Vg_edges_meV": Vg_edges_meV,
    "best_Vg_meV": best_Vg_meV,
    "best_phiG_deg": best_phiG_deg,
    "best_dist_meV": best_dist_meV,
    "best_w1p_ev": best_w1p_ev,
    "best_w1d_ev": best_w1d_ev,
    "w1p_vals_meV": w1p_vals * 1000,
    "w1d_vals_meV": w1d_vals * 1000,
    "dist_w_2d_meV": dist_w_2d_meV,
    "w1p_edges_meV": w1p_edges * 1000,
    "w1d_edges_meV": w1d_edges * 1000,
}

# ─── Selection mode: recompute EDC at chosen cell ────────────────────────────

if have_selection:
    tol_vg = 1e-6
    tol_phig = 1e-6
    mask_sel = (
        np.abs(Vg - vg_selected) < tol_vg
    ) & (
        np.abs(phiG - phig_selected) < tol_phig
    ) & ~np.isnan(dist)

    if not mask_sel.any():
        print(f"No valid fits at Vg={vg_selected*1000:.0f} meV, phiG={phig_selected:.0f} deg")
        vg_vals_list = sorted(set(Vg))
        pg_vals_list = sorted(set(phiG))
        print(f"Available Vg [meV]: {[v*1000 for v in vg_vals_list]}")
        print(f"Available phiG [deg]: {pg_vals_list}")
        sys.exit(1)

    sel_idx = np.nanargmin(dist[mask_sel])
    idx_selected = np.where(mask_sel)[0][sel_idx]

    _vg = float(Vg[idx_selected])
    _phig_deg = float(phiG[idx_selected])
    _w1p_val = float(w1p[idx_selected])
    _w1d_val = float(w1d[idx_selected])

    print(f"Selected: Vg={_vg*1000:.1f} meV, phiG={_phig_deg:.1f} deg, "
          f"w1p={_w1p_val:.4f}, w1d={_w1d_val:.4f}")

    print("Recomputing EDC intensity profile...")

    sample = "S11"
    n_shells = 2
    theta = TWIST_ANGLES[sample]
    spreadE = 0.03
    n_cells = MoireGeometry.n_cells(n_shells)

    Vk_ev = fixed_params.get("Vk_ev", 0.0077)
    phiK_deg = fixed_params.get("phiK_deg", 106)
    phiG_rad = _phig_deg / 180.0 * np.pi
    phiK_rad = phiK_deg / 180.0 * np.pi
    pars_V = (_vg, Vk_ev, phiG_rad, phiK_rad)

    monolayer_fns = {
        "WSe2": master_folder + "/Inputs/monolayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
        "WS2": master_folder + "/Inputs/monolayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
    }

    _wse2 = TMDMaterial("WSe2")
    _wse2.load_fitted(monolayer_fns["WSe2"])
    _ws2 = TMDMaterial("WS2")
    _ws2.load_fitted(monolayer_fns["WS2"])

    pars_interlayer = {
        "stacking": "P",
        "w1p": _w1p_val, "w2p": w2p_fixed,
        "w1d": _w1d_val, "w2d": w2d_fixed,
    }

    geometry = MoireGeometry(theta)

    moire_ham = MoireHamiltonian(_wse2, _ws2, geometry)
    evals_raw, evecs_raw = moire_ham.diagonalize(
        np.array([np.zeros(2)]), n_shells, pars_interlayer, pars_V
    )
    evals_raw = evals_raw[0] + ENERGY_OFFSETS.get(sample, 0.0)
    evecs_raw = evecs_raw[0]

    abv = np.absolute(evecs_raw) ** 2
    weights = np.sum(abv[:22, :], axis=0) + np.sum(abv[22 * n_cells:22 * (1 + n_cells), :], axis=0)

    index_tvb = 28 * n_cells - 1
    index_lvb = 26 * n_cells - 1
    index_l = index_lvb - 2 * n_cells + 1

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
        return amp * gam ** 2 / ((x - cen) ** 2 + gam ** 2)

    def _four_lorentzians(x, a1, c1, g1, a2, c2, g2, a3, c3, g3, a4, c4, g4):
        return (_lorentz_peak(x, a1, c1, g1) +
                _lorentz_peak(x, a2, c2, g2) +
                _lorentz_peak(x, a3, c3, g3) +
                _lorentz_peak(x, a4, c4, g4))

    sorted_idx = np.argsort(full_weight_values)[::-1]
    peak_states = []
    seen_centers = []
    for si in sorted_idx:
        e = full_energy_values[si]
        w = full_weight_values[si]
        if w < 1e-4:
            break
        too_close = any(abs(e - c) < 0.01 for c in seen_centers)
        if not too_close:
            peak_states.append((e, w))
            seen_centers.append(e)
        if len(peak_states) == 4:
            break

    if len(peak_states) < 4:
        print("Warning: fewer than 4 peak states found, using hardcoded seeds")
        peak_states = [(-1.15, 1.5), (-1.25, 0.8), (-1.82, 1.0), (-1.87, 0.5)]

    peak_states.sort(key=lambda x: x[0], reverse=True)

    import lmfit as lmfit_mod
    model = lmfit_mod.Model(_four_lorentzians)
    params_fit = model.make_params(
        a1=peak_states[0][1], c1=peak_states[0][0], g1=spreadE,
        a2=peak_states[1][1], c2=peak_states[1][0], g2=spreadE,
        a3=peak_states[2][1], c3=peak_states[2][0], g3=spreadE,
        a4=peak_states[3][1], c4=peak_states[3][0], g4=spreadE,
    )
    for p in ["a1", "a2", "a3", "a4"]:
        params_fit[p].set(min=0)
    for p in ["g1", "g2", "g3", "g4"]:
        params_fit[p].set(min=1e-4, max=0.2)
    for i, p in enumerate(["c1", "c2", "c3", "c4"]):
        seed = peak_states[i][0]
        params_fit[p].set(min=seed - 0.05, max=seed + 0.05)

    try:
        result = model.fit(weight_list, params_fit, x=energy_list)
        if result.success:
            fits = [
                (result.best_values["a1"], result.best_values["c1"], result.best_values["g1"]),
                (result.best_values["a2"], result.best_values["c2"], result.best_values["g2"]),
                (result.best_values["a3"], result.best_values["c3"], result.best_values["g3"]),
                (result.best_values["a4"], result.best_values["c4"], result.best_values["g4"]),
            ]
            fits.sort(key=lambda x: x[1], reverse=True)
            fit_4L_centers = np.array([fits[0][1], fits[1][1], fits[2][1], fits[3][1]])
            fit_4L_curve = _four_lorentzians(energy_list,
                fits[0][0], fits[0][1], fits[0][2],
                fits[1][0], fits[1][1], fits[1][2],
                fits[2][0], fits[2][1], fits[2][2],
                fits[3][0], fits[3][1], fits[3][2],
            )
            fit_4L_redchi = float(result.redchi)
        else:
            raise RuntimeError("lmfit did not converge")
    except Exception as exc:
        print(f"4-Lorentzian fit failed: {exc}")
        fit_4L_curve = np.full_like(energy_list, np.nan)
        fit_4L_centers = np.full(4, np.nan)
        fit_4L_redchi = np.nan

    print(f"4-Lorentzian fit centers: {[f'{c:.4f}' for c in fit_4L_centers]} eV, "
          f"redchi={fit_4L_redchi:.6f}")

    export["energy_list"] = energy_list
    export["weight_list"] = weight_list
    export["fit_4L_curve"] = fit_4L_curve
    export["fit_4L_centers"] = fit_4L_centers
    export["fit_4L_redchi"] = fit_4L_redchi
    export["exp_positions_ev"] = exp
    export["selected_Vg_meV"] = float(_vg * 1000)
    export["selected_phiG_deg"] = float(_phig_deg)
    export["selected_w1p_ev"] = float(_w1p_val)
    export["selected_w1d_ev"] = float(_w1d_val)
    export["selected_w2p_ev"] = float(w2p_fixed) if w2p_fixed is not None else np.nan
    export["selected_w2d_ev"] = float(w2d_fixed) if w2d_fixed is not None else np.nan

# ─── Save ────────────────────────────────────────────────────────────────────

if output is None:
    out_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "plotsPaper" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if have_selection:
        suffix = f"_Vg_{_vg*1000:.0f}meV_phiG_{_phig_deg:.0f}deg"
    output = out_dir / f"edc_gamma_{run_id}{suffix}.npz"

np.savez_compressed(output, **export)
print(f"\nExported to {output}")
print(f"Keys: {list(export.keys())}")
