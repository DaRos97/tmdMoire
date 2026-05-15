"""EDC intensity grid sweep at K for HPC.

Loads grid parameters from Inputs/bilayer_fitting/grid_config_k.json,
sweeps 2D parameter space (Vk, phiK) with fixed Gamma parameters and
interlayer couplings from the config, fits 2 Lorentzians to each EDC,
and saves results.

Usage:
    python scripts/edc_grid_k.py --chunk <id>/<total>
    python scripts/edc_grid_k.py --chunk 0/128 --run-id 001
    python scripts/edc_grid_k.py --chunk 0/128 --run-id test
"""
import sys
import os
import json
import time
import shutil
import datetime
from pathlib import Path
from itertools import product, islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lmfit
import h5py

from tmdmoire import TMDMaterial, MoireGeometry, MoireHamiltonian
from tmdmoire import TWIST_ANGLES, ENERGY_OFFSETS, LATTICE_CONSTANTS
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

# ─── Parse arguments ─────────────────────────────────────────────────────────

run_id = "default"
chunk_id = 0
n_chunks = 1
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--chunk" and i + 1 < len(args):
        val = args[i + 1]
        if "/" in val:
            chunk_id, n_chunks = map(int, val.split("/"))
        elif "=" in val:
            chunk_id, n_chunks = map(int, val.split("="))
        i += 2
    elif args[i] == "--run-id" and i + 1 < len(args):
        run_id = args[i + 1]
        i += 2
    else:
        i += 1

# ─── Load grid config ────────────────────────────────────────────────────────

config_path = master_folder + "/Inputs/bilayer_fitting/grid_config_k.json"
with open(config_path) as f:
    grid_cfg = json.load(f)

fixed = grid_cfg.get("fixed", {})
Vg = fixed.get("Vg_ev", 0.015)
phiG_deg = fixed.get("phiG_deg", 180)
phiG = phiG_deg / 180 * np.pi
w1p = fixed.get("w1p_ev", -1.3232)
w1d = fixed.get("w1d_ev", 0.5012)
w2p = fixed.get("w2p_ev", -0.1774)
w2d = fixed.get("w2d_ev", 0.0295)

# ─── Load materials ──────────────────────────────────────────────────────────

monolayer_fns = {
    "WSe2": master_folder + "/Inputs/monolayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "/Inputs/monolayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

wse2 = TMDMaterial("WSe2")
wse2.load_fitted(monolayer_fns["WSe2"])
ws2 = TMDMaterial("WS2")
ws2.load_fitted(monolayer_fns["WS2"])

# ─── Build grid ──────────────────────────────────────────────────────────────

mo = grid_cfg["moire"]

list_Vk = np.arange(mo["Vk"]["min_ev"], mo["Vk"]["max_ev"] + mo["Vk"]["step_ev"] * 0.5,
                    mo["Vk"]["step_ev"])
list_phiK = np.arange(mo["phiK"]["min_deg"], mo["phiK"]["max_deg"] + mo["phiK"]["step_deg"] * 0.5,
                      mo["phiK"]["step_deg"])

total_jobs = len(list_Vk) * len(list_phiK)

chunk_size = total_jobs // n_chunks
remainder = total_jobs % n_chunks
start = int(chunk_id * chunk_size + min(chunk_id, remainder))
end = int(start + chunk_size + (1 if chunk_id < remainder else 0))

grid = product(list_Vk, list_phiK)
grid_chunk = islice(grid, start, end)

print(f"Grid: Vk={len(list_Vk)}, phiK={len(list_phiK)}")
print(f"Total grid points: {total_jobs:,}")
print(f"Chunk {chunk_id}/{n_chunks}: points {start}–{end-1} ({end - start} points)")

# ─── Geometry and constants ─────────────────────────────────────────────────

sample = "S11"
n_shells = 2
theta = TWIST_ANGLES[sample]
spreadE = 0.03
n_cells = MoireGeometry.n_cells(n_shells)
geometry = MoireGeometry(theta)

# K-point of the monolayer BZ
k_K = np.array([[4 * np.pi / (3 * LATTICE_CONSTANTS["WSe2"]), 0]])

pars_interlayer = {"stacking": "P", "w1p": w1p, "w2p": w2p, "w1d": w1d, "w2d": w2d}

# ─── Fit helpers ─────────────────────────────────────────────────────────────

def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

def _two_lorentzian(x, a1, c1, g1, a2, c2, g2):
    return _lorentzian(x, a1, c1, g1) + _lorentzian(x, a2, c2, g2)

def compute_and_fit(Vk, phiK_deg):
    phiK = phiK_deg / 180 * np.pi
    pars_V = (Vg, Vk, phiG, phiK)

    moire_ham = MoireHamiltonian(wse2, ws2, geometry)
    evals_raw, evecs_raw = moire_ham.diagonalize(k_K, n_shells, pars_interlayer, pars_V)
    evals_raw = evals_raw[0] + ENERGY_OFFSETS.get(sample, 0.0)
    evecs_raw = evecs_raw[0]

    ab = np.absolute(evecs_raw) ** 2
    weights = np.sum(ab[:22, :], axis=0) + np.sum(ab[22 * n_cells:22 * (1 + n_cells), :], axis=0)

    index_tvb = 28 * n_cells - 1
    index_l = index_tvb - n_cells + np.argmax(weights[index_tvb - n_cells:index_tvb]) + 1

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

    for i in range(len(full_energy_values)):
        weight_list += spreadE / np.pi * full_weight_values[i] / (
            (energy_list - full_energy_values[i]) ** 2 + spreadE ** 2
        )

    # Find 2 peak seeds from weights
    sorted_indices = np.argsort(full_weight_values)[::-1]
    peak_states = []
    seen_centers = []
    for si in sorted_indices:
        e = full_energy_values[si]
        w = full_weight_values[si]
        if w < 1e-4:
            break
        too_close = any(abs(e - c) < 0.01 for c in seen_centers)
        if not too_close:
            peak_states.append((e, w))
            seen_centers.append(e)
        if len(peak_states) == 2:
            break

    if len(peak_states) < 2:
        return None

    peak_states.sort(key=lambda x: x[0], reverse=True)

    model = lmfit.Model(_two_lorentzian)
    params_fit = model.make_params(
        a1=peak_states[0][1], c1=peak_states[0][0], g1=spreadE,
        a2=peak_states[1][1], c2=peak_states[1][0], g2=spreadE,
    )
    for p in ["a1", "a2"]:
        params_fit[p].set(min=0)
    for p in ["g1", "g2"]:
        params_fit[p].set(min=1e-4, max=0.2)

    result = model.fit(weight_list, params_fit, x=energy_list)

    if not result.success:
        return None

    return {
        "c1": result.best_values["c1"],
        "c2": result.best_values["c2"],
        "a1": result.best_values["a1"],
        "a2": result.best_values["a2"],
        "g1": result.best_values["g1"],
        "g2": result.best_values["g2"],
        "redchi": result.redchi,
    }

# ─── Run grid ────────────────────────────────────────────────────────────────

out_dir = Path("Data") / f"edc_grid_k_run_{run_id}"
out_dir.mkdir(parents=True, exist_ok=True)

# Copy grid config and interlayer params into run directory for reproducibility
grid_config_src = master_folder + "/Inputs/bilayer_fitting/grid_config_k.json"
grid_config_dst = out_dir / "grid_config.json"
if not grid_config_dst.exists() or os.path.getmtime(grid_config_src) > grid_config_dst.stat().st_mtime:
    shutil.copy2(grid_config_src, grid_config_dst)

# Save run metadata
meta_fn = out_dir / "run_metadata.json"
if not meta_fn.exists():
    meta = {
        "run_id": run_id,
        "timestamp_start": datetime.datetime.now().isoformat(),
        "n_chunks": n_chunks,
        "grid_sizes": {
            "Vk": len(list_Vk), "phiK": len(list_phiK),
        },
        "fixed_params": {
            "Vg_ev": Vg,
            "phiG_deg": phiG_deg,
            "w1p_ev": w1p,
            "w1d_ev": w1d,
            "w2p_ev": w2p,
            "w2d_ev": w2d,
        },
        "total_points": total_jobs,
    }
    with open(meta_fn, "w") as f:
        json.dump(meta, f, indent=2)

out_fn = out_dir / f"chunk_{chunk_id}_{n_chunks}.h5"

columns = ["Vk", "phiK",
           "c1", "c2", "a1", "a2", "g1", "g2", "redchi"]

t_start = time.perf_counter()
n_success = 0
n_fail = 0

with h5py.File(out_fn, "w") as hf:
    dsets = {col: hf.create_dataset(col, (end - start,), dtype="f8", fillvalue=np.nan)
             for col in columns}

    for i, (Vk, phiK_deg) in enumerate(grid_chunk):
        result = compute_and_fit(Vk, phiK_deg)

        dsets["Vk"][i] = Vk
        dsets["phiK"][i] = phiK_deg

        if result is not None:
            for col in ["c1", "c2", "a1", "a2", "g1", "g2", "redchi"]:
                dsets[col][i] = result[col]
            n_success += 1
        else:
            n_fail += 1

        if (i + 1) % 10 == 0 or i == end - start - 1:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (end - start - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{end-start}] success={n_success} fail={n_fail} "
                  f"rate={rate:.1f} pts/s eta={eta:.0f}s")

elapsed = time.perf_counter() - t_start
print(f"\nChunk {chunk_id}/{n_chunks} done: {n_success} success, {n_fail} fail in {elapsed:.1f}s")
print(f"Results saved to {out_fn}")
