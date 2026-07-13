"""Export moire band data around Gamma for standalone plotting.

Computes the supercell band structure along a G->K line through Gamma
for V_G = 0 and 12 meV, packages eigenvalues + weights + metadata into a
self-contained .npz file. Output goes to scripts/plotsPaper/data/.

Usage:
    source ../PyEnv/bin/activate
    python scripts/export_moire_bands.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tmdmoire.material import TMDMaterial
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.bilayer.intensity import compute_weights
from tmdmoire.constants import TWIST_ANGLES, ENERGY_OFFSETS

INPUT_DIR = Path("Inputs") / "plot_bilayer"
OUTPUT_DIR = Path("scripts") / "plotsPaper" / "data"

K_RANGE = 0.4
N_K_PTS = 301
N_SHELLS = 2
SAMPLE = "S11"

BAND_LO = 26
BAND_HI = 28

INTERLAYER = {"w1p": -1.220, "w1d": 0.460, "w2p": -0.1694, "w2d": 0.0215}

PHI_G = 175.0 * np.pi / 180.0
V_G_VALUES = [0.0, 0.0105]
V_G_LABELS = ["0 meV", "10.5 meV"]


def main():
    print("Loading monolayer parameters from Inputs/plot_bilayer/")
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")

    wse2 = TMDMaterial("WSe2", params=tb_wse2)
    ws2 = TMDMaterial("WS2", params=tb_ws2)

    theta = TWIST_ANGLES[SAMPLE]
    geometry = MoireGeometry(theta)
    moire_ham = MoireHamiltonian(wse2, ws2, geometry)

    n_cells = MoireGeometry.n_cells(N_SHELLS)
    band_start = BAND_LO * n_cells
    band_end = BAND_HI * n_cells
    print(f"n_shells={N_SHELLS}, n_cells={n_cells}, bands {band_start}:{band_end}")

    k_vals = np.linspace(-K_RANGE, K_RANGE, N_K_PTS)
    k_list = np.column_stack([k_vals, np.zeros(N_K_PTS)])

    energy_offset = ENERGY_OFFSETS.get(SAMPLE, 0.0)

    export = {
        "k_vals": k_vals,
        "Vg_values_meV": np.array([0, 10.5], dtype=np.float64),
        "Vg_labels": np.array(V_G_LABELS, dtype=object),
        "n_shells": N_SHELLS,
        "n_cells": n_cells,
        "n_kpts": N_K_PTS,
        "k_range": K_RANGE,
        "phiG_deg": 175.0,
        "interlayer_w1p": INTERLAYER["w1p"],
        "interlayer_w1d": INTERLAYER["w1d"],
        "interlayer_w2p": INTERLAYER["w2p"],
        "interlayer_w2d": INTERLAYER["w2d"],
    }

    for i, (vg, label) in enumerate(zip(V_G_VALUES, V_G_LABELS)):
        pars_V = (vg, 0.0, PHI_G, 0.0)
        print(f"Diagonalizing V_G = {vg*1000:.0f} meV ({N_K_PTS} k-points, {n_cells} cells) ...", flush=True)

        evals_full, evecs_full = moire_ham.diagonalize(
            k_list, N_SHELLS, INTERLAYER, pars_V
        )

        evals = evals_full[:, band_start:band_end] + energy_offset
        evecs = evecs_full[:, :, band_start:band_end]

        weights = compute_weights(evecs, n_cells, pow_factor=2.0, shade_factor_ws2=0.1)

        export[f"evals_{i}"] = evals
        export[f"weights_{i}"] = weights
        print(f"  Done.", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fn = OUTPUT_DIR / f"moire_bands_k{N_K_PTS}_n{N_SHELLS}_Vg0_10.5.npz"
    np.savez(out_fn, **export)
    print(f"Exported: {out_fn}")


if __name__ == "__main__":
    main()
