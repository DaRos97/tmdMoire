"""Export moire band data around Gamma for standalone plotting.

Computes the supercell band structure along a G->K line through Gamma
for V_G = 0 and a chosen V_G value, packages eigenvalues + weights + metadata into a
self-contained .npz file. Output goes to scripts/plotsPaper/data/.

Usage:
    source ../PyEnv/bin/activate
    python scripts/export_moire_bands.py
    python scripts/export_moire_bands.py --sample S3 --Vg 11.5 --w1p -1.2 --w1d 0.455 --phiG 175
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

BAND_LO = 26
BAND_HI = 28


def parse_args():
    sample = "S11"
    w1p = -1.220
    w1d = 0.460
    w2p = -0.1694
    w2d = 0.0215
    phiG_deg = 175.0
    vg_meV = 10.5

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--sample" and i + 1 < len(args):
            sample = args[i + 1]
            i += 2
        elif args[i] == "--w1p" and i + 1 < len(args):
            w1p = float(args[i + 1])
            i += 2
        elif args[i] == "--w1d" and i + 1 < len(args):
            w1d = float(args[i + 1])
            i += 2
        elif args[i] == "--w2p" and i + 1 < len(args):
            w2p = float(args[i + 1])
            i += 2
        elif args[i] == "--w2d" and i + 1 < len(args):
            w2d = float(args[i + 1])
            i += 2
        elif args[i] == "--phiG" and i + 1 < len(args):
            phiG_deg = float(args[i + 1])
            i += 2
        elif args[i] == "--Vg" and i + 1 < len(args):
            vg_meV = float(args[i + 1])
            i += 2
        else:
            i += 1

    return sample, w1p, w1d, w2p, w2d, phiG_deg, vg_meV


def main():
    sample, w1p, w1d, w2p, w2d, phiG_deg, vg_meV = parse_args()

    interlayer = {"w1p": w1p, "w1d": w1d, "w2p": w2p, "w2d": w2d}
    phiG_rad = phiG_deg * np.pi / 180.0
    vg_ev = vg_meV / 1000.0
    vg_values = [0.0, vg_ev]
    vg_labels = np.array(["0 meV", f"{vg_meV:.1f} meV"], dtype=object)

    print("Loading monolayer parameters from Inputs/plot_bilayer/")
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")

    wse2 = TMDMaterial("WSe2", params=tb_wse2)
    ws2 = TMDMaterial("WS2", params=tb_ws2)

    theta = TWIST_ANGLES[sample]
    geometry = MoireGeometry(theta)
    moire_ham = MoireHamiltonian(wse2, ws2, geometry)

    n_cells = MoireGeometry.n_cells(N_SHELLS)
    band_start = BAND_LO * n_cells
    band_end = BAND_HI * n_cells
    print(f"n_shells={N_SHELLS}, n_cells={n_cells}, bands {band_start}:{band_end}")

    k_vals = np.linspace(-K_RANGE, K_RANGE, N_K_PTS)
    k_list = np.column_stack([k_vals, np.zeros(N_K_PTS)])

    energy_offset = ENERGY_OFFSETS.get(sample, 0.0)

    export = {
        "k_vals": k_vals,
        "Vg_values_meV": np.array([0.0, vg_meV], dtype=np.float64),
        "Vg_labels": vg_labels,
        "n_shells": N_SHELLS,
        "n_cells": n_cells,
        "n_kpts": N_K_PTS,
        "k_range": K_RANGE,
        "phiG_deg": phiG_deg,
        "interlayer_w1p": w1p,
        "interlayer_w1d": w1d,
        "interlayer_w2p": w2p,
        "interlayer_w2d": w2d,
    }

    for i_probe, (vg, label) in enumerate(zip(vg_values, vg_labels)):
        pars_V = (vg, 0.0, phiG_rad, 0.0)
        print(f"Diagonalizing V_G = {vg*1000:.1f} meV ({N_K_PTS} k-points, {n_cells} cells) ...", flush=True)

        evals_full, evecs_full = moire_ham.diagonalize(
            k_list, N_SHELLS, interlayer, pars_V
        )

        evals = evals_full[:, band_start:band_end] + energy_offset
        evecs = evecs_full[:, :, band_start:band_end]

        weights = compute_weights(evecs, n_cells, pow_factor=2.0, shade_factor_ws2=0.1)

        export[f"evals_{i_probe}"] = evals
        export[f"weights_{i_probe}"] = weights
        print(f"  Done.", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vg_str = vg_meV if vg_meV != int(vg_meV) else int(vg_meV)
    out_fn = OUTPUT_DIR / f"moire_bands_{sample}_k{N_K_PTS}_n{N_SHELLS}_Vg0_{vg_str}.npz"
    np.savez(out_fn, **export)
    print(f"Exported: {out_fn}")


if __name__ == "__main__":
    main()
