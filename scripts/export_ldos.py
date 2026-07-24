"""Export LDOS data for standalone plotting.

Computes the real-space local density of states along the a1+a2 moire
diagonal and packages everything into a self-contained .npz file.
Output goes to scripts/plotsPaper/data/.

Usage:
    source ../PyEnv/bin/activate
    python scripts/export_ldos.py
    python scripts/export_ldos.py --k-pts 10 --n-shells 2 --e-min -1.28 --e-max -1.11
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tmdmoire.material import TMDMaterial
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.bilayer.ldos import (
    compute_ldos,
    compute_r_list,
    compute_k_grid,
    get_moire_lattice_vectors,
)
from tmdmoire.constants import TWIST_ANGLES, ENERGY_OFFSETS, LATTICE_CONSTANTS
from tmdmoire.utils.kpoints import R_z

INPUT_DIR = Path("Inputs") / "plot_bilayer"
OUTPUT_DIR = Path("scripts") / "plotsPaper" / "data"


def parse_args():
    p = argparse.ArgumentParser(description="Export LDOS data for standalone plotting")
    p.add_argument("--k-pts", type=int, default=10)
    p.add_argument("--r-pts", type=int, default=80)
    p.add_argument("--n-shells", type=int, default=2)
    p.add_argument("--e-min", type=float, default=-1.28)
    p.add_argument("--e-max", type=float, default=-1.11)
    p.add_argument("--delta-e", type=float, default=0.002)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--r-extra", type=float, default=0.166)
    p.add_argument("--sample", default="S11")
    p.add_argument("--theta", type=float, default=None)
    p.add_argument("--Vg", type=float, default=None)
    p.add_argument("--phiG", type=float, default=170.0)
    p.add_argument("--Vk", type=float, default=None)
    p.add_argument("--phiK", type=float, default=None)
    p.add_argument("--w1p", type=float, default=None,
                   help="Override interlayer w1p from interlayer_G.json")
    p.add_argument("--w1d", type=float, default=None,
                   help="Override interlayer w1d from interlayer_G.json")
    p.add_argument("--w2p", type=float, default=None,
                   help="Override interlayer w2p from interlayer_G.json")
    p.add_argument("--w2d", type=float, default=None,
                   help="Override interlayer w2d from interlayer_G.json")
    p.add_argument("--out", type=str, default=None,
                   help="Output filename (default: auto-generated)")
    return p.parse_args()


def compute_monolayer_K(tmd="WSe2"):
    b2 = 4 * np.pi / np.sqrt(3) / LATTICE_CONSTANTS[tmd] * np.array([0, 1])
    b1 = R_z(-np.pi / 3) @ b2
    b6 = R_z(-2 * np.pi / 3) @ b2
    return (b1 + b6) / 3


def main():
    args = parse_args()

    import json
    print("Loading parameters from Inputs/plot_bilayer/")
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")
    with open(INPUT_DIR / "interlayer_G.json") as f:
        ig = json.load(f)
    with open(INPUT_DIR / "interlayer_K.json") as f:
        ik = json.load(f)

    interlayer = {
        "w1p": args.w1p if args.w1p is not None else ig["w1p"],
        "w1d": args.w1d if args.w1d is not None else ig["w1d"],
        "w2p": args.w2p if args.w2p is not None else ig["w2p"],
        "w2d": args.w2d if args.w2d is not None else ig["w2d"],
    }
    Vg = args.Vg if args.Vg is not None else ig["Vg"]
    Vk = args.Vk if args.Vk is not None else ik["Vk"]
    phiG = args.phiG * np.pi / 180
    phiK = (args.phiK if args.phiK is not None else ik["phiK_deg"]) * np.pi / 180

    theta = args.theta if args.theta is not None else TWIST_ANGLES[args.sample]
    energy_offset = ENERGY_OFFSETS.get(args.sample, 0.0)

    wse2 = TMDMaterial("WSe2", params=tb_wse2)
    ws2 = TMDMaterial("WS2", params=tb_ws2)
    geometry = MoireGeometry(theta)
    moire_ham = MoireHamiltonian(wse2, ws2, geometry)

    n_cells = MoireGeometry.n_cells(args.n_shells)
    G_M = geometry.reciprocal_vectors()
    lu = MoireGeometry.lu_table(args.n_shells)
    pars_v = (Vg, Vk, phiG, phiK)

    e_list = np.linspace(args.e_min, args.e_max,
                         int((args.e_max - args.e_min) / args.delta_e))

    print(f"Computing k-grid (k_pts={args.k_pts})")
    k_flat = compute_k_grid(args.k_pts, G_M)

    print(f"Diagonalizing (n_cells={n_cells}, dim={n_cells*44})")
    evals, evecs = moire_ham.diagonalize(k_flat, args.n_shells, interlayer, pars_v)
    evals += energy_offset

    a1, a2 = get_moire_lattice_vectors(G_M)
    r_list, rL = compute_r_list(args.r_pts, a1, a2, extra=args.r_extra)

    print(f"Computing LDOS (r_pts={args.r_pts}, n_e={len(e_list)})")
    ldos = compute_ldos(evals, evecs, k_flat, G_M, lu, n_cells,
                        r_list, e_list, args.eta)

    export = {
        "ldos": ldos,
        "r_list": r_list,
        "e_list": e_list,
        "rL": rL,
        "k_pts": args.k_pts,
        "r_pts": args.r_pts,
        "n_shells": args.n_shells,
        "n_cells": n_cells,
        "e_min": args.e_min,
        "e_max": args.e_max,
        "delta_e": args.delta_e,
        "eta": args.eta,
        "r_extra": args.r_extra,
        "sample": args.sample,
        "theta_deg": theta,
        "energy_offset": energy_offset,
        "Vg": Vg,
        "Vk": Vk,
        "phiG_deg": phiG * 180 / np.pi,
        "phiK_deg": phiK * 180 / np.pi,
        "interlayer_w1p": interlayer["w1p"],
        "interlayer_w1d": interlayer["w1d"],
        "interlayer_w2p": interlayer["w2p"],
        "interlayer_w2d": interlayer["w2d"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_fn = OUTPUT_DIR / args.out
    else:
        phi_str = f"{phiG*180/np.pi:.0f}deg"
        Vg_str = f"{Vg*1000:.1f}meV"
        out_fn = OUTPUT_DIR / f"ldos_{args.sample}_n{args.n_shells}_k{args.k_pts}"
        out_fn = out_fn.with_name(out_fn.name + f"_{Vg_str}_{phi_str}.npz")

    np.savez(out_fn, **export)
    print(f"Exported: {out_fn}")


if __name__ == "__main__":
    main()
