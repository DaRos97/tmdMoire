"""Compute real-space local density of states (LDOS) for moire bilayer.

Diagonalizes the supercell Hamiltonian over a uniform k-grid covering
the mini-Brillouin zone, then reconstructs the spatial wavefunction at
each real-space point to compute:

    LDOS(r, E) = (1/N_k) * Σ_{k,n} |ψ_{nk}(r)|² * η / [π ((E - E_{nk})² + η²)]

Usage:
    source ../PyEnv/bin/activate
    python scripts/compute_ldos.py [--k-pts 12] [--r-pts 300] ...

All monolayer and interlayer parameters are loaded from Inputs/plot_bilayer/.

Output:
    Data/ldos/<diag_params>/diag.npz, metadata.json
    Data/ldos/<diag_params>/ldos_<ldos_args>/ldos.npz, ldos.png
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from tmdmoire.constants import ENERGY_OFFSETS, TWIST_ANGLES, LATTICE_CONSTANTS
from tmdmoire.utils.kpoints import R_z

CACHE_ROOT = Path("Data") / "ldos"
INPUT_DIR = Path("Inputs") / "plot_bilayer"


def compute_monolayer_K(tmd="WSe2"):
    b2 = 4 * np.pi / np.sqrt(3) / LATTICE_CONSTANTS[tmd] * np.array([0, 1])
    b1 = R_z(-np.pi / 3) @ b2
    b6 = R_z(-2 * np.pi / 3) @ b2
    return (b1 + b6) / 3


def parse_args():
    parser = argparse.ArgumentParser(description="Compute real-space LDOS for moire bilayer")
    parser.add_argument("--k-pts", type=int, default=12,
                        help="Number of k-points per mini-BZ side (default: 12)")
    parser.add_argument("--r-pts", type=int, default=300,
                        help="Number of real-space grid points (default: 300)")
    parser.add_argument("--r-extra", type=float, default=0.0,
                        help="Fraction of period to extend past the full a1+a2 path (default: 0)")
    parser.add_argument("--n-shells", type=int, default=2,
                        help="Number of moire shells (default: 2)")
    parser.add_argument("--e-min", type=float, default=-1.0,
                        help="Minimum energy in eV (default: -1.0)")
    parser.add_argument("--e-max", type=float, default=0.0,
                        help="Maximum energy in eV (default: 0.0)")
    parser.add_argument("--delta-e", type=float, default=0.005,
                        help="Energy grid spacing in eV (default: 0.005)")
    parser.add_argument("--eta", type=float, default=0.01,
                        help="Lorentzian broadening in eV (default: 0.01)")
    parser.add_argument("--center", choices=["G", "K"], default="G",
                        help="BZ point to center the k-grid on (default: G)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cache and recompute")
    parser.add_argument("--sample", default="S11",
                        help="Sample name for energy offset and twist angle (default: S11)")
    parser.add_argument("--theta", type=float, default=None,
                        help="Twist angle in degrees (overrides sample)")
    parser.add_argument("--Vg", type=float, default=None,
                        help="Override moire potential at Gamma (eV)")
    parser.add_argument("--Vk", type=float, default=None,
                        help="Override moire potential at K (eV)")
    parser.add_argument("--phiG", type=float, default=None,
                        help="Override moire phase at Gamma (degrees)")
    parser.add_argument("--phiK", type=float, default=None,
                        help="Override moire phase at K (degrees)")
    return parser.parse_args()


def load_params():
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")
    with open(INPUT_DIR / "interlayer_G.json") as f:
        interlayer_g = json.load(f)
    with open(INPUT_DIR / "interlayer_K.json") as f:
        interlayer_k = json.load(f)

    interlayer = {
        "w1p": interlayer_g["w1p"],
        "w1d": interlayer_g["w1d"],
        "w2p": interlayer_g["w2p"],
        "w2d": interlayer_g["w2d"],
    }
    moire = {
        "Vg": interlayer_g["Vg"],
        "phiG": interlayer_g["phiG_deg"] * np.pi / 180,
        "Vk": interlayer_k["Vk"],
        "phiK": interlayer_k["phiK_deg"] * np.pi / 180,
    }
    return tb_wse2, tb_ws2, interlayer, moire


def mono_hash(tb_wse2, tb_ws2):
    h = hashlib.sha256(tb_wse2.tobytes() + tb_ws2.tobytes())
    return h.hexdigest()[:8]


def diag_dir_name(interlayer, moire, theta, n_shells, k_pts, tb_wse2, tb_ws2, center):
    Vg_meV = moire["Vg"] * 1000
    Vk_meV = moire["Vk"] * 1000
    phiG_deg = moire["phiG"] * 180 / np.pi
    phiK_deg = moire["phiK"] * 180 / np.pi
    mhash = mono_hash(tb_wse2, tb_ws2)
    return (f"diag_{mhash}_{interlayer['w1p']:.4f}_{interlayer['w1d']:.4f}_"
            f"{interlayer['w2p']:.4f}_{interlayer['w2d']:.4f}_"
            f"{Vg_meV:.1f}_{phiG_deg:.1f}_{Vk_meV:.1f}_{phiK_deg:.1f}_"
            f"t{theta:.1f}_n{n_shells}_k{k_pts}_{center}")


def ldos_dir_name(r_pts, e_min, e_max, delta_e, eta, r_extra):
    return f"ldos_{r_pts}_{e_min:.2f}_{e_max:.2f}_{delta_e:.4f}_{eta:.4f}_re{r_extra:.3f}"


def main():
    args = parse_args()

    print("Loading parameters from Inputs/plot_bilayer/")
    tb_wse2, tb_ws2, interlayer, moire = load_params()

    if args.Vg is not None:
        moire["Vg"] = args.Vg
    if args.Vk is not None:
        moire["Vk"] = args.Vk
    if args.phiG is not None:
        moire["phiG"] = args.phiG * np.pi / 180
    if args.phiK is not None:
        moire["phiK"] = args.phiK * np.pi / 180

    theta = args.theta
    if theta is None:
        theta = TWIST_ANGLES.get(args.sample, 2.8)

    energy_offset = ENERGY_OFFSETS.get(args.sample, 0.0)

    wse2 = TMDMaterial("WSe2", params=tb_wse2)
    ws2 = TMDMaterial("WS2", params=tb_ws2)
    geometry = MoireGeometry(theta)
    moire_ham = MoireHamiltonian(wse2, ws2, geometry)

    n_cells = MoireGeometry.n_cells(args.n_shells)
    G_M = geometry.reciprocal_vectors()
    lu = MoireGeometry.lu_table(args.n_shells)
    pars_v = (moire["Vg"], moire["Vk"], moire["phiG"], moire["phiK"])

    e_list = np.linspace(args.e_min, args.e_max,
                         int((args.e_max - args.e_min) / args.delta_e))

    dd_name = diag_dir_name(interlayer, moire, theta, args.n_shells,
                            args.k_pts, tb_wse2, tb_ws2, args.center)
    ld_name = ldos_dir_name(args.r_pts, args.e_min, args.e_max,
                            args.delta_e, args.eta, args.r_extra)

    diag_dir = CACHE_ROOT / dd_name
    ldos_dir = diag_dir / ld_name
    diag_file = diag_dir / "diag.npz"
    ldos_file = ldos_dir / "ldos.npz"

    diag_cached = diag_file.exists() and not args.no_cache
    ldos_cached = ldos_file.exists() and not args.no_cache

    if diag_cached:
        print(f"Loading diagonalization from {dd_name}")
        d = np.load(diag_file, allow_pickle=True)
        evals = d["evals"]
        evecs = d["evecs"]
        k_flat = d["k_flat"]
    else:
        print(f"Computing k-grid (k_pts={args.k_pts}, center={args.center})")
        k_center = compute_monolayer_K("WSe2") if args.center == "K" else None
        k_flat = compute_k_grid(args.k_pts, G_M, center=k_center)

        print(f"Diagonalizing supercell (k_pts²={args.k_pts**2}, "
              f"n_cells={n_cells}, dim={n_cells * 44})")
        evals, evecs = moire_ham.diagonalize(k_flat, args.n_shells, interlayer, pars_v)
        evals += energy_offset

        print(f"Saving diagonalization to {diag_file}")
        diag_dir.mkdir(parents=True, exist_ok=True)
        np.savez(diag_file, evals=evals, evecs=evecs, k_flat=k_flat)

        meta = {
            "k_pts": args.k_pts,
            "n_shells": args.n_shells,
            "n_cells": n_cells,
            "theta_deg": theta,
            "sample": args.sample,
            "energy_offset": energy_offset,
            "center": args.center,
            "interlayer": interlayer,
            "moire": {
                "Vg_ev": moire["Vg"],
                "Vg_meV": moire["Vg"] * 1000,
                "phiG_deg": moire["phiG"] * 180 / np.pi,
                "phiG_rad": moire["phiG"],
                "Vk_ev": moire["Vk"],
                "Vk_meV": moire["Vk"] * 1000,
                "phiK_deg": moire["phiK"] * 180 / np.pi,
                "phiK_rad": moire["phiK"],
            },
        }
        with open(diag_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    if ldos_cached:
        print(f"Loading LDOS from {ld_name}")
        ld = np.load(ldos_file, allow_pickle=True)
        ldos_result = ld["ldos"]
        r_list = ld["r_list"]
        e_list = ld["e_list"]
        rL = ld["rL"]
    else:
        a1, a2 = get_moire_lattice_vectors(G_M)
        r_list, rL = compute_r_list(args.r_pts, a1, a2, extra=args.r_extra)

        print(f"Computing LDOS (r_pts={args.r_pts}, n_e={len(e_list)}, "
              f"n_k={args.k_pts**2})")
        ldos_result = compute_ldos(evals, evecs, k_flat, G_M, lu, n_cells,
                                   r_list, e_list, args.eta)

        print(f"Saving LDOS to {ldos_file}")
        ldos_dir.mkdir(parents=True, exist_ok=True)
        np.savez(ldos_file, ldos=ldos_result, r_list=r_list, e_list=e_list, rL=rL)

        meta = {
            "r_pts": args.r_pts,
            "r_extra": args.r_extra,
            "rL": float(rL),
            "e_min": args.e_min,
            "e_max": args.e_max,
            "delta_e": args.delta_e,
            "eta": args.eta,
        }
        with open(ldos_dir / "ldos_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    print("Plotting LDOS")
    r_norm = np.linalg.norm(r_list, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    s_ = 20
    X, Y = np.meshgrid(e_list, r_norm)
    mesh = ax.pcolormesh(X, Y, ldos_result,
                         cmap="hot", shading="auto")
    ax.invert_yaxis()

    ax.set_yticks([0, rL / 3, 2 * rL / 3, rL],
                  [r"W/W", r"Se/W", r"W/S", r"W/W"],
                  size=s_)
    ax.set_xlabel("Energy [eV]", size=s_)

    title = (f"LDOS  V$_g$={moire['Vg']*1000:.1f} meV  "
             f"$\\phi_G$={moire['phiG']*180/np.pi:.1f}°  "
             f"center={args.center}")
    ax.set_title(title, size=s_)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_ticks([])
    cbar.ax.text(1.3, 0.02, "low", ha="left", va="bottom",
                 transform=cbar.ax.transAxes, fontsize=s_)
    cbar.ax.text(1.3, 0.98, "high", ha="left", va="top",
                 transform=cbar.ax.transAxes, fontsize=s_)

    plot_file = ldos_dir / "ldos.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_file}")

    print("Done")


if __name__ == "__main__":
    main()
