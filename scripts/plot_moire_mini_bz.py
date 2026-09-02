"""Compute and plot moire band structure along a mini-BZ path.

Path: Gamma -> K -> M -> K' -> Gamma along the G1+G2 direction.

Usage:
    source ../PyEnv/bin/activate
    python scripts/plot_moire_mini_bz.py [--k-pts 50] [--n-shells 2] [--no-cache]

Parameters loaded from Inputs/plot_bilayer/.
Output: Data/moire_mini_bz/diag_<hash>/diag.npz
        Data/moire_mini_bz/diag_<hash>/bands.png
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
from tmdmoire.constants import ENERGY_OFFSETS, TWIST_ANGLES, LATTICE_CONSTANTS
from tmdmoire.utils.kpoints import R_z

INPUT_DIR = Path("Inputs") / "plot_bilayer"
CACHE_ROOT = Path("Data") / "moire_mini_bz"


def compute_monolayer_K(tmd="WSe2"):
    b2 = 4 * np.pi / np.sqrt(3) / LATTICE_CONSTANTS[tmd] * np.array([0, 1])
    b1 = R_z(-np.pi / 3) @ b2
    return (b1 + b2) / 3


def build_path(G1, G2, k_pts_per_seg):
    """Build Gamma -> K -> M -> K' -> Gamma path along G1+G2 direction.
    
    k_pts_per_seg points per segment: Gamma->K, K->K', K'->Gamma.
    M is the midpoint of K->K'.
    Total k-points = 3 * k_pts_per_seg.
    """
    direction = G1 + G2
    k_list = []
    labels = [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"]
    label_positions = []

    # Gamma -> K (t: 0 -> 1/3)
    for i in range(k_pts_per_seg):
        t = (1/3) * i / k_pts_per_seg
        k_list.append(t * direction)
    label_positions.append(0)

    # K -> K' (t: 1/3 -> 2/3), M at midpoint (t=1/2)
    for i in range(k_pts_per_seg):
        t = 1/3 + (1/3) * i / k_pts_per_seg
        k_list.append(t * direction)
    label_positions.append(k_pts_per_seg)
    m_idx = k_pts_per_seg + k_pts_per_seg // 2

    # K' -> Gamma (t: 2/3 -> 1)
    for i in range(k_pts_per_seg):
        t = 2/3 + (1/3) * i / k_pts_per_seg
        k_list.append(t * direction)
    label_positions.append(2 * k_pts_per_seg)

    # Add final Gamma point
    k_list.append(direction)
    label_positions.append(3 * k_pts_per_seg)

    k_list = np.array(k_list)
    labels.insert(2, "M")
    label_positions.insert(2, m_idx)
    return k_list, labels, label_positions


def param_hash(tb_wse2, tb_ws2, interlayer, moire, theta, n_shells, k_pts, center):
    h = hashlib.sha256()
    h.update(tb_wse2.tobytes())
    h.update(tb_ws2.tobytes())
    for v in interlayer.values():
        h.update(f"{v:.8f}".encode())
    for v in moire.values():
        h.update(f"{v:.8f}".encode())
    h.update(f"{theta:.8f}{n_shells}{k_pts}{center}".encode())
    return h.hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser(description="Plot moire mini-BZ band structure")
    parser.add_argument("--k-pts", type=int, default=50,
                        help="K-points per segment (default: 50)")
    parser.add_argument("--n-shells", type=int, default=2,
                        help="Number of moire shells (default: 2)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cache and recompute")
    parser.add_argument("--coupling-type", type=str, default="parallel",
                        choices=["parallel", "anti_parallel"],
                        help="Interlayer coupling type (default: parallel)")
    parser.add_argument("--center", choices=["G", "K"], default="G",
                        help="BZ point to center the path on (default: G)")
    parser.add_argument("--theta", type=float, default=None,
                        help="Twist angle in degrees (overrides sample)")
    parser.add_argument("--Vg", type=float, default=None,
                        help="Override moire potential at Gamma (eV)")
    parser.add_argument("--Vk", type=float, default=None,
                        help="Override moire potential at K (eV)")
    parser.add_argument("--phiG", type=float, default=None,
                        help="Override moire phase at Gamma (degrees)")
    args = parser.parse_args()

    print("Loading parameters from Inputs/plot_bilayer/")
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")
    with open(INPUT_DIR / "interlayer_G.json") as f:
        interlayer_g = json.load(f)

    sample = "S11"
    theta = args.theta if args.theta is not None else TWIST_ANGLES.get(sample, 2.8)
    energy_offset = ENERGY_OFFSETS.get(sample, 0.0)

    if args.coupling_type == "parallel":
        interlayer = {
            "w1p": interlayer_g["w1p"],
            "w1d": interlayer_g["w1d"],
            "w2p": interlayer_g["w2p"],
            "w2d": interlayer_g["w2d"],
        }
    else:
        interlayer = {
            "w3p": 0.5781,
            "w3d": -0.4076,
        }
    moire = {
        "Vg": interlayer_g["Vg"],
        "phiG": interlayer_g["phiG_deg"] * np.pi / 180,
        "Vk": 0.0,
        "phiK": 0.0,
    }
    if args.Vg is not None:
        moire["Vg"] = args.Vg
    if args.Vk is not None:
        moire["Vk"] = args.Vk
    if args.phiG is not None:
        moire["phiG"] = args.phiG * np.pi / 180

    n_cells = MoireGeometry.n_cells(args.n_shells)
    h = param_hash(tb_wse2, tb_ws2, interlayer, moire, theta, args.n_shells, args.k_pts, args.center)
    diag_dir = CACHE_ROOT / f"diag_{h}"
    diag_file = diag_dir / "diag.npz"

    if diag_file.exists() and not args.no_cache:
        print(f"Loading diagonalization from {diag_dir.name}")
        d = np.load(diag_file, allow_pickle=True)
        evals = d["evals"]
        k_list = d["k_list"]
        label_positions = list(d["label_positions"])
        labels = list(d["labels"])
    else:
        wse2 = TMDMaterial("WSe2", params=tb_wse2)
        ws2 = TMDMaterial("WS2", params=tb_ws2)
        geometry = MoireGeometry(theta)
        moire_ham = MoireHamiltonian(wse2, ws2, geometry)

        G_M = geometry.reciprocal_vectors()
        G1, G2 = G_M[1], G_M[2]

        k_center = compute_monolayer_K("WSe2") if args.center == "K" else None

        print(f"Building path (k_pts={args.k_pts}/seg, n_cells={n_cells}, dim={n_cells * 44}, center={args.center})")
        k_list, labels, label_positions = build_path(G1, G2, args.k_pts)
        if k_center is not None:
            k_list = k_list + k_center

        pars_v = (moire["Vg"], moire["Vk"], moire["phiG"], moire["phiK"])
        print(f"Diagonalizing (n_k={len(k_list)}, dim={n_cells * 44}, coupling={args.coupling_type})")
        evals, _ = moire_ham.diagonalize(k_list, args.n_shells, interlayer, pars_v,
                                         coupling_type=args.coupling_type)
        evals += energy_offset

        print(f"Saving diagonalization to {diag_file}")
        diag_dir.mkdir(parents=True, exist_ok=True)
        np.savez(diag_file, evals=evals, k_list=k_list,
                 label_positions=np.array(label_positions), labels=np.array(labels))

        meta = {
            "k_pts": args.k_pts,
            "n_shells": args.n_shells,
            "n_cells": n_cells,
            "theta_deg": theta,
            "sample": sample,
            "energy_offset": energy_offset,
            "coupling_type": args.coupling_type,
            "center": args.center,
            "interlayer": interlayer,
            "moire": {
                "Vg_ev": moire["Vg"],
                "Vg_meV": moire["Vg"] * 1000,
                "phiG_deg": moire["phiG"] * 180 / np.pi,
            },
        }
        with open(diag_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    print("Plotting")
    n_bands = 10
    band_indices = list(range(28 * n_cells - n_bands, 28 * n_cells))

    dist = np.linalg.norm(np.diff(k_list, axis=0, prepend=k_list[:1]), axis=1)
    x = np.cumsum(dist)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ib in band_indices:
        ax.plot(x, evals[:, ib], color="steelblue", lw=1.0)

    for pos, label in zip(label_positions, labels):
        ax.axvline(x[pos], color="gray", lw=0.5, ls="--")
        ax.text(x[pos], ax.get_ylim()[0], label, ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_xlabel("Path in mini-BZ", fontsize=12)
    if args.coupling_type == "parallel":
        param_str = (f"w1p={interlayer['w1p']:.2f}  w1d={interlayer['w1d']:.2f}  "
                     f"w2p={interlayer['w2p']:.3f}  w2d={interlayer['w2d']:.3f}  "
                     f"Vg={moire['Vg']*1000:.1f} meV  phiG={interlayer_g['phiG_deg']:.0f}°")
    else:
        param_str = (f"w3p={interlayer['w3p']:.4f}  w3d={interlayer['w3d']:.4f}  "
                     f"Vg={moire['Vg']*1000:.1f} meV  phiG={interlayer_g['phiG_deg']:.0f}°")
    ax.set_title(f"Moire mini-BZ bands ({args.coupling_type}, center={args.center})\n{param_str}", fontsize=11)
    ax.set_xlim(x[0], x[-1])

    fig.savefig(diag_dir / "bands.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {diag_dir / 'bands.png'}")


if __name__ == "__main__":
    main()
