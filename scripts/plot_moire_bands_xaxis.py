"""Plot moire bands along x-axis in mini-BZ for theta=0.

Uses 0 interlayer coupling and 0 moire potential.
Caches eigenvalues to Data/plot_moire_bands_xaxis/ for reuse.

Usage:
    python scripts/plot_moire_bands_xaxis.py [--n-k 200]

Output: Figures/fig_moire_bands_xaxis.png
"""
import argparse
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

INPUT_DIR = Path("Inputs") / "plot_bilayer"
FIGURES_DIR = Path("Figures")
CACHE_DIR = Path("Data") / "plot_moire_bands_xaxis"
OUTPUT_FILE = FIGURES_DIR / "fig_moire_bands_xaxis.png"


def compute_moire_bands(n_k=200, n_shells=2, theta=1.0):
    """Compute moire bands along x-axis for given twist angle."""
    cache_file = CACHE_DIR / f"bands_theta{theta}_nk{n_k}_ns{n_shells}.npz"
    if cache_file.exists():
        print(f"Loading cached bands from {cache_file}")
        d = np.load(cache_file)
        return d["k_vals"], d["evals"], d["K_mag"]

    print("Loading parameters...")
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")

    # Zero SOC parameters (indices 41, 42)
    tb_wse2 = tb_wse2.copy()
    tb_ws2 = tb_ws2.copy()
    tb_wse2[41] = 0.0
    tb_wse2[42] = 0.0
    tb_ws2[41] = 0.0
    tb_ws2[42] = 0.0

    wse2 = TMDMaterial("WSe2", params=tb_wse2)
    ws2 = TMDMaterial("WS2", params=tb_ws2)
    geo = MoireGeometry(theta)
    moire_ham = MoireHamiltonian(wse2, ws2, geo)

    print(f"Twist angle: {geo.theta_deg} deg")
    print(f"Moire length: {geo.moire_length:.4f} A")
    print(f"Mini-BZ rotation (eta): {geo.mini_bz_rotation:.6f} rad ({geo.mini_bz_rotation * 180 / np.pi:.4f} deg)")

    G_M = geo.reciprocal_vectors()
    G1, G2 = G_M[1], G_M[2]
    K_mag = np.linalg.norm((G1 + G2) / 3)
    print(f"|G1| = {np.linalg.norm(G1):.6f} A^-1")
    print(f"|K| = {K_mag:.6f} A^-1")
    print(f"Path: {-5*K_mag:.6f} to {5*K_mag:.6f} A^-1 along x-axis")

    print(f"\nReciprocal lattice vectors:")
    for i in range(1, 7):
        print(f"  G{i} = [{G_M[i][0]:.6f}, {G_M[i][1]:.6f}] A^-1")

    a_moire = geo.moire_length
    eta = geo.mini_bz_rotation
    from tmdmoire.utils.kpoints import R_z
    a1 = a_moire * np.array([1, 0])
    a2 = a_moire * np.array([0.5, np.sqrt(3) / 2])
    a1 = R_z(eta) @ a1
    a2 = R_z(eta) @ a2
    print(f"\nReal-space moire lattice vectors:")
    print(f"  a1 = [{a1[0]:.4f}, {a1[1]:.4f}] A")
    print(f"  a2 = [{a2[0]:.4f}, {a2[1]:.4f}] A")

    k_vals = np.linspace(-5 * K_mag, 5 * K_mag, n_k)
    k_list = np.column_stack([k_vals, np.zeros_like(k_vals)])

    interlayer = {"w1p": 0.0, "w1d": 0.0, "w2p": 0.0, "w2d": 0.0}
    pars_v = (0.0, 0.0, 0.0, 0.0)

    n_cells = MoireGeometry.n_cells(n_shells)
    print(f"\nDiagonalizing (n_k={n_k}, n_shells={n_shells}, n_cells={n_cells}, dim={n_cells * 44})")
    evals, _ = moire_ham.diagonalize(k_list, n_shells, interlayer, pars_v)

    band_indices = list(range(28 * n_cells - 14, 28 * n_cells))
    k_vals_out = k_vals
    evals_out = evals[:, band_indices]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, k_vals=k_vals_out, evals=evals_out, K_mag=K_mag)
    print(f"Saved cache to {cache_file}")
    return k_vals_out, evals_out, K_mag


def print_lattice_info(theta=1.0):
    """Print moire lattice vectors and reciprocal vectors for given twist angle."""
    geo = MoireGeometry(theta)

    print(f"Twist angle: {geo.theta_deg} deg")
    print(f"Moire length: {geo.moire_length:.4f} A")
    print(f"Mini-BZ rotation (eta): {geo.mini_bz_rotation:.6f} rad ({geo.mini_bz_rotation * 180 / np.pi:.4f} deg)")

    G_M = geo.reciprocal_vectors()
    print(f"\nReciprocal lattice vectors (|G1| = {np.linalg.norm(G_M[1]):.6f} A^-1):")
    for i in range(1, 7):
        print(f"  G{i} = [{G_M[i][0]:.6f}, {G_M[i][1]:.6f}] A^-1")

    a_moire = geo.moire_length
    eta = geo.mini_bz_rotation
    from tmdmoire.utils.kpoints import R_z
    a1 = a_moire * np.array([1, 0])
    a2 = a_moire * np.array([0.5, np.sqrt(3) / 2])
    a1 = R_z(eta) @ a1
    a2 = R_z(eta) @ a2
    print(f"\nReal-space moire lattice vectors:")
    print(f"  a1 = [{a1[0]:.4f}, {a1[1]:.4f}] A")
    print(f"  a2 = [{a2[0]:.4f}, {a2[1]:.4f}] A")

    K_mag = np.linalg.norm((G_M[1] + G_M[2]) / 3)
    print(f"\n|K| = {K_mag:.6f} A^-1")
    print(f"Path: {-5*K_mag:.6f} to {5*K_mag:.6f} A^-1 along x-axis")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-k", type=int, default=200, help="Number of k-points")
    parser.add_argument("--n-shells", type=int, default=2, help="Number of moire shells")
    parser.add_argument("--theta", type=float, default=1.0, help="Twist angle in degrees")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print_lattice_info(args.theta)

    k_vals, evals, K_mag = compute_moire_bands(args.n_k, args.n_shells, args.theta)

    fig, ax = plt.subplots(figsize=(6, 4))

    c1 = "black"
    c2 = "red"
    c3 = "blue"
    c_default = "gray"

    n_bands = evals.shape[1]

    for i in range(n_bands):
        for j in range(len(k_vals) - 1):
            k_mid = (k_vals[j] + k_vals[j + 1]) / 2
            k_abs = abs(k_mid)

            band_from_top = n_bands - 1 - i
            k_norm = k_abs / K_mag

            color = c_default

            if k_norm >= 3:
                if band_from_top < 6:
                    color = c3
                elif band_from_top < 8:
                    color = c2
            elif k_norm >= 2:
                if band_from_top < 2:
                    color = c3
                elif band_from_top < 4:
                    color = c2
                elif band_from_top < 12:
                    color = c3
            elif k_norm >= 1.5:
                if band_from_top < 4:
                    color = c2
                elif band_from_top < 6:
                    color = c3
                elif band_from_top < 8:
                    color = c1
            elif k_norm >= 1:
                if band_from_top < 4:
                    color = c2
                elif band_from_top < 6:
                    color = c1
                elif band_from_top < 8:
                    color = c3
            else:
                if band_from_top < 2:
                    color = c1
                elif band_from_top < 11:
                    color = c2

            ax.plot(k_vals[j:j+2], evals[j:j+2, i], color=color, lw=1.0)

    for k in np.arange(-5 * K_mag, 5 * K_mag + K_mag, K_mag):
        ax.axvline(k, color="gray", lw=0.5, ls="--")

    ax.set_xticks([-4*K_mag, -3*K_mag, -2*K_mag, -K_mag, 0, K_mag, 2*K_mag, 3*K_mag, 4*K_mag])
    ax.set_xticklabels(["K'", r"$\Gamma$", "K", "K'", r"$\Gamma$", "K", "K'", r"$\Gamma$", "K"])
    ax.set_yticks([])
    ax.set_ylim(-1.054, -1.048)

    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
