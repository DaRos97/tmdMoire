"""Show the effect of the moire potential on bands around Gamma.

Computes the supercell bands along a G->K line through Gamma (±0.4 Å⁻¹)
for three values of V_G (0, 12, 25 meV), overlays weight-proportional
blue dots, and saves a 3-panel comparison figure.

Usage:
    source ../PyEnv/bin/activate
    python scripts/dev/show_moire_effect.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tmdmoire.material import TMDMaterial
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.bilayer.intensity import compute_weights
from tmdmoire.constants import TWIST_ANGLES, ENERGY_OFFSETS

INPUT_DIR = Path("Inputs") / "plot_bilayer"
OUTPUT_DIR = Path("Data") / "show_moire_effect"

K_RANGE = 0.4
N_K_PTS = 100
N_SHELLS = 1
SAMPLE = "S11"

BAND_LO = 18
BAND_HI = 28
BAND_LO_YLIM = 22
BAND_HI_YLIM = 28

INTERLAYER = {"w1p": -1.378, "w1d": 0.511, "w2p": -0.139, "w2d": 0.011}

PHI_G = 176.0 * np.pi / 180.0
V_G_VALUES = [0.0, 0.012, 0.025]
V_G_LABELS = ["0 meV", "12 meV", "25 meV"]


def _cache_filename():
    parts = [
        f"moire_data",
        f"k{N_K_PTS}",
        f"n{N_SHELLS}",
        f"w1p{INTERLAYER['w1p']:.3f}",
        f"w1d{INTERLAYER['w1d']:.3f}",
        f"w2p{INTERLAYER['w2p']:.3f}",
        f"w2d{INTERLAYER['w2d']:.3f}",
    ]
    vg_str = "_".join(f"Vg{int(v*1000)}" for v in V_G_VALUES)
    parts.append(vg_str)
    return OUTPUT_DIR / ("_".join(parts) + ".npz")


def main():
    cache_fn = _cache_filename()

    if cache_fn.exists():
        print(f"Loading cached data from {cache_fn.name}")
        data = np.load(cache_fn, allow_pickle=True)
        k_vals = data["k_vals"]
        all_evals = [data[f"evals_{i}"] for i in range(len(V_G_VALUES))]
        all_weights = [data[f"weights_{i}"] for i in range(len(V_G_VALUES))]
    else:
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
        print(f"n_cells = {n_cells}, keeping bands {band_start}:{band_end} (global)")

        k_vals = np.linspace(-K_RANGE, K_RANGE, N_K_PTS)
        k_list = np.column_stack([k_vals, np.zeros(N_K_PTS)])

        energy_offset = ENERGY_OFFSETS.get(SAMPLE, 0.0)

        all_evals = []
        all_weights = []

        for vg, label in zip(V_G_VALUES, V_G_LABELS):
            pars_V = (vg, 0.0, PHI_G, 0.0)
            print(f"Diagonalizing V_G = {vg*1000:.0f} meV ({N_K_PTS} k-points, {n_cells} cells) ...", flush=True)

            evals_full, evecs_full = moire_ham.diagonalize(
                k_list, N_SHELLS, INTERLAYER, pars_V
            )

            evals = evals_full[:, band_start:band_end] + energy_offset
            evecs = evecs_full[:, :, band_start:band_end]

            weights = compute_weights(evecs, n_cells, pow_factor=2.0, shade_factor_ws2=0.1)

            all_evals.append(evals)
            all_weights.append(weights)
            print(f"  Done.", flush=True)

        print(f"Saving data to {cache_fn.name}")
        save_dict = {"k_vals": k_vals}
        for i, (evals, weights) in enumerate(zip(all_evals, all_weights)):
            save_dict[f"evals_{i}"] = evals
            save_dict[f"weights_{i}"] = weights
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_fn, **save_dict)

    y_min = -1.5
    y_max = -1.0

    print("Plotting (energy range: -1.5 to -1.0 eV)")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True, constrained_layout=True)

    for ax, evals, weights, label in zip(axes, all_evals, all_weights, V_G_LABELS):
        for ib in range(evals.shape[1]):
            ax.plot(k_vals, evals[:, ib], color="lightgray", lw=0.5, alpha=0.5, zorder=1)

        w_max = weights.max()
        if w_max > 0:
            w_norm = weights / w_max
            dot_sizes = 80 * w_norm
            for ib in range(evals.shape[1]):
                mask = dot_sizes[:, ib] > 0
                if mask.any():
                    ax.scatter(
                        k_vals[mask], evals[mask, ib],
                        s=dot_sizes[mask, ib],
                        c="#1f77b4", alpha=0.6, zorder=2,
                        edgecolors="none", linewidths=0,
                    )

        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel(r"$k$ (\AA$^{-1}$)", fontsize=12)
        ax.set_title(f"$V_G = {label}$", fontsize=14, fontweight="bold")
        ax.set_xlim(-K_RANGE, K_RANGE)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("Energy (eV)", fontsize=12)

    interlayer_str = ", ".join(
        f"{k}={v}" for k, v in INTERLAYER.items()
    )
    fig.suptitle(
        f"Moiré potential effect on bands around Γ\n"
        f"(n_shells={N_SHELLS}, ϕ_G=176°, {interlayer_str})",
        fontsize=14, fontweight="bold", y=1.02,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fn = OUTPUT_DIR / (cache_fn.stem + ".png")
    fig.savefig(out_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fn}")


if __name__ == "__main__":
    main()
