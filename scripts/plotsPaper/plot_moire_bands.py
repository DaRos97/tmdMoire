"""Standalone moire band plot around Gamma.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_moire_bands.py.

Produces:
  moire_bands_gamma.png  -- 1x2 panels (V_G = 0, 12 meV) with band lines
                            and weight-proportional circles.

Usage:
    python plot_moire_bands.py <data.npz>
    python plot_moire_bands.py data.npz --output-dir ./figures
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python plot_moire_bands.py <data.npz> [--output-dir <dir>]")
        sys.exit(1)

    data_path = Path(args[0])
    output_dir = Path("figures")

    i = 1
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    d = np.load(data_path, allow_pickle=True)

    k_vals = d["k_vals"]
    Vg_labels = d["Vg_labels"]
    evals_0 = d["evals_0"]
    weights_0 = d["weights_0"]
    evals_1 = d["evals_1"]
    weights_1 = d["weights_1"]

    all_evals = [evals_0, evals_1]
    all_weights = [weights_0, weights_1]

    k_range = float(d["k_range"])
    n_shells = int(d["n_shells"])
    phiG_deg = float(d["phiG_deg"])
    w1p = float(d["interlayer_w1p"])
    w1d = float(d["interlayer_w1d"])
    w2p = float(d["interlayer_w2p"])
    w2d = float(d["interlayer_w2d"])

    y_min = -1.5
    y_max = -1.0

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True, constrained_layout=True)

    for ax, evals, weights, label in zip(axes, all_evals, all_weights, Vg_labels):
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
                        c="#1f77b4", alpha=1.0, zorder=2,
                        edgecolors="none", linewidths=0,
                    )

        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)", fontsize=12)
        ax.set_title(f"$V_G = {label}$", fontsize=14, fontweight="bold")
        ax.set_xlim(-k_range, k_range)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("Energy (eV)", fontsize=12)

    fig.suptitle(
        f"Moir\u00e9 potential effect on bands around \u0393\n"
        f"(n_shells={n_shells}, \u03d5_G={phiG_deg:.0f}\u00b0, "
        f"w1p={w1p}, w1d={w1d}, w2p={w2p}, w2d={w2d})",
        fontsize=14, fontweight="bold", y=1.08,
    )

    out_fn = output_dir / "moire_bands_gamma.png"
    fig.savefig(out_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fn}")


if __name__ == "__main__":
    main()
