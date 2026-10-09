"""Standalone moire band plot around Gamma.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_moire_bands.py.

Produces:
  moire_bands_gamma.png  -- one panel per exported V_G, with band lines and
                             weight-proportional circles.
  edc_profile_4L_<run_id>.png -- companion Gamma EDC profile when present in
                                 the input export.

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

if __package__:
    from .plot_edc_profile import plot_edc_profile
else:
    from plot_edc_profile import plot_edc_profile


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python plot_moire_bands.py <data.npz> [--output-dir <dir>]")
        sys.exit(1)

    data_path = Path(args[0])
    output_dir = Path(__file__).resolve().parent / "figures"

    i = 1
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    d = np.load(data_path, allow_pickle=True)

    k_vals = d["k_vals"]
    vg_labels = d["Vg_labels"]
    all_evals = [d[f"evals_{i}"] for i in range(len(vg_labels))]
    all_weights = [d[f"weights_{i}"] for i in range(len(vg_labels))]
    if not all_evals:
        raise ValueError(f"No V_G band data found in {data_path}")

    k_range = float(d["k_range"])
    n_shells = int(d["n_shells"])
    phiG_deg = float(d["phiG_deg"])
    w1p = float(d["interlayer_w1p"])
    w1d = float(d["interlayer_w1d"])
    w2p = float(d["interlayer_w2p"])
    w2d = float(d["interlayer_w2d"])

    all_e = np.concatenate([evals.ravel() for evals in all_evals])
    e_min = np.nanmin(all_e)
    e_max = np.nanmax(all_e)
    pad = 0.1 * (e_max - e_min) if (e_max - e_min) > 0 else 0.1
    y_min = e_min - pad
    y_max = e_max + pad

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        len(all_evals),
        figsize=(6.5 * len(all_evals), 7),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, evals, weights, label in zip(axes, all_evals, all_weights, vg_labels):
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

    if "energy_list" in d.files:
        plot_edc_profile(data_path, output_dir)


if __name__ == "__main__":
    main()
