"""Standalone EDC TVB–side band distance vs V_G plot.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_edc_vs_V.py.

Produces:
  edc_vs_V.png  -- distance between TVB main and side-band peaks (meV)
                   vs V_G (meV), with ARPES reference line.

Usage:
    python plot_edc_vs_V.py <data.npz>
    python plot_edc_vs_V.py data.npz --output-dir ./figures
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
        print("Usage: python plot_edc_vs_V.py <data.npz> [--output-dir <dir>]")
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

    Vg_vals_meV = d["Vg_vals_meV"]
    distances_meV = d["distances_meV"]
    arpes_distance_meV = float(d["arpes_distance_meV"])
    w1p = float(d["interlayer_w1p"])
    w1d = float(d["interlayer_w1d"])
    w2p = float(d["interlayer_w2p"])
    w2d = float(d["interlayer_w2d"])
    phiG_deg = float(d["phiG_deg"])

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    ax.plot(Vg_vals_meV, distances_meV, "ko-", lw=1.5, markersize=7, label="Computed")
    ax.axhline(arpes_distance_meV, color="red", lw=1.5, ls="--",
               label=f"ARPES = {arpes_distance_meV:.1f} meV")

    ax.set_xlabel(r"$V_G$ (meV)", fontsize=12)
    ax.set_ylabel(r"$\Delta E$ (meV)", fontsize=12)
    ax.set_title(
        f"TVB \u2013 side band distance at \u0393\n"
        f"(w1p={w1p}, w1d={w1d}, w2p={w2p}, w2d={w2d}, "
        f"\u03d5_G={phiG_deg:.0f}\u00b0)",
        fontsize=12,
    )
    ax.legend(fontsize=10)

    out_fn = output_dir / "edc_vs_V.png"
    fig.savefig(out_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fn}")


if __name__ == "__main__":
    main()
