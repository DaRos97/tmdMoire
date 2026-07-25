"""Plot fitted TB parameters of WSe2 and WS2 side-by-side.

Bars show fitted values, horizontal lines show DFT reference values,
dashed lines show bounds. Excludes offset and SOC parameters.

Usage
-----
::

    python scripts/plot_monolayer_params.py
    python scripts/plot_monolayer_params.py --output-dir Figures
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmdmoire.material import TMDMaterial
from tmdmoire.constants import FORMATTED_NAMES
from tmdmoire.utils.paths import get_repo_root


GROUP_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
GROUP_LABELS = [r"$\varepsilon$", r"$t_1$", r"$t_5$", r"$t_6$"]
GROUP_BOUNDS = [(0, 6), (7, 27), (28, 35), (36, 39)]
BOX_STYLE = dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor="black", linewidth=1, alpha=1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Plot fitted TB parameters for WSe2 and WS2."
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: Figures/).")
    args = parser.parse_args()

    master_folder = get_repo_root()
    bilayer_dir = Path(master_folder) / "Inputs" / "bilayer_fitting"

    materials = ["WSe2", "WS2"]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("#F7F7F7")

    for col, (tmd, ax) in enumerate(zip(materials, axes.flat)):
        matches = sorted(bilayer_dir.glob(f"tb_{tmd}*.npy"))
        if not matches:
            print(f"[WARNING] No tb_{tmd}*.npy found, skipping.")
            continue
        pars = np.load(matches[0])
        # Exclude offset (40) and SOC (41, 42)
        pars_plot = pars[:40]
        npars = len(pars_plot)

        mat = TMDMaterial(tmd)
        dft = mat.dft_params[:40]

        ax.set_facecolor("#F7F7F7")
        x = np.arange(npars)

        # Group background bands
        for gi, (start, end) in enumerate(GROUP_BOUNDS):
            ax.axvspan(start - 0.5, end + 0.5, color=GROUP_COLORS[gi],
                       alpha=0.07, zorder=0)

        # Colours and bounds
        param_colors = [""] * npars
        param_bound = [None] * npars
        Bs = [8, 4, 5, 2]
        for gi, (start, end) in enumerate(GROUP_BOUNDS):
            for i in range(start, end + 1):
                param_colors[i] = GROUP_COLORS[gi]
                param_bound[i] = Bs[gi]

        bar_w = 0.8
        for i in range(npars):
            val, ref = pars_plot[i], dft[i]
            ax.bar(i, val, width=bar_w, color=param_colors[i], alpha=0.80,
                   linewidth=0.3, edgecolor="white", zorder=3)
            hw = bar_w * 0.48
            ax.plot([i - hw, i + hw], [ref, ref], color="#111", lw=1.2,
                    zorder=6, solid_capstyle="butt", linestyle="-")
            yo = val + (0.05 if val >= 0 else -0.05)
            va = "bottom" if val >= 0 else "top"
            ax.text(i, yo, f"{val:.4f}", ha="center", va=va,
                    fontsize=8, color="#333", rotation=90, zorder=7,
                    fontweight="bold")

            if param_bound[i] is not None:
                b = param_bound[i]
                for sign in (1, -1):
                    ax.plot([i - 0.5, i + 0.5], [sign * b, sign * b],
                            color="#CC3311", lw=1.0, ls="--", zorder=5, alpha=0.8)

        s_ = 10
        ax.set_xticks(x)
        ax.set_xticklabels(FORMATTED_NAMES[:40], rotation=55, ha="center",
                           fontsize=s_, fontfamily="monospace")
        ax.set_xlim(-0.4, npars - 0.6)
        ax.axhline(0, color="#555", lw=0.8, zorder=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(bottom=False)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", ls=":", lw=0.5, color="#bbb", zorder=0)

        # Group separators
        for gi, (start, end) in enumerate(GROUP_BOUNDS[:-1]):
            ax.axvline(end + 0.5, color="#aaa", lw=0.7, zorder=2)

        # Group labels — only on top plot, larger, in a box
        if col == 0:
            ylim_top = ax.get_ylim()[1]
            for gi, (start, end) in enumerate(GROUP_BOUNDS):
                ax.text((start + end) / 2, ylim_top * 0.83, GROUP_LABELS[gi],
                        ha="center", va="top", fontsize=16,
                        color=GROUP_COLORS[gi], fontweight="bold", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor=GROUP_COLORS[gi],
                                  edgecolor="none", alpha=0.15))

        ax.set_ylabel("Value [eV]", fontsize=14, labelpad=6)

        # Material name in a box inside the plot
        name = r"\textbf{WSe$_2$}" if tmd == "WSe2" else r"\textbf{WS$_2$}"
        ax.text(0.015, 0.96, name, transform=ax.transAxes,
                fontsize=14, bbox=BOX_STYLE, ha="left", va="top", zorder=10)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.04)

    out_dir = Path(args.output_dir) if args.output_dir else Path(master_folder) / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt, dpi in [("png", 600), ("pdf", None)]:
        kw = {"dpi": dpi} if dpi else {}
        fn = out_dir / f"params_WSe2_WS2.{fmt}"
        fig.savefig(fn, **kw)
        print(f"Saved: {fn}", flush=True)


if __name__ == "__main__":
    main()
