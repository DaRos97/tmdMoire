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
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmdmoire.material import TMDMaterial
from tmdmoire.constants import FORMATTED_NAMES
from tmdmoire.utils.paths import get_repo_root


GROUP_COLORS = ["#C44E52", "#1861b3", "#257022", "#85132a", "#ecce16"]
GROUP_LABELS = [r"$\varepsilon$", r"$t^{\rm nn}_{\rm MX}$", r"$t_{XX}$", r"$t_{MM}$", r"$t^{\rm nnn}_{\rm MX}$"]
GROUP_BOUNDS = [(0, 6), (7, 14), (15, 26), (27, 35), (36, 39)]
SUB_RANGES = {}
BOX_STYLE = dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor="black", linewidth=1, alpha=1.0)

NEW_ORDER = [
    1, 2, 5, 6,
    0, 3, 4,
    28, 29, 30, 31, 32, 33, 34, 35,
    9, 10, 11, 15, 16, 17,
    18, 20, 22, 23, 26, 27,
    7, 8, 12, 13, 14,
    19, 21, 24, 25,
    36, 37, 38, 39,
]

ORB_NAMES = {
    1: r"$d_{xz}$",
    2: r"$d_{yz}$",
    3: r"$p_z^o$",
    4: r"$p_x^o$",
    5: r"$p_y^o$",
    6: r"$d_{z^2}$",
    7: r"$d_{xy}$",
    8: r"$d_{x^2\text{-}y^2}$",
    9: r"$p_z^e$",
    10: r"$p_x^e$",
    11: r"$p_y^e$",
}

TICK_INDICES = [
    "3", "4", "9", "10",
    "1", "6", "7",
    "4-1", "3-2", "5-2", "9-6", "11-6", "10-7", "9-8", "11-8",
    "3,3", "4,4", "5,5", "9,9", "10,10", "11,11",
    "3,5", "9,11", "3,4", "4,5", "9,10", "10,11",
    "1,1", "2,2", "6,6", "7,7", "8,8",
    "6,8", "1,2", "6,7", "7,8",
    "9,6", "11,6", "9,8", "11,8",
]

def _orbital_label(s):
    if "-" in s:
        a, b = s.split("-")
        return f"{ORB_NAMES[int(a)]}-{ORB_NAMES[int(b)]}"
    if "," in s:
        a, b = s.split(",")
        return f"{ORB_NAMES[int(a)]}-{ORB_NAMES[int(b)]}"
    return ORB_NAMES[int(s)]

TICK_LABELS = [_orbital_label(s) for s in TICK_INDICES]


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

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.7), sharex=True)
    fig.patch.set_facecolor("white")

    for col, (tmd, ax) in enumerate(zip(materials, axes.flat)):
        matches = sorted(bilayer_dir.glob(f"tb_{tmd}*.npy"))
        if not matches:
            print(f"[WARNING] No tb_{tmd}*.npy found, skipping.")
            continue
        pars = np.load(matches[0])
        # Exclude offset (40) and SOC (41, 42)
        pars_plot = pars[NEW_ORDER]
        npars = len(pars_plot)

        mat = TMDMaterial(tmd)
        dft = mat.dft_params[NEW_ORDER]

        ax.set_facecolor("white")
        x = np.arange(npars)

        # Group background bands (sub-blocks use lighter alpha)
        band_specs = []
        for gi, (start, end) in enumerate(GROUP_BOUNDS):
            color = GROUP_COLORS[gi]
            sub = SUB_RANGES.get(gi, [])
            if not sub:
                band_specs.append((start, end, color, 0.1))
            else:
                cur = start
                for ss, se, sa in sub:
                    if ss > cur:
                        band_specs.append((cur, ss - 1, color, 0.1))
                    band_specs.append((ss, se, color, sa))
                    cur = se + 1
                if cur <= end:
                    band_specs.append((cur, end, color, 0.1))
        for s, e, c, a in band_specs:
            ax.axvspan(s - 0.5, e + 0.5, color=c, alpha=a, zorder=0)

        # Dashed vertical lines at each parameter index
        for i in range(npars):
            ls = (0, (1, 1.5)) if i % 2 == 1 else "--"
            ax.axvline(i, color="#888", lw=0.3, ls=ls, alpha=0.5, zorder=1)

        # Colours and bounds
        param_colors = [""] * npars
        param_bound = [None] * npars
        Bs = [8, 5, 4, 4, 2]
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
            ax.plot([i - hw, i + hw], [ref, ref], color="#111", lw=0.4,
                    zorder=6, solid_capstyle="butt", linestyle="-")
            yo = val + (0.05 if val >= 0 else -0.05)
            va = "bottom" if val >= 0 else "top"
            ax.text(i, yo, f"{val:.3f}", ha="center", va=va,
                    fontsize=3, color="#333", rotation=90, zorder=7,
                    fontweight="bold")

            if param_bound[i] is not None:
                b = param_bound[i]
                for sign in (1, -1):
                    ax.plot([i - 0.5, i + 0.5], [sign * b, sign * b],
                            color="#CC3311", lw=0.7, ls="--", zorder=5, alpha=0.8)

        ax.set_xlim(-0.4, npars + 0.0)
        ax.set_xticks(range(0, npars, 2))
        if col == 0:
            ax.set_xticklabels([""] * (npars // 2))
            top_ax = ax.twiny()
            top_ax.set_xlim(-0.4, npars + 0.0)
            top_ax.set_xticks(range(1, npars, 2))
            top_ax.set_xticklabels(
                [TICK_LABELS[i] for i in range(1, npars, 2)],
                rotation=45, ha="center", fontsize=3
            )
            top_ax.tick_params(axis="x", which="both", length=2, pad=1)
            for spine in top_ax.spines.values():
                spine.set_visible(False)
            for lbl in top_ax.get_xticklabels():
                lbl.set_clip_on(False)
        else:
            ax.set_xticklabels(
                [TICK_LABELS[i] for i in range(0, npars, 2)],
                rotation=45, ha="center", fontsize=3
            )
            for lbl in ax.get_xticklabels():
                lbl.set_clip_on(False)
        ax.axhline(0, color="#555", lw=0.6, zorder=4)
        if col == 0:
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(True)
            ax.spines["left"].set_visible(True)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        else:
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        if col == 1:
            ax.tick_params(bottom=True, direction="out", length=2, pad=1)
        else:
            ax.tick_params(bottom=False, pad=1)
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_yticks([-6, -3, 0, 3, 6])
        ax.tick_params(axis="y", labelsize=5, pad=1)

        # Group separators
        for gi, (start, end) in enumerate(GROUP_BOUNDS[:-1]):
            ax.axvline(end + 0.5, color="#aaa", lw=0.4, zorder=2)

        # Sub-block separators (none — handled by group separators)

        # Group labels — only on top plot, larger, in a box
        if col == 0:
            ylim_top = ax.get_ylim()[1]
            for gi, (start, end) in enumerate(GROUP_BOUNDS):
                color = GROUP_COLORS[gi]
                ax.text((start + end) / 2, ylim_top * 0.83, GROUP_LABELS[gi],
                        ha="center", va="top", fontsize=7,
                        color=color, fontweight="bold", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor=color,
                                  edgecolor="none", alpha=0.18))

        ax.set_ylabel("Value [eV]", fontsize=7, labelpad=1)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.08, right=0.99, left=0.11, hspace=0)

    pos1 = axes.flat[1].get_position()
    pos0 = axes.flat[0].get_position()
    fig.add_artist(Line2D(
        [pos0.x0, pos0.x1], [pos1.y1, pos1.y1],
        color="black", lw=0.5, transform=fig.transFigure
    ))

    for ax, tmd in zip(axes.flat, ["WSe2", "WS2"]):
        name = r"\textbf{WSe$_2$}" if tmd == "WSe2" else r"\textbf{WS$_2$}"
        pos = ax.get_position()
        fig.text(pos.x0 - 0.02, pos.y1 - 0.01, name, fontsize=7, ha="right", va="top", zorder=10)

    out_dir = Path(args.output_dir) if args.output_dir else Path(master_folder) / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fn = out_dir / "fig_params_WSe2_WS2.pdf"
    fig.savefig(fn)
    print(f"Saved: {fn}", flush=True)


if __name__ == "__main__":
    main()
