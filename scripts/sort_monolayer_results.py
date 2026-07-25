#!/usr/bin/env python
"""Sort, visualize, and export monolayer fitting results.

Loads merged HDF5 files (from v3.0) or individual .npz results, plots 2D
heatmaps of min(chi2) and min(chi2+K2_M) versus K2 and K3, and offers
interactive inspection and export of the best parameter sets.

Usage
-----
::

    python scripts/sort_monolayer_results.py --tmd WSe2 --input Data/WSe2_run1/merged_WSe2_absolute.h5
    python scripts/sort_monolayer_results.py --tmd WS2 --input Data/WS2_run1/merged_WS2_absolute.h5
    python scripts/sort_monolayer_results.py --tmd WSe2 --input-dir Data/WSe2_default  # npz mode
"""
import argparse
import sys
import os
import copy
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmdmoire.material import TMDMaterial
from tmdmoire.monolayer.data import MonolayerData
from tmdmoire.monolayer.scoring import GridScorer
from tmdmoire.utils.paths import get_repo_root

matplotlib.use("TkAgg", force=True)


OFFSET_K1 = -1e-7
K2_MIN = -(2 ** (-8))
K2_MAX = 10
OFFSET_K3 = -0.012
OFFSET_K6 = -1


def load_from_h5(path: Path, tmd: str) -> tuple:
    """Load and filter results from a merged HDF5 file."""
    with h5py.File(path, "r") as h5:
        elements = h5["elements"][:]
        Ks = h5["Ks"][:]
        Bs = h5["Bs"][:]
        pars = h5["pars"][:]

    mask = (
        (Ks[:, 0] > OFFSET_K1)
        & (Ks[:, 1] > K2_MIN)
        & (Ks[:, 1] < K2_MAX)
        & (Ks[:, 2] > OFFSET_K3)
        & (Ks[:, 5] > OFFSET_K6)
    )
    elements = elements[mask]
    Ks = Ks[mask]
    Bs = Bs[mask]
    pars = pars[mask]

    if tmd == "WSe2":
        tol = 1e-2
        mask2 = (
            (np.abs(pars[:, 0:7]) < Bs[:, [0]] - tol).all(axis=1)
            & (np.abs(pars[:, 7:28]) < Bs[:, [1]] - tol).all(axis=1)
            & (np.abs(pars[:, 28:36]) < Bs[:, [2]] - tol).all(axis=1)
            & (np.abs(pars[:, 36:40]) < Bs[:, [3]] - tol).all(axis=1)
        )
        elements = elements[mask2]
        Ks = Ks[mask2]
        Bs = Bs[mask2]
        pars = pars[mask2]

    return elements, Ks, Bs, pars


def load_from_npz(data_dir: Path, tmd: str) -> tuple:
    """Load and filter results from individual .npz files via GridScorer."""
    scorer = GridScorer(tmd, data_dir=str(data_dir))
    df = scorer.load_results()
    if df.empty:
        return None, None, None, None

    scorer._apply_K_range_mask = lambda df: df.copy()
    scorer._apply_bounds_saturation_mask = lambda df: df.copy()
    df = scorer._apply_K_range_mask(df)
    if tmd == "WSe2":
        df = scorer._apply_bounds_saturation_mask(df)

    elements = np.column_stack([
        df["band_K6"].values, df["K1_val"].values, df["K2_val"].values,
        df["K3_val"].values, df["K4_val"].values, df["K5_val"].values,
    ])
    Ks = np.column_stack([
        df["K1_w"].values, df["K2_w"].values, df["K3_w"].values,
        df["K4_w"].values, df["K5_w"].values, df["K6_w"].values,
    ])
    Bs = np.array([list(b) for b in df["Bs"].values])
    pars = np.array([np.asarray(p).flatten() for p in df["params"].values])
    return elements, Ks, Bs, pars


def build_grid(measure: np.ndarray, Ks: np.ndarray) -> tuple:
    """Build 2D grid: min(measure) over all dimensions except K2 and K3."""
    x_vals = np.unique(Ks[:, 1])
    y_vals = np.unique(Ks[:, 2])
    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}
    for i in range(len(measure)):
        xi = x_idx[Ks[i, 1]]
        yi = y_idx[Ks[i, 2]]
        if np.isnan(grid[yi, xi]) or measure[i] < grid[yi, xi]:
            grid[yi, xi] = measure[i]
    return x_vals, y_vals, grid


def plot_heatmap(measure, Ks, Bs, global_idx, tmd, cutoff, title, fig, ax):
    """Plot a 2D heatmap of a measure vs K2 and K3."""
    x_vals, y_vals, grid = build_grid(measure, Ks)
    grid[grid > cutoff] = np.nan

    min_measure = measure[global_idx]
    min_Ks = Ks[global_idx]
    min_Bs = Bs[global_idx]

    print(f"Global {title} minimum: {min_measure:.6g}  (index {global_idx})")
    print(f"  Ks = {min_Ks}")
    print(f"  Bs = {min_Bs}")

    img = ax.pcolormesh(x_vals, y_vals, grid, shading="nearest", cmap="viridis_r")
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(r"$\min$ measure", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

    ax.scatter(min_Ks[1], min_Ks[2], marker="*", s=220, color="red", zorder=5,
               label="global minimum")

    ks_str = "\n".join(f"  K{i + 1} = {min_Ks[i]:.6g}" for i in range(len(min_Ks)))
    bs_str = "\n".join(f"  B{i} = {min_Bs[i]:.6g}" for i in range(len(min_Bs)))
    legend_text = f"measure min = {min_measure:.6g}\nKs:\n{ks_str}\nBs:\n{bs_str}"
    ax.scatter([], [], marker="*", color="red", label=legend_text)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85,
              handlelength=1.2, borderpad=0.8)

    ax.set_xlabel(r"$K_2$", fontsize=13)
    ax.set_ylabel(r"$K_3$", fontsize=13)
    ax.set_title(f"{tmd} : measure = {title}", fontsize=13)


def add_legend(legend_info, ax):
    """Add a text box with fit configuration and scoring to an axis."""
    if len(legend_info) >= 7:
        tmd, Ks, bound_type, Bs, chi2_elements, rank, idx = legend_info[:7]
    else:
        tmd, Ks, bound_type, Bs, chi2_elements = legend_info
        rank, idx = None, None

    txt = tmd + "\n"
    if rank is not None:
        txt += f"Rank #{rank}  (idx {idx})\n"
    names_b = ["gen", "z  ", "xy ", "soc"] if bound_type == "relative" else ["eps", "t_1", "t_5", "t_6", "soc"]
    txt += "-" * 10 + "\nBoundaries: " + bound_type + "\n" + "-" * 10 + "\n"
    for i in range(len(Bs)):
        txt += names_b[i] + ": %s" % Bs[i] + "\n"
    txt += "-" * 10 + "\nConstants\n" + "-" * 10 + "\n"
    for i in range(6):
        txt += "K%s: %6f" % (i + 1, Ks[i]) + "\n"
    txt += "-" * 10 + "\nFunction values\n" + "-" * 10 + "\n"
    names_v = ["Chi2 energy bands", "K1 pars distance", "K2 M orb content",
               "K3 G/K orb content", "K4 minimum at K", "K5 band gap"]
    for i in range(6):
        txt += names_v[i] + ":\n    %.6f" % chi2_elements[i] + "\n"
    box_dic = dict(boxstyle="round", facecolor="white", alpha=1)
    ax.text(0.0, 0., txt, bbox=box_dic, transform=ax.transAxes,
            fontsize=15, fontfamily="monospace")
    ax.axis("off")


def plot_result(pars, tmd, Ks, bound_type, Bs, elements):
    """Plot bands, parameters, and orbital content for a result."""
    pts = 91
    master_folder = get_repo_root()
    data = MonolayerData(tmd, master_folder, pts=pts)

    mat = TMDMaterial(tmd)
    full_pars = pars.copy()
    HSO = mat.build_soc_hamiltonian(full_pars[-2:])
    from tmdmoire.monolayer.fitter import ParameterFitter
    dummy_config = {"Ks": tuple(Ks), "boundType": bound_type, "Bs": tuple(int(b) for b in Bs)}
    fitter = ParameterFitter(mat, data, dummy_config)
    tb_en = fitter.chi2(full_pars[:-2], HSO, full_pars[-2:], return_energy=True)

    legend_info = (tmd, tuple(Ks), bound_type, tuple(int(b) for b in Bs),
                   tuple(elements), "", 0)

    _plot_bands(tb_en, data, legend_info)
    plt.show(block=False)

    _plot_parameters(full_pars, tmd, Bs, legend_info)
    plt.show(block=False)

    _plot_orbitals(full_pars, tmd, legend_info)
    plt.show()


def _plot_bands(tb_en, data, legend_info):
    """Plot TB band energies vs ARPES data."""
    fit_data = data.fit_data
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[10, 1], hspace=0)
    ax = fig.add_subplot(gs[0])
    for b in range(fit_data.shape[1] - 3):
        targ = np.argwhere(np.isnan(fit_data[:, 3 + b]))
        en_pars = copy.copy(tb_en[b, :])
        en_pars[targ] = np.nan
        ax.plot(fit_data[:, 0], fit_data[:, 3 + b], label="ARPES" if b == 0 else "",
                zorder=1, color="r", marker="o", markersize=10, mew=1, mec="k",
                mfc="firebrick")
        ax.plot(fit_data[:, 0], en_pars, ls="-", label="Fit" if b == 0 else "",
                zorder=3, color="skyblue", marker="s", markersize=10, mew=1,
                mec="k", mfc="deepskyblue")
    s_m, s_, s_p = 15, 20, 30
    mod_k = np.linalg.norm(data.K)
    ks = [fit_data[0, 0], mod_k, fit_data[-1, 0]]
    ax.set_xticks(ks, [r"$\Gamma$", r"$K$", r"$M$"], size=s_)
    for i in range(len(ks)):
        ax.axvline(ks[i], color="k", lw=0.5)
    ax.set_xlim(ks[0], ks[-1])
    ax.set_ylabel("Energy [eV]", size=s_)
    if fit_data.shape[1] == 9:
        ticks_y = np.linspace(np.max(fit_data[:, 3]) + 0.2,
                              np.min(fit_data[~np.isnan(fit_data[:, 6]), 6]) - 0.2, 5)
        ax.set_yticks(ticks_y, ["{:.1f}".format(i) for i in ticks_y], size=s_m)
    plt.legend(fontsize=20)
    ax.set_title("Bands comparison", size=s_p)
    ax2 = fig.add_subplot(gs[1])
    add_legend(legend_info, ax2)
    plt.subplots_adjust(left=0.083, bottom=0.045, right=0.893, top=0.95,
                        wspace=0.06, hspace=0.2)


def _plot_parameters(pars, tmd, Bs, legend_info):
    """Plot parameter values as bar chart with DFT reference lines."""
    mat = TMDMaterial(tmd)
    DFT_pars = mat.dft_params[:pars.shape[0]]
    npars = pars.shape[0]

    fig = plt.figure(figsize=(19, 9))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[10, 1], hspace=0)
    fig.patch.set_facecolor("#F7F7F7")
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#F7F7F7")
    x = np.arange(npars)
    group_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#64B5CD"]
    if npars >= 43:
        group_bounds = [(0, 6), (7, 27), (28, 35), (36, 39), (40, 40), (41, 42)]
        group_labels = ["Epsilon", "t_1", "t_5", "t_6", "", "SOC"]
        has_bound = [0, 1, 2, 3, 5]
    else:
        group_bounds = [(0, 6), (7, 27), (28, 35), (36, 39), (40, 40)]
        group_labels = ["Epsilon", "t_1", "t_5", "t_6", ""]
        has_bound = [0, 1, 2, 3]
    for gi, (start, end) in enumerate(group_bounds):
        ax.axvspan(start - 0.5, end + 0.5, color=group_colors[gi],
                   alpha=0.07, zorder=0)
    param_colors, param_bound = [""] * npars, [None] * npars
    b_idx = 0
    for gi, (start, end) in enumerate(group_bounds):
        for i in range(start, end + 1):
            param_colors[i] = group_colors[gi]
        if gi in has_bound:
            for i in range(start, end + 1):
                param_bound[i] = Bs[b_idx]
            b_idx += 1
    bar_w = 0.8
    from tmdmoire.constants import FORMATTED_NAMES
    for i in range(npars):
        val, ref = pars[i], DFT_pars[i]
        ax.bar(i, val, width=bar_w, color=param_colors[i], alpha=0.80,
               linewidth=0.3, edgecolor="white", zorder=3)
        hw = bar_w * 0.48
        ax.plot([i - hw, i + hw], [ref, ref], color="#111", lw=1.5,
                zorder=6, solid_capstyle="butt", linestyle="-")
        offset_y = 0.05 if val >= 0 else -0.05
        va_ = "bottom" if val >= ref else "top"
        ax.text(i, val + offset_y, f"{abs(val - ref):.3f}", ha="center",
                va=va_, fontsize=9, color="#333", rotation=90, zorder=7,
                fontweight="bold")
        if param_bound[i] is not None:
            b = param_bound[i]
            for sign in (1, -1):
                ax.plot([i - 0.5, i + 0.5], [sign * b, sign * b],
                        color="#CC3311", lw=1.2, ls="--", zorder=5, alpha=0.8)
    s_, s_p = 12, 15
    ax.set_xticks(x)
    ax.set_xticklabels(FORMATTED_NAMES[:npars], rotation=55, ha="center",
                       fontsize=s_, fontfamily="monospace")
    ax.set_xlim(-0.4, npars - 0.6)
    ax.set_ylabel("Value", fontsize=s_p, labelpad=6)
    ax.axhline(0, color="#555", lw=0.8, zorder=4)
    ax.set_title("Parameter Overview", fontsize=20, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(bottom=False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", ls=":", lw=0.5, color="#bbb", zorder=0)
    ylim_top = ax.get_ylim()[1]
    for gi, (start, end) in enumerate(group_bounds[:-1]):
        ax.axvline(end + 0.5, color="#aaa", lw=0.7, zorder=2)
    for gi, (start, end) in enumerate(group_bounds):
        ax.text((start + end) / 2, ylim_top * 0.97, group_labels[gi],
                ha="center", va="top", fontsize=s_,
                color=group_colors[gi], fontweight="bold", zorder=7)
    axl = fig.add_subplot(gs[1])
    add_legend(legend_info, axl)
    fig.tight_layout()


def _plot_orbitals(pars, tmd, legend_info):
    """Plot orbital content of bands along high-symmetry path."""
    from tmdmoire.constants import LATTICE_CONSTANTS
    from tmdmoire.material import _find_t, _find_e, _find_HSO
    from tmdmoire.monolayer.hamiltonian import MonolayerHamiltonian

    if pars.shape[0] == 41:
        mat_dft = TMDMaterial(tmd)
        full_pars = np.append(pars, mat_dft.dft_params[-2:])
    else:
        full_pars = pars

    n_gk, n_km = 200, 100
    n_mg = int(n_gk / 2 * np.sqrt(3))
    n_k = n_gk + n_km + n_mg + 1
    a_TMD = LATTICE_CONSTANTS[tmd]
    K = np.array([4 * np.pi / 3 / a_TMD, 0])
    M = np.array([np.pi / a_TMD, np.pi / np.sqrt(3) / a_TMD])
    data_k = np.zeros((n_k, 2))
    data_k[:n_gk, 0] = np.linspace(0, K[0], n_gk, endpoint=False)
    for ik in range(n_km):
        data_k[n_gk + ik] = K + (M - K) / n_km * ik
    for ik in range(n_mg + 1):
        data_k[n_gk + n_km + ik] = M + M / n_mg * ik

    hopping, epsilon = _find_t(full_pars), _find_e(full_pars)
    HSO = _find_HSO(full_pars[-2:])
    args_H = (hopping, epsilon, HSO, full_pars[-3])
    mat = TMDMaterial(tmd)
    ham = MonolayerHamiltonian(mat)
    all_H = ham.build(data_k, *args_H)
    ens, evs = np.zeros((n_k, 22)), np.zeros((n_k, 22, 22), dtype=complex)
    for i in range(n_k):
        ens[i], evs[i] = np.linalg.eigh(all_H[i])

    orbitals = np.zeros((5, 22, n_k))
    list_orbs = ([6, 7], [0, 1], [5], [3, 4, 9, 10], [2, 8])
    for orb in range(5):
        for ib in range(22):
            for ik in range(n_k):
                for iorb in list_orbs[orb]:
                    orbitals[orb, ib, ik] += (np.linalg.norm(evs[ik, iorb, ib]) ** 2
                                              + np.linalg.norm(evs[ik, iorb + 11, ib]) ** 2)

    fig = plt.figure(figsize=(15, 9))
    s_m, s_, s_p = 15, 20, 30
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[10, 1], hspace=0)
    ax = fig.add_subplot(gs[0])
    colors = ["red", "brown", "blue", "green", "aqua"]
    labels = [r"$d_{xy}+d_{x^2-y^2}$", r"$d_{xz}+d_{yz}$", r"$d_{z^2}$",
              r"$p_x+p_y$", r"$p_z$"]
    xvals = np.linspace(0, n_k - 1, n_k)
    leg = []
    for orb in range(5):
        for ib in range(22):
            ax.scatter(xvals, ens[:, ib], s=(orbitals[orb, ib] * 100),
                       marker="o", facecolor=colors[orb], lw=0, alpha=0.3)
        leg.append(Line2D([0], [0], marker="o", markeredgecolor="none",
                          markerfacecolor=colors[orb], markersize=10,
                          label=labels[orb], lw=0))
    legend = ax.legend(handles=leg, loc=(0.7, 0.45), fontsize=s_,
                       handletextpad=0.35, handlelength=0.5)
    ax.add_artist(legend)
    ax.set_ylim(-4, 3)
    ax.set_xlim(0, n_k - 1)
    ax.axvline(n_gk, color="k", lw=1, zorder=-1)
    ax.axvline(n_gk + n_km, color="k", lw=1, zorder=-1)
    ax.axhline(0, color="k", lw=1, zorder=-1)
    ax.set_xticks([0, n_gk - 1, n_gk + n_km - 1, n_k - 1],
                  [r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"], size=s_)
    ax.set_ylabel("Energy [eV]", size=s_)
    ax.tick_params(axis="y", labelsize=s_m)
    ax.set_title("Orbital content of bands", size=s_p)
    axl = fig.add_subplot(gs[1])
    add_legend(legend_info, axl)
    plt.subplots_adjust(left=0.083, bottom=0.045, right=0.893, top=0.95,
                        wspace=0.06, hspace=0.2)


def auto_detect_input(input_path: str | None, input_dir: str | None, tmd: str) -> tuple:
    """Auto-detect input format and load data."""
    if input_path is not None:
        path = Path(input_path)
        if not path.exists():
            sys.exit(f"[ERROR] File not found: {path}")
        if path.suffix == ".h5":
            print(f"Loading from HDF5: {path}")
            elements, Ks, Bs, pars = load_from_h5(path, tmd)
            bound_type = str(path.stem).split("_")[-1]
            return elements, Ks, Bs, pars, bound_type, "h5"
        else:
            sys.exit(f"[ERROR] Unsupported file format: {path.suffix}")

    if input_dir is not None:
        path = Path(input_dir)
        if not path.is_dir():
            sys.exit(f"[ERROR] Directory not found: {path}")
        npz_files = list(path.glob("fit_idx*.npz"))
        if npz_files:
            print(f"Loading from .npz directory: {path} ({len(npz_files)} files)")
            elements, Ks, Bs, pars = load_from_npz(path, tmd)
            config_path = path / "fit_config.json"
            bound_type = "absolute"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                bound_type = cfg.get("bounds", {}).get("boundType", "absolute")
            return elements, Ks, Bs, pars, bound_type, "npz"
        h5_files = list(path.glob("*.h5"))
        if h5_files:
            print(f"Loading from HDF5: {h5_files[0]}")
            elements, Ks, Bs, pars = load_from_h5(h5_files[0], tmd)
            bound_type = str(h5_files[0].stem).split("_")[-1]
            return elements, Ks, Bs, pars, bound_type, "h5"
        sys.exit(f"[ERROR] No .npz or .h5 files found in {path}")

    sys.exit("[ERROR] Please specify --input or --input-dir")


def main():
    parser = argparse.ArgumentParser(
        description="Sort, visualize, and export monolayer fitting results."
    )
    parser.add_argument("--tmd", required=True, choices=["WSe2", "WS2"],
                        help="Target material.")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to merged .h5 file (v3.0 format).")
    parser.add_argument("--input-dir", "-d", type=str, default=None,
                        help="Directory of .npz or .h5 files.")
    parser.add_argument("--cutoff", "-c", type=float, default=0.3,
                        help="Cutoff value for chi2 heatmap (default: 0.3).")
    args = parser.parse_args()

    tmd = args.tmd
    elements, Ks, Bs, pars, bound_type, source = auto_detect_input(
        args.input, args.input_dir, tmd
    )
    if elements is None or len(elements) == 0:
        sys.exit("[ERROR] No results loaded.")

    print(f"Loaded {elements.shape[0]} runs ({source} format)")

    chi2 = elements[:, 0]
    K2_M = elements[:, 2]
    ind_chosen = 1 if tmd == "WSe2" else 0

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(121)
    global_idx1 = np.argsort(chi2)[ind_chosen]
    plot_heatmap(chi2, Ks, Bs, global_idx1, tmd, args.cutoff, "chi2", fig, ax1)
    print("-" * 40)

    ax2 = fig.add_subplot(122)
    global_idx2 = np.argsort(chi2 + K2_M)[ind_chosen]
    plot_heatmap(chi2 + K2_M, Ks, Bs, global_idx2, tmd, args.cutoff,
                 "chi2+K2_M", fig, ax2)

    plt.tight_layout()
    plt.show()

    inp = input("\nPlot best result? [1-2-a/N] (a is for all): ")
    global_idxs = []
    if inp == "1":
        global_idxs = [(global_idx1, "chi2")]
    elif inp == "2":
        global_idxs = [(global_idx2, "chi2+K2_M")]
    elif inp == "a":
        global_idxs = [(global_idx1, "chi2"), (global_idx2, "chi2+K2_M")]

    for global_idx, label in global_idxs:
        print(f"\n--- Result: {label} ---")
        plot_result(pars[global_idx], tmd, Ks[global_idx], bound_type,
                    Bs[global_idx], elements[global_idx])

    if inp in ("1", "2"):
        save = input("Save result? [y/N] ") == "y"
        if save:
            out_fn = f"Data/result_{tmd}.npy"
            pars_result = pars[global_idxs[0][0]]
            np.save(out_fn, pars_result)
            print(f"Saved result '{inp}' to {out_fn}")


if __name__ == "__main__":
    main()
