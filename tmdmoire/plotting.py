import numpy as np
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from pathlib import Path
from .constants import (
    FORMATTED_NAMES, IND_OFF, IND_PZ, IND_PXY, IND_SOC,
)


def plot_data_pipeline(arpes, tmd=None, save_dir=None):
    """Plot the three stages of ARPES data processing: raw → symmetrized → interpolated.

    Produces a 3-column figure showing, for each band on each path:
    - Left: raw experimental data (momentum vs energy)
    - Center: symmetrized data (averaged K→Γ and Γ→K segments)
    - Right: interpolated data on the uniform fitting grid

    Parameters
    ----------
    arpes : ARPESData
        ARPESData instance with raw_data, sym_data, and fit_data populated.
    tmd : str, optional
        Material name for the filename. Inferred from ``arpes.tmd`` if not given.
    save_dir : str or Path, optional
        Directory to save the figure. If None, uses ``Figures/`` relative to
        the repository root.
    """
    tmd = tmd or arpes.tmd
    paths = arpes.paths
    nbands = arpes.nbands

    fig, axes = plt.subplots(len(paths), 3, figsize=(18, 4 * len(paths)),
                             sharey=True, constrained_layout=True)
    if len(paths) == 1:
        axes = axes[np.newaxis, :]

    path_labels = {"KpGK": r"$K' \rightarrow \Gamma \rightarrow K$",
                   "KMKp": r"$K \rightarrow M \rightarrow K'$"}

    stage_titles = ["Raw data", "Symmetrized", "Interpolated (fitting grid)"]

    for ip, path in enumerate(paths):
        for ib in range(nbands[path]):
            ax_raw = axes[ip, 0]
            ax_sym = axes[ip, 1]
            ax_interp = axes[ip, 2]

            # Raw data
            rd = arpes.raw_data[path][ib]
            valid = ~np.isnan(rd[:, 1])
            ax_raw.scatter(rd[valid, 0], rd[valid, 1], s=4, c="steelblue", alpha=0.6)
            ax_raw.axhline(0, color="gray", lw=0.5, ls="--")
            ax_raw.set_title(f"Band {ib + 1}", fontsize=11)
            if ip == 0:
                ax_raw.set_title(stage_titles[0], fontsize=12, fontweight="bold")
            if ip == len(paths) - 1:
                ax_raw.set_xlabel("Momentum (Å⁻¹)", fontsize=10)
            if ib == 0:
                ax_raw.set_ylabel("Energy (eV)", fontsize=10)

            # Symmetrized data
            sd = arpes.sym_data[path][ib]
            valid = ~np.isnan(sd[:, 1])
            ax_sym.scatter(sd[valid, 0], sd[valid, 1], s=8, c="darkorange", alpha=0.7)
            ax_sym.axhline(0, color="gray", lw=0.5, ls="--")
            if ip == 0:
                ax_sym.set_title(stage_titles[1], fontsize=12, fontweight="bold")
            if ip == len(paths) - 1:
                ax_sym.set_xlabel("Momentum (Å⁻¹)", fontsize=10)

            # Interpolated data
            fd = arpes.fit_data
            ptsGK = fd.shape[0] // 3 * 2 if "KMKp" in paths else fd.shape[0]
            if path == "KpGK":
                xs = fd[:ptsGK, 0]
                ys = fd[:ptsGK, 3 + ib]
            else:
                xs = fd[ptsGK:, 0]
                ys = fd[ptsGK:, 3 + ib]
            valid = ~np.isnan(ys)
            ax_interp.scatter(xs[valid], ys[valid], s=12, c="forestgreen", alpha=0.8, zorder=3)
            ax_interp.axhline(0, color="gray", lw=0.5, ls="--")

            # Add high-symmetry point markers on interpolated plot
            modK = np.linalg.norm(arpes.K)
            if path == "KpGK":
                ax_interp.axvline(0, color="k", lw=0.8, ls=":")
                ax_interp.axvline(modK, color="k", lw=0.8, ls=":")
                if ib == 0:
                    ax_interp.text(0, ax_interp.get_ylim()[1] * 0.95, r"$\Gamma$",
                                   ha="center", va="top", fontsize=9, fontweight="bold")
                    ax_interp.text(modK, ax_interp.get_ylim()[1] * 0.95, r"$K$",
                                   ha="center", va="top", fontsize=9, fontweight="bold")
            else:
                ax_interp.axvline(modK, color="k", lw=0.8, ls=":")
                modKM = modK + np.linalg.norm(arpes.M - arpes.K)
                ax_interp.axvline(modKM, color="k", lw=0.8, ls=":")
                if ib == 0:
                    ax_interp.text(modK, ax_interp.get_ylim()[1] * 0.95, r"$K$",
                                   ha="center", va="top", fontsize=9, fontweight="bold")
                    ax_interp.text(modKM, ax_interp.get_ylim()[1] * 0.95, r"$M$",
                                   ha="center", va="top", fontsize=9, fontweight="bold")

            if ip == 0:
                ax_interp.set_title(stage_titles[2], fontsize=12, fontweight="bold")
            if ip == len(paths) - 1:
                ax_interp.set_xlabel(r"$|k|$ (Å⁻¹)", fontsize=10)

        # Row label
        for ax in axes[ip]:
            ax.text(-0.15, 0.5, path_labels.get(path, path),
                    transform=ax.transAxes, rotation=90, va="center",
                    fontsize=12, fontweight="bold")

    fig.suptitle(f"ARPES data processing pipeline — {tmd}", fontsize=14, fontweight="bold", y=1.01)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[1] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / f"data_pipeline_{tmd}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


def plot_bands(tb_en, data, legend_info, save_path=None):
    """Plot TB band energies vs ARPES data.

    Parameters
    ----------
    tb_en : np.ndarray
        Band energies, shape (6, n_kpts).
    data : ARPESData
        ARPES data with fit_data and K attributes.
    legend_info : tuple
        (tmd, Ks, boundType, Bs, chi2_elements) or with optional rank appended.
    save_path : str or Path, optional
        If given, save figure to this path and close.
    """
    fit_data = data.fit_data
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[10, 1], hspace=0)
    ax = fig.add_subplot(gs[0])
    for b in range(fit_data.shape[1] - 3):
        targ = np.argwhere(np.isnan(fit_data[:, 3 + b]))
        en_pars = copy.copy(tb_en[b, :])
        en_pars[targ] = np.nan
        ax.plot(fit_data[:, 0], fit_data[:, 3 + b], label="ARPES" if b == 0 else "",
                zorder=1, color="r", marker="o", markersize=10, mew=1, mec="k", mfc="firebrick")
        ax.plot(fit_data[:, 0], en_pars, ls="-", label="Fit" if b == 0 else "",
                zorder=3, color="skyblue", marker="s", markersize=10, mew=1, mec="k", mfc="deepskyblue")
    s_m, s_, s_p = 15, 20, 30
    modK = np.linalg.norm(data.K)
    ks = [fit_data[0, 0], modK, fit_data[-1, 0]]
    ax.set_xticks(ks, [r"$\Gamma$", r"$K$", r"$M$"], size=s_)
    for i in range(len(ks)):
        ax.axvline(ks[i], color="k", lw=0.5)
    ax.set_xlim(ks[0], ks[-1])
    ax.set_ylabel("Energy [eV]", size=s_)
    if fit_data.shape[1] == 9:
        ticks_y = np.linspace(np.max(fit_data[:, 3]) + 0.2, np.min(fit_data[~np.isnan(fit_data[:, 6]), 6]) - 0.2, 5)
        ax.set_yticks(ticks_y, ["{:.1f}".format(i) for i in ticks_y], size=s_m)
    plt.legend(fontsize=20)
    ax.set_title("Bands comparison", size=s_p)
    ax2 = fig.add_subplot(gs[1])
    _add_legend_result(legend_info, ax2)
    plt.subplots_adjust(left=0.083, bottom=0.045, right=0.893, top=0.95, wspace=0.06, hspace=0.2)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_parameters_absolute(pars, tmd, Bs, legend_info, save_path=None):
    """Plot parameter values as bar chart.

    Parameters
    ----------
    pars : np.ndarray
        43 tight-binding parameters.
    tmd : str
        Material name.
    Bs : tuple
        Bound parameters.
    legend_info : tuple
        (tmd, Ks, boundType, Bs, chi2_elements) or with optional rank appended.
    save_path : str or Path, optional
        If given, save figure to this path and close.
    """
    npars = pars.shape[0]
    DFT_pars = np.array([0] * npars)
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
        ax.axvspan(start - 0.5, end + 0.5, color=group_colors[gi], alpha=0.07, zorder=0)
    param_colors = [""] * npars
    param_bound = [None] * npars
    b_idx = 0
    for gi, (start, end) in enumerate(group_bounds):
        for i in range(start, end + 1):
            param_colors[i] = group_colors[gi]
        if gi in has_bound:
            for i in range(start, end + 1):
                param_bound[i] = Bs[b_idx]
            b_idx += 1
    bar_w = 0.8
    for i in range(npars):
        val = pars[i]
        ref = DFT_pars[i]
        c = param_colors[i]
        ax.bar(i, abs(val), width=bar_w, bottom=min(0, val),
               color=c, alpha=0.80, linewidth=0.3, edgecolor="white", zorder=3)
        hw = bar_w * 0.48
        ax.plot([i - hw, i + hw], [ref, ref],
                color="#111", lw=1.2, zorder=6, solid_capstyle="butt", linestyle="-")
        label = f"{val:+.2f}"
        if abs(val) > 0.20:
            ax.text(i, val / 2, label, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold", rotation=90, zorder=7)
        else:
            yo = val + (0.035 if val >= 0 else -0.035)
            va_ = "bottom" if val >= 0 else "top"
            ax.text(i, yo, label, ha="center", va=va_,
                    fontsize=8, color="#333", rotation=90, zorder=7)
        if param_bound[i] is not None:
            b = param_bound[i]
            for sign in (1, -1):
                ax.plot([i - 1 / 2, i + 1 / 2], [sign * b, sign * b],
                        color="#111", lw=1.4, ls="-", zorder=5, alpha=0.75)
    s_, s_p = 12, 15
    ax.set_xticks(x)
    ax.set_xticklabels(FORMATTED_NAMES[:npars], rotation=55, ha="right", fontsize=s_, fontfamily="monospace")
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
                ha="center", va="top", fontsize=s_, color=group_colors[gi], fontweight="bold", zorder=7)
    axl = fig.add_subplot(gs[1])
    _add_legend_result(legend_info, axl)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_orbital_content(pars, tmd, legend_info, save_path=None):
    """Plot orbital content of bands along high-symmetry path.

    Parameters
    ----------
    pars : np.ndarray
        43 tight-binding parameters (or 41 if SOC is frozen).
    tmd : str
        Material name.
    legend_info : tuple
        (tmd, Ks, boundType, Bs, chi2_elements) or with optional rank appended.
    save_path : str or Path, optional
        If given, save figure to this path and close.
    """
    from .material import _find_t, _find_e, _find_HSO
    from .constants import LATTICE_CONSTANTS, IND_OPO, IND_IPO
    from .material import TMDMaterial

    if pars.shape[0] == 41:
        mat_dft = TMDMaterial(tmd)
        full_pars = np.append(pars, mat_dft.dft_params[-2:])
    else:
        full_pars = pars

    Ngk = 200
    Nkm = int(Ngk / 2)
    Nmg = int(Ngk / 2 * np.sqrt(3))
    Nk = Ngk + Nkm + Nmg + 1
    a_TMD = LATTICE_CONSTANTS[tmd]
    K = np.array([4 * np.pi / 3 / a_TMD, 0])
    M = np.array([np.pi / a_TMD, np.pi / np.sqrt(3) / a_TMD])
    data_k = np.zeros((Nk, 2))
    data_k[:Ngk, 0] = np.linspace(0, K[0], Ngk, endpoint=False)
    for ik in range(Nkm):
        data_k[Ngk + ik] = K + (M - K) / Nkm * ik
    for ik in range(Nmg + 1):
        data_k[Ngk + Nkm + ik] = M + M / Nmg * ik

    hopping = _find_t(full_pars)
    epsilon = _find_e(full_pars)
    offset = full_pars[-3]
    HSO = _find_HSO(full_pars[-2:])
    args_H = (hopping, epsilon, HSO, offset)

    from .hamiltonian import MonolayerHamiltonian
    from .material import TMDMaterial
    mat = TMDMaterial(tmd)
    ham = MonolayerHamiltonian(mat)
    all_H = ham.build(data_k, *args_H)
    ens = np.zeros((Nk, 22))
    evs = np.zeros((Nk, 22, 22), dtype=complex)
    for i in range(Nk):
        ens[i], evs[i] = np.linalg.eigh(all_H[i])

    orbitals = np.zeros((5, 22, Nk))
    list_orbs = ([6, 7], [0, 1], [5], [3, 4, 9, 10], [2, 8])
    for orb in range(5):
        inds_orb = list_orbs[orb]
        for ib in range(22):
            for ik in range(Nk):
                for iorb in inds_orb:
                    orbitals[orb, ib, ik] += np.linalg.norm(evs[ik, iorb, ib]) ** 2 + np.linalg.norm(evs[ik, iorb + 11, ib]) ** 2

    indM = Ngk + Nkm
    fig = plt.figure(figsize=(15, 9))
    s_m, s_, s_p = 15, 20, 30
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[10, 1], hspace=0)
    ax = fig.add_subplot(gs[0])
    color = ["red", "brown", "blue", "green", "aqua"]
    labels = [r"$d_{xy}+d_{x^2-y^2}$", r"$d_{xz}+d_{yz}$", r"$d_{z^2}$", r"$p_x+p_y$", r"$p_z$"]
    leg = []
    xvals = np.linspace(0, Nk - 1, Nk)
    for orb in range(5):
        for ib in range(22):
            ax.scatter(xvals, ens[:, ib], s=(orbitals[orb, ib] * 100),
                       marker="o", facecolor=color[orb], lw=0, alpha=0.3)
        leg.append(Line2D([0], [0], marker="o", markeredgecolor="none",
                          markerfacecolor=color[orb], markersize=10, label=labels[orb], lw=0))
    legend = ax.legend(handles=leg, loc=(0.7, 0.45), fontsize=s_, handletextpad=0.35, handlelength=0.5)
    ax.add_artist(legend)
    ax.set_ylim(-4, 3)
    ax.set_xlim(0, Nk - 1)
    ax.axvline(Ngk, color="k", lw=1, zorder=-1)
    ax.axvline(Ngk + Nkm, color="k", lw=1, zorder=-1)
    ax.axhline(0, color="k", lw=1, zorder=-1)
    ax.set_xticks([0, Ngk - 1, Ngk + Nkm - 1, Nk - 1], [r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"], size=s_)
    ax.set_ylabel("Energy [eV]", size=s_)
    ax.tick_params(axis="y", labelsize=s_m)
    ax.set_title("Orbital content of bands", size=s_p)
    axl = fig.add_subplot(gs[1])
    _add_legend_result(legend_info, axl)
    plt.subplots_adjust(left=0.083, bottom=0.045, right=0.893, top=0.95, wspace=0.06, hspace=0.2)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _add_legend_result(legend_info, ax):
    """Add a text box with fit configuration and scoring to an axis.

    Parameters
    ----------
    legend_info : tuple
        Either (tmd, Ks, boundType, Bs, chi2_elements) or
        (tmd, Ks, boundType, Bs, chi2_elements, rank, idx).
    ax : matplotlib.axes.Axes
        Axis to draw the text box on.
    """
    if len(legend_info) >= 7:
        tmd, Ks, boundType, Bs, chi2_elements, rank, idx = legend_info[:7]
    else:
        tmd, Ks, boundType, Bs, chi2_elements = legend_info
        rank, idx = None, None

    txt = tmd + "\n"
    if rank is not None:
        txt += f"Rank #{rank}  (idx {idx})\n"
    txt_Bs = ["gen", "z  ", "xy ", "soc"] if boundType == "relative" else ["eps", "t_1", "t_5", "t_6", "soc"]
    txt += "-" * 10 + "\nBoundaries: " + boundType + "\n" + "-" * 10 + "\n"
    for i in range(len(Bs)):
        txt += txt_Bs[i] + ": %s" % Bs[i] + "\n"
    txt += "-" * 10 + "\nConstants\n" + "-" * 10 + "\n"
    for i in range(6):
        txt += "K%s: %6f" % (i + 1, Ks[i]) + "\n"
    txt += "-" * 10 + "\nFunction values\n" + "-" * 10 + "\n"
    txt_chiv = ["Chi2 energy bands", "K1 pars distance", "K2 M orb content",
                "K3 G/K orb content", "K4 minimum at K", "K5 band gap"]
    for i in range(6):
        txt += txt_chiv[i] + ":\n    %.6f" % chi2_elements[i] + "\n"
    box_dic = dict(boxstyle="round", facecolor="white", alpha=1)
    ax.text(0.0, 0., txt, bbox=box_dic, transform=ax.transAxes,
            fontsize=15, fontfamily="monospace")
    ax.axis("off")


def plot_top_results(scored_df, material_name, master_folder, run_dir, top_n=None):
    """Generate band, parameter, and orbital content plots for top scoring results.

    For each result in the scored DataFrame, produces three figures:
    1. Bands comparison (TB vs ARPES)
    2. Parameter overview (bar chart)
    3. Orbital content of bands

    Figures are saved to ``run_dir/Figures/`` with filenames like
    ``rank001_bands.png``, ``rank001_params.png``, ``rank001_orbitals.png``.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Scored results from GridScorer.score(), must have 'rank' column.
    material_name : str
        "WSe2" or "WS2".
    master_folder : str
        Repository root (for loading ARPES data).
    run_dir : str
        Run directory containing grid_config.json and fit_*.npz files.
    top_n : int, optional
        Number of top results to plot. If None, plots all in scored_df.
    """
    from .arpes_data import ARPESData
    import json

    df = scored_df if top_n is None else scored_df.head(top_n)
    if df.empty:
        print("No results to plot.")
        return

    fig_dir = Path(run_dir) / "Figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pts = 91
    config_path = Path(run_dir) / "grid_config.json"
    if config_path.exists():
        with open(config_path) as f:
            pts = json.load(f).get("pts", 91)

    if not df.empty:
        first_tb = df.iloc[0]["tb_en"]
        if hasattr(first_tb, "shape"):
            pts = first_tb.shape[1]

    arpes = ARPESData(material_name, master_folder, pts=pts)

    for _, row in df.iterrows():
        rank = int(row["rank"])
        idx = int(row["idx"])
        params = row["params"]
        tb_en = row["tb_en"]
        if "Ks" in row.index:
            Ks = tuple(row["Ks"])
        else:
            Ks = tuple([row[f"K{i}_w"] for i in range(1, 7)])
        Bs = tuple(row["Bs"])
        boundType = "absolute"

        chi2_elements = [
            row["chi2_band"], row["K1_val"], row["K2_val"],
            row["K3_val"], row["K4_val"], row["K5_val"],
        ]
        legend_info = (material_name, Ks, boundType, Bs, chi2_elements, rank, idx)

        prefix = f"rank{rank:03d}_idx{idx}"

        plot_bands(tb_en, arpes, legend_info,
                   save_path=fig_dir / f"{prefix}_bands.png")

        plot_parameters_absolute(params, material_name, Bs, legend_info,
                                 save_path=fig_dir / f"{prefix}_params.png")

        plot_orbital_content(params, material_name, legend_info,
                             save_path=fig_dir / f"{prefix}_orbitals.png")

        print(f"  Plots saved for rank {rank} (idx {idx})")

    print(f"All figures saved to {fig_dir}/")
