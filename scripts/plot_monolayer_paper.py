"""Paper-level figure of monolayer tight-binding fitting results.

Produces a 3-panel figure comparing ARPES data, TB fit, and DFT:
- Top-left: ARPES intensity + fit intensity (pcolormesh)
- Bottom-left: Orbital content (DFT left, fit right)
- Right: Band comparison (ARPES dots, fit lines, DFT lines)

Uses fitted parameters from Inputs/bilayer_fitting/tb_{TMD}.npy and
ARPES intensity data from Inputs/monolayer_plot/.

Usage
-----
::

    python scripts/plot_monolayer_paper.py WSe2
    python scripts/plot_monolayer_paper.py WS2
    python scripts/plot_monolayer_paper.py WSe2 --tb-file path/to/tb_WSe2.npy
    python scripts/plot_monolayer_paper.py WSe2 --no-cache
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmdmoire.material import TMDMaterial, _find_t, _find_e, _find_HSO
from tmdmoire.monolayer.data import MonolayerData
from tmdmoire.monolayer.hamiltonian import MonolayerHamiltonian
from tmdmoire.utils.kpoints import get_k_list
from tmdmoire.constants import LATTICE_CONSTANTS, DFT_INITIAL_PARAMS
from tmdmoire.utils.paths import get_repo_root


# ─── Spectral weight spreading ───────────────────────────────────────────────

def _lorentz_spreading(weight, k_temp, e_temp, k_list, e_grid, spread_k, spread_e):
    """Lorentzian spreading of spectral weight in k and energy.

    Parameters
    ----------
    weight : float
        Spectral weight to spread.
    k_temp : np.ndarray
        Momentum position (2D vector).
    e_temp : float
        Energy position.
    k_list : np.ndarray
        Grid of k-points, shape (N, 2).
    e_grid : np.ndarray
        Grid of energy values, shape (1, M).
    spread_k : float
        Lorentzian width in momentum (1/Angstrom).
    spread_e : float
        Lorentzian width in energy (eV).

    Returns
    -------
    np.ndarray
        Spread weight, shape (N, M).
    """
    k_grid = np.linalg.norm(k_list - k_temp, axis=1)[:, None]
    k2 = spread_k ** 2
    e2 = spread_e ** 2
    return weight / (k_grid ** 2 + k2) / ((e_grid - e_temp) ** 2 + e2)


# ─── K-point path for intensity (G-K-M) ──────────────────────────────────────

def _get_gkm_path(n_k, tmd, endpoint=True):
    """Generate G-K-M k-point path with cumulative norm.

    Parameters
    ----------
    n_k : int
        Number of k-points.
    tmd : str
        Material name.
    endpoint : bool
        Include endpoint.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        k-points (N, 2) and cumulative norm (N,).
    """
    return get_k_list("G-K-M", n_k, tmd=tmd, endpoint=endpoint, return_norm=True)


# ─── Main plotting ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paper-level monolayer fitting figure."
    )
    parser.add_argument("material", choices=["WSe2", "WS2"],
                        help="Target material.")
    parser.add_argument("--tb-file", type=str, default=None,
                        help="Path to fitted TB params .npy file. "
                             "Default: Inputs/bilayer_fitting/tb_{TMD}*.npy (first match).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute eigenvalues/intensities instead of loading cache.")
    parser.add_argument("--run-id", type=str, default="de01",
                        help="Run ID for monolayer data loading (default: de01).")
    parser.add_argument("--n-k-int", type=int, default=582,
                        help="Number of k-points for intensity plot (default: 582).")
    parser.add_argument("--n-k-orb", type=int, default=201,
                        help="Number of k-points for orbital plot (default: 201).")
    parser.add_argument("--n-k-bands", type=int, default=61,
                        help="Number of k-points for bands plot (default: 61).")
    parser.add_argument("--spread-k", type=float, default=0.005,
                        help="Lorentzian k-width for intensity (default: 0.005).")
    parser.add_argument("--spread-e", type=float, default=0.05,
                        help="Lorentzian E-width for intensity (default: 0.05).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures. Default: Data/run_<id>/Figures/.")

    args = parser.parse_args()
    tmd = args.material
    master_folder = get_repo_root()

    # ── Load fitted parameters ───────────────────────────────────────────
    if args.tb_file is not None:
        tb_path = Path(args.tb_file)
    else:
        bilayer_dir = Path(master_folder) / "Inputs" / "bilayer_fitting"
        matches = list(bilayer_dir.glob(f"tb_{tmd}*.npy"))
        if not matches:
            raise FileNotFoundError(f"No tb_{tmd}*.npy found in {bilayer_dir}")
        tb_path = matches[0]
    print(f"Loading TB params from: {tb_path}")
    fit_pars = np.load(tb_path)

    # ── Load ARPES intensity data ────────────────────────────────────────
    arpes_intensity_path = Path(master_folder) / "Inputs" / "monolayer_plot" / f"intensity_GKM_{tmd}_sum_BE_crop.txt"
    print(f"Loading ARPES intensity from: {arpes_intensity_path}")
    arpes_intensity = np.loadtxt(arpes_intensity_path, delimiter="\t")
    arpes_intensity /= np.max(arpes_intensity)

    n_k_int, n_e = arpes_intensity.shape
    list_e = np.linspace(-3.5, 0, n_e)
    k_end = 1.9896 if tmd == "WS2" else 1.9075
    norm_k_arpes = np.linspace(0, k_end, n_k_int, endpoint=True)

    # ── Setup ────────────────────────────────────────────────────────────
    material = TMDMaterial(tmd)
    ham = MonolayerHamiltonian(material)
    hopping = _find_t(fit_pars)
    epsilon = _find_e(fit_pars)
    offset = fit_pars[-3]
    hso = _find_HSO(fit_pars[-2:])
    args_h = (hopping, epsilon, hso, offset)

    dft_pars = np.array(DFT_INITIAL_PARAMS[tmd])
    hopping_dft = _find_t(dft_pars)
    epsilon_dft = _find_e(dft_pars)
    offset_dft = dft_pars[-3]
    hso_dft = _find_HSO(dft_pars[-2:])
    args_h_dft = (hopping_dft, epsilon_dft, hso_dft, offset_dft)

    cache_dir = Path("Data") / "monolayer_figure_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Eigenvalues/eigenvectors for intensity (G-K-M path) ──────────────
    list_k_fit_i, norm_fit_i = _get_gkm_path(args.n_k_int, tmd, endpoint=True)
    n_k_fit_i = len(list_k_fit_i)

    en_fn = cache_dir / f"intensity_fit_en_{n_k_fit_i}_{tmd}.npz"
    if not args.no_cache and en_fn.is_file():
        print("Loading cached energies for intensity")
        ens_fit = np.load(en_fn)["ens"]
        evs_fit = np.load(en_fn)["evs"]
    else:
        print("Computing energies for intensity")
        all_h = ham.build(list_k_fit_i, *args_h)
        ens_fit = np.zeros((n_k_fit_i, 22))
        evs_fit = np.zeros((n_k_fit_i, 22, 22), dtype=complex)
        for i in tqdm(range(n_k_fit_i), desc="Energies (intensity)"):
            ens_fit[i], evs_fit[i] = np.linalg.eigh(all_h[i])
        np.savez(en_fn, ens=ens_fit, evs=evs_fit)

    # ── Intensity matrix (spectral weight spreading) ─────────────────────
    we_fn = cache_dir / f"intensity_fit_we_{n_k_fit_i}_{tmd}_{args.spread_k:.6f}_{args.spread_e:.6f}_Lorentz.npy"
    if not args.no_cache and we_fn.is_file():
        print("Loading cached intensity weights")
        intensity_fit = np.load(we_fn)
    else:
        print("Computing intensity weights")
        intensity_fit = np.zeros((n_k_fit_i, n_e))
        for i in tqdm(range(n_k_fit_i), desc="Weights (intensity)"):
            for n in range(22):
                intensity_fit += _lorentz_spreading(
                    1.0,
                    list_k_fit_i[i],
                    ens_fit[i, n],
                    list_k_fit_i,
                    list_e[None, :],
                    args.spread_k,
                    args.spread_e,
                )
        np.save(we_fn, intensity_fit)
    intensity_fit /= np.max(intensity_fit)

    # ── Orbital occupations (DFT and fit) ────────────────────────────────
    n_k_orb = args.n_k_orb
    list_k_orb, norm_orb = _get_gkm_path(n_k_orb, tmd, endpoint=True)
    n_k_orb = len(list_k_orb)

    orb_fn = cache_dir / f"orbitals_{n_k_orb}_{tmd}.npz"
    if not args.no_cache and orb_fn.is_file():
        print("Loading cached orbitals")
        ens_fit_orb = np.load(orb_fn)["ens_fit"]
        orbitals_fit = np.load(orb_fn)["orb_fit"]
        ens_dft_orb = np.load(orb_fn)["ens_dft"]
        orbitals_dft = np.load(orb_fn)["orb_dft"]
    else:
        print("Computing orbitals")
        all_h_fit = ham.build(list_k_orb, *args_h)
        all_h_dft = ham.build(list_k_orb, *args_h_dft)
        ens_fit_orb = np.zeros((n_k_orb, 22))
        evs_fit_orb = np.zeros((n_k_orb, 22, 22), dtype=complex)
        ens_dft_orb = np.zeros((n_k_orb, 22))
        evs_dft_orb = np.zeros((n_k_orb, 22, 22), dtype=complex)
        for i in tqdm(range(n_k_orb), desc="Orbitals"):
            ens_fit_orb[i], evs_fit_orb[i] = np.linalg.eigh(all_h_fit[i])
            ens_dft_orb[i], evs_dft_orb[i] = np.linalg.eigh(all_h_dft[i])

        orbitals_fit = np.zeros((5, 22, n_k_orb))
        orbitals_dft = np.zeros((5, 22, n_k_orb))
        list_orbs = ([6, 7], [0, 1], [5], [3, 4, 9, 10], [2, 8])
        for orb in range(5):
            inds_orb = list_orbs[orb]
            for ib in range(22):
                for ik in range(n_k_orb):
                    for iorb in inds_orb:
                        orbitals_fit[orb, ib, ik] += (
                            np.linalg.norm(evs_fit_orb[ik, iorb, ib]) ** 2
                            + np.linalg.norm(evs_fit_orb[ik, iorb + 11, ib]) ** 2
                        )
                        orbitals_dft[orb, ib, ik] += (
                            np.linalg.norm(evs_dft_orb[ik, iorb, ib]) ** 2
                            + np.linalg.norm(evs_dft_orb[ik, iorb + 11, ib]) ** 2
                        )
        np.savez(orb_fn,
                 ens_fit=ens_fit_orb, orb_fit=orbitals_fit,
                 ens_dft=ens_dft_orb, orb_dft=orbitals_dft)

    # ── ARPES bands (from monolayer data pipeline) ───────────────────────
    run_dir = os.path.join("Data", f"{tmd}_run_{args.run_id}")
    data = MonolayerData(tmd, master_folder, pts=args.n_k_bands)
    arpes_bands = data.fit_data

    # ── Fit and DFT bands ────────────────────────────────────────────────
    bands_fn = cache_dir / f"bands_en_{tmd}_{args.n_k_bands}.npz"
    if not args.no_cache and bands_fn.is_file():
        print("Loading cached bands")
        bands_fit = np.load(bands_fn)["fit"]
        bands_dft = np.load(bands_fn)["dft"]
    else:
        print("Computing energy bands")
        k_pts_bands = arpes_bands[:, :2]
        bands_fit = ham.eigenvalues(k_pts_bands, *args_h).T
        bands_dft = ham.eigenvalues(k_pts_bands, *args_h_dft).T
        np.savez(bands_fn, fit=bands_fit, dft=bands_dft)

    # ── Plotting ─────────────────────────────────────────────────────────
    s_norm = 10
    s_small = 8
    col_vline = "b"
    lw_vline = 0.5
    ls_vline = (0, (10, 7))
    dict_box = dict(
        facecolor="white",
        edgecolor="black",
        linewidth=1,
        boxstyle="round",
        alpha=0.8,
        pad=0.3,
    )
    xl_text = 0.03
    xr_text = 0.93
    y_text = 0.86

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig = plt.figure(figsize=(7, 4))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1])
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[:, 0], hspace=0.0
    )

    # ── Panel 1: ARPES + Fit intensity ───────────────────────────────────
    norm_k_full = np.concatenate([norm_k_arpes, norm_k_arpes[-1] + norm_fit_i[-1] - norm_fit_i[::-1]])
    intensity_full = np.concatenate([arpes_intensity ** 0.5, intensity_fit[::-1, :]], axis=0)
    ax = fig.add_subplot(gs_left[0])
    ax.pcolormesh(
        norm_k_full, list_e, intensity_full.T,
        cmap="gray_r", rasterized=True
    )
    ax.set_ylabel("Energy [eV]", size=s_norm)
    ax.tick_params(axis="y", which="both", left=True, right=True,
                   labelleft=True, labelright=False, labelsize=s_norm)
    ax.set_xticks([])
    for i in range(3):
        ind = [2, 3, 4]
        xval = norm_k_full[-1] / 6 * ind[i]
        if i == 0:
            xval -= 1e-3
        ax.axvline(xval, color=col_vline, lw=lw_vline, ls=ls_vline, zorder=3)
    ax.text(xl_text, y_text, "ARPES", transform=ax.transAxes,
            ha="left", va="center", fontsize=s_norm, bbox=dict_box)
    ax.text(xr_text, y_text, "Fit", transform=ax.transAxes,
            ha="left", va="center", fontsize=s_norm, bbox=dict_box)

    # ── Panel 2: Orbital content (DFT left, Fit right) ───────────────────
    ax = fig.add_subplot(gs_left[1])
    color = ["red", "brown", "blue", "green", "aqua"]
    labels = [
        r"$d_{xy}+d_{x^2-y^2}$", r"$d_{xz}+d_{yz}$",
        r"$d_{z^2}$", r"$p_x+p_y$", r"$p_z$",
    ]
    leg1 = []
    leg2 = []
    for orb in range(5):
        for ib in range(22):
            ax.scatter(
                norm_orb, ens_dft_orb[:, ib],
                s=(orbitals_dft[orb, ib] * 30), marker="o",
                facecolor=color[orb], lw=0, alpha=0.3, rasterized=True
            )
            ax.scatter(
                2 * norm_orb[-1] - norm_orb, ens_fit_orb[:, ib],
                s=(orbitals_fit[orb, ib] * 30), marker="o",
                facecolor=color[orb], lw=0, alpha=0.3, rasterized=True
            )
        if orb < 3:
            leg1.append(Line2D([0], [0], marker="o", markeredgecolor="none",
                               markerfacecolor=color[orb], markersize=6,
                               label=labels[orb], lw=0))
        else:
            leg2.append(Line2D([0], [0], marker="o", markeredgecolor="none",
                               markerfacecolor=color[orb], markersize=6,
                               label=labels[orb], lw=0))
    legend1 = ax.legend(handles=leg1, loc=(0.23, 0.33), fontsize=s_small,
                        handletextpad=0.35, handlelength=0.5, labelspacing=0.1)
    legend2 = ax.legend(handles=leg2, loc=(0.6, 0.33), fontsize=s_small,
                        handletextpad=0.35, handlelength=0.5, labelspacing=0.1)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.set_ylim(list_e[0], list_e[-1])
    ax.set_xlim(0, 2 * norm_orb[-1])
    ax.set_xticks(
        [norm_orb[-1] / 3 * i for i in [0, 2, 3, 4, 6]],
        [r"$\Gamma$", r"$K$", r"$M$", r"$K$", r"$\Gamma$"],
        size=s_norm,
    )
    ax.set_ylabel("Energy [eV]", size=s_norm)
    ax.tick_params(axis="y", which="both", left=True, right=True,
                   labelleft=True, labelright=False, labelsize=s_norm)
    for i in range(3):
        ind = [2, 3, 4]
        ax.axvline(norm_orb[-1] / 3 * ind[i], color=col_vline,
                   lw=lw_vline, ls=ls_vline, zorder=3)
    ax.text(xl_text, y_text, "DFT-derived", transform=ax.transAxes,
            ha="left", va="center", fontsize=s_norm, bbox=dict_box)
    ax.text(xr_text, y_text, "Fit", transform=ax.transAxes,
            ha="left", va="center", fontsize=s_norm, bbox=dict_box)

    # ── Panel 3: Band comparison ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    lw = 0.7
    for b in range(arpes_bands.shape[1] - 3):
        xline = arpes_bands[:, 0]
        ax.plot(
            xline, arpes_bands[:, 3 + b],
            color="r", marker="s", markersize=2, lw=0,
            label="ARPES" if b == 0 else "", zorder=-1, mec="k", mew=0.2
        )
        ax.plot(
            xline, bands_fit[b, :],
            color="blue", ls="-", lw=lw,
            label="Fit" if b == 0 else "", zorder=2, alpha=0.7
        )
        ax.plot(
            xline, bands_dft[b, :],
            color="orange", ls="-", lw=lw,
            label="DFT" if b == 0 else "", zorder=1, alpha=0.7
        )
    leg = ax.legend(
        loc=(0.6, 0.0), fontsize=s_small, labelspacing=0.1,
        facecolor="white", framealpha=1, edgecolor="black"
    )
    leg.get_frame().set_alpha(1)
    ax.set_xlim(arpes_bands[0, 0], arpes_bands[-1, 0])
    ax.set_ylim(list_e[0], list_e[-1])
    a_mono = LATTICE_CONSTANTS[tmd]
    ks = [arpes_bands[0, 0], 4 / 3 * np.pi / a_mono, arpes_bands[-1, 0]]
    ax.set_xticks(ks, [r"$\Gamma$", r"$K$", r"$M$"], size=s_norm)

    plt.subplots_adjust(
        bottom=0.064, top=0.983, right=0.974, left=0.076,
        wspace=0.136, hspace=0.036
    )

    # ── Save ─────────────────────────────────────────────────────────────
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(master_folder) / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / f"fig_monolayer_{tmd}.svg")
    fig.savefig(out_dir / f"fig_monolayer_{tmd}.png", dpi=600)
    print(f"Saved: {out_dir / f'fig_monolayer_{tmd}.svg'}")
    print(f"Saved: {out_dir / f'fig_monolayer_{tmd}.png'}")


if __name__ == "__main__":
    main()
