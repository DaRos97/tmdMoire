"""Effective mass of the top valence band at Gamma for monolayer and bilayer WSe2.

Three-panel figure:
- Left:   monolayer WSe2 TVB (DFT, fitted, experimental ARPES)
- Middle: monolayer WS2  TVB (DFT, fitted, experimental ARPES)
- Right:  WSe2/WS2 bilayer TVB (from full 44x44 moiré Hamiltonian)

Each panel shows the 2 top valence bands and a parabolic fit near Gamma
along the Gamma-K direction. Effective mass m*/m_e appears in the legend.

Usage
-----
::

    python scripts/effective_mass.py
    python scripts/effective_mass.py --k-max-plot 1.0 --k-max-fit 0.20
    python scripts/effective_mass.py --theta 2.8 --output-dir Figures

Output
------
``Figures/effective_mass_WSe2.png`` — three side-by-side panels.
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmdmoire.material import TMDMaterial, _find_t, _find_e, _find_HSO
from tmdmoire.monolayer.hamiltonian import MonolayerHamiltonian
from tmdmoire.monolayer.data import MonolayerData
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.utils.paths import get_repo_root
from tmdmoire.constants import LATTICE_CONSTANTS


HBAR2_2ME = 3.81
"""hbar^2 / (2 m_e) in eV*Ang^2.

For E(k) = E_0 - A*k^2 (A > 0 for a valence band), the ratio
m*/m_e = HBAR2_2ME / A.
"""

TVB_INDEX = 13
"""Index of the top valence band in the 22 monolayer eigenvalues."""

COLOR_DFT = "C1"
COLOR_FIT = "C0"
COLOR_EXP = "C2"
COLOR_BILAYER = "C4"
COLOR_BILAYER_FIT = "black"


def _parabolic(k, E0, A):
    """Parabolic dispersion E(k) = E0 - A*k^2 (valence band, A > 0)."""
    return E0 - A * k * k


def _tvb_dispersion(k_points, params, tmd="WSe2"):
    """Compute the 2 top valence band energies of a monolayer along a k-path.

    Parameters
    ----------
    k_points : np.ndarray
        Momentum points, shape (N, 2) in 1/Ang.
    params : np.ndarray
        43-element TB parameter array.
    tmd : str
        Material name, "WSe2" or "WS2".

    Returns
    -------
    np.ndarray
        Energies of the top 2 valence bands, shape (N, 2).
        Column 0 is TVB (index 13), column 1 is TVB-1 (index 12).
    """
    ham = MonolayerHamiltonian(TMDMaterial(tmd))
    args_h = (
        _find_t(params),
        _find_e(params),
        _find_HSO(params[-2:]),
        params[-3],
    )
    all_H = ham.build(k_points, *args_h)
    out = np.zeros((k_points.shape[0], 2))
    for i in range(k_points.shape[0]):
        ev = la.eigvalsh(all_H[i])
        out[i, 0] = ev[TVB_INDEX]
        out[i, 1] = ev[TVB_INDEX - 1]
    return out


def _bilayer_tvb_dispersion(k_points, bilayer_params):
    """Compute the top 4 valence band energies of the bilayer along a k-path.

    Uses a single moiré cell (``n_shells=0``, 44x44 Hamiltonian) built
    from the WSe2/WS2 monolayer parameters, interlayer coupling, and
    moiré potential supplied in ``bilayer_params``.

    At Gamma the top 4 valence bands are:
    - rank 0, 1: WSe2 TVB and its SOC partner (degenerate at Gamma)
    - rank 2, 3: WS2 TVB and its SOC partner (degenerate at Gamma)

    Parameters
    ----------
    k_points : np.ndarray
        Momentum points, shape (N, 2) in 1/Ang.
    bilayer_params : dict
        Dictionary with keys: ``tb_wse2``, ``tb_ws2``, ``interlayer_G``,
        ``interlayer_K``, ``theta_deg``.

    Returns
    -------
    np.ndarray
        Energies of the top 4 valence bands, shape (N, 4), sorted
        descending (highest first): cols 0,1 = WSe2; cols 2,3 = WS2.
    """
    wse2 = TMDMaterial("WSe2")
    wse2.fitted_params = bilayer_params["tb_wse2"]
    ws2 = TMDMaterial("WS2")
    ws2.fitted_params = bilayer_params["tb_ws2"]

    geo = MoireGeometry(bilayer_params["theta_deg"])
    inter_G = bilayer_params["interlayer_G"]
    inter_K = bilayer_params["interlayer_K"]
    Vg = inter_G["Vg"]
    phiG = np.deg2rad(inter_G["phiG_deg"])
    Vk = inter_K["Vk"]
    phiK = np.deg2rad(inter_K["phiK_deg"])
    pars_V = (Vg, Vk, phiG, phiK)

    mh = MoireHamiltonian(wse2, ws2, geo)
    evals, _ = mh.diagonalize(
        k_points, n_shells=0, interlayer_params=inter_G, pars_V=pars_V
    )

    out = np.zeros((k_points.shape[0], 4))
    for i in range(k_points.shape[0]):
        ev = evals[i]
        below_idx = np.where(ev < 0.0)[0]
        order = below_idx[np.argsort(ev[below_idx])[::-1]][:4]
        for j, idx in enumerate(order):
            out[i, j] = ev[idx]
    return out


def _fit_parabolic(k_mag, energies):
    """Fit E(k) = E_0 - A*k^2 and return (E0, A, m_eff_rel).

    m*/m_e is reported as a positive number.
    """
    p0 = (energies[0], max((energies[0] - energies[-1]) / max(k_mag[-1] ** 2, 1e-6), 0.1))
    popt, _ = curve_fit(_parabolic, k_mag, energies, p0=p0)
    E0, A = popt
    m_eff_rel = HBAR2_2ME / A
    return E0, A, m_eff_rel


def _load_fitted_params(repo_root, tmd, tb_file=None):
    """Load fitted monolayer TB parameters for a given TMD."""
    if tb_file is not None:
        return np.load(tb_file)
    bilayer_dir = Path(repo_root) / "Inputs" / "bilayer_fitting"
    matches = sorted(bilayer_dir.glob(f"tb_{tmd}*.npy"))
    if not matches:
        raise FileNotFoundError(f"No tb_{tmd}*.npy found in {bilayer_dir}")
    return np.load(matches[0])


def _load_bilayer_params(repo_root):
    """Load the bilayer plot parameters from ``Inputs/plot_bilayer/``."""
    base = Path(repo_root) / "Inputs" / "plot_bilayer"
    return {
        "tb_wse2": np.load(base / "tb_WSe2.npy"),
        "tb_ws2": np.load(base / "tb_WS2.npy"),
        "interlayer_G": json.load(open(base / "interlayer_G.json")),
        "interlayer_K": json.load(open(base / "interlayer_K.json")),
        "theta_deg": 2.8,
    }


def _load_exp_bands(repo_root, tmd, k_max, pts=121):
    """Load the 2 top experimental valence bands along Gamma-K for a TMD.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (k_mag, tvb, tvb1) in (1/Ang, eV, eV).
    """
    data = MonolayerData(tmd, repo_root, pts=pts)
    fit = data.fit_data
    k_mag = fit[:, 0]
    tvb = fit[:, 3]
    tvb1 = fit[:, 4]
    mask_tvb = (k_mag <= k_max) & ~np.isnan(tvb)
    mask_tvb1 = (k_mag <= k_max) & ~np.isnan(tvb1)
    return k_mag[mask_tvb], tvb[mask_tvb], k_mag[mask_tvb1], tvb1[mask_tvb1]


def _plot_monolayers_panel(ax, materials_data, k_plot, k_fit, k_plot_max,
                            k_exp, en_exp, k_exp1, en_exp1):
    """Fill in one panel showing WSe2 and WS2 monolayers together.

    Each material's first 2 experimental bands (TVB + TVB-1) are
    plotted as continuous solid lines: TVB thick, TVB-1 thin (matching
    the bilayer panel style). Materials are distinguished only by color
    (WSe2 -> "C2" green, WS2 -> "C5" brown). The three parabolic fits
    per material (Exp, DFT, Fit) are overlaid as dashed lines. The DFT
    and Fit band dispersions themselves are not drawn; only their
    parabolic fits.

    Returns
    -------
    dict
        ``{tmd: {"dft": (E0,A,m), "fit": (E0,A,m), "exp": (E0,A,m)}}``
    """
    exp_color_per_tmd = {"WSe2": "C2", "WS2": "C5"}
    fit_dash = (0, (3, 2))
    results = {}
    for tmd in ["WSe2", "WS2"]:
        dft_p = materials_data[tmd]["dft"]
        fit_p = materials_data[tmd]["fit"]
        exp_color = exp_color_per_tmd[tmd]

        k_points_fit = np.column_stack([k_fit, np.zeros_like(k_fit)])
        en_dft_fit = _tvb_dispersion(k_points_fit, dft_p, tmd=tmd)
        en_fit_fit = _tvb_dispersion(k_points_fit, fit_p, tmd=tmd)

        E0_dft, A_dft, m_dft = _fit_parabolic(k_fit, en_dft_fit[:, 0])
        E0_fit, A_fit, m_fit = _fit_parabolic(k_fit, en_fit_fit[:, 0])

        k_e, en_e = k_exp[tmd], en_exp[tmd]
        k_e1, en_e1 = k_exp1[tmd], en_exp1[tmd]
        mask = k_e <= k_fit[-1]
        E0_exp, A_exp, m_exp = _fit_parabolic(k_e[mask], en_e[mask])

        # Experimental bands: TVB thick, TVB-1 thin (mirrors the bilayer
        # panel style). Both continuous solid lines.
        ax.plot(k_e, en_e, color=exp_color, lw=1.4,
                label=f"{tmd} TVB, $m^*/m_e={m_exp:.2f}$")
        ax.plot(k_e1, en_e1, color=exp_color, lw=0.7, alpha=0.7)

        # Three parabolic fits per material, all dashed
        ax.plot(k_plot, _parabolic(k_plot, E0_exp, A_exp),
                color=exp_color, lw=0.9, ls=fit_dash)
        ax.plot(k_plot, _parabolic(k_plot, E0_fit, A_fit),
                color=COLOR_FIT, lw=0.9, ls=fit_dash,
                label=f"{tmd} Fit fit, $m^*/m_e={m_fit:.2f}$")
        ax.plot(k_plot, _parabolic(k_plot, E0_dft, A_dft),
                color=COLOR_DFT, lw=0.9, ls=fit_dash,
                label=f"{tmd} DFT fit, $m^*/m_e={m_dft:.2f}$")

        results[tmd] = {
            "dft": (E0_dft, A_dft, m_dft),
            "fit": (E0_fit, A_fit, m_fit),
            "exp": (E0_exp, A_exp, m_exp),
        }

    ax.set_title("Monolayer WSe$_2$ and WS$_2$")
    ax.legend(loc="lower center", fontsize=5.5, ncol=2,
              framealpha=0.95, handlelength=2.5, columnspacing=1.0)
    return results


def _plot_monolayer_panel(ax, tmd, dft_params, fit_params, k_plot, k_fit,
                          k_plot_max, k_exp_tvb, en_exp_tvb, k_exp_tvb1, en_exp_tvb1,
                          en_dft_full, en_fit_full):
    """Fill in one monolayer panel (DFT, fit, exp + parabolic fits)."""
    en_dft_fit = _tvb_dispersion(
        np.column_stack([k_fit, np.zeros_like(k_fit)]), dft_params, tmd=tmd
    )
    en_fit_fit = _tvb_dispersion(
        np.column_stack([k_fit, np.zeros_like(k_fit)]), fit_params, tmd=tmd
    )

    E0_dft, A_dft, m_dft = _fit_parabolic(k_fit, en_dft_fit[:, 0])
    E0_fit, A_fit, m_fit = _fit_parabolic(k_fit, en_fit_fit[:, 0])
    mask = k_exp_tvb <= k_fit[-1]
    E0_exp, A_exp, m_exp = _fit_parabolic(
        k_exp_tvb[mask], en_exp_tvb[mask]
    )

    fit_curve_exp = _parabolic(k_plot, E0_exp, A_exp)
    fit_curve_dft = _parabolic(k_plot, E0_dft, A_dft)
    fit_curve_fit = _parabolic(k_plot, E0_fit, A_fit)

    ax.plot(k_plot, en_dft_full[:, 1], color=COLOR_DFT, lw=0.8, ls="--", alpha=0.5)
    ax.plot(k_plot, en_fit_full[:, 1], color=COLOR_FIT, lw=0.8, ls="--", alpha=0.5)
    ax.scatter(k_exp_tvb1, en_exp_tvb1, color=COLOR_EXP, s=10, marker="x",
               alpha=0.6, linewidths=1.0)

    ax.plot(k_plot, en_dft_full[:, 0], color=COLOR_DFT, lw=1.2,
            label=fr"DFT, $m^*/m_e = {m_dft:.3f}$")
    ax.plot(k_plot, en_fit_full[:, 0], color=COLOR_FIT, lw=1.2,
            label=fr"Fit, $m^*/m_e = {m_fit:.3f}$")
    ax.scatter(k_exp_tvb, en_exp_tvb, color=COLOR_EXP, s=18, marker="o",
               edgecolors="k", linewidths=0.3, zorder=5,
               label=fr"Exp, $m^*/m_e = {m_exp:.3f}$")

    ax.plot(k_plot, fit_curve_exp, color=COLOR_EXP, lw=1.6, ls="--")
    ax.plot(k_plot, fit_curve_dft, color=COLOR_DFT, lw=1.6, ls="--")
    ax.plot(k_plot, fit_curve_fit, color=COLOR_FIT, lw=1.6, ls="--")

    ax.set_title(fr"{tmd.replace('W', r'W')}$_2$ monolayer")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    return {"exp": (E0_exp, A_exp, m_exp),
            "dft": (E0_dft, A_dft, m_dft),
            "fit": (E0_fit, A_fit, m_fit)}


def _plot_bilayer_panel(ax, bilayer_params, k_plot, k_fit):
    """Fill in the bilayer panel (WSe2 band + WS2 band + parabolic fits).

    Fits the WSe2-derived TVB (column 0) and the WS2-derived band
    (column 2). The SOC-split partners (columns 1 and 3) are drawn
    faint as context.
    """
    k_points_plot = np.column_stack([k_plot, np.zeros_like(k_plot)])
    k_points_fit = np.column_stack([k_fit, np.zeros_like(k_fit)])

    en_bil_plot = _bilayer_tvb_dispersion(k_points_plot, bilayer_params)
    en_bil_fit = _bilayer_tvb_dispersion(k_points_fit, bilayer_params)

    E0_wse2, A_wse2, m_wse2 = _fit_parabolic(k_fit, en_bil_fit[:, 0])
    E0_ws2, A_ws2, m_ws2 = _fit_parabolic(k_fit, en_bil_fit[:, 2])

    fit_curve_wse2 = _parabolic(k_plot, E0_wse2, A_wse2)
    fit_curve_ws2 = _parabolic(k_plot, E0_ws2, A_ws2)

    # SOC-split partners (faint context)
    ax.plot(k_plot, en_bil_plot[:, 1], color=COLOR_BILAYER, lw=0.6, alpha=0.4)
    ax.plot(k_plot, en_bil_plot[:, 3], color=COLOR_BILAYER, lw=0.6, alpha=0.4)

    # Main bands (bold purple)
    ax.plot(k_plot, en_bil_plot[:, 0], color=COLOR_BILAYER, lw=1.5,
            label=r"WSe$_2$ band")
    ax.plot(k_plot, en_bil_plot[:, 2], color=COLOR_BILAYER, lw=1.5,
            ls="--", label=r"WS$_2$ band")

    # Parabolic fits (thin black, clearly distinct from bands)
    ax.plot(k_plot, fit_curve_wse2, color=COLOR_BILAYER_FIT, lw=1.0,
            label=fr"WSe$_2$ fit, $m^*/m_e = {m_wse2:.2f}$")
    ax.plot(k_plot, fit_curve_ws2, color=COLOR_BILAYER_FIT, lw=1.0,
            ls="--", label=fr"WS$_2$ fit, $m^*/m_e = {m_ws2:.2f}$")

    ax.set_title(r"WSe$_2$/WS$_2$ bilayer")
    ax.legend(loc="lower left", fontsize=6, framealpha=0.95, ncol=1)

    return {"wse2": (E0_wse2, A_wse2, m_wse2),
            "ws2": (E0_ws2, A_ws2, m_ws2)}


def main():
    parser = argparse.ArgumentParser(
        description="Effective mass of WSe2/WS2 TVB at Gamma: monolayer + bilayer."
    )
    parser.add_argument("--k-max-plot", type=float, default=0.6,
                        help="Max |k| (1/Ang) for the band plot (default 0.6).")
    parser.add_argument("--k-max-fit", type=float, default=0.35,
                        help="Max |k| (1/Ang) for the parabolic fit window (default 0.35).")
    parser.add_argument("--n-k-plot", type=int, default=251,
                        help="Number of k-points for the TB band plot (default 251).")
    parser.add_argument("--n-k-fit", type=int, default=51,
                        help="Number of k-points for the TB parabolic fit (default 51).")
    parser.add_argument("--exp-pts", type=int, default=121,
                        help="Number of interpolation points for the experimental data (default 121).")
    parser.add_argument("--tb-file-wse2", type=str, default=None,
                        help="Path to fitted monolayer TB params for WSe2.")
    parser.add_argument("--tb-file-ws2", type=str, default=None,
                        help="Path to fitted monolayer TB params for WS2.")
    parser.add_argument("--theta", type=float, default=None,
                        help="Twist angle (deg) for the bilayer. Default: 2.8.")
    parser.add_argument("--output-dir", type=str, default="Figures",
                        help="Output directory (default: Figures/).")
    args = parser.parse_args()

    repo_root = get_repo_root()
    out_dir = Path(repo_root) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    k_plot_max = args.k_max_plot
    k_fit_max = args.k_max_fit

    # ── Monolayer parameters ─────────────────────────────────────────────
    materials = {
        "WSe2": {
            "dft": np.array(TMDMaterial("WSe2").dft_params),
            "fit": _load_fitted_params(repo_root, "WSe2", args.tb_file_wse2),
            "a": LATTICE_CONSTANTS["WSe2"],
        },
        "WS2": {
            "dft": np.array(TMDMaterial("WS2").dft_params),
            "fit": _load_fitted_params(repo_root, "WS2", args.tb_file_ws2),
            "a": LATTICE_CONSTANTS["WS2"],
        },
    }
    bilayer_params = _load_bilayer_params(repo_root)
    if args.theta is not None:
        bilayer_params["theta_deg"] = args.theta

    print(f"  k_max_plot = {k_plot_max},  k_max_fit = {k_fit_max}")
    print(f"  WSe2 a = {materials['WSe2']['a']} A, |K| = {4*np.pi/(3*materials['WSe2']['a']):.4f} 1/A")
    print(f"  WS2  a = {materials['WS2']['a']} A,  |K| = {4*np.pi/(3*materials['WS2']['a']):.4f} 1/A")
    print(f"  theta_bilayer = {bilayer_params['theta_deg']:.2f} deg")

    # ── k-grids ──────────────────────────────────────────────────────────
    k_plot = np.linspace(0.0, k_plot_max, args.n_k_plot)
    k_fit = np.linspace(0.0, k_fit_max, args.n_k_fit)

    # ── Dispersions and experimental data per material ──────────────────
    en_dft_full = {}
    en_fit_full = {}
    k_exp = {}
    en_exp = {}
    k_exp1 = {}
    en_exp1 = {}
    for tmd in ["WSe2", "WS2"]:
        m = materials[tmd]
        k_points_plot = np.column_stack([k_plot, np.zeros_like(k_plot)])
        en_dft_full[tmd] = _tvb_dispersion(k_points_plot, m["dft"], tmd=tmd)
        en_fit_full[tmd] = _tvb_dispersion(k_points_plot, m["fit"], tmd=tmd)
        k_e, en_e, k_e1, en_e1 = _load_exp_bands(repo_root, tmd, k_plot_max, pts=args.exp_pts)
        k_exp[tmd], en_exp[tmd], k_exp1[tmd], en_exp1[tmd] = k_e, en_e, k_e1, en_e1

    # ── Plot ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 0.9,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })

    prb_col_in = 3.375
    fig, (ax_mono, ax_bil) = plt.subplots(
        2, 1, figsize=(prb_col_in, 1.6 * prb_col_in), sharex=True
    )

    res_mono = _plot_monolayers_panel(
        ax_mono, materials,
        k_plot, k_fit, k_plot_max,
        k_exp, en_exp, k_exp1, en_exp1,
    )
    res_bil = _plot_bilayer_panel(ax_bil, bilayer_params, k_plot, k_fit)

    ax_mono.set_ylabel(r"$E$ [eV]")
    ax_bil.set_ylabel(r"$E$ [eV]")
    ax_bil.set_xlabel(r"$|k|$ along $\Gamma \to K$ $[\AA^{-1}]$")

    for ax in (ax_mono, ax_bil):
        ax.set_xlim(0, k_plot_max)

    # Subplot labels (a) and (b)
    panel_label_kwargs = dict(
        transform=ax_mono.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_mono.text(-0.10, 1.04, "(a)", **panel_label_kwargs)
    panel_label_kwargs["transform"] = ax_bil.transAxes
    ax_bil.text(-0.10, 1.04, "(b)", **panel_label_kwargs)

    plt.tight_layout()

    # ── Print results ────────────────────────────────────────────────────
    print("")
    print("WSe2/WS2 TVB effective mass (Gamma-K direction)")
    for tmd in ["WSe2", "WS2"]:
        res = res_mono[tmd]
        print(f"  {tmd} monolayer Exp:  m*/m_e = {res['exp'][2]:.4f}   (E0 = {res['exp'][0]:.4f} eV)")
        print(f"  {tmd} monolayer DFT:  m*/m_e = {res['dft'][2]:.4f}   (E0 = {res['dft'][0]:.4f} eV)")
        print(f"  {tmd} monolayer Fit:  m*/m_e = {res['fit'][2]:.4f}   (E0 = {res['fit'][0]:.4f} eV)")
    print(f"  Bilayer WSe2 band: m*/m_e = {res_bil['wse2'][2]:.4f}   (E0 = {res_bil['wse2'][0]:.4f} eV)")
    print(f"  Bilayer WS2 band:  m*/m_e = {res_bil['ws2'][2]:.4f}   (E0 = {res_bil['ws2'][0]:.4f} eV)")

    fn = out_dir / "SM_effective_mass.pdf"
    fig.savefig(fn, bbox_inches="tight")
    print("")
    print(f"Saved: {fn}")


if __name__ == "__main__":
    main()
