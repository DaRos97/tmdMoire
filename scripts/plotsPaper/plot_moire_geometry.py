"""Standalone plot of moire geometry quantities vs twist angle.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.

Produces:
  moire_geometry.png  -- a_M(theta) and eta(theta) on dual y-axes

Usage:
    python plot_moire_geometry.py [--output-dir ./figures]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Core formulas copied from tmdmoire/bilayer/geometry.py ────────────────────

A_WS2 = 3.18
A_WSE2 = 3.32

def moire_length(theta_deg):
    theta = theta_deg / 180.0 * np.pi
    return 1.0 / np.sqrt(
        1.0 / A_WSE2**2 + 1.0 / A_WS2**2
        - 2.0 * np.cos(theta) / (A_WSE2 * A_WS2)
    )

def mini_bz_rotation_deg(theta_deg):
    theta = theta_deg / 180.0 * np.pi
    eta_rad = np.arctan(
        np.tan(theta / 2.0) * (A_WSE2 + A_WS2) / (A_WSE2 - A_WS2)
    )
    return eta_rad / np.pi * 180.0


def main():
    output_dir = Path(__file__).resolve().parent / "figures"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    theta = np.linspace(0.5, 5.0, 500)
    a_m = moire_length(theta)
    eta_deg = mini_bz_rotation_deg(theta)

    theta_marks = [2.6, 2.8, 3.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1, 3]})
    ax1b = ax1.twinx()

    # ── Left subplot: a_M(theta) + eta(theta) on dual y-axes ────────────────────

    line_am, = ax1.plot(theta, a_m, "C0-", lw=1.5, label=r"$a_M$")
    line_eta, = ax1b.plot(theta, eta_deg, "C1-", lw=1.5, label=r"$\eta$")

    ax1.set_xlabel(r"Twist angle $\theta$ (deg)", fontsize=12)
    ax1.set_ylabel("Moiré period $a_M$ (Å)", fontsize=12, color="C0")
    ax1b.set_ylabel(r"Mini-BZ rotation $\eta$ (deg)", fontsize=12, color="C1")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1b.tick_params(axis="y", labelcolor="C1")

    x_min, x_max = theta[0], theta[-1]
    label_y_offsets = {2.6: 23, 2.8: 20, 3.0: 17}
    for tm in theta_marks:
        am_val = moire_length(tm)
        et_val = mini_bz_rotation_deg(tm)
        x_frac = (tm - x_min) / (x_max - x_min)
        ax1.axvline(x=tm, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax1.plot(tm, am_val, "o", color="C0", ms=5)
        ax1b.plot(tm, et_val, "o", color="C1", ms=5)
        ax1.axhline(y=am_val, xmax=x_frac, color="C0", ls="--", lw=0.8, alpha=0.6)
        ax1b.axhline(y=et_val, xmin=x_frac, color="C1", ls="--", lw=0.8, alpha=0.6)
        ax1b.text(tm + 0.03, label_y_offsets[tm], f"{tm}\u00b0", fontsize=10, color="gray",
                  va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # ── Right subplot: L2 distance heatmaps ──────────────────────────────────────

    data_dir = Path(__file__).resolve().parent / "data"

    d_a = np.load(data_dir / "edc_gamma_4Dsm8a_Vg_10.5meV_phiG_176deg.npz")
    phi_edges_a = d_a["phi_edges"]
    Vg_edges_meV = d_a["Vg_edges_meV"]
    dist_sep_a = d_a["dist_sep_2d_meV"]

    d_b = np.load(data_dir / "edc_gamma_4Dsm8d_S11_m26_Vg_11.5meV_phiG_175deg.npz")
    phi_edges_b = d_b["phi_edges"]
    shift_b = 20.0
    dist_sep_b = d_b["dist_sep_2d_meV"].copy()
    phi_centers_b = (phi_edges_b[:-1] + phi_edges_b[1:]) / 2.0
    dist_sep_b[:, phi_centers_b > 180] = np.nan

    d_c = np.load(data_dir / "edc_gamma_4Dsm8e_S11_p30_Vg_9.5meV_phiG_175deg.npz")
    phi_edges_c = d_c["phi_edges"]
    shift_c = 20.0
    dist_sep_c = d_c["dist_sep_2d_meV"].copy()
    phi_centers_c = (phi_edges_c[:-1] + phi_edges_c[1:]) / 2.0
    dist_sep_c[:, phi_centers_c < 180] = np.nan

    im_a = ax2.pcolormesh(phi_edges_a, Vg_edges_meV, dist_sep_a,
                          cmap="plasma_r", shading="flat")
    im_b = ax2.pcolormesh(phi_edges_b - shift_b, Vg_edges_meV, dist_sep_b,
                          cmap="Blues", shading="flat")
    im_c = ax2.pcolormesh(phi_edges_c + shift_c, Vg_edges_meV, dist_sep_c,
                          cmap="Reds", shading="flat")

    for v in np.arange(2, 21, 2):
        ax2.axhline(y=v, color="gray", ls="--", lw=0.5, alpha=0.5)

    ax2.text(180, 19, r"S11 moiré analysis with twist angle uncertainty $\pm0.2^\circ$",
             ha="center", va="top", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"))
    ax2.axvline(x=180, color="red", ls="--", lw=0.7, alpha=0.5)
    ax2.axvline(x=180 - shift_b, ymin=8 / 20, ymax=18 / 20, color="red", ls="--", lw=0.7)
    ax2.annotate(
        "", xy=(165 - shift_b, 8), xytext=(180 - shift_b, 8),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    for x_pos, lbl in [(180 - shift_b, "180"), (170 - shift_b, "170")]:
        ax2.plot([x_pos, x_pos], [7.8, 8.0], "k-", lw=0.8)
        ax2.text(x_pos, 7.65, lbl, fontsize=10, ha="center", va="top")

    ax2.axvline(x=180 + shift_c, ymin=5 / 20, ymax=15 / 20, color="red", ls="--", lw=0.7)
    ax2.annotate(
        "", xy=(195 + shift_c, 5), xytext=(180 + shift_c, 5),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    for x_pos, lbl in [(180 + shift_c, "180"), (190 + shift_c, "190")]:
        ax2.plot([x_pos, x_pos], [4.8, 5.0], "k-", lw=0.8)
        ax2.text(x_pos, 4.65, lbl, fontsize=10, ha="center", va="top")

    ax2.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
    ax2.set_ylabel(r"$\bar{V}$ (meV)", fontsize=12)
    ax2.set_xticks(np.arange(140, 221, 20))
    ax2.set_yticks(np.arange(0, 21, 2))
    ax2.set_ylim(0, 20)
    ax2.set_xlim(140, 220)

    divider = make_axes_locatable(ax2)
    cax_b = divider.append_axes("right", size="4%", pad=0.05)
    cb_b = plt.colorbar(im_b, cax=cax_b, orientation="vertical")
    cb_b.set_ticks([])

    cax_a = divider.append_axes("right", size="4%", pad=0.05)
    cb_a = plt.colorbar(im_a, cax=cax_a, orientation="vertical")
    cb_a.set_ticks([])

    cax_c = divider.append_axes("right", size="4%", pad=0.05)
    cb_c = plt.colorbar(im_c, cax=cax_c, orientation="vertical")
    cb_c.set_label("Min distance $f$ (meV)", fontsize=12)

    cax_b.set_title("2.6\u00b0", fontsize=12)
    cax_a.set_title("2.8\u00b0", fontsize=12)
    cax_c.set_title("3.0\u00b0", fontsize=12)

    fig.savefig(output_dir / "SM_twist_angle.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'SM_twist_angle.pdf'}")


if __name__ == "__main__":
    main()
