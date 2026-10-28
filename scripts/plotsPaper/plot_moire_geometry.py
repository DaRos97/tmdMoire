"""Standalone twist-angle geometry and Gamma EDC summary plot.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.

Produces:
  SM_twist_angle.pdf  -- geometry and S11/S3 Gamma distance heatmaps

Usage:
    python plot_moire_geometry.py [--output-dir ./figures]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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


def _plot_heatmap(ax, data_path, cmap, norm, shift_deg=0.0, side=None):
    with np.load(data_path) as data:
        phi_edges = data["phi_edges"].copy()
        vg_edges_mev = data["Vg_edges_meV"].copy()
        dist_sep = data["dist_sep_2d_meV"].copy()

    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2.0
    if side == "low":
        dist_sep[:, phi_centers > 180] = np.nan
    elif side == "high":
        dist_sep[:, phi_centers < 180] = np.nan

    return ax.pcolormesh(
        phi_edges + shift_deg,
        vg_edges_mev,
        dist_sep,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )


def _plot_heatmap_panel(
    ax,
    data_dir,
    sample,
    file_names,
    angle_uncertainty,
    norm,
):
    shift_deg = 25.0
    image_low = _plot_heatmap(
        ax,
        data_dir / file_names["low"],
        "Blues",
        norm,
        shift_deg=-shift_deg,
        side="low",
    )
    image_center = _plot_heatmap(
        ax,
        data_dir / file_names["center"],
        "plasma_r",
        norm,
    )
    image_high = _plot_heatmap(
        ax,
        data_dir / file_names["high"],
        "Reds",
        norm,
        shift_deg=shift_deg,
        side="high",
    )

    for v in np.arange(2, 23, 2):
        ax.axhline(y=v, color="gray", ls="--", lw=0.5, alpha=0.5)

    ax.set_xlim(130, 230)
    ax.set_ylim(0, 22)
    ax.set_ylabel(r"$V_\Gamma$ (meV)", fontsize=12)
    ax.set_xticks(np.arange(140, 221, 20))
    ax.set_yticks(np.arange(0, 23, 2))

    ax.text(
        180,
        21,
        rf"{sample} moiré analysis with twist angle uncertainty $\pm{angle_uncertainty:.1f}^\circ$",
        ha="center",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"),
    )

    center_x = 180.0
    low_center_x = center_x - shift_deg
    high_center_x = center_x + shift_deg
    ax.axvline(x=center_x, color="red", ls="--", lw=0.7, alpha=0.5)

    ax.axvline(
        x=low_center_x,
        ymin=8 / 22,
        ymax=18 / 22,
        color="red",
        ls="--",
        lw=0.7,
    )
    ax.annotate(
        "",
        xy=(low_center_x - shift_deg, 8),
        xytext=(low_center_x, 8),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    for x_pos, label in [
        (low_center_x, "180"),
        (low_center_x - 10, "170"),
        (low_center_x - 20, "160"),
    ]:
        ax.plot([x_pos, x_pos], [7.8, 8.0], "k-", lw=0.8)
        ax.text(x_pos, 7.65, label, fontsize=10, ha="center", va="top")

    ax.axvline(
        x=high_center_x,
        ymin=5 / 22,
        ymax=15 / 22,
        color="red",
        ls="--",
        lw=0.7,
    )
    ax.annotate(
        "",
        xy=(high_center_x + shift_deg, 5),
        xytext=(high_center_x, 5),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    for x_pos, label in [
        (high_center_x, "180"),
        (high_center_x + 10, "190"),
        (high_center_x + 20, "200"),
    ]:
        ax.plot([x_pos, x_pos], [4.8, 5.0], "k-", lw=0.8)
        ax.text(x_pos, 4.65, label, fontsize=10, ha="center", va="top")

    return image_low, image_center, image_high


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

    theta_marks_s11 = [2.6, 2.8, 3.0]
    theta_marks_s3 = [1.5, 1.8, 2.1]

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    layout = fig.add_gridspec(1, 2, width_ratios=[1.1, 3.5])
    ax1 = fig.add_subplot(layout[0, 0])
    ax1b = ax1.twinx()
    ax1.set_box_aspect(1.5)

    heatmap_layout = layout[0, 1].subgridspec(
        2,
        4,
        width_ratios=[1.0, 0.04, 0.04, 0.04],
        hspace=0.05,
        wspace=0.04,
    )
    ax_s11 = fig.add_subplot(heatmap_layout[0, 0])
    ax_s3 = fig.add_subplot(heatmap_layout[1, 0], sharex=ax_s11)
    colorbar_axes = [fig.add_subplot(heatmap_layout[:, i]) for i in (1, 2, 3)]

    # ── Left subplot: a_M(theta) + eta(theta) on dual y-axes ────────────────────

    line_am, = ax1.plot(theta, a_m, "C0-", lw=1.5, label=r"$a_M$")
    line_eta, = ax1b.plot(theta, eta_deg, "C1-", lw=1.5, label=r"$\eta$")

    ax1.set_xlabel(r"Twist angle $\theta$ (deg)", fontsize=12)
    ax1.set_ylabel("Moiré period $a_M$ (Å)", fontsize=12, color="C0")
    ax1b.set_ylabel(r"Mini-BZ rotation $\eta$ (deg)", fontsize=12, color="C1")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1b.tick_params(axis="y", labelcolor="C1")

    x_min, x_max = theta[0], theta[-1]
    label_y_offsets_s11 = {2.6: 23, 2.8: 20, 3.0: 17}
    for tm in theta_marks_s11:
        am_val = moire_length(tm)
        et_val = mini_bz_rotation_deg(tm)
        x_frac = (tm - x_min) / (x_max - x_min)
        ax1.axvline(x=tm, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax1.plot(tm, am_val, "o", color="C0", ms=5)
        ax1b.plot(tm, et_val, "o", color="C1", ms=5)
        ax1.axhline(y=am_val, xmax=x_frac, color="C0", ls="--", lw=0.8, alpha=0.6)
        ax1b.axhline(y=et_val, xmin=x_frac, color="C1", ls="--", lw=0.8, alpha=0.6)
        ax1b.text(tm + 0.03, label_y_offsets_s11[tm], f"{tm}\u00b0", fontsize=10, color="gray",
                  va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1))

    for tm in theta_marks_s3:
        am_val = moire_length(tm)
        et_val = mini_bz_rotation_deg(tm)
        x_frac = (tm - x_min) / (x_max - x_min)
        ax1.axvline(x=tm, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax1.plot(tm, am_val, "o", color="C0", ms=5)
        ax1b.plot(tm, et_val, "o", color="C1", ms=5)
        ax1.axhline(y=am_val, xmax=x_frac, color="C0", ls="--", lw=0.8, alpha=0.6)
        ax1b.axhline(y=et_val, xmin=x_frac, color="C1", ls="--", lw=0.8, alpha=0.6)
        ax1b.text(
            tm + 0.03,
            et_val - 5,
            f"{tm}\u00b0",
            fontsize=10,
            color="gray",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )

    # ── Right panels: L2 distance heatmaps with sample-specific L1 filters ─────

    data_dir = Path(__file__).resolve().parent / "data"
    shared_norm = Normalize(vmin=0.0, vmax=10.0)
    images_s11 = _plot_heatmap_panel(
        ax_s11,
        data_dir,
        "S11",
        {
            "low": "edc_gamma_4Dsm8d_155_205_L1_10meV.npz",
            "center": "edc_gamma_4Dsm8a_155_205_L1_10meV.npz",
            "high": "edc_gamma_4Dsm8e_155_205_L1_10meV.npz",
        },
        angle_uncertainty=0.2,
        norm=shared_norm,
    )
    _plot_heatmap_panel(
        ax_s3,
        data_dir,
        "S3",
        {
            "low": "edc_gamma_S3_2b_155_205_L1_20meV.npz",
            "center": "edc_gamma_S3_2a_155_205_L1_20meV.npz",
            "high": "edc_gamma_S3_2c_155_205_L1_20meV.npz",
        },
        angle_uncertainty=0.3,
        norm=shared_norm,
    )

    # One shared set of colorbars: blue is the lower-twist endpoint, purple the
    # central angle, and red the upper-twist endpoint in both sample panels.
    shared_images = [images_s11[0], images_s11[1], images_s11[2]]
    angle_pairs = [(2.6, 1.5), (2.8, 1.8), (3.0, 2.1)]
    colorbars = []
    for image, cax, (s11_angle, s3_angle) in zip(
        shared_images, colorbar_axes, angle_pairs
    ):
        colorbar = fig.colorbar(image, cax=cax, orientation="vertical")
        colorbar.ax.set_title(
            f"{s11_angle:.1f}\u00b0\n{s3_angle:.1f}\u00b0",
            fontsize=9,
            pad=4,
        )
        colorbars.append(colorbar)

    colorbars[0].set_ticks([])
    colorbars[1].set_ticks([])
    colorbars[2].set_ticks(np.arange(0, 11, 2))
    colorbars[2].set_label(r"Min distance $f$ (meV)", fontsize=11)

    ax_s11.tick_params(axis="x", labelbottom=False)
    ax_s3.set_xlabel(r"$\phi_\Gamma$ (deg)", fontsize=12)

    fig.savefig(output_dir / "SM_twist_angle.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'SM_twist_angle.pdf'}")


if __name__ == "__main__":
    main()
