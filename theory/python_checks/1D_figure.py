import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _1D_common import g, alpha, eps, H


def draw_cartoon(ax, ks, eigenvalues, eigenvectors):
    central_weight = eigenvectors[:, 1, :] ** 2
    max_size = 80.0
    v0_bands = np.stack([eps(ks, n) for n in (-1, 0, 1)], axis=1)
    for n in range(3):
        ax.plot(ks / g, v0_bands[:, n], color="firebrick", lw=0.2, zorder=4)
    for band in range(3):
        ax.scatter(
            ks / g,
            eigenvalues[:, band],
            s=central_weight[:, band] * max_size,
            c="C0",
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("k (G)")
    ax.set_ylabel("Energy")
    ax.set_yticks([])
    ax.set_xlim(ks[0] / g, ks[-1] / g)
    ax.set_ylim(-20, 1)

    axins = ax.inset_axes([0.66, 0.03, 0.30, 0.30])
    for n in range(3):
        axins.plot(ks / g, v0_bands[:, n], color="firebrick", lw=0.2, zorder=4)
    for band in range(3):
        axins.scatter(
            ks / g,
            eigenvalues[:, band],
            s=central_weight[:, band] * max_size * 5.0,
            c="C0",
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
    axins.set_xlim(-0.3, 0.3)
    axins.set_ylim(-8.7, -6.7)
    axins.set_xticks([])
    axins.set_yticks([])

    idx0 = ks.size // 2

    k_gap = -0.5 * g
    i_gap = int(np.argmin(np.abs(ks - k_gap)))
    e_gap = np.linalg.eigvalsh(H(ks[i_gap], 1.0))
    e_top = e_gap[2]
    e_mid = e_gap[1]
    ax.annotate(
        "",
        xy=(k_gap / g, e_top),
        xytext=(k_gap / g, e_mid),
        arrowprops=dict(arrowstyle="<->", color="c", lw=0.7, mutation_scale=8,
                        shrinkA=0, shrinkB=0),
        zorder=5,
    )
    ax.text(
        k_gap / g + 0.08,
        0.5 * (e_top + e_mid) - 0.5,
        r"$\Delta$",
        color="c",
        fontsize=10,
        va="center",
    )

    i_src = int(np.argmin(np.abs(ks - (-1.5 * g))))
    y_src = eigenvalues[i_src, 1]
    y_tgt = eigenvalues[i_src, 2]
    top_band = eigenvalues[:, 2]
    diffs = top_band - y_src
    signs = np.sign(diffs)
    cross_idx = np.where(np.diff(signs))[0]
    cross_idx = cross_idx[cross_idx < i_src]
    j = int(cross_idx[0])
    k_cross = ks[j] - diffs[j] * (ks[j + 1] - ks[j]) / (diffs[j + 1] - diffs[j])

    fig_local = ax.figure
    bbox = ax.get_window_extent().transformed(fig_local.dpi_scale_trans.inverted())
    x_range_data = (ks[-1] - ks[0]) / g
    y_lo, y_hi = ax.get_ylim()
    y_range_data = y_hi - y_lo
    r_pts = np.sqrt(80.0 / np.pi)
    r_inches = r_pts / 72.0
    r_x = r_inches / (bbox.width / x_range_data)
    r_y = r_inches / (bbox.height / y_range_data)

    import matplotlib.patches as mpatches
    from matplotlib.path import Path

    thetas_ur = np.deg2rad(np.linspace(-45.0, 135.0, 60))
    xs_ur = -1.5 + r_x * np.cos(thetas_ur)
    ys_ur = y_src + r_y * np.sin(thetas_ur)
    verts_ur = np.column_stack([xs_ur, ys_ur])
    verts_ur = np.vstack([verts_ur, [[-1.5, y_src], [-1.5 + r_x * np.cos(thetas_ur[0]),
                                                     y_src + r_y * np.sin(thetas_ur[0])]]])
    codes_ur = [Path.MOVETO] + [Path.LINETO] * (len(thetas_ur) + 1)
    ax.add_patch(mpatches.PathPatch(Path(verts_ur, codes_ur),
                                    facecolor="orange", edgecolor="black",
                                    linewidth=0.5, zorder=6))

    thetas_ll = np.deg2rad(np.linspace(135.0, 315.0, 60))
    xs_ll = -1.5 + r_x * np.cos(thetas_ll)
    ys_ll = y_src + r_y * np.sin(thetas_ll)
    verts_ll = np.column_stack([xs_ll, ys_ll])
    verts_ll = np.vstack([verts_ll, [[-1.5, y_src], [-1.5 + r_x * np.cos(thetas_ll[0]),
                                                     y_src + r_y * np.sin(thetas_ll[0])]]])
    codes_ll = [Path.MOVETO] + [Path.LINETO] * (len(thetas_ll) + 1)
    ax.add_patch(mpatches.PathPatch(Path(verts_ll, codes_ll),
                                    facecolor="gold", edgecolor="black",
                                    linewidth=0.5, zorder=6))

    ax.scatter(k_cross / g, y_src, s=80, c="gold", edgecolors="black",
               linewidths=0.5, zorder=6)
    ax.scatter(-1.5, y_tgt, s=80, c="orange", edgecolors="black",
               linewidths=0.5, zorder=6)
    ax.plot([-1.5, k_cross / g], [y_src, y_src], color="gold",
            lw=0.8, ls="--", zorder=5)
    ax.plot([-1.5, -1.5], [y_src, y_tgt], color="orange",
            lw=0.8, ls="--", zorder=5)
    x_mid_h = 0.5 * (-1.5 + k_cross / g)
    ax.text(x_mid_h, y_src + 1.5, r"$\rho$", color="gold",
            fontsize=10, ha="center")
    ax.text(-1.5 - 0.12, 0.5 * (y_src + y_tgt), r"$\lambda$", color="orange",
            fontsize=10, va="center", ha="right")

    e_0 = np.linalg.eigvalsh(H(ks[idx0], 1.0))
    e_top_0 = e_0[2]
    e_bot_0 = e_0[0]
    ax.annotate(
        "",
        xy=(0.0, e_top_0),
        xytext=(0.0, e_bot_0),
        arrowprops=dict(arrowstyle="<->", color="limegreen", lw=0.7, mutation_scale=12,
                        shrinkA=0, shrinkB=0),
        zorder=5,
    )
    ax.text(
        0.08,
        0.5 * (e_top_0 + e_bot_0) + 1.5,
        r"$\chi$",
        color="limegreen",
        fontsize=10,
        va="center",
    )


def main():
    v = 1.0
    ks = np.linspace(-3.0 * g, 3.0 * g, 1201)
    eigenvalues = np.empty((ks.size, 3))
    eigenvectors = np.empty((ks.size, 3, 3))
    for i, k in enumerate(ks):
        evals, evecs = np.linalg.eigh(H(k, v))
        eigenvalues[i] = evals
        eigenvectors[i] = evecs

    data_path = Path(__file__).with_name("data") / "1D_analysis.npz"
    data = np.load(data_path)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 9,
    })

    fig = plt.figure(figsize=(6.75, 2.5))
    gs = gridspec.GridSpec(3, 2, figure=fig,
                            left=0.07, right=0.97, top=0.94, bottom=0.18,
                            width_ratios=[1.2, 0.8],
                            hspace=0.05, wspace=0.18)

    ax_cartoon = fig.add_subplot(gs[:, 0])
    draw_cartoon(ax_cartoon, ks, eigenvalues, eigenvectors)

    vs = data["vs"]
    a_val = float(data["a_val"])

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(vs, data["gap_at_k_g2"], color="c", lw=1.5)
    ax1.plot(vs, -data["gap_analytic_v3"], color="black", lw=0.5, ls="--",
             label=r"$2V$")
    ax1.axhline(0.0, color="k", lw=0.5, ls=":", zorder=0)
    ax1.tick_params(labelbottom=False, bottom=False)
    ax1.text(2.8, float(data["gap_at_k_g2"][np.argmin(np.abs(vs - 4.6))]) - 0.6,
             r"$\Delta$", color="black", fontsize=10, va="top")
    ax1.legend(fontsize=8, loc="upper left")

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    shift_4a = 4.0 * a_val
    ax2.plot(vs, data["dist_bot_at_0"] - shift_4a, color="limegreen", lw=1.5)
    ax2.plot(vs, data["analytic_full_2"], color="black", lw=0.5, ls="--",
             label=r"$V^2/A^2$")
    ax2.set_ylim(-0.3, 5.5)
    ax2.axhline(0.0, color="k", lw=0.5, ls=":", zorder=0)
    ax2.tick_params(labelbottom=False, bottom=False)
    ax2.text(2.5, 4.5, r"$\chi-4A$", color="black", fontsize=10,
             ha="center", va="center")
    ax2.legend(fontsize=8, loc="upper left")

    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax3.plot(vs, data["weight_ratio_same_k"], color="orange", lw=1.5)
    ax3.plot(vs, data["analytic_ratio_same"], color="black", lw=0.5, ls="--",
             label=r"$V^2/(64A^2)$")
    ax3.plot(vs, data["weight_ratio_cross"], color="gold", lw=1.5)
    ax3.plot(vs, data["analytic_ratio_cross"], color="darkgray", lw=0.5, ls="--",
             label=r"$V^2/(256A^2)$")
    ax3.set_xlabel("V")
    ax3.tick_params(labelbottom=True)
    ax3.axhline(0.0, color="k", lw=0.5, ls=":", zorder=0)
    ax3.text(2.4, float(data["weight_ratio_same_k"][np.argmin(np.abs(vs - 4.6))]),
             r"$\lambda$", color="black", fontsize=10, va="top")
    ax3.text(2.4, float(data["weight_ratio_cross"][np.argmin(np.abs(vs - 4.6))]),
             r"$\rho$", color="black", fontsize=10, va="bottom")
    ax3.legend(fontsize=8, loc="upper left")

    out = Path(__file__).with_name("figures") / "fig_1D_theory.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
