import numpy as np
from pathlib import Path


hbar = 6.582119569e-13
m0 = 9.1093837e-31
g = 0.1257
m = 3.5 * m0
hbar_si = 1.054571817e-34
eV = 1.602176634e-19

alpha = (hbar_si ** 2 / (2.0 * m)) / (eV * 1e-3) * 1e20


def eps(k, n):
    return -alpha * (k + n * g) ** 2


def H(k, v):
    return np.array([
        [eps(k, -1), v, 0.0],
        [v,         eps(k, 0), v],
        [0.0,       v,         eps(k, 1)],
    ])


def main():
    v = 2.0
    ks = np.linspace(-3.0 * g, 3.0 * g, 1201)

    eigenvalues = np.empty((ks.size, 3))
    eigenvectors = np.empty((ks.size, 3, 3))
    for i, k in enumerate(ks):
        evals, evecs = np.linalg.eigh(H(k, v))
        eigenvalues[i] = evals
        eigenvectors[i] = evecs

    print(f"alpha = {alpha:.6f} meV*A^2")
    print(f"V = {v} meV")
    print(f"G = {g} A^-1")
    print(f"k range = [{ks[0]:+.5f}, {ks[-1]:+.5f}] A^-1  ({ks.size} points)")
    print()

    idx0 = ks.size // 2
    print(f"H(k={ks[idx0]:+.5f}, V={v}) =")
    print(np.array2string(H(ks[idx0], v), precision=4, suppress_small=True))
    print()
    print(f"Eigenvalues at k={ks[idx0]:+.5f} (meV):")
    print(eigenvalues[idx0])
    print(f"Eigenvectors at k={ks[idx0]:+.5f} (columns):")
    print(np.array2string(eigenvectors[idx0], precision=4, suppress_small=True))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available; skipping plot.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    central_weight = eigenvectors[:, 1, :] ** 2
    max_size = 80.0
    v0_bands = np.stack([eps(ks, n) for n in (-1, 0, 1)], axis=1)
    for n in range(3):
        ax.plot(ks / g, v0_bands[:, n], color="red", lw=0.2, zorder=4)
    for band in range(3):
        ax.scatter(
            ks / g,
            eigenvalues[:, band],
            s=central_weight[:, band] * max_size,
            c="C0",
            linewidths=0,
            zorder=3,
        )
    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("k (G)")
    ax.set_ylabel("Energy")
    ax.set_yticks([])
    ax.set_xlim(ks[0] / g, ks[-1] / g)
    ax.set_ylim(-50, 5)

    axins = fig.add_axes([0.66, 0.17, 0.28, 0.25])
    for n in range(3):
        axins.plot(ks / g, v0_bands[:, n], color="red", lw=0.2, zorder=4)
    for band in range(3):
        axins.scatter(
            ks / g,
            eigenvalues[:, band],
            s=central_weight[:, band] * max_size * 5.0,
            c="C0",
            linewidths=0,
            zorder=3,
        )
    axins.set_xlim(-0.25, 0.25)
    axins.set_ylim(-20, -15)
    axins.set_xticks([])
    axins.set_yticks([])

    k_gap = -0.5 * g
    i_gap = int(np.argmin(np.abs(ks - k_gap)))
    e_gap = np.linalg.eigvalsh(H(ks[i_gap], v))
    e_top = e_gap[2]
    e_mid = e_gap[1]
    ax.annotate(
        "",
        xy=(k_gap / g, e_top),
        xytext=(k_gap / g, e_mid),
        arrowprops=dict(arrowstyle="<->", color="red", lw=0.7, mutation_scale=8,
                        shrinkA=0, shrinkB=0),
        zorder=5,
    )
    ax.text(
        k_gap / g + 0.08,
        0.5 * (e_top + e_mid),
        r"$\Delta$",
        color="red",
        fontsize=12,
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

    ax.scatter(-1.5, y_src, s=80, c="orange", edgecolors="black",
               linewidths=0.5, zorder=6)
    ax.scatter(k_cross / g, y_src, s=80, c="orange", edgecolors="black",
               linewidths=0.5, zorder=6)
    ax.scatter(-1.5, y_tgt, s=80, c="orange", edgecolors="black",
               linewidths=0.5, zorder=6)
    ax.plot([-1.5, k_cross / g], [y_src, y_src], color="orange",
            lw=0.8, ls="--", zorder=5)
    ax.plot([-1.5, -1.5], [y_src, y_tgt], color="orange",
            lw=0.8, ls="--", zorder=5)
    x_mid_h = 0.5 * (-1.5 + k_cross / g)
    ax.text(x_mid_h, y_src + 1.5, r"$\rho$", color="orange",
            fontsize=12, ha="center")
    ax.text(-1.5 - 0.12, 0.5 * (y_src + y_tgt), r"$\lambda$", color="orange",
            fontsize=12, va="center", ha="right")

    e_0 = np.linalg.eigvalsh(H(ks[idx0], v))
    e_top_0 = e_0[2]
    e_bot_0 = e_0[0]
    ax.annotate(
        "",
        xy=(0.0, e_top_0),
        xytext=(0.0, e_bot_0),
        arrowprops=dict(arrowstyle="<->", color="lime", lw=0.7, mutation_scale=12,
                        shrinkA=0, shrinkB=0),
        zorder=5,
    )
    ax.text(
        0.08,
        0.5 * (e_top_0 + e_bot_0),
        r"$\chi$",
        color="lime",
        fontsize=12,
        va="center",
    )
    out = Path(__file__).with_name("figures") / "1D_bands_V1meV.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.13)
    fig.savefig(out)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
