import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _1D_common import g, alpha, eps, H


def main():
    n_v = 201
    vs = np.linspace(0.0, 3.0, n_v)

    a_val = alpha * g ** 2 / 4.0
    b_val = 4.0 * a_val

    gap_at_k_g2 = np.empty(n_v)
    dist_mid_at_0 = np.empty(n_v)
    dist_bot_at_0 = np.empty(n_v)
    weight_ratio_same_k = np.empty(n_v)
    weight_ratio_cross = np.empty(n_v)

    ks_fine = np.linspace(-3.0 * g, 3.0 * g, 5001)

    for i, v in enumerate(vs):
        e_g2, _ = np.linalg.eigh(H(0.5 * g, v))
        e_0, _ = np.linalg.eigh(H(0.0, v))
        gap_at_k_g2[i] = e_g2[2] - e_g2[1]
        dist_mid_at_0[i] = e_0[2] - e_0[1]
        dist_bot_at_0[i] = e_0[2] - e_0[0]

        e1_at_15, evecs_15 = np.linalg.eigh(H(1.5 * g, v))
        e1 = e1_at_15[1]
        w1 = evecs_15[1, 1] ** 2
        w2_same = evecs_15[1, 2] ** 2
        weight_ratio_same_k[i] = w2_same / w1 if w1 > 0 else np.nan

        e_all = np.array([np.linalg.eigvalsh(H(k, v))[2] for k in ks_fine])
        diffs = e_all - e1
        signs = np.sign(diffs)
        cross = np.where(np.diff(signs))[0]
        cross = cross[ks_fine[cross] > 1.5 * g]
        if len(cross) == 0:
            weight_ratio_cross[i] = np.nan
            continue
        j = int(cross[0])
        k_cross = ks_fine[j] - diffs[j] * (ks_fine[j + 1] - ks_fine[j]) / (diffs[j + 1] - diffs[j])
        _, evecs_cross = np.linalg.eigh(H(k_cross, v))
        w2_cross = evecs_cross[1, 2] ** 2
        weight_ratio_cross[i] = w2_cross / w1 if w1 > 0 else np.nan

    gap_analytic_full = (-2.0 * vs
                         + 3.0 * vs ** 3 / (256.0 * a_val ** 2)
                         - 55.0 * vs ** 5 / (262144.0 * a_val ** 4))
    gap_analytic_v3 = -2.0 * vs + 3.0 * vs ** 3 / (256.0 * a_val ** 2)

    analytic_full_2 = 4.0 * vs ** 2 / b_val - 8.0 * vs ** 4 / b_val ** 3
    analytic_full_1 = 0.5 * analytic_full_2

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 4.0))

    ax1.plot(vs, gap_at_k_g2, color="C0", lw=1.5,
             label="numeric: $\\lambda_2 - \\lambda_1$ at $k=G/2$")
    ax1.plot(vs, -gap_analytic_v3, color="C1", lw=1.0, ls="--",
             label="analytic up to $V^3$: $2V - 3V^3/(256A^2)$")
    ax1.set_xlabel("V")
    ax1.set_ylabel("Gap at $k=G/2$")
    ax1.axhline(0.0, color="k", lw=0.5, ls=":")
    ax1.legend(fontsize=8)

    shift_4a = 4.0 * a_val
    ax2.plot(vs, dist_bot_at_0 - shift_4a, color="C0", lw=1.5,
             label="numeric: $(\\lambda_2 - \\lambda_0) - 4A$ at $k=0$")
    ax2.plot(vs, analytic_full_2, color="C1", lw=1.0, ls="--",
             label="analytic: $4V^2/B - 8V^4/B^3$")
    ax2.set_xlabel("V")
    ax2.set_ylabel("Distance from $4A$ at $k=0$")
    ax2.set_ylim(0.0, 6.5)
    ax2.axhline(0.0, color="k", lw=0.5, ls=":")
    ax2.legend(fontsize=8)

    analytic_ratio_same = vs ** 2 / (64.0 * a_val ** 2)
    analytic_ratio_cross = vs ** 2 / (256.0 * a_val ** 2)
    ax3.plot(vs, weight_ratio_same_k, color="C0", lw=1.5,
             label="numeric same $k=1.5G$")
    ax3.plot(vs, analytic_ratio_same, color="C1", lw=1.0, ls="--",
             label="analytic: $V^2/(64A^2)$")
    ax3.plot(vs, weight_ratio_cross, color="C2", lw=1.5,
             label="numeric cross $k$")
    ax3.plot(vs, analytic_ratio_cross, color="C3", lw=1.0, ls="--",
             label="analytic: $V^2/(256A^2)$")
    ax3.set_xlabel("V")
    ax3.set_ylabel("weight ratio at $k=1.5G$")
    ax3.set_yscale("log")
    ax3.axhline(1.0, color="k", lw=0.5, ls=":")
    ax3.legend(fontsize=7)

    out = Path(__file__).with_name("figures") / "1D_analysis.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"\nSaved figure to {out}")

    data_out = Path(__file__).with_name("data") / "1D_analysis.npz"
    data_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        data_out,
        vs=vs,
        alpha=alpha,
        g=g,
        a_val=a_val,
        b_val=b_val,
        gap_at_k_g2=gap_at_k_g2,
        gap_analytic_v3=gap_analytic_v3,
        gap_analytic_full=gap_analytic_full,
        dist_bot_at_0=dist_bot_at_0,
        analytic_full_2=analytic_full_2,
        weight_ratio_same_k=weight_ratio_same_k,
        weight_ratio_cross=weight_ratio_cross,
        analytic_ratio_same=analytic_ratio_same,
        analytic_ratio_cross=analytic_ratio_cross,
    )
    print(f"Saved data to {data_out}")


if __name__ == "__main__":
    main()
