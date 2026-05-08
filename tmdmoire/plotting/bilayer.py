"""Bilayer plotting: data pipeline and fit comparison."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_bilayer_data(bilayer_data, save_dir=None):
    """Plot bilayer ARPES raw data, symmetrized data, and interpolated fitting grid."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True, constrained_layout=True)
    ax_raw, ax_sym, ax_interp = axes

    colors_raw = ["steelblue", "darkorange", "forestgreen"]
    colors_sym = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    colors_interp = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    band_labels = [r"Band 1", r"Band 2", r"Band 3"]

    for ib in range(bilayer_data.n_bands):
        rd = bilayer_data.raw_data[ib]
        valid = ~np.isnan(rd[:, 1])
        ax_raw.scatter(rd[valid, 0], rd[valid, 1], s=6, c=colors_raw[ib],
                       alpha=0.6, label=band_labels[ib])

    ax_raw.axhline(0, color="gray", lw=0.5, ls="--")
    ax_raw.axvline(0, color="k", lw=0.8, ls=":")
    ax_raw.set_xlabel("Momentum (Å⁻¹)", fontsize=12)
    ax_raw.set_ylabel("Energy (eV)", fontsize=12)
    ax_raw.set_title("Raw bilayer ARPES data", fontsize=13, fontweight="bold")
    ax_raw.legend(fontsize=10)
    ax_raw.text(0, ax_raw.get_ylim()[1] * 0.95, r"$\Gamma$",
                ha="center", va="top", fontsize=10, fontweight="bold")

    for ib in range(bilayer_data.n_bands):
        sd = bilayer_data.sym_data[ib]
        valid = ~np.isnan(sd[:, 1])
        ax_sym.scatter(sd[valid, 0], sd[valid, 1], s=8, c=colors_sym[ib],
                       alpha=0.7, label=band_labels[ib])

    ax_sym.axhline(0, color="gray", lw=0.5, ls="--")
    ax_sym.set_xlabel(r"$|k|$ (Å⁻¹)", fontsize=12)
    ax_sym.set_title("Symmetrized (K'↔K averaged)", fontsize=13, fontweight="bold")
    ax_sym.legend(fontsize=10)

    fd = bilayer_data.fit_data
    k_grid = fd[:, 0]
    for ib in range(bilayer_data.n_bands):
        energies = fd[:, ib + 1]
        valid = ~np.isnan(energies)
        ax_interp.scatter(k_grid[valid], energies[valid], s=10,
                          c=colors_interp[ib], alpha=0.8, zorder=3,
                          label=band_labels[ib])

    ax_interp.axhline(0, color="gray", lw=0.5, ls="--")
    ax_interp.set_xlabel(r"$|k|$ (Å⁻¹)", fontsize=12)
    ax_interp.set_title("Interpolated (fitting grid)", fontsize=13, fontweight="bold")
    ax_interp.legend(fontsize=10)

    fig.suptitle("WSe2/WS2 bilayer ARPES data — K'→Γ→K path",
                 fontsize=14, fontweight="bold", y=1.05)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "bilayer_data.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


def plot_bilayer_fit(bilayer_data, k_list, evals, evals_no_coupling=None, save_dir=None):
    """Plot fitted bilayer bands against experimental ARPES data."""
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    comp_k = np.linalg.norm(k_list, axis=1)
    n_bands = bilayer_data.n_bands
    band_indices = [27, 25, 23]
    computed = evals[:, band_indices]

    colors_exp = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    colors_fit = ["#aec7e8", "#ffbb78", "#98df8a"]
    band_labels = [r"Band 1", r"Band 2", r"Band 3"]

    exp_data = bilayer_data.fit_data
    exp_k = exp_data[:, 0]
    for ib in range(n_bands):
        exp_e = exp_data[:, ib + 1]
        valid = ~np.isnan(exp_e)
        ax.scatter(exp_k[valid], exp_e[valid], s=20, c=colors_exp[ib],
                   alpha=0.7, zorder=3, label=f"{band_labels[ib]} (ARPES)")
        ax.plot(comp_k, computed[:, ib], color=colors_fit[ib],
                lw=2.5, zorder=2, label=f"{band_labels[ib]} (fit)")

    if evals_no_coupling is not None:
        no_coup = evals_no_coupling[:, band_indices]
        for ib in range(n_bands):
            ax.plot(comp_k, no_coup[:, ib], color="gray",
                    lw=1.5, ls="--", zorder=1,
                    label=f"{band_labels[ib]} (no coupling)" if ib == 0 else "")

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(r"$|k|$ (Å⁻¹)", fontsize=12)
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_title("Bilayer interlayer coupling fit", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "bilayer_fit.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)
