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


def plot_bilayer_diagnostic(bilayer_data, k_list, evals_zero_coupling, save_dir=None):
    """Plot ARPES bilayer bands vs computed bands with zero interlayer coupling."""
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    comp_k = np.linalg.norm(k_list, axis=1)
    comp_band_indices = [27, 26, 25, 24]

    colors_exp = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    colors_comp = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
    band_labels = [r"Band 1", r"Band 2", r"Band 3", r"Band 4"]

    exp_data = bilayer_data.fit_data
    exp_k = exp_data[:, 0]
    n_exp_bands = bilayer_data.n_bands

    y_min = np.inf
    y_max = -np.inf
    for ib in range(n_exp_bands):
        exp_e = exp_data[:, ib + 1]
        valid = ~np.isnan(exp_e)
        if valid.any():
            y_min = min(y_min, exp_e[valid].min())
            y_max = max(y_max, exp_e[valid].max())
        ax.scatter(exp_k[valid], exp_e[valid], s=20, c=colors_exp[ib],
                   alpha=0.7, zorder=3, label=f"{band_labels[ib]} (ARPES)")
        ax.plot(comp_k, evals_zero_coupling[:, comp_band_indices[ib]],
                color=colors_comp[ib], lw=2.5, zorder=2,
                label=f"{band_labels[ib]} (TB, no coupling)")

    ax.plot(comp_k, evals_zero_coupling[:, comp_band_indices[3]],
            color=colors_comp[3], lw=2.5, zorder=2,
            label=f"{band_labels[3]} (TB, no coupling)")

    for ib in range(n_exp_bands):
        exp_e = exp_data[:, ib + 1]
        valid = ~np.isnan(exp_e)
        if valid.any():
            y_min = min(y_min, exp_e[valid].min())
            y_max = max(y_max, exp_e[valid].max())

    padding = (y_max - y_min) * 0.1 if y_min != np.inf else 0.5
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(r"$|k|$ (Å⁻¹)", fontsize=12)
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_title("Bilayer bands — ARPES vs TB (no interlayer coupling)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "bilayer_diagnostic.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


def plot_bilayer_supercell(bilayer_data, k_list, evals, weights,
                           evals_no_coupling=None,
                           evals_nc0=None, edc_peaks=None, edc_e_grid=None, edc_intensities=None, save_dir=None):
    """Plot supercell bands (n_shells > 0) with weight-proportional dots.

    n_shells bands are shown as faint gray lines with dot size proportional
    to central-cell weight. n_shells=0 bands are shown as dashed lines.

    Parameters
    ----------
    bilayer_data : BilayerData
        Experimental ARPES data.
    k_list : np.ndarray
        k-points used for computation.
    evals : np.ndarray
        (n_kpts, n_total_bands) eigenvalues with energy offset applied.
    weights : np.ndarray
        (n_kpts, n_total_bands) central-cell weights.
    evals_no_coupling : np.ndarray, optional
        Eigenvalues with zero interlayer coupling (same n_shells).
    evals_nc0 : np.ndarray, optional
        Full eigenvalues from n_shells=0, no coupling reference.
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    comp_k = np.linalg.norm(k_list, axis=1)
    n_total = evals.shape[1]
    n_cells = n_total // 44

    # Energy window: top ~3*n_cells valence bands
    band_lo = 24 * n_cells
    band_hi = 28 * n_cells

    # Plot n_shells=0 reference bands as dashed red lines
    if evals_nc0 is not None:
        nc0_lo = 24
        nc0_hi = 28
        for ib in range(nc0_lo, nc0_hi):
            ax.plot(comp_k, evals_nc0[:, ib], color="red", lw=1.0, ls="--",
                    alpha=0.5, zorder=1)

    # Plot n_shells bands in the ARPES window as faint gray lines
    for ib in range(band_lo, band_hi):
        ax.plot(comp_k, evals[:, ib], color="lightgray", lw=0.5, alpha=0.3, zorder=1)

    # Plot weight-proportional dots - all same color
    w_max = weights[:, band_lo:band_hi].max()
    if w_max > 0:
        w_norm = weights[:, band_lo:band_hi] / w_max
        dot_sizes = 80 * w_norm  # size 0-80, zero weight = invisible

    exp_data = bilayer_data.fit_data
    exp_k = exp_data[:, 0]
    n_exp_bands = bilayer_data.n_bands

    y_min = np.inf
    y_max = -np.inf
    for ib in range(n_exp_bands):
        exp_e = exp_data[:, ib + 1]
        valid = ~np.isnan(exp_e)
        if valid.any():
            y_min = min(y_min, exp_e[valid].min())
            y_max = max(y_max, exp_e[valid].max())

    # Plot dots for each band, all same color, size proportional to weight
    # Only plot where weight > 0 to avoid invisible dots cluttering
    bilayer_color = "#1f77b4"
    for ib in range(band_lo, band_hi):
        w_col = dot_sizes[:, ib - band_lo]
        mask = w_col > 0
        if mask.any():
            ax.scatter(comp_k[mask], evals[mask, ib], s=w_col[mask],
                       c=bilayer_color, alpha=0.6, zorder=2,
                       edgecolors='none', linewidths=0)

    # Overlay EDC peak positions in green
    if edc_peaks is not None:
        sel_color = "green"
        for ik, pk in enumerate(edc_peaks):
            if len(pk) > 0:
                ax.scatter([comp_k[ik]] * len(pk), pk, s=30, c=sel_color,
                           alpha=0.8, zorder=3, edgecolors='none', linewidths=0,
                           label="EDC peaks" if ik == 0 else "")

    # Plot ARPES data on top - all same color, same marker
    arpes_color = "#ff7f0e"
    for ib in range(n_exp_bands):
        exp_e = exp_data[:, ib + 1]
        valid = ~np.isnan(exp_e)
        ax.scatter(exp_k[valid], exp_e[valid], s=25, c=arpes_color,
                   alpha=0.9, zorder=4, edgecolors='white', linewidths=0.5,
                   marker='s', label=f"ARPES Band {ib + 1}")

    if y_min != np.inf:
        padding = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(r"$|k|$ (Å⁻¹)", fontsize=12)
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_title(f"Supercell bands (n_cells={n_cells}) — dot size = central-cell weight",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "bilayer_supercell.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


def plot_bilayer_fit(bilayer_data, k_list, evals, evals_no_coupling=None,
                     interlayer_params=None, save_dir=None):
    """Plot fitted bilayer bands against experimental ARPES data.

    Parameters
    ----------
    interlayer_params : dict, optional
        Dict of fitted parameter names and values, e.g. {"w1p": 1.2, ...}.
        If provided, displayed as a text box on the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    comp_k = np.linalg.norm(k_list, axis=1)
    n_bands = bilayer_data.n_bands
    band_indices = [27, 26, 25]
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
                   alpha=0.5, zorder=3, label=f"{band_labels[ib]} (ARPES)")
        ax.plot(comp_k, computed[:, ib], color=colors_fit[ib],
                lw=2.5, zorder=2, alpha=0.6,
                label=f"{band_labels[ib]} (fit)")

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

    if interlayer_params is not None:
        text_str = "\n".join(
            f"{k} = {v:+.4f} eV" for k, v in interlayer_params.items()
        )
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "bilayer_fit.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)
