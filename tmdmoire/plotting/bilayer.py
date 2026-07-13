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


def _apply_energy_shading(spread, e_list, shade_factor_e):
    """Apply linear energy-dependent shading.

    Starts at factor 0.1 at E_min and goes to shade_factor_e at E_max.
    """
    shade = np.linspace(0.1, shade_factor_e, len(e_list))
    return spread * shade[np.newaxis, :]


def _label_high_symmetry(ax, k_positions, labels, y_min, y_max):
    """Add vertical dashed lines and labels at high-symmetry points."""
    for k_pos, label in zip(k_positions, labels):
        ax.axvline(k_pos, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.text(k_pos, y_max * 0.95, label, ha="center", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="gray", alpha=0.7))


def plot_moire_bands_simulated(k_kgk, k_kmkp, e_list, spread_kgk, spread_kmkp,
                                shade_factor_e=3.0, save_dir=None):
    """Plot simulated moire band intensity for K'->G->K and K->M->K' paths."""
    spread_kgk = _apply_energy_shading(spread_kgk, e_list, shade_factor_e)
    spread_kmkp = _apply_energy_shading(spread_kmkp, e_list, shade_factor_e)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    ax1 = axes[0]
    ax1.pcolormesh(k_kgk, e_list, spread_kgk.T, cmap="Greys", shading="auto")
    ax1.set_ylabel("Energy (eV)", fontsize=12)
    ax1.set_xlabel("Momentum (A$^{-1}$)", fontsize=12)
    ax1.set_title("K'$\\rightarrow\\Gamma\\rightarrow$K", fontsize=13, fontweight="bold")
    ax1.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_xlim(-1.4, 1.4)

    ax2 = axes[1]
    ax2.pcolormesh(k_kmkp, e_list, spread_kmkp.T, cmap="Greys", shading="auto")
    ax2.set_xlabel("Momentum (A$^{-1}$)", fontsize=12)
    ax2.set_title("K$\\rightarrow$M$\\rightarrow$K'", fontsize=13, fontweight="bold")
    ax2.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax2.set_xlim(-1.2, 1.2)

    fig.suptitle("Moiré bilayer bands — simulated intensity",
                 fontsize=14, fontweight="bold", y=1.02)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "moire_bands_simulated.png"
    fig.savefig(fn, dpi=200, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


def plot_arpes_data(k_kgk, k_kmkp, e_list, intensity_kgk, intensity_kmkp,
                     save_dir=None):
    """Plot experimental ARPES intensity for K'->G->K and K->M->K' paths."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    ax1 = axes[0]
    ax1.pcolormesh(k_kgk, e_list, intensity_kgk.T, cmap="Greys", shading="auto")
    ax1.set_ylabel("Energy (eV)", fontsize=12)
    ax1.set_xlabel("Momentum (A$^{-1}$)", fontsize=12)
    ax1.set_title("K'$\\rightarrow\\Gamma\\rightarrow$K", fontsize=13, fontweight="bold")
    ax1.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_xlim(-1.4, 1.4)

    ax2 = axes[1]
    ax2.pcolormesh(k_kmkp, e_list, intensity_kmkp.T, cmap="Greys", shading="auto")
    ax2.set_xlabel("Momentum (A$^{-1}$)", fontsize=12)
    ax2.set_title("K$\\rightarrow$M$\\rightarrow$K'", fontsize=13, fontweight="bold")
    ax2.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax2.set_xlim(-1.2, 1.2)

    fig.suptitle("Experimental ARPES intensity",
                 fontsize=14, fontweight="bold", y=1.02)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "arpes_data.png"
    fig.savefig(fn, dpi=200, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


def plot_moire_bands_half_arpes(k_kgk, k_kmkp, e_list, spread_kgk, spread_kmkp,
                                 arpes_kgk, arpes_kmkp,
                                 shade_factor_e=3.0, save_dir=None):
    """Plot half-ARPES / half-simulated moire band comparison.

    Left side (k < 0) is ARPES, right side (k > 0) is simulated, for both subplots.
    """
    spread_kgk = _apply_energy_shading(spread_kgk, e_list, shade_factor_e)
    spread_kmkp = _apply_energy_shading(spread_kmkp, e_list, shade_factor_e)

    zero_kgk = np.argmin(np.abs(k_kgk))
    zero_kmkp = np.argmin(np.abs(k_kmkp))

    k_kgk_left = k_kgk[:zero_kgk + 1]
    k_kgk_right = k_kgk[zero_kgk:]
    arpes_kgk_left = arpes_kgk[:zero_kgk + 1]
    spread_kgk_right = spread_kgk[zero_kgk:]

    k_kmkp_left = k_kmkp[:zero_kmkp + 1]
    k_kmkp_right = k_kmkp[zero_kmkp:]
    arpes_kmkp_left = arpes_kmkp[:zero_kmkp + 1]
    spread_kmkp_right = spread_kmkp[zero_kmkp:]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    ax1 = axes[0]
    arpes_left = arpes_kgk_left.copy()
    sim_right = spread_kgk_right.copy()
    arpes_left = arpes_left / np.max(arpes_left) if np.max(arpes_left) > 0 else arpes_left
    sim_right = sim_right / np.max(sim_right) if np.max(sim_right) > 0 else sim_right
    ax1.pcolormesh(k_kgk_left, e_list, arpes_left.T, cmap="Greys",
                   shading="auto", alpha=0.7)
    ax1.pcolormesh(k_kgk_right, e_list, sim_right.T, cmap="Greys",
                   shading="auto")
    ax1.set_ylabel("Energy (eV)", fontsize=12)
    ax1.set_xlabel("Momentum (A$^{-1}$)", fontsize=12)
    ax1.set_title("K'$\\rightarrow\\Gamma\\rightarrow$K", fontsize=13, fontweight="bold")
    ax1.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_xlim(-1.4, 1.4)

    ax2 = axes[1]
    arpes_km_left = arpes_kmkp_left.copy()
    sim_km_right = spread_kmkp_right.copy()
    arpes_km_left = arpes_km_left / np.max(arpes_km_left) if np.max(arpes_km_left) > 0 else arpes_km_left
    sim_km_right = sim_km_right / np.max(sim_km_right) if np.max(sim_km_right) > 0 else sim_km_right
    ax2.pcolormesh(k_kmkp_left, e_list, arpes_km_left.T, cmap="Greys",
                   shading="auto", alpha=0.7)
    ax2.pcolormesh(k_kmkp_right, e_list, sim_km_right.T, cmap="Greys",
                   shading="auto")
    ax2.set_xlabel("Momentum (A$^{-1}$)", fontsize=12)
    ax2.set_title("K$\\rightarrow$M$\\rightarrow$K'", fontsize=13, fontweight="bold")
    ax2.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax2.set_xlim(-1.2, 1.2)

    fig.suptitle("Moiré bilayer bands — half ARPES / half simulated",
                 fontsize=14, fontweight="bold", y=1.02)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "moire_bands_half_arpes.png"
    fig.savefig(fn, dpi=200, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


def plot_moire_bands_simulated_with_arpes(k_kgk, k_kmkp, e_list, spread_kgk, spread_kmkp,
                                           bilayer_data, shade_factor_e=3.0, save_dir=None):
    """Plot simulated moire bands with ARPES band overlay on K'->G->K left half.

    Left subplot (K'->Gamma->K): left side (k < 0) shows simulated intensity
    with ARPES bands overlaid as thin red lines; right side (k > 0) is
    simulated-only. Right subplot (K->M->K') is simulated-only.
    """
    spread_kgk = _apply_energy_shading(spread_kgk, e_list, shade_factor_e)
    spread_kmkp = _apply_energy_shading(spread_kmkp, e_list, shade_factor_e)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    ax1 = axes[0]
    ax1.pcolormesh(k_kgk, e_list, spread_kgk.T, cmap="Greys", shading="auto")
    for ib in range(bilayer_data.n_bands):
        rd = bilayer_data.raw_data[ib]
        left_mask = rd[:, 0] < 0
        rd_left = rd[left_mask]
        valid = ~np.isnan(rd_left[:, 1])
        ax1.plot(rd_left[valid, 0], rd_left[valid, 1],
                 color="red", lw=1.0, alpha=0.8, zorder=3)

    ax1.set_ylabel("Energy (eV)", fontsize=12)
    ax1.set_xlabel("Momentum (A$^{-1}$)", fontsize=12)
    ax1.set_title("K'$\\rightarrow\\Gamma\\rightarrow$K", fontsize=13, fontweight="bold")
    ax1.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_xlim(-1.4, 1.4)

    ax2 = axes[1]
    ax2.pcolormesh(k_kmkp, e_list, spread_kmkp.T, cmap="Greys", shading="auto")
    ax2.set_xlabel("Momentum (A$^{-1}$)", fontsize=12)
    ax2.set_title("K$\\rightarrow$M$\\rightarrow$K'", fontsize=13, fontweight="bold")
    ax2.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax2.set_xlim(-1.2, 1.2)

    fig.suptitle("Moiré bilayer bands — simulated + ARPES overlay",
                 fontsize=14, fontweight="bold", y=1.02)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "moire_bands_simulated_with_arpes.png"
    fig.savefig(fn, dpi=200, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


def plot_diag_half_bands(k_kgk_comp, k_kmkp_comp, evals_kgk, evals_kmkp,
                          k_kgk_arpes, k_kmkp_arpes, e_arpes,
                          arpes_kgk, arpes_kmkp, save_dir=None):
    """Half ARPES / half computed bands split for both BZ cuts.

    Left side (k < 0) shows ARPES intensity; right side (k >= 0)
    shows all computed moire bands as thin red lines on a white background.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    ax1 = axes[0]
    zero_arpes = np.argmin(np.abs(k_kgk_arpes))
    ax1.pcolormesh(k_kgk_arpes[:zero_arpes + 1], e_arpes,
                    arpes_kgk[:zero_arpes + 1].T,
                    cmap="Greys", shading="auto")
    mask_right = k_kgk_comp >= 0
    for ib in range(evals_kgk.shape[1]):
        ax1.plot(k_kgk_comp[mask_right], evals_kgk[mask_right, ib],
                  color="red", lw=0.5, alpha=0.7, zorder=5)
    ax1.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_ylabel("Energy (eV)", fontsize=12)
    ax1.set_xlabel("Momentum (Å$^{-1}$)", fontsize=12)
    ax1.set_title("K'$\\rightarrow\\Gamma\\rightarrow$K", fontsize=13, fontweight="bold")
    ax1.set_xlim(-1.4, 1.4)

    ax2 = axes[1]
    zero_arpes_km = np.argmin(np.abs(k_kmkp_arpes))
    ax2.pcolormesh(k_kmkp_arpes[:zero_arpes_km + 1], e_arpes,
                    arpes_kmkp[:zero_arpes_km + 1].T,
                    cmap="Greys", shading="auto")
    mask_right_km = k_kmkp_comp >= 0
    for ib in range(evals_kmkp.shape[1]):
        ax2.plot(k_kmkp_comp[mask_right_km], evals_kmkp[mask_right_km, ib],
                  color="red", lw=0.5, alpha=0.7, zorder=5)
    ax2.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax2.set_xlabel("Momentum (Å$^{-1}$)", fontsize=12)
    ax2.set_title("K$\\rightarrow$M$\\rightarrow$K'", fontsize=13, fontweight="bold")
    ax2.set_xlim(-1.2, 1.2)

    fig.suptitle("Moiré bilayer bands — half ARPES / half computed bands",
                  fontsize=14, fontweight="bold", y=1.02)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "diag_half_bands.png"
    fig.savefig(fn, dpi=200, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


def plot_diag_bands_over_arpes(k_kgk_comp, k_kmkp_comp, evals_kgk, evals_kmkp,
                                k_kgk_arpes, k_kmkp_arpes, e_arpes,
                                arpes_kgk, arpes_kmkp, save_dir=None):
    """ARPES intensity background with all computed moire bands overlaid.

    Full ARPES pcolormesh at full opacity with all computed bands
    superimposed as thin red lines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    ax1 = axes[0]
    ax1.pcolormesh(k_kgk_arpes, e_arpes, arpes_kgk.T,
                    cmap="Greys", shading="auto")
    for ib in range(evals_kgk.shape[1]):
        ax1.plot(k_kgk_comp, evals_kgk[:, ib],
                  color="red", lw=0.5, alpha=0.7, zorder=5)
    ax1.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_ylabel("Energy (eV)", fontsize=12)
    ax1.set_xlabel("Momentum (Å$^{-1}$)", fontsize=12)
    ax1.set_title("K'$\\rightarrow\\Gamma\\rightarrow$K", fontsize=13, fontweight="bold")
    ax1.set_xlim(-1.4, 1.4)

    ax2 = axes[1]
    ax2.pcolormesh(k_kmkp_arpes, e_arpes, arpes_kmkp.T,
                    cmap="Greys", shading="auto")
    for ib in range(evals_kmkp.shape[1]):
        ax2.plot(k_kmkp_comp, evals_kmkp[:, ib],
                  color="red", lw=0.5, alpha=0.7, zorder=5)
    ax2.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax2.set_xlabel("Momentum (Å$^{-1}$)", fontsize=12)
    ax2.set_title("K$\\rightarrow$M$\\rightarrow$K'", fontsize=13, fontweight="bold")
    ax2.set_xlim(-1.2, 1.2)

    fig.suptitle("Moiré bilayer bands — ARPES with computed bands overlay",
                  fontsize=14, fontweight="bold", y=1.02)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[2] / "Figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = save_dir / "diag_bands_over_arpes.png"
    fig.savefig(fn, dpi=200, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)
