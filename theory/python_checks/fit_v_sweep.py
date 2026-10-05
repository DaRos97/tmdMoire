"""Fit the V-sweep of the 2D moire analysis.

Loads data/2D_analysis.npz (produced by 2D_analysis.py) and fits the four
observables at constant phi (phi = 180 deg, V in [0, 10] meV):

  - delta:   linear fit             delta(V) = c0 * V + c1
  - chi:     full + centered quadratic
  - lambda:  full + centered quadratic
  - rho:     full + centered quadratic

The full quadratic is    y(V) = a + b * V + c * V^2     (np.polyfit deg 2).
The centered quadratic is  y(V) - y(0) = a_c + b_c * V^2 (no linear term,
                                                            anchored at first data point).

R^2 is reported for every fit. Results are printed to stdout and saved to
data/2D_figure_fits.json next to this script.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmdmoire.bilayer.geometry import MoireGeometryFixed


def _r_squared(y, y_pred):
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def linear_fit(x, y):
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    return {
        "slope": float(coeffs[0]),
        "intercept": float(coeffs[1]),
        "r_squared": _r_squared(y, y_pred),
    }


def quadratic_fit(x, y):
    coeffs = np.polyfit(x, y, 2)
    y_pred = np.polyval(coeffs, x)
    return {
        "c": float(coeffs[0]),
        "b": float(coeffs[1]),
        "a": float(coeffs[2]),
        "r_squared": _r_squared(y, y_pred),
    }


def centered_quadratic_fit(x, y):
    y0 = float(y[0])
    y_shifted = y - y0
    design = np.column_stack([np.ones_like(x), x ** 2])
    a_c, b_c = np.linalg.lstsq(design, y_shifted, rcond=None)[0]
    y_pred = y0 + a_c + b_c * x ** 2
    return {
        "a_offset": float(a_c),
        "b_quadratic": float(b_c),
        "y0": y0,
        "r_squared": _r_squared(y, y_pred),
    }


def best_fraction(x, max_denom=100):
    """Closest rational p/q to x with q <= max_denom."""
    return Fraction(x).limit_denominator(max_denom)


def closest_unit_fraction(x, max_n=1000):
    """Closest 1/n to x, scanning n in [1, max_n]."""
    target_n = max(1, int(round(1.0 / x)))
    lo = max(1, target_n - 5)
    hi = min(max_n, target_n + 5)
    best_n = target_n
    best_diff = abs(1.0 / best_n - x)
    for n in range(lo, hi + 1):
        diff = abs(1.0 / n - x)
        if diff < best_diff:
            best_diff = diff
            best_n = n
    return best_n


def fmt_quadratic_full(name, fit, A_sq=None):
    line = (
        f"Full:     {name}(V) = {fit['c']:+.6e} V^2 + "
        f"{fit['b']:+.6f} V + {fit['a']:+.6f}     "
        f"R^2 = {fit['r_squared']:.6f}"
    )
    if A_sq is not None:
        line += f"   | A^2*c = {A_sq * fit['c']:+.6e}"
    return line


def fmt_quadratic_centered(name, fit, A_sq=None):
    line = (
        f"Centered: {name}(V) - {name}(0) = {fit['a_offset']:+.6e} + "
        f"{fit['b_quadratic']:+.6e} V^2     "
        f"R^2 = {fit['r_squared']:.6f}"
    )
    if A_sq is not None:
        line += f"   | A^2*b = {A_sq * fit['b_quadratic']:+.6e}"
    return line


def fmt_linear(name, fit):
    return (
        f"{name}(V) = {fit['slope']:+.6f} V + {fit['intercept']:+.6f}     "
        f"R^2 = {fit['r_squared']:.6f}"
    )


def main():
    data_path = Path(__file__).with_name("data") / "2D_analysis.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run theory/python_checks/2D_analysis.py "
            f"first to generate the sweep data."
        )

    sw = np.load(data_path)
    Vs = sw["V_V"]
    delta = sw["delta_V"]
    chi = sw["chi_V"]
    lam = sw["lambda_V"]
    rho = sw["rho_V"]

    n_pts = len(Vs)
    theta = float(sw["theta"])
    a_moire = float(sw["a_moire"])

    geo = MoireGeometryFixed(theta, a_moire_override=a_moire)
    g_mag = float(np.linalg.norm(geo.reciprocal_vectors()[1]))
    alpha = (
        (1.054571817e-34) ** 2 / (2.0 * 1.19 * 9.1093837e-31)
        / (1.602176634e-19 * 1e-3) * 1e20
    )
    A = alpha * g_mag ** 2 / 4.0
    A_sq = A * A

    print(f"V-sweep fit: phi = {float(sw['phi_V_sweep']) * 180.0 / np.pi:.1f} deg, "
          f"{n_pts} points in V = [{Vs[0]:.4f}, {Vs[-1]:.4f}] meV")
    print(f"Geometry: theta = {theta} deg, a_moire = {a_moire} A, "
          f"|G_M| = {g_mag:.6f} 1/A")
    print(f"Constants: A = alpha*|G_M|^2/4 = {A:.6f} meV, "
          f"A^2 = {A_sq:.6e} meV^2")
    print()

    results = {
        "V_sweep": {
            "Vs": Vs.tolist(),
            "delta": {"linear": linear_fit(Vs, delta)},
            "chi": {
                "full_quadratic": quadratic_fit(Vs, chi),
                "centered_quadratic": centered_quadratic_fit(Vs, chi),
            },
            "lambda": {
                "full_quadratic": quadratic_fit(Vs, lam),
                "centered_quadratic": centered_quadratic_fit(Vs, lam),
            },
            "rho": {
                "full_quadratic": quadratic_fit(Vs, rho),
                "centered_quadratic": centered_quadratic_fit(Vs, rho),
            },
        },
        "metadata": {
            "theta": theta,
            "a_moire": a_moire,
            "phi_V_sweep": float(sw["phi_V_sweep"]),
            "phi_V_sweep_deg": float(sw["phi_V_sweep"]) * 180.0 / np.pi,
            "v_for_phi_sweep": float(sw["v_for_phi_sweep"]),
            "n_pts": int(n_pts),
            "G_mag": g_mag,
            "alpha": alpha,
            "A": A,
            "A_sq": A_sq,
        },
    }

    for name in ("chi", "lambda", "rho"):
        fq = results["V_sweep"][name]["full_quadratic"]
        cq = results["V_sweep"][name]["centered_quadratic"]
        fq["A_squared_times_c"] = A_sq * fq["c"]
        cq["A_squared_times_b"] = A_sq * cq["b_quadratic"]

    print("=== Delta (linear) ===")
    print(fmt_linear("Delta", results["V_sweep"]["delta"]["linear"]))
    print()

    one_d_analytic = {
        "chi": None,
        "lambda": (1, 64),
        "rho": (1, 256),
    }

    for name, sym in [("chi", "chi"), ("lambda", "lambda"), ("rho", "rho")]:
        section = results["V_sweep"][name]
        print(f"=== {name.capitalize()} (quadratic) ===")
        print(fmt_quadratic_full(sym, section["full_quadratic"], A_sq=A_sq))
        print(fmt_quadratic_centered(sym, section["centered_quadratic"], A_sq=A_sq))
        a_sq_b = section["centered_quadratic"]["A_squared_times_b"]
        frac = best_fraction(a_sq_b, max_denom=200)
        unit_n = closest_unit_fraction(a_sq_b)
        print(f"   => A^2*b as fraction: {a_sq_b:.6f} ~ {frac.numerator}/{frac.denominator}")
        print(f"   => closest 1/n: {a_sq_b:.6f} ~ 1/{unit_n} = {1.0/unit_n:.6f}")
        one_d = one_d_analytic[name]
        if one_d is not None:
            num, denom = one_d
            one_d_val = num / denom
            ratio = a_sq_b / one_d_val
            print(
                f"   1D analytic: A^2*b = {num}/{denom} = {one_d_val:.6f}    "
                f"|  2D/1D ratio = {ratio:.3f}"
            )
        print()

    out_path = Path(__file__).with_name("data") / "2D_figure_fits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved fits to {out_path}")


if __name__ == "__main__":
    main()
