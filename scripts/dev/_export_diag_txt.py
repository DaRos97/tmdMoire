"""Generate txt export files and intensity metadata for all existing diag folders.

- n_shells=0: bands_KpGK.txt, bands_KpMK.txt (9 columns: k + top 8 valence bands)
- All intensity dirs: intensity_KpGK.txt, intensity_KpMK.txt, intensity_meta.json

Usage:
    source ../PyEnv/bin/activate
    python scripts/dev/_export_diag_txt.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

DATA_ROOT = Path("Data") / "plot_bilayer_moire"
BAND_HEADER = "k_Angstrom\tBand27_eV\tBand26_eV\tBand25_eV\tBand24_eV\tBand23_eV\tBand22_eV\tBand21_eV\tBand20_eV"


def main():
    for diag_dir in sorted(DATA_ROOT.glob("diag_*")):
        diag_file = diag_dir / "diag.npz"
        if not diag_file.exists():
            continue

        n_shells = int(diag_dir.name.split("_")[2][1:])  # n0, n1, n2

        if n_shells == 0:
            bands_kgk_fn = diag_dir / "bands_KpGK.txt"
            bands_kmkp_fn = diag_dir / "bands_KpMK.txt"
            if not bands_kgk_fn.exists() or not bands_kmkp_fn.exists():
                data = np.load(diag_file, allow_pickle=True)
                evals_kgk = data["evals_kgk"]
                evals_kmkp = data["evals_kmkp"]
                norm_kgk = data["norm_kgk"]
                norm_kmkp = data["norm_kmkp"]
                band_cols = list(range(-1, -9, -1))

                out_kgk = np.column_stack([norm_kgk] + [evals_kgk[:, i] for i in band_cols])
                np.savetxt(bands_kgk_fn, out_kgk, fmt="%.8f", delimiter="\t",
                           header=BAND_HEADER, comments="")
                print(f"  {bands_kgk_fn}")

                out_kmkp = np.column_stack([norm_kmkp] + [evals_kmkp[:, i] for i in band_cols])
                np.savetxt(bands_kmkp_fn, out_kmkp, fmt="%.8f", delimiter="\t",
                           header=BAND_HEADER, comments="")
                print(f"  {bands_kmkp_fn}")

        for int_dir in sorted(diag_dir.glob("intensity_*")):
            spread_file = int_dir / "spread.npz"
            if not spread_file.exists():
                continue

            int_kgk_fn = int_dir / "intensity_KpGK.txt"
            int_kmkp_fn = int_dir / "intensity_KpMK.txt"
            meta_fn = int_dir / "intensity_meta.json"

            needs_kgk = not int_kgk_fn.exists()
            needs_kmkp = not int_kmkp_fn.exists()
            needs_meta = not meta_fn.exists()
            needs_any = needs_kgk or needs_kmkp or needs_meta

            s = None
            if needs_kgk or needs_kmkp:
                s = np.load(spread_file, allow_pickle=True)
            diag_data = None
            if needs_meta:
                diag_data = np.load(diag_file, allow_pickle=True)

            if needs_kgk:
                np.savetxt(int_kgk_fn, s["spread_kgk"], fmt="%.8f", delimiter="\t")
                print(f"  {int_kgk_fn}")
            if needs_kmkp:
                np.savetxt(int_kmkp_fn, s["spread_kmkp"], fmt="%.8f", delimiter="\t")
                print(f"  {int_kmkp_fn}")
            if needs_meta:
                norm_kgk = diag_data["norm_kgk"]
                norm_kmkp = diag_data["norm_kmkp"]
                if s is None:
                    s = np.load(spread_file, allow_pickle=True)
                e_list = _build_e_list(s["spread_kgk"].shape[1])
                with open(meta_fn, "w") as f:
                    json.dump({
                        "k_KpGK": norm_kgk.tolist(),
                        "k_KpMK": norm_kmkp.tolist(),
                        "e_list": e_list.tolist(),
                        "n_e": len(e_list),
                    }, f, indent=2)
                print(f"  {meta_fn}")

    print("Done")


def _build_e_list(n_e):
    """Reconstruct energy grid from number of points using default params."""
    return np.linspace(-3.5, 0.0, n_e)


if __name__ == "__main__":
    main()
