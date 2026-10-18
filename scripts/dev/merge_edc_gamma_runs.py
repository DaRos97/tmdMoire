"""Merge two disjoint Gamma EDC sweeps into a new, analysis-ready run.

The source run directories are read-only. The merged data and metadata are
written to a new Data/edc_gamma_<output-id>/ directory, which can be passed
to scripts/analyze_edc_gamma.py.

Example:
    python scripts/dev/merge_edc_gamma_runs.py \
        --base-id 4Dsm8a \
        --extension-id 4Dsm8a_phi155_164 \
        --output-id 4Dsm8a_extended
"""
import argparse
import copy
import datetime
import json
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
COORDINATES = ("Vg", "phiG", "w1p", "w1d")
FIXED_PARAMS = ("Vk_ev", "phiK_deg", "w2p_ev", "w2d_ev")


def _read_run(run_id):
    run_dir = DATA_DIR / f"edc_gamma_{run_id}"
    combined_path = run_dir / "combined.h5"
    metadata_path = run_dir / "metadata.json"
    if not combined_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Expected both combined.h5 and metadata.json in {run_dir}. "
            "Combine the extension chunks first."
        )

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    with h5py.File(combined_path, "r") as file:
        columns = list(file.keys())
        data = {name: file[name][:] for name in columns}

    lengths = {len(values) for values in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"Unequal dataset lengths in {combined_path}")
    if not all(name in data for name in COORDINATES):
        raise ValueError(f"Missing grid-coordinate columns in {combined_path}")

    n_rows = next(iter(lengths), 0)
    if metadata.get("total_points") is not None and metadata["total_points"] != n_rows:
        raise ValueError(
            f"{combined_path} has {n_rows} rows, but metadata declares "
            f"{metadata['total_points']} points"
        )
    return run_dir, metadata, data


def _axes_and_check_complete(data, run_id):
    axes = {name: np.unique(data[name]) for name in COORDINATES}
    n_rows = len(data["Vg"])
    expected_rows = int(np.prod([len(axes[name]) for name in COORDINATES]))
    if n_rows != expected_rows:
        raise ValueError(
            f"Run {run_id} is not a complete unique grid: "
            f"{n_rows} rows, expected {expected_rows} from its axes"
        )

    order = np.lexsort(tuple(data[name] for name in reversed(COORDINATES)))
    duplicate = np.ones(max(n_rows - 1, 0), dtype=bool)
    for name in COORDINATES:
        values = data[name][order]
        duplicate &= np.isclose(values[1:], values[:-1], rtol=0, atol=1e-12)
    if np.any(duplicate):
        raise ValueError(f"Run {run_id} contains duplicate grid points")
    return axes


def _same_axis(left, right):
    return left.shape == right.shape and np.allclose(left, right, rtol=0, atol=1e-12)


def _validate_run_axes(metadata, axes, run_id):
    grid_config = metadata.get("grid_config", {})
    specifications = {
        "Vg": (grid_config.get("moire", {}).get("Vg"), "min_ev", "max_ev", "step_ev"),
        "phiG": (grid_config.get("moire", {}).get("phiG"), "min_deg", "max_deg", "step_deg"),
        "w1p": (grid_config.get("interlayer", {}).get("w1p"), "min_ev", "max_ev", "step_ev"),
        "w1d": (grid_config.get("interlayer", {}).get("w1d"), "min_ev", "max_ev", "step_ev"),
    }
    for name, (spec, min_key, max_key, step_key) in specifications.items():
        if not isinstance(spec, dict) or not all(
            key in spec for key in (min_key, max_key, step_key)
        ):
            raise ValueError(f"Run {run_id} metadata has no absolute grid for {name}")
        expected = np.arange(
            spec[min_key], spec[max_key] + spec[step_key] * 0.5, spec[step_key]
        )
        if not _same_axis(axes[name], expected):
            raise ValueError(f"Run {run_id} data axis {name} disagrees with its metadata")


def _validate_fixed_params(base_metadata, extension_metadata):
    base = base_metadata.get("fixed_params", {})
    extension = extension_metadata.get("fixed_params", {})
    for name in FIXED_PARAMS:
        if name not in base or name not in extension:
            raise ValueError(f"Missing fixed parameter {name} in run metadata")
        if not np.isclose(base[name], extension[name], rtol=0, atol=1e-12):
            raise ValueError(
                f"Runs have different fixed {name}: {base[name]} vs {extension[name]}"
            )


def _merged_metadata(base_metadata, extension_metadata, output_id, axes, n_rows,
                     base_id, extension_id):
    metadata = copy.deepcopy(base_metadata)
    phi_values = axes["phiG"]
    phi_steps = np.diff(phi_values)
    if len(phi_steps) == 0 or not np.allclose(phi_steps, phi_steps[0], rtol=0, atol=1e-12):
        raise ValueError("The merged phiG axis must be regularly spaced")

    metadata.update({
        "run_id": output_id,
        "sample": base_metadata.get("sample", "S11"),
        "timestamp_merged": datetime.datetime.now().isoformat(),
        "source_runs": [base_id, extension_id],
        "n_chunks": int(base_metadata.get("n_chunks", 0)
                         + extension_metadata.get("n_chunks", 0)),
        "grid_sizes": {name: len(axes[name]) for name in COORDINATES},
        "total_points": n_rows,
    })

    grid_config = metadata.get("grid_config")
    if not isinstance(grid_config, dict) or "moire" not in grid_config:
        raise ValueError("Base run metadata has no usable grid_config")
    grid_config["moire"]["phiG"] = {
        "min_deg": float(phi_values[0]),
        "max_deg": float(phi_values[-1]),
        "step_deg": float(phi_steps[0]),
    }
    metadata["grid_config"] = grid_config
    metadata["fixed_params"] = copy.deepcopy(base_metadata["fixed_params"])
    return metadata


def merge_runs(base_id, extension_id, output_id):
    ids = (base_id, extension_id, output_id)
    if any(Path(run_id).name != run_id for run_id in ids):
        raise ValueError("Run IDs must be simple names, not paths")
    if len(set(ids)) != 3:
        raise ValueError("Base, extension, and output IDs must all differ")

    output_dir = DATA_DIR / f"edc_gamma_{output_id}"
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new --output-id; source/output data will not be overwritten."
        )

    base_dir, base_metadata, base_data = _read_run(base_id)
    extension_dir, extension_metadata, extension_data = _read_run(extension_id)
    if set(base_data) != set(extension_data):
        raise ValueError("The two HDF5 runs have different columns")
    if base_metadata.get("sample", "S11") != extension_metadata.get("sample", "S11"):
        raise ValueError("The two runs use different samples")
    _validate_fixed_params(base_metadata, extension_metadata)

    base_axes = _axes_and_check_complete(base_data, base_id)
    extension_axes = _axes_and_check_complete(extension_data, extension_id)
    _validate_run_axes(base_metadata, base_axes, base_id)
    _validate_run_axes(extension_metadata, extension_axes, extension_id)
    for name in ("Vg", "w1p", "w1d"):
        if not _same_axis(base_axes[name], extension_axes[name]):
            raise ValueError(f"Runs use different {name} grid values")
    if np.intersect1d(base_axes["phiG"], extension_axes["phiG"]).size:
        raise ValueError("The two runs overlap in phiG; refusing to duplicate grid points")

    merged_data = {
        name: np.concatenate((base_data[name], extension_data[name]))
        for name in base_data
    }
    sort_order = np.lexsort(tuple(merged_data[name] for name in reversed(COORDINATES)))
    merged_data = {name: values[sort_order] for name, values in merged_data.items()}
    merged_axes = {name: np.unique(merged_data[name]) for name in COORDINATES}

    n_rows = len(merged_data["Vg"])
    expected_rows = int(np.prod([len(merged_axes[name]) for name in COORDINATES]))
    if n_rows != expected_rows:
        raise ValueError(
            f"Merged grid is incomplete or duplicated: {n_rows} rows, "
            f"expected {expected_rows}"
        )

    metadata = _merged_metadata(
        base_metadata,
        extension_metadata,
        output_id,
        merged_axes,
        n_rows,
        base_id,
        extension_id,
    )

    base_params = base_dir / "interlayer_params.npy"
    extension_params = extension_dir / "interlayer_params.npy"
    if base_params.is_file() and extension_params.is_file():
        if not np.array_equal(np.load(base_params), np.load(extension_params)):
            raise ValueError("The run-local interlayer snapshots differ")
    params_to_copy = base_params if base_params.is_file() else extension_params

    output_dir.mkdir(parents=True)
    output_h5 = output_dir / "combined.h5"
    with h5py.File(output_h5, "w") as file:
        for name, values in merged_data.items():
            file.create_dataset(name, data=values, dtype="f8")

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    if params_to_copy.is_file():
        np.save(output_dir / "interlayer_params.npy", np.load(params_to_copy))

    print(f"Merged {len(base_data['Vg']):,} + {len(extension_data['Vg']):,} rows")
    print(
        f"Saved {n_rows:,} points with phiG={merged_axes['phiG'][0]:g}–"
        f"{merged_axes['phiG'][-1]:g} deg to {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-id", default="4Dsm8a")
    parser.add_argument("--extension-id", default="4Dsm8a_phi155_164")
    parser.add_argument("--output-id", default="4Dsm8a_extended")
    args = parser.parse_args()
    merge_runs(args.base_id, args.extension_id, args.output_id)


if __name__ == "__main__":
    main()
