"""Combine EDC grid chunk files into a single result file.

Scans Data/edc_grid_<bz>_run_<id>/ for all chunk_*.h5 files,
concatenates datasets, and saves a combined file.

Usage:
    python scripts/combine_edc_chunks.py --bz-point gamma --run-id 001
    python scripts/combine_edc_chunks.py --bz-point k --run-id 001
    python scripts/combine_edc_chunks.py --bz-point gamma --run-id 001 --output Data/edc_grid_gamma_run_001/combined.h5
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py

bz_point = "gamma"
run_id = "default"
output = None

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--bz-point" and i + 1 < len(args):
        bz_point = args[i + 1]
        i += 2
    elif args[i] == "--run-id" and i + 1 < len(args):
        run_id = args[i + 1]
        i += 2
    elif args[i] == "--output" and i + 1 < len(args):
        output = Path(args[i + 1])
        i += 2
    else:
        i += 1

out_dir = Path("Data") / f"edc_grid_{bz_point}_run_{run_id}"
if output is None:
    output = out_dir / "combined.h5"

chunk_files = sorted(out_dir.glob("chunk_*.h5"))

if not chunk_files:
    print(f"No chunk files found in {out_dir}")
    sys.exit(1)

print(f"Found {len(chunk_files)} chunk files in {out_dir}")

# Read first file to get column names
with h5py.File(chunk_files[0], "r") as f:
    columns = list(f.keys())

print(f"Columns: {columns}")

# Concatenate all chunks
total_points = 0
data = {col: [] for col in columns}

for fn in chunk_files:
    with h5py.File(fn, "r") as f:
        for col in columns:
            data[col].append(f[col][:])
        total_points += f[columns[0]].shape[0]

print(f"Total points: {total_points}")

# Save combined file
output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(output, "w") as hf:
    for col in columns:
        arr = np.concatenate(data[col])
        hf.create_dataset(col, data=arr, dtype="f8")

print(f"Combined file saved to {output}")
