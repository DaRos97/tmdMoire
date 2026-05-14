"""Verify all fit_config.json keys are passed correctly through the pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import itertools
from tmdmoire.utils.paths import get_repo_root, SOURCE_CONFIG

master_folder = get_repo_root()

# 1. Load raw config
with open(SOURCE_CONFIG) as f:
    config = json.load(f)

print("=== fit_config.json keys ===")
for k, v in config.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for k2, v2 in v.items():
            print(f"    {k2}: {v2}")
    else:
        print(f"  {k}: {v}")

# 2. Simulate build_grid
grid = config["grid"]
keys = ["K1", "K2", "K3", "K4", "K5", "K6"]
values = [grid[k] for k in keys]
combos = list(itertools.product(*values))

cfg = {
    "idx": 0,
    "Ks": combos[0],
    "pts": config.get("pts", 91),
    "seed": config.get("seed", 42),
    "boundType": config["bounds"]["boundType"],
    "Bs": tuple(config["bounds"]["Bs"]),
    "optimizer": config.get("optimizer", {}),
    "use_dft_x0": config.get("use_dft_x0", True),
}

print("\n=== Config passed to ParameterFitter ===")
for k, v in cfg.items():
    print(f"  {k}: {v}")

# 3. Check what ParameterFitter.run() reads
opt = cfg.get("optimizer", {})
method = opt.get("method", "dual_annealing")
use_dft_x0 = cfg.get("use_dft_x0", True)
nm_maxiter = opt.get("nm_maxiter", 50)
nm_fatol = opt.get("nm_fatol", 1e-3)
da_maxiter = opt.get("da_maxiter", 100)
de_maxiter = opt.get("de_maxiter", 100)
de_popsize = opt.get("de_popsize", 15)
de_strategy = opt.get("de_strategy", "best1bin")
de_mutation = opt.get("de_mutation", (0.5, 1.0))
de_recombination = opt.get("de_recombination", 0.7)

print("\n=== Values read by ParameterFitter.run() ===")
print(f"  method:          {method}")
print(f"  use_dft_x0:      {use_dft_x0}")
print(f"  nm_maxiter:      {nm_maxiter}")
print(f"  nm_fatol:        {nm_fatol}")
print(f"  da_maxiter:      {da_maxiter}")
print(f"  de_maxiter:      {de_maxiter}")
print(f"  de_popsize:      {de_popsize}")
print(f"  de_strategy:     {de_strategy}")
print(f"  de_mutation:     {de_mutation}")
print(f"  de_recombination:{de_recombination}")
print(f"  Bs:              {cfg['Bs']}")
print(f"  boundType:       {cfg['boundType']}")
print(f"  seed:            {cfg['seed']}")
print(f"  pts:             {cfg['pts']}")

# 4. Check for missing keys
all_config_keys = set(config.keys())
used_keys = {"description", "grid", "bounds", "pts", "seed", "use_dft_x0", "optimizer"}
optimizer_keys = set(config.get("optimizer", {}).keys())
used_optimizer_keys = {"method", "da_maxiter", "de_maxiter", "de_popsize", "de_strategy",
                       "de_mutation", "de_recombination", "nm_maxiter", "nm_fatol"}

missing_config = all_config_keys - used_keys
missing_optimizer = optimizer_keys - used_optimizer_keys

print("\n=== Missing keys check ===")
if missing_config:
    print(f"  UNREAD config keys: {missing_config}")
else:
    print("  All config keys are read")
if missing_optimizer:
    print(f"  UNREAD optimizer keys: {missing_optimizer}")
else:
    print("  All optimizer keys are read")

# 5. Check grid size
total = len(combos)
print(f"\n=== Grid size: {total} combinations ===")
print(f"  K1: {len(grid['K1'])} values")
print(f"  K2: {len(grid['K2'])} values")
print(f"  K3: {len(grid['K3'])} values")
print(f"  K4: {len(grid['K4'])} values")
print(f"  K5: {len(grid['K5'])} values")
print(f"  K6: {len(grid['K6'])} values")
