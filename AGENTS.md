#AGENTS.md

## Project

Tight-binding model of WSe2/WS2 heterobilayer moire superlattices. Two-stage workflow:
1. **Monolayer** — fit 43 TB parameters per TMD (WSe2, WS2) to ARPES data
2. **Bilayer** — build moire supercell Hamiltonian, compute EDCs, extract moire potential & interlayer coupling

## Commands

```
python monolayer.py <WSe2|WS2> <index>     # Fit monolayer TB params (Nelder-Mead)
python edc.py <G|K> <index>                 # EDC analysis at Gamma or K (chunk 0-127)
```

No build/test/lint tooling exists. Verification = does it run and produce `.npy`/`.h5` output.

## Architecture

| File | Role |
|---|---|
| `CORE_functions.py` (1135 lines) | Physics core: `H_monolayer()` (22x22), `find_t/find_e/find_HSO`, `monolayerData` class, DFT params, moire utilities |
| `monolayer.py` (139 lines) | Entry point 1: minimization driver |
| `edc.py` (200 lines) | Entry point 2: parameter sweep + EDC fitting |
| `utils_mono.py` (830 lines) | Fitting utilities: `chi2`, bounds, orbital constraints, plotting |
| `utils1.py` (603 lines) | Bilayer utilities: `big_H()` (supercell diagonalization), EDC fitting with Voigt profiles, LDOS |

## Key facts

- **Hamiltonian basis**: 11 orbitals x 2 spins = 22x22 monolayer. Orbitals: `[d_xz, d_yz, p_z^o, p_x^o, p_y^o, d_z2, d_xy, d_x2-y2, p_z^e, p_x^e, p_y^e]`
- **43 parameters**: 7 on-site + 21 t1 + 8 t5 + 4 t6 + 1 offset + 2 SOC (L_W, L_S)
- **Supercell**: `(44 * nCells) x (44 * nCells)` where `nCells = 1 + 3*nShells*(nShells+1)`. nShells=2 → 19 cells → 836x836 matrix
- **Samples**: S3 (theta=1.8°), S11 (theta=2.8°, energy offset=-0.47 eV)
- **Lattice constants**: WS2=3.18 Å, WSe2=3.32 Å
- **Inputs/**: ARPES band data (`KpGK_*.txt`, `KMKp_*.txt`) and pre-fitted TB params (`tb_*.npy`)
- **Outputs**: `Data/*.npz` (fitting intermediates), `Data/*.h5` (EDC results)

## Machine detection

`get_machine(cwd)` checks path prefix: `dario`→`loc`, `/home/users/r/rossid`→`hpc`, `/users/rossid`→`maf`. Entry points derive `master_folder` from this. Hardcoded paths also exist in `utils_mono.get_home_dn()` and `utils1.get_home_dn()`.

## Conventions

- `cfs` = alias for `CORE_functions` (used everywhere)
- `utils` = alias for `utils_mono` in monolayer context, `utils1` in bilayer context
- `disp = machine=='loc'` controls verbose printing
- `machine=='maf'` shifts CLI index by -1
- SOC bounds set to 0 → fit only TB params (HSO fixed from DFT)
- `chi2` uses global `min_chi2` / `evaluation_step` for early-exit at `max_eval`
