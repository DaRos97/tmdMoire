# TMD heterobilayer WSe₂/WS₂

Tight-binding model of WSe₂/WS₂ heterobilayer moiré superlattices. Three-stage computational workflow:

1. **Monolayer fitting** — Fit 43 TB parameters per TMD to ARPES band dispersions via Nelder-Mead optimization with physical constraints
2. **Bilayer interlayer coupling** — Fit 4 interlayer hopping parameters to reproduce the 3 top valence bands from bilayer ARPES
3. **Bilayer moiré potential** — Sweep moiré potential parameters at Γ and K to match experimental EDC peak positions, compute band structure and LDOS

## Environment

```bash
source ../PyEnv/bin/activate
```

## Quick Start

### Monolayer fitting

```bash
python scripts/fit_monolayer.py WSe2 0          # single fit
python scripts/run_monolayer_grid.py WSe2        # full grid search
python scripts/run_monolayer_grid.py WSe2 --score --export  # score + export
python scripts/sort_monolayer_results.py --tmd WSe2 --input Data/WSe2_run1/merged_WSe2_absolute.h5
```

### Bilayer interlayer coupling

```bash
python scripts/fit_bilayer_coupling.py
```

### Moiré potential & band plotting

```bash
python scripts/edc_grid_gamma.py --chunk 0/1000 -id myrun
python scripts/combine_edc_chunks.py --bz-point gamma -id myrun
python scripts/analyze_edc_gamma.py -id myrun
python scripts/plot_moire_bands.py --n-shells 2 --k-pts 300
```

### Real-space LDOS

```bash
python scripts/compute_ldos.py --n-shells 2 --k-pts 10 --r-pts 80
```

## Documentation

| Topic | File |
|---|---|
| Monolayer fitting: physics, objective, constraints, workflow | [`docs/monolayer.md`](docs/monolayer.md) |
| Pre-computed results (WSe₂, WS₂) | [`docs/results.md`](docs/results.md) |
| Bilayer interlayer coupling | [`docs/bilayer.md`](docs/bilayer.md) |
| Moiré potential EDC analysis (Γ + K sweeps) | [`docs/bilayer_moire.md`](docs/bilayer_moire.md) |
| Moiré band structure plotting | [`docs/bilayer_bandplot.md`](docs/bilayer_bandplot.md) |
| Real-space LDOS | [`docs/bilayer_ldos.md`](docs/bilayer_ldos.md) |

## Key facts

- **Hamiltonian**: 11 orbitals × 2 spins = 22×22 monolayer; 44·N × 44·N supercell for moiré
- **43 parameters**: 7 on-site + 21 t₁ + 8 t₅ + 4 t₆ + 1 offset + 2 SOC
- **Samples**: S3 (θ=1.8°), S11 (θ=2.8°, offset=−0.47 eV)
- **Lattice constants**: WS₂=3.18 Å, WSe₂=3.32 Å

## References

*(Documentation forthcoming)*
