# Real-Space LDOS

## LDOS Overview

The `compute_ldos.py` script computes the **local density of states** in real space along a 1D cut through the moiré unit cell. The LDOS reveals which stacking sites (W/W, Se/W, W/S) host the flat moiré bands at a given energy, and how strongly the electronic states localize at different positions in the moiré pattern.

## Physical formula

The LDOS is the **k-integrated spectral function** evaluated in real space:

$$LDOS(\mathbf{r}, E) = \frac{1}{N_k} \sum_{\mathbf{k}} \sum_{n} \left|\psi_{n\mathbf{k}}(\mathbf{r})\right|^2 \cdot \frac{\eta}{\pi\Big((E - E_{n\mathbf{k}})^2 + \eta^2\Big)}$$

where:
- $\mathbf{k}$ — k-points spanning the mini-Brillouin zone (uniform $k_\text{pts} \times k_\text{pts}$ grid)
- $n$ — eigenstate index ($836$ states for $n_\text{shells}=2$, $19$ cells)
- $E_{n\mathbf{k}}$ — eigenvalue of eigenstate $n$ at k-point $\mathbf{k}$
- $\eta$ — Lorentzian broadening (energy spreading, in eV)
- $N_k$ — total number of k-points

The spatial probability density $|\psi_{n\mathbf{k}}(\mathbf{r})|^2$ is summed over all 44 orbitals (both layers):

$$\left|\psi_{n\mathbf{k}}(\mathbf{r})\right|^2 = \sum_{\alpha=1}^{44} \left|\psi_{n\mathbf{k}}^\alpha(\mathbf{r})\right|^2$$

## Wavefunction reconstruction

Each orbital's real-space wavefunction is reconstructed from the supercell eigenvectors $c_{\alpha n}^{i_c}(\mathbf{k})$ by Fourier transform over the moiré reciprocal lattice:

$$\psi_{n\mathbf{k}}^\alpha(\mathbf{r}) = \sum_{i_c=1}^{n_\text{cells}} e^{i(\mathbf{k} + \mathbf{G}_{i_c})\cdot\mathbf{r}} \cdot c_{\alpha n}^{i_c}(\mathbf{k})$$

where:
- $\alpha$ — orbital index (0–43: WSe₂ indices 0–21, WS₂ indices 22–43)
- $i_c$ — mini-BZ cell index (0 to $n_\text{cells}-1$)
- $\mathbf{G}_{i_c}$ — reciprocal lattice vector offset of cell $i_c$:
  $$\mathbf{G}_{i_c} = \mathbf{G}_1 \cdot l_{i_c}[0] + \mathbf{G}_2 \cdot l_{i_c}[1]$$
  where $\mathbf{G}_1, \mathbf{G}_2$ are the two basis moiré reciprocal vectors and $l_{i_c}$ is the lookup table entry for cell $i_c$

The implementation is vectorized: all $n_\text{cells}$ and 44 orbitals are handled simultaneously via a precomputed index array of shape $(44, n_\text{cells})$ that maps each (orbital, cell) pair to its position in the $44 \cdot n_\text{cells}$ eigenvector.

## Real-space path

The spatial cut follows the direction $\mathbf{a}_1 + \mathbf{a}_2$ (diagonal of the moiré unit cell), starting at one W/W high-symmetry point and passing through the other stacking configurations:

```
W/W ──→ Se/W ──→ W/S ──→ W/W
```

| Position | r | Stacking | Description |
|---|---|---|---|
| W/W | 0 | AA-like | W atoms of both layers aligned |
| Se/W | $\|\mathbf{a}_1+\mathbf{a}_2\| / 3$ | AB/BA | Se atom over W site |
| W/S | $2\|\mathbf{a}_1+\mathbf{a}_2\| / 3$ | AB/BA | W atom over S site |
| W/W | $\|\mathbf{a}_1+\mathbf{a}_2\|$ | AA-like | Back to start |

The total length of this path is $|\mathbf{a}_1 + \mathbf{a}_2| = a_M \cdot \sqrt{3}$, where $a_M$ is the moiré lattice constant (~50 Å for S11 at 2.8°).

## Real-space sharpness

The spatial resolution of the LDOS is controlled by **`--n-shells`**, not `--k-pts`. Each shell adds higher-order moiré reciprocal lattice vectors $\mathbf{G}_{i_c}$ to the Fourier sum, analogous to adding higher harmonics:

| n_shells | Cells | G vectors | Fourier resolution |
|---|---|---|---|
| 1 | 7 | 7 | Coarse |
| 2 | 19 | 19 | Good (sweet spot) |
| 3 | 37 | 37 | Marginal gain |

Increasing `--k-pts` refines the **k-integral convergence** but does not add new Fourier components — once the integral is converged (≥10 k-points is usually sufficient), further increases have negligible effect on spatial sharpness.

## LDOS CLI options

All monolayer and interlayer parameters are loaded from `Inputs/plot_bilayer/`.

```bash
python scripts/compute_ldos.py [options]
```

| Option | Default | Description |
|---|---|---|
| `--k-pts` | 12 | Number of k-points per mini-BZ side ($k_\text{pts}^2$ total) |
| `--r-pts` | 300 | Number of real-space grid points along the cut |
| `--n-shells` | 2 | Number of moiré shells ($n_\text{cells} = 1 + 3n(n+1)$) |
| `--e-min` | −1.0 | Minimum energy (eV) |
| `--e-max` | 0.0 | Maximum energy (eV) |
| `--delta-e` | 0.005 | Energy grid spacing (eV) |
| `--eta` | 0.01 | Lorentzian broadening width (eV) |
| `--r-extra` | 0.0 | Extra fraction of the moiré period to extend past the W/W point (e.g. 0.166 for padding) |
| `--center` | G | BZ point to center the k-grid on: `G` (Γ) or `K` |
| `--sample` | S11 | Sample name for energy offset and twist angle |
| `--theta` | — | Twist angle in degrees (overrides sample) |
| `--Vg` | — | Override moiré potential at Γ (eV) |
| `--Vk` | — | Override moiré potential at K (eV) |
| `--phiG` | — | Override moiré phase at Γ (degrees) |
| `--phiK` | — | Override moiré phase at K (degrees) |
| `--no-cache` | — | Ignore cache and recompute |

## LDOS Caching

The script uses a two-level cache under `Data/ldos/`:

```
Data/ldos/
  diag_<mono_hash>_<w1p>_<w1d>_<w2p>_<w2d>_<Vg>_<phiG>_<Vk>_<phiK>_t<theta>_n<n_shells>_k<k_pts>_<center>/
    metadata.json                      ← run parameters snapshot
    diag.npz                           ← evals, evecs, k_flat (diagonalization)
    ldos_<r_pts>_<e_min>_<e_max>_<delta_e>_<eta>_re<r_extra>/
      ldos.npz                         ← LDOS array [n_r, n_e], r_list, e_list, rL
      ldos_meta.json                   ← LDOS computation parameters
      ldos.png                         ← pcolormesh plot
    ldos_<other_params>/
      ...
  diag_<other_hamiltonian_params>/
    ...
```

**Diagonalization cache** (expensive, minutes): hashed by monolayer param hash + interlayer + moiré + theta + n_shells + k_pts + center. Changing any of these triggers a new diagonalization.

**LDOS cache** (cheap, seconds): hashed by r_pts + energy range + broadening + r_extra. Exploring different energy windows, resolutions, or broadening values reuses the cached diagonalization.

## LDOS Output

**Data file** (`ldos.npz`):
- `ldos`: array of shape `(n_r, n_e)` — LDOS values
- `r_list`: array of shape `(n_r, 2)` — real-space positions (Å)
- `e_list`: array of shape `(n_e,)` — energy grid (eV)
- `rL`: float — moiré period $|\mathbf{a}_1 + \mathbf{a}_2|$ (Å)

**Plot** (`ldos.png`):
- X-axis: Energy (eV)
- Y-axis: Position along $\mathbf{a}_1 + \mathbf{a}_2$, with stacking site labels (W/W, Se/W, W/S, W/W)
- Colormap: `hot` (yellow–red–black)
- Colorbar: "low" / "high" labels, no ticks
- Y-axis inverted (r = 0 at top)

## LDOS Quick start

```bash
# Basic run at Gamma
python scripts/compute_ldos.py --k-pts 10 --r-pts 80 --n-shells 2 \
    --e-min -1.28 --e-max -1.11 --delta-e 0.002 --eta 0.005 --r-extra 0.166

# With different moiré phase
python scripts/compute_ldos.py --k-pts 10 --r-pts 80 --n-shells 2 \
    --e-min -1.28 --e-max -1.11 --delta-e 0.002 --eta 0.005 \
    --phiG 170 --r-extra 0.166

# K-centered LDOS
python scripts/compute_ldos.py --k-pts 10 --r-pts 80 --n-shells 2 \
    --e-min -1.28 --e-max -1.11 --delta-e 0.002 --center K

# Force recompute (ignore cache)
python scripts/compute_ldos.py --k-pts 10 --r-pts 80 --n-shells 2 --no-cache
```

