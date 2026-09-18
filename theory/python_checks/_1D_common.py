import numpy as np


hbar_si = 1.054571817e-34
m0 = 9.1093837e-31
eV = 1.602176634e-19

L_moire = 76.0
g = 2.0 * np.pi / L_moire

m = 3.5 * m0

alpha = (hbar_si ** 2 / (2.0 * m)) / (eV * 1e-3) * 1e20


def eps(k, n):
    return -alpha * (k + n * g) ** 2


def H(k, v):
    return np.array([
        [eps(k, -1), v, 0.0],
        [v,         eps(k, 0), v],
        [0.0,       v,         eps(k, 1)],
    ])
