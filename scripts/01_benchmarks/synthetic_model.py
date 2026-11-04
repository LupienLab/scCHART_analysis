
"""Synthetic chromatin accessibility model for the scCHART benchmarks.

The model generates latent cell states and binary accessibility matrices
containing structural, programme-specific, and background peaks.

Simulation parameters and random-number generation are fixed to preserve
the validated benchmark results.
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.special import expit

from scchart import external_programme_reference


# Latent cell-state geometry
K_DIM = 10
BASE_SEED_S = 20260925
BASE_SEED_P = 20260928

mu = np.linspace(1.0, 2.0, K_DIM)
MU = mu / np.linalg.norm(mu)

_v = np.where(np.arange(K_DIM) < K_DIM // 2, 1.0, -1.0)
_v = _v - (_v @ MU) * MU
V = _v / np.linalg.norm(_v)


# Accessibility model
N_SPK = 2000   # Structural peaks
N_PPK = 300    # Peaks per regulatory programme
N_BGK = 3000   # Background peaks

B_ANCH = 0.0128
Q_ANCH = 0.1058
SIG_DEPTH = 0.35

W_S = np.random.default_rng(
    (BASE_SEED_P, 1)
).standard_normal((N_SPK, K_DIM))

ALPHA_S = -3.0 + 0.5 * np.random.default_rng(
    (BASE_SEED_P, 2)
).standard_normal(N_SPK)

TARGET_S = float(expit(ALPHA_S).mean())

BG_RATE = np.exp(
    np.log(0.022)
    + 0.73 * np.random.default_rng(
        (BASE_SEED_P, 3)
    ).standard_normal(N_BGK)
)


# Two-state simulation parameters
LAM = 1.0
SIG_Z = 0.10
TAU = 0.10
KAPPA = 3.0
W_COMMON = 0.8


def milli(x: float) -> int:
    """Convert a parameter value to its integer seed component."""
    return int(round(1000 * x))


def two_state_latent(delta, rng, n, frac_b=0.5, sigma=0.05):
    """Generate two latent states separated by delta along V.

    State A is shifted by frac_b * delta and state B by
    -(1 - frac_b) * delta. Cells are returned in random order.
    """
    n_b = int(round(frac_b * n))
    state = np.array(["A"] * (n - n_b) + ["B"] * n_b)

    shift = np.where(
        state == "A", frac_b, -(1.0 - frac_b)
    ) * delta

    Z = (
        MU[None, :]
        + shift[:, None] * V[None, :]
        + sigma * rng.standard_normal((n, K_DIM)) / np.sqrt(K_DIM)
    )

    order = rng.permutation(n)
    return Z[order], state[order]


def state_axis_Z(s, rng):
    """Generate latent states along the A-to-B axis with Gaussian noise."""
    eps = rng.standard_normal((s.size, K_DIM)) / np.sqrt(K_DIM)

    return (
        MU[None]
        + LAM * s[:, None] * V[None]
        + SIG_Z * eps
    )


def state_axis_activity(s):
    """Programme activities for A, B, and a shared programme."""
    wA = expit(-KAPPA * s)

    return np.c_[
        wA,
        1.0 - wA,
        np.full(s.size, W_COMMON),
    ]


def beta_for(Zmc):
    """Find the offset that matches mean structural accessibility to TARGET_S."""
    L = ALPHA_S[None] + (Zmc - MU) @ W_S.T

    return brentq(
        lambda t: expit(L + t).mean() - TARGET_S,
        -40,
        10,
        xtol=1e-12,
    )


def observe(Z, W, beta, rng):
    """Sample a binary cell-by-peak accessibility matrix.

    Structural peak probabilities depend on latent state Z.
    Programme and background probabilities also vary with
    the cell-specific multiplicative depth factor rho.
    """
    Kp = W.shape[1]
    T = np.kron(np.eye(Kp), np.ones((N_PPK, 1)))

    n = Z.shape[0]
    rho = np.exp(
        SIG_DEPTH * rng.standard_normal(n)
        - SIG_DEPTH ** 2 / 2
    )

    p = np.hstack([
        expit(ALPHA_S[None] + (Z - MU) @ W_S.T + beta),
        np.minimum(
            1.0,
            rho[:, None] * (B_ANCH + Q_ANCH * (W @ T.T)),
        ),
        np.minimum(
            1.0,
            rho[:, None] * BG_RATE[None],
        ),
    ])

    return sp.csr_matrix(
        (rng.random(p.shape) < p).astype(np.float64)
    )


def feature_labels(progs):
    """Assign programme labels to peaks; other peaks are Shared."""
    return np.array(
        ["Shared"] * N_SPK
        + [p for p in progs for _ in range(N_PPK)]
        + ["Shared"] * N_BGK,
        dtype=object,
    )


def no_activity_reference(progs, seed, n_ref=1200):
    """Build an external reference from cells without programme activity."""
    rng = np.random.default_rng(seed)

    Zr = (
        MU[None]
        + SIG_Z
        * rng.standard_normal((n_ref, K_DIM))
        / np.sqrt(K_DIM)
    )

    Br = observe(
        Zr,
        np.zeros((n_ref, len(progs))),
        beta_for(Zr),
        rng,
    )

    return external_programme_reference(
        feature_labels(progs),
        Br,
        np.array(["reference_outgroup"] * n_ref),
        programmes=list(progs),
    )
