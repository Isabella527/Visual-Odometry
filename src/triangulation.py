"""
triangulation.py
────────────────
3-D point triangulation from two camera views.

Contents
  - triangulation        : basic DLT triangulation (no filtering)
  - triangulation_robust : DLT + singular-value ratio + reprojection filter
"""

import numpy as np
import scipy.linalg

from utils import hat, hom, deh


# ─────────────────────────────────────────────────────────────
# BASIC TRIANGULATION
# ─────────────────────────────────────────────────────────────

def triangulation(p1: np.ndarray,
                  p2: np.ndarray,
                  M1: np.ndarray,
                  M2: np.ndarray) -> np.ndarray:
    """
    Triangulate N 3-D points from two sets of homogeneous pixel coordinates
    using the DLT (linear) method.

    Parameters
    ----------
    p1, p2 : (3, N) homogeneous pixel coordinates in camera 1 and 2
    M1, M2 : (3, 4) perspective projection matrices  K @ [R | T]

    Returns
    -------
    P : (3, N) non-homogeneous world coordinates
    """
    assert p1.shape[0] == 3 and p2.shape[0] == 3
    assert p1.shape[1] == p2.shape[1]

    n = p1.shape[1]
    P = np.zeros((3, n))

    for i in range(n):
        Q  = np.vstack([hat(p1[:, i]) @ M1,
                        hat(p2[:, i]) @ M2])
        _, _, VT = scipy.linalg.svd(Q)
        Pi        = VT[-1, :]
        P[:, i]   = Pi[:3] / Pi[3]

    return P


# ─────────────────────────────────────────────────────────────
# ROBUST TRIANGULATION
# ─────────────────────────────────────────────────────────────

def triangulation_robust(p1: np.ndarray,
                         p2: np.ndarray,
                         M1: np.ndarray,
                         M2: np.ndarray,
                         tol_mu:  float = 1e-3,
                         tol_rep: float = 1.0) -> tuple:
    """
    Triangulate N points and return inliers based on two quality criteria:

      1. Singular-value ratio  μ = σ₄ / σ₃  <  tol_mu
         (small ratio → well-conditioned system → reliable depth)

      2. Symmetric reprojection error  max(err1, err2)  <  tol_rep  (px)

    Parameters
    ----------
    p1, p2   : (3, N) homogeneous pixel coordinates
    M1, M2   : (3, 4) projection matrices
    tol_mu   : upper bound on singular-value ratio (default 1e-3)
    tol_rep  : upper bound on reprojection error in pixels (default 1.0)

    Returns
    -------
    p_W           : (3, N) triangulated world points (all N, not just inliers)
    index_inliers : 1-D array of column indices that pass both filters
    mu_ratio      : (N,) singular-value ratios
    err           : (N,) max symmetric reprojection error per point
    """
    assert p1.shape[0] == 3 and p2.shape[0] == 3
    assert p1.shape[1] == p2.shape[1]

    n    = p1.shape[1]
    p_W  = np.zeros((3, n))
    mu   = np.zeros((4, n))

    for i in range(n):
        Q           = np.vstack([hat(p1[:, i]) @ M1,
                                 hat(p2[:, i]) @ M2])
        _, S, VT    = scipy.linalg.svd(Q)
        mu[:, i]    = S
        Pi          = VT[-1, :]
        p_W[:, i]   = Pi[:3] / Pi[3]

    mu_ratio = mu[3, :] / mu[2, :]

    err1 = np.sqrt(np.sum((deh(M1 @ hom(p_W)) - deh(p1)) ** 2, axis=0))
    err2 = np.sqrt(np.sum((deh(M2 @ hom(p_W)) - deh(p2)) ** 2, axis=0))
    err  = np.max(np.vstack([err1, err2]), axis=0)

    index_inliers = np.where((mu_ratio < tol_mu) & (err < tol_rep))[0]

    return p_W, index_inliers, mu_ratio, err
