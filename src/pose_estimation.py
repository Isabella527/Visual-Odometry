"""
pose_estimation.py
──────────────────
Camera pose estimation routines.

Contents
  - calibrate_camera       : single-image checkerboard calibration (hw2 method)
  - estimate_pose_DLT      : DLT pose from known K + 2-D/3-D correspondences
  - estimate_pose_RANSAC_DLT : RANSAC wrapper around estimate_pose_DLT
  - essentialmatrix2RT     : decompose E into (R, T) via cheirality
  - eightpoint_algorithm   : normalised 8-point algorithm → E, R, T
  - pnp_pose               : solvePnPRansac wrapper used in the VO loop
"""

import numpy as np
import cv2
import scipy.linalg

from triangulation import triangulation
from utils import hom, deh


# ─────────────────────────────────────────────────────────────
# CAMERA CALIBRATION  (hw2 single-image method)
# ─────────────────────────────────────────────────────────────

def calibrate_camera(calib_image_path: str,
                     checker: tuple = (6, 4),
                     checker_width: float = 1.0) -> tuple:
    """
    Calibrate a camera from a single checkerboard image using the hw2 method.

    Parameters
    ----------
    calib_image_path : path to the checkerboard photo taken by the VIDEO camera
    checker          : (inner_cols, inner_rows) of the board  e.g. (6, 4)
    checker_width    : physical side length of one square (any consistent unit)

    Returns
    -------
    K           : (3, 3) intrinsic matrix
    dist_coeffs : (1, 5) distortion coefficients  [k1 k2 p1 p2 k3]
    rms         : RMS reprojection error in pixels (aim for < 1.0)

    Raises
    ------
    AssertionError if the checkerboard is not detected in the image.
    """
    img = cv2.imread(calib_image_path)
    assert img is not None, f"Cannot read calibration image: {calib_image_path}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # detect inner corners
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, checker, flags)
    assert found, (
        "Checkerboard not detected!\n"
        "• Verify CHECKER = (inner_cols, inner_rows)\n"
        "• Ensure the full board is visible and well-lit\n"
        "• The upper-left square should be BLACK"
    )

    # sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 3-D object points in board frame  (z = 0 plane)
    obj_pts = np.zeros((checker[0] * checker[1], 1, 3), np.float32)
    obj_pts[:, 0, :2] = (
        np.mgrid[0:checker[0], 0:checker[1]].T.reshape(-1, 2) * checker_width
    )

    rms, K, dist_coeffs, _, _ = cv2.calibrateCamera(
        [obj_pts], [corners], (w, h), None, None
    )

    print(f"Calibration RMS: {rms:.4f} px  ({'good' if rms < 1.0 else 'WARNING: > 1.0'})")
    print(f"K =\n{K}")
    return K, dist_coeffs, rms


# ─────────────────────────────────────────────────────────────
# DLT POSE ESTIMATION  (known K)
# ─────────────────────────────────────────────────────────────

def estimate_pose_DLT(p: np.ndarray,
                      P: np.ndarray,
                      K: np.ndarray) -> tuple:
    """
    Estimate camera pose [R | T] via DLT given known intrinsics K.

    Parameters
    ----------
    p : (3, N) homogeneous pixel coordinates
    P : (4, N) homogeneous world coordinates
    K : (3, 3) intrinsic matrix

    Returns
    -------
    R   : (3, 3) rotation matrix
    T   : (3, 1) translation vector  (X_cam = R @ X_world + T)
    R_T : (3, 4) concatenated [R | T]
    """
    assert p.shape[0] == 3 and P.shape[0] == 4
    assert p.shape[1] == P.shape[1]

    # normalise pixel coordinates by K
    p_n    = scipy.linalg.inv(K) @ p
    p_uv   = p_n[:2, :] / p_n[2, :]
    n      = p.shape[1]

    Q = np.empty((0, 12))
    for i in range(n):
        Qi_0 = np.array([[1, 0, -p_uv[0, i]],
                         [0, 1, -p_uv[1, i]]])
        Q = np.vstack([Q, np.kron(Qi_0, P[:, i])])

    _, _, VT    = scipy.linalg.svd(Q)
    M_tilde     = VT.T[:, -1].reshape(3, 4)

    if scipy.linalg.det(M_tilde[:, :3]) < 0:
        M_tilde = -M_tilde

    U, _, Vt = scipy.linalg.svd(M_tilde[:, :3])
    R        = U @ Vt
    s        = np.sqrt(np.sum(R ** 2)) / np.sqrt(np.sum(M_tilde[:, :3] ** 2))
    T        = (s * M_tilde[:, 3]).reshape(3, 1)
    R_T      = np.hstack([R, T])

    return R, T, R_T


# ─────────────────────────────────────────────────────────────
# RANSAC DLT POSE ESTIMATION
# ─────────────────────────────────────────────────────────────

def estimate_pose_RANSAC_DLT(p_matched: np.ndarray,
                              p_W_matched: np.ndarray,
                              K: np.ndarray,
                              N_iter: int = 1000,
                              tol_inlier: float = 10.0,
                              display_iter: bool = True) -> tuple:
    """
    RANSAC wrapper around estimate_pose_DLT.

    Parameters
    ----------
    p_matched   : (3, N) homogeneous pixel coordinates
    p_W_matched : (4, N) homogeneous world coordinates
    K           : (3, 3) intrinsic matrix
    N_iter      : number of RANSAC iterations
    tol_inlier  : reprojection error threshold in pixels
    display_iter: print progress when a new best is found

    Returns
    -------
    R, T, M         : best pose and projection matrix
    N_inliers_max   : inlier count of the best hypothesis
    i_inliers_best  : indices of inliers for the best hypothesis
    """
    N_inliers_max  = 0
    N_data         = p_matched.shape[1]
    i_inliers_best = np.array([], dtype=int)
    R_best = T_best = None

    for i_iter in range(N_iter):
        idx      = np.random.choice(N_data, 6, replace=False)
        R, T, RT = estimate_pose_DLT(p_matched[:, idx], p_W_matched[:, idx], K)

        proj  = K @ RT @ p_W_matched
        error = np.sqrt(np.sum(
            (proj[:2] / proj[2] - p_matched[:2] / p_matched[2]) ** 2,
            axis=0))
        i_inliers = np.where(error < tol_inlier)[0]

        if len(i_inliers) > N_inliers_max:
            N_inliers_max  = len(i_inliers)
            i_inliers_best = i_inliers
            R_best, T_best = R, T
            if display_iter:
                print(f"iter={i_iter}, N_inliers={N_inliers_max}, "
                      f"w={N_inliers_max/N_data:.2f}")

    if N_inliers_max >= 6:
        R, T, M = estimate_pose_DLT(
            p_matched[:, i_inliers_best],
            p_W_matched[:, i_inliers_best], K)
    else:
        print(f"N_inliers={N_inliers_max} < 6 — pose estimation failed.")
        R = np.zeros((3, 3))
        T = np.zeros((3, 1))
        M = np.zeros((3, 4))

    return R, T, M, N_inliers_max, i_inliers_best


# ─────────────────────────────────────────────────────────────
# ESSENTIAL MATRIX → (R, T)
# ─────────────────────────────────────────────────────────────

def essentialmatrix2RT(E: np.ndarray,
                       p1: np.ndarray,
                       p2: np.ndarray,
                       K1: np.ndarray,
                       K2: np.ndarray) -> tuple:
    """
    Decompose an essential matrix into (R, T) by selecting the hypothesis
    with the fewest points at negative depth (cheirality check).

    Parameters
    ----------
    E       : (3, 3) essential matrix
    p1, p2  : (3, N) homogeneous pixel coordinates in camera 1 and 2
    K1, K2  : (3, 3) intrinsic matrices

    Returns
    -------
    R : (3, 3) rotation   camera-2 relative to camera-1
    T : (3, 1) translation
    """
    U, _, VT = scipy.linalg.svd(E)
    W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    R_cands = np.zeros((3, 3, 4))
    T_cands = np.zeros((3, 4))

    R_cands[:, :, 0], T_cands[:, 0] = U @ W @ VT,    U[:, 2]
    R_cands[:, :, 1], T_cands[:, 1] = U @ W @ VT,   -U[:, 2]
    R_cands[:, :, 2], T_cands[:, 2] = U @ W.T @ VT,  U[:, 2]
    R_cands[:, :, 3], T_cands[:, 3] = U @ W.T @ VT, -U[:, 2]

    for i in range(4):
        if scipy.linalg.det(R_cands[:, :, i]) < 0:
            R_cands[:, :, i] = -R_cands[:, :, i]

    M1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    neg_depths = np.zeros(4, dtype=int)

    for i in range(4):
        M2 = K2 @ np.hstack([R_cands[:, :, i],
                              T_cands[:, i].reshape(3, 1)])
        P  = triangulation(p1, p2, M1, M2)
        neg_depths[i] = np.sum(P[2, :] < 0)

    i_opt = np.argmin(neg_depths)
    return R_cands[:, :, i_opt], T_cands[:, i_opt].reshape(3, 1)


# ─────────────────────────────────────────────────────────────
# EIGHT-POINT ALGORITHM
# ─────────────────────────────────────────────────────────────

def eightpoint_algorithm(p1: np.ndarray,
                         p2: np.ndarray,
                         K1: np.ndarray,
                         K2: np.ndarray) -> tuple:
    """
    Normalised 8-point algorithm to compute E, R, T.

    Parameters
    ----------
    p1, p2  : (3, N) homogeneous pixel coordinates (N ≥ 8)
    K1, K2  : (3, 3) intrinsic matrices

    Returns
    -------
    E : (3, 3) essential matrix
    R : (3, 3) rotation
    T : (3, 1) translation
    """
    n = p1.shape[1]
    p_bar_1 = scipy.linalg.inv(K1) @ p1
    p_bar_2 = scipy.linalg.inv(K2) @ p2

    Q = np.zeros((n, 9))
    for i in range(n):
        Q[i, :] = np.kron(p_bar_1[:, i], p_bar_2[:, i])

    _, _, VT = scipy.linalg.svd(Q)
    E        = VT[-1, :].reshape(3, 3).T

    R, T = essentialmatrix2RT(E, p1, p2, K1, K2)
    return E, R, T


# ─────────────────────────────────────────────────────────────
# PNP WRAPPER  (used in the main VO loop)
# ─────────────────────────────────────────────────────────────

def pnp_pose(pts3d: np.ndarray,
             pts2d: np.ndarray,
             K: np.ndarray) -> tuple:
    """
    Estimate camera pose from 3-D/2-D correspondences via solvePnPRansac.

    Parameters
    ----------
    pts3d : (3, N) or (N, 3) world-frame 3-D points
    pts2d : (2, N) or (N, 2) corresponding pixel coordinates
    K     : (3, 3) intrinsic matrix

    Returns
    -------
    R        : (3, 3) rotation  (X_cam = R @ X_world + T)
    T        : (3, 1) translation
    i_inliers: 1-D array of inlier indices, or None on failure
    """
    # accept both (3,N) and (N,3) layouts
    if pts3d.shape[0] == 3 and pts3d.shape[1] != 3:
        pts3d = pts3d.T
    if pts2d.shape[0] == 2 and pts2d.shape[1] != 2:
        pts2d = pts2d.T

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float64),
        pts2d.astype(np.float64),
        K, None)

    if not ok or inliers is None:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1), inliers.ravel()
