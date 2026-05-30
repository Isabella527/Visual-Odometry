"""
bootstrap.py
────────────
VO initialisation: detect keypoints in frame-0, track to frame-1,
recover pose via Essential matrix, triangulate initial landmarks.

Contents
  - run_bootstrap : full bootstrap pipeline → keypoints, p_W, R, T
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from tracking       import harris_corner, select_keypoints, KLT
from triangulation  import triangulation_robust
from visualization  import draw_frame
from utils          import hom


# ─────────────────────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────────────────────

def run_bootstrap(img0: np.ndarray,
                  img1: np.ndarray,
                  K: np.ndarray,
                  param: dict,
                  display: bool = False) -> tuple:
    """
    Initialise VO from two frames.

    Pipeline
    --------
    1. Detect Harris keypoints in img0
    2. Track them to img1 via KLT (with bidirectional check)
    3. Estimate Essential matrix via RANSAC
    4. Recover (R, T) with cheirality check
    5. Triangulate and filter with robust triangulation

    Parameters
    ----------
    img0    : first  grayscale frame (world origin)
    img1    : second grayscale frame (bootstrap frame)
    K       : (3, 3) camera intrinsic matrix
    param   : dict with keys
                'W_harris_patch', 'kappa_harris', 'N_keypoint', 'W_nms'
                'W_KLT', 'tol_KLT_bidir'
                'tol_E', 'tol_E_RANSAC_prob'
                'tol_TRI_mu', 'tol_TRI_rep'
    display : if True, show (a) tracked points in img1 and
              (b) a 3-D scatter of the triangulated landmarks

    Returns
    -------
    keypoints : list of (v, u) pixel positions of landmarks in img0
    p_W       : (3, N) world-frame 3-D positions of those landmarks
    R1        : (3, 3) rotation of camera-1 relative to camera-0
    T1        : (3, 1) translation of camera-1 relative to camera-0
    """
    # ── unpack parameters ─────────────────────────────────────
    W_harris      = param['W_harris_patch']
    kappa         = param['kappa_harris']
    N_kp          = param['N_keypoint']
    W_nms         = param['W_nms']
    W_KLT         = param['W_KLT']
    tol_bidir     = param['tol_KLT_bidir']
    tol_E         = param['tol_E']
    tol_E_prob    = param['tol_E_RANSAC_prob']
    tol_TRI_mu    = param['tol_TRI_mu']
    tol_TRI_rep   = param['tol_TRI_rep']

    # ── Step 1: detect keypoints in img0 ──────────────────────
    scores     = harris_corner(img0, W_harris, kappa)
    keypoints0 = select_keypoints(scores, N_kp, W_nms)
    p0         = np.array(keypoints0, dtype='float32').T
    p0         = p0[[1, 0], :]       # (row,col) → (u,v)

    # ── Step 2: KLT tracking img0 → img1 ──────────────────────
    p1, idx_track = KLT(img0, img1, p0, W_KLT, tol_bidir)
    p0 = p0[:, idx_track]
    p1 = p1[:, idx_track]

    # ── Step 3: Essential matrix (RANSAC) ─────────────────────
    E, mask_E = cv2.findEssentialMat(
        p0.T, p1.T, K, cv2.RANSAC, tol_E_prob, tol_E)
    idx_E = np.where(mask_E.ravel() > 0)[0]
    p0 = p0[:, idx_E];  p1 = p1[:, idx_E]

    # ── Step 4: recover pose ───────────────────────────────────
    _, R1, T1, mask_pose = cv2.recoverPose(E, p0.T, p1.T, K)
    idx_pose = np.where(mask_pose.ravel() > 0)[0]
    p0 = p0[:, idx_pose];  p1 = p1[:, idx_pose]

    # ── Step 5: robust triangulation ──────────────────────────
    M0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    M1 = K @ np.hstack([R1, T1])

    p_W_all, idx_tri, _, _ = triangulation_robust(
        hom(p0), hom(p1), M0, M1, tol_TRI_mu, tol_TRI_rep)

    # keep only triangulation inliers
    keypoints = [(vu[1], vu[0]) for vu in p0[:, idx_tri].T]
    p_W       = p_W_all[:, idx_tri]

    print(f"[Bootstrap] tracked={p0.shape[1]}  "
          f"triangulated={p_W.shape[1]}  "
          f"T={T1.flatten()}")

    # ── Optional display ───────────────────────────────────────
    if display:
        fig = plt.figure(dpi=150, figsize=(6, 10))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.imshow(img1, cmap='gray')
        ax1.axis('off')
        ax1.plot(p0[0, :], p0[1, :], 'b+', markersize=4, label='frame-0 kp')
        ax1.plot(p1[0, :], p1[1, :], 'r+', markersize=4, label='frame-1 kp')
        ax1.legend(fontsize=6)
        ax1.set_title(f'Bootstrap: {p_W.shape[1]} landmarks', fontsize=8)

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        ax2.plot(p_W[0, :], p_W[1, :], p_W[2, :], 'b.', markersize=1)
        ax2.set_xlim(-20, 20)
        ax2.set_ylim(-10,  5)
        ax2.set_zlim(-10, 50)
        ax2.set_xlabel('x', fontsize=7)
        ax2.set_ylabel('y', fontsize=7)
        ax2.set_zlabel('z', fontsize=7)
        ax2.view_init(elev=0., azim=-90)
        draw_frame(ax2, np.eye(3), np.zeros((3, 1)), axis_length=0.5)
        draw_frame(ax2, R1, T1, axis_length=0.5)
        plt.tight_layout()

    return keypoints, p_W, R1, T1
