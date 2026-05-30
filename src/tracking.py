"""
tracking.py
───────────
Feature detection and KLT optical-flow tracking.

Contents
  - harris_corner    : compute Harris corner response map
  - select_keypoints : NMS-based keypoint selection from Harris scores
  - KLT              : forward-backward KLT tracker with bi-directional check
"""

import numpy as np
import cv2
import scipy.signal
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# HARRIS CORNER DETECTION
# ─────────────────────────────────────────────────────────────

def harris_corner(img: np.ndarray, W: int, kappa: float) -> np.ndarray:
    """
    Compute the Harris corner response for every pixel.

    Parameters
    ----------
    img   : grayscale image (H, W) as float or uint8
    W     : half-size of the Harris summation patch  (patch = 2W+1 × 2W+1)
    kappa : Harris score parameter (typically 0.04 – 0.08)

    Returns
    -------
    harris_score : (H, W) float array, negative values clamped to 0,
                   border pixels within W of the edge set to 0
    """
    S_1 = np.array([1, 2, 1], dtype='float32')
    S_2 = np.array([-1, 0, 1], dtype='float32')

    I_x = scipy.signal.sepfir2d(img, S_2, S_1)
    I_y = scipy.signal.sepfir2d(img, S_1, S_2)

    I_x2 = I_x ** 2
    I_y2 = I_y ** 2
    I_xy = I_x * I_y

    filt = np.ones(2 * W + 1, dtype='float32')
    M_x2 = scipy.signal.sepfir2d(I_x2, filt, filt)
    M_y2 = scipy.signal.sepfir2d(I_y2, filt, filt)
    M_xy = scipy.signal.sepfir2d(I_xy, filt, filt)

    score = (M_x2 * M_y2 - M_xy ** 2) - kappa * (M_x2 + M_y2) ** 2

    # zero borders
    score[: W + 1, :]    = 0
    score[:, : W + 1]    = 0
    score[:, -(W + 1):]  = 0
    score[-(W + 1):, :]  = 0
    score[score < 0]     = 0

    return score


# ─────────────────────────────────────────────────────────────
# KEYPOINT SELECTION (NMS)
# ─────────────────────────────────────────────────────────────

def select_keypoints(score: np.ndarray,
                     N_keypoint: int,
                     W_nms: int) -> list:
    """
    Greedily select the N strongest keypoints with non-maximum suppression.

    Parameters
    ----------
    score      : (H, W) Harris score array (non-negative)
    N_keypoint : maximum number of keypoints to return
    W_nms      : suppression half-window; after selecting a peak, a square
                 of side (2*W_nms+1) centred on it is zeroed out

    Returns
    -------
    keypoints : list of (row, col) tuples in descending score order
    """
    score    = score.copy()          # don't modify caller's array
    n_rows, n_cols = score.shape
    keypoints = []

    for _ in range(N_keypoint):
        r, c = np.unravel_index(score.argmax(), score.shape)
        if score[r, c] == 0:
            break
        keypoints.append((r, c))
        r0 = max(r - W_nms, 0);      r1 = min(r + W_nms + 1, n_rows)
        c0 = max(c - W_nms, 0);      c1 = min(c + W_nms + 1, n_cols)
        score[r0:r1, c0:c1] = 0

    return keypoints


# ─────────────────────────────────────────────────────────────
# KLT OPTICAL-FLOW TRACKER
# ─────────────────────────────────────────────────────────────

def KLT(img_pre: np.ndarray,
        img_cur: np.ndarray,
        p_pre: np.ndarray,
        W: int = 7,
        tol_bidir: float = 1.0,
        display: bool = False) -> tuple:
    """
    Track 2-D points from img_pre to img_cur using pyramidal Lucas-Kanade,
    validated by a backward (bidirectional) consistency check.

    Parameters
    ----------
    img_pre   : previous grayscale frame
    img_cur   : current  grayscale frame
    p_pre     : (2, N) pixel coordinates in img_pre  (row 0 = u, row 1 = v)
    W         : KLT window half-size  (window = 2W+1 × 2W+1)
    tol_bidir : maximum allowed squared bidirectional error (pixels²)
    display   : if True, draw tracked points and flow vectors

    Returns
    -------
    p_cur       : (2, N) pixel coordinates in img_cur  (all N points,
                  including untracked ones — use index_track to filter)
    index_track : 1-D array of column indices that survived both forward
                  and backward tracking with error < tol_bidir
    """
    win = (2 * W + 1, 2 * W + 1)

    # forward pass
    p_cur_T, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
        img_pre, img_cur, p_pre.T, None, winSize=win)
    p_cur = p_cur_T.T

    # backward pass
    p_pre_rec_T, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
        img_cur, img_pre, p_cur.T, None, winSize=win)
    p_pre_recovered = p_pre_rec_T.T

    # bidirectional error (squared L2)
    err_bidir   = np.sum((p_pre - p_pre_recovered) ** 2, axis=0)
    index_track = np.where(
        (st_fwd.ravel() > 0) &
        (st_bwd.ravel() > 0) &
        (err_bidir < tol_bidir)
    )[0]

    if display:
        plt.figure(dpi=150)
        plt.imshow(img_cur, cmap='gray')
        plt.axis('off')
        for i in index_track:
            u_q, v_q = p_cur[0, i], p_cur[1, i]
            u_d, v_d = p_pre[0, i], p_pre[1, i]
            plt.plot(u_q, v_q, 'r+')
            plt.plot(u_d, v_d, 'b+')
            plt.plot([u_d, u_q], [v_d, v_q], 'r', linewidth=0.5)

    return p_cur, index_track
