"""
utils.py
────────
Shared data structures and helper functions used across the VO pipeline.

Contents
  - state      : NamedTuple holding current landmarks and pose history
  - candidate  : NamedTuple holding candidate keypoints not yet triangulated
  - hom        : convert 2-D/3-D array to homogeneous coordinates
  - deh        : convert homogeneous array back to Euclidean coordinates
  - hat        : skew-symmetric (hat) operator for a 3-vector
  - read_frame_at   : seek-and-read one video frame with undistortion
  - read_next_frame : read the next frame in sequence with undistortion
"""

import numpy as np
import cv2
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────
# VO STATE STRUCTURES
# ─────────────────────────────────────────────────────────────

class state(NamedTuple):
    """
    Active VO state passed between frames.

    keypoints   : list of (v, u) pixel positions of tracked landmarks
    p_W         : (3, N) world-frame 3-D positions of landmarks
    T_W_history : list of (3,1) camera-centre positions in world frame
    """
    keypoints:   list
    p_W:         np.ndarray
    T_W_history: list


class candidate(NamedTuple):
    """
    Candidate keypoints that are being tracked but not yet triangulated.

    keypoints     : list of (v, u) current pixel positions
    keypoints_org : list of (v, u) pixel positions at first observation
    R_org         : list of 3×3 rotation matrices at first observation
    T_org         : list of (3,1) translation vectors at first observation
    """
    keypoints:     list
    keypoints_org: list
    R_org:         list
    T_org:         list


# ─────────────────────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────────────────────

def hom(p: np.ndarray) -> np.ndarray:
    """
    Append a row of ones to convert Euclidean → homogeneous coordinates.

    Parameters
    ----------
    p : (2, N) or (3, N) array

    Returns
    -------
    (3, N) or (4, N) array
    """
    assert p.shape[0] in (2, 3), "Input must be 2×N or 3×N"
    return np.concatenate((p, np.ones((1, p.shape[1]))), axis=0)


def deh(p_h: np.ndarray) -> np.ndarray:
    """
    Divide by the last row to convert homogeneous → Euclidean coordinates.

    Parameters
    ----------
    p_h : (3, N) or (4, N) array

    Returns
    -------
    (2, N) or (3, N) array
    """
    n = p_h.shape[0] - 1
    assert n in (2, 3), "Input must be 3×N or 4×N"
    return p_h[:n, :] / p_h[n, :]


def hat(s: np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix (hat operator) for a 3-vector s.

    Parameters
    ----------
    s : array-like of length 3

    Returns
    -------
    (3, 3) skew-symmetric matrix S such that S @ v = s × v
    """
    S = np.zeros((3, 3))
    S[1, 2] = -s[0];  S[2, 1] =  s[0]
    S[0, 2] =  s[1];  S[2, 0] = -s[1]
    S[0, 1] = -s[2];  S[1, 0] =  s[2]
    return S


# ─────────────────────────────────────────────────────────────
# VIDEO FRAME READERS
# ─────────────────────────────────────────────────────────────

def read_frame_at(cap: cv2.VideoCapture,
                  frame_number: int,
                  K: np.ndarray,
                  dist_coeffs: np.ndarray) -> tuple:
    """
    Seek to frame_number and return an undistorted (gray, rgb) pair.

    Parameters
    ----------
    cap          : cv2.VideoCapture object (already opened)
    frame_number : 0-based frame index to seek to
    K            : (3, 3) camera intrinsic matrix
    dist_coeffs  : distortion coefficients from calibration

    Returns
    -------
    (gray, rgb) as uint8 numpy arrays, or (None, None) on failure
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None, None
    frame = cv2.undistort(frame, K, dist_coeffs)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return gray, rgb


def read_next_frame(cap: cv2.VideoCapture,
                    K: np.ndarray,
                    dist_coeffs: np.ndarray) -> tuple:
    """
    Read the next frame from cap and return an undistorted (gray, rgb) pair.

    Parameters
    ----------
    cap         : cv2.VideoCapture object positioned at the desired frame
    K           : (3, 3) camera intrinsic matrix
    dist_coeffs : distortion coefficients from calibration

    Returns
    -------
    (gray, rgb) as uint8 numpy arrays, or (None, None) on failure
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        return None, None
    frame = cv2.undistort(frame, K, dist_coeffs)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return gray, rgb
