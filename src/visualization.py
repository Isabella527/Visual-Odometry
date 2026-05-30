"""
visualization.py
────────────────
All rendering and video-output functions for the VO pipeline.

Contents
  - render_img_panel      : overlay tracked/candidate points on a colour frame
  - render_topdown_fixed  : fixed-origin, fixed-scale top-down X-Z map
  - render_3d_view        : rotating perspective 3-D view (used in prob2)
  - write_video_frame     : compose image panel + map side-by-side → VideoWriter
  - draw_frame            : draw RGB axes of a camera frame in a 3-D matplotlib axes
  - save_bootstrap_figure : save the bootstrap output figure to disk
  - save_trajectory_map   : save the final X-Z and X-Y trajectory plots
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.spatial.distance


# ─────────────────────────────────────────────────────────────
# IMAGE PANEL
# ─────────────────────────────────────────────────────────────

def render_img_panel(img_rgb: np.ndarray,
                     S_cur,
                     C_cur,
                     i_frame: int,
                     prev_vis_pts: list = None) -> tuple:
    """
    Draw tracked landmarks (red) and candidate points (blue) on the frame,
    plus short motion-track lines if previous positions are supplied.

    Parameters
    ----------
    img_rgb       : (H, W, 3) RGB image
    S_cur         : current state NamedTuple
    C_cur         : current candidate NamedTuple
    i_frame       : frame index (shown as text overlay)
    prev_vis_pts  : list of (u, v) from the previous frame for track lines

    Returns
    -------
    vis           : annotated RGB image
    cur_pts       : list of (u, v) for the current landmarks
                    (pass back as prev_vis_pts on the next call)
    """
    vis     = img_rgb.copy()
    cur_pts = [(int(u), int(v)) for (v, u) in S_cur.keypoints]

    # motion track lines
    if prev_vis_pts is not None:
        n = min(len(prev_vis_pts), len(cur_pts))
        for i in range(n):
            p1, p2 = prev_vis_pts[i], cur_pts[i]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            if dx * dx + dy * dy < 400:          # skip large jumps
                cv2.line(vis, p1, p2, (255, 0, 0), 1, cv2.LINE_AA)

    # tracked landmarks (red)
    for (v, u) in S_cur.keypoints:
        cv2.circle(vis, (int(u), int(v)), 2, (0, 0, 255), -1, cv2.LINE_AA)

    # candidates (blue)
    for (v, u) in C_cur.keypoints:
        cv2.circle(vis, (int(u), int(v)), 1, (255, 0, 0), -1, cv2.LINE_AA)

    # HUD
    n_lm = S_cur.p_W.shape[1]
    n_c  = len(C_cur.keypoints)
    cv2.putText(vis,
                f"F:{i_frame:04d}  LM:{n_lm}  C:{n_c}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 200), 2, cv2.LINE_AA)

    return vis, cur_pts


# ─────────────────────────────────────────────────────────────
# FIXED-SCALE TOP-DOWN MAP
# ─────────────────────────────────────────────────────────────

def render_topdown_fixed(T_W_history: list,
                         p_W_accum: np.ndarray,
                         map_size: int = 500,
                         scale: float = 10.0) -> np.ndarray:
    """
    Render a top-down X-Z map with a fixed world origin and fixed scale.
    This avoids the "frozen map" illusion that auto-scaling can create.

    Parameters
    ----------
    T_W_history : list of (3,1) camera-centre positions in world frame
    p_W_accum   : (3, M) accumulated world-frame landmarks
    map_size    : canvas side length in pixels
    scale       : pixels per world unit (tune to fit your scene)

    Returns
    -------
    canvas : (map_size, map_size, 3) BGR image
    """
    canvas = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    traj   = np.array([t.flatten() for t in T_W_history])

    # world origin is fixed at bottom-centre of the canvas
    origin_px = (map_size // 2, int(map_size * 0.85))

    def w2p(x, z):
        u = int(origin_px[0] + x * scale)
        v = int(origin_px[1] - z * scale)   # z forward → upward in image
        return u, v

    # landmarks (gray dots)
    for i in range(p_W_accum.shape[1]):
        u, v = w2p(p_W_accum[0, i], p_W_accum[2, i])
        if 0 <= u < map_size and 0 <= v < map_size:
            cv2.circle(canvas, (u, v), 1, (70, 70, 70), -1)

    # trajectory (green line)
    for i in range(1, len(traj)):
        p1 = w2p(traj[i - 1, 0], traj[i - 1, 2])
        p2 = w2p(traj[i,     0], traj[i,     2])
        if (0 <= p1[0] < map_size and 0 <= p1[1] < map_size and
                0 <= p2[0] < map_size and 0 <= p2[1] < map_size):
            cv2.line(canvas, p1, p2, (0, 200, 0), 1)

    # start (orange) and current camera position (red)
    sp = w2p(traj[0,  0], traj[0,  2])
    cp = w2p(traj[-1, 0], traj[-1, 2])
    cv2.circle(canvas, sp, 5, (200, 100,   0), -1)
    cv2.circle(canvas, cp, 6, (0,     0, 255), -1)

    # axis arrows
    cv2.arrowedLine(canvas, origin_px,
                    (origin_px[0] + 40, origin_px[1]),
                    (100, 100, 100), 1, tipLength=0.2)
    cv2.arrowedLine(canvas, origin_px,
                    (origin_px[0], origin_px[1] - 40),
                    (100, 100, 100), 1, tipLength=0.2)
    cv2.putText(canvas, 'X', (origin_px[0] + 44, origin_px[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, 'Z', (origin_px[0] - 10, origin_px[1] - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # scale bar  (5 world units)
    bar_px = int(5 * scale)
    cv2.line(canvas, (10, map_size - 15), (10 + bar_px, map_size - 15),
             (180, 180, 180), 1)
    cv2.putText(canvas, '5 units', (10, map_size - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    cv2.putText(canvas, 'Top-down X-Z (fixed scale)', (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    return canvas


# ─────────────────────────────────────────────────────────────
# ROTATING 3-D VIEW  (prob2 style)
# ─────────────────────────────────────────────────────────────

def render_3d_view(T_W_history: list,
                   p_W: np.ndarray,
                   angle: float = 0.0,
                   map_size: int = 500) -> np.ndarray:
    """
    Render a fake-perspective 3-D top-down/side view that rotates around
    the scene as the video progresses.

    Parameters
    ----------
    T_W_history : list of (3,1) camera-centre positions in world frame
    p_W         : (3, M) current landmark cloud
    angle       : rotation angle in degrees (increment each frame for spin)
    map_size    : canvas side length in pixels

    Returns
    -------
    canvas : (map_size, map_size, 3) BGR image
    """
    canvas = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    traj   = np.array([t.flatten() for t in T_W_history])

    theta = np.radians(angle)
    Ry    = np.array([[ np.cos(theta), 0, np.sin(theta)],
                      [ 0,             1, 0            ],
                      [-np.sin(theta), 0, np.cos(theta)]])

    P_rot    = Ry @ p_W
    traj_rot = (Ry @ traj.T).T

    def project(pt):
        x, y, z = pt
        z += 40
        if z <= 1:
            return None
        f = 220 / z
        return int(map_size / 2 + x * f), int(map_size / 2 - y * f)

    # landmarks (depth-shaded gray)
    step = max(1, P_rot.shape[1] // 3000)
    for i in range(0, P_rot.shape[1], step):
        px = project(P_rot[:, i])
        if px is None:
            continue
        u, v = px
        if 0 <= u < map_size and 0 <= v < map_size:
            depth      = float(P_rot[2, i])
            brightness = max(50, 255 - int(depth * 3))
            cv2.circle(canvas, (u, v), 1,
                       (brightness, brightness, brightness), -1)

    # trajectory (green)
    for i in range(1, len(traj_rot)):
        p1 = project(traj_rot[i - 1])
        p2 = project(traj_rot[i])
        if p1 is not None and p2 is not None:
            cv2.line(canvas, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

    # current camera (red dot)
    cp = project(traj_rot[-1])
    if cp is not None:
        cv2.circle(canvas, cp, 6, (0, 0, 255), -1)

    cv2.putText(canvas, '3D VIEW', (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


# ─────────────────────────────────────────────────────────────
# VIDEO FRAME COMPOSER
# ─────────────────────────────────────────────────────────────

def write_video_frame(writer: cv2.VideoWriter,
                      vis_rgb: np.ndarray,
                      map_bgr: np.ndarray,
                      out_w: int,
                      out_h: int) -> None:
    """
    Compose the image panel and the map side-by-side and write to the
    VideoWriter.

    Parameters
    ----------
    writer  : open cv2.VideoWriter
    vis_rgb : annotated frame in RGB
    map_bgr : map canvas in BGR
    out_w   : total output width in pixels
    out_h   : total output height in pixels
    """
    vis_bgr   = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    cam_w     = int(out_w * 0.68)
    map_w     = out_w - cam_w
    vis_res   = cv2.resize(vis_bgr, (cam_w, out_h))
    map_res   = cv2.resize(map_bgr, (map_w, out_h))
    canvas    = np.hstack([vis_res, map_res])
    writer.write(canvas)


# ─────────────────────────────────────────────────────────────
# MATPLOTLIB HELPERS
# ─────────────────────────────────────────────────────────────

def draw_frame(ax, R: np.ndarray, T: np.ndarray,
               axis_length: float = 1.0,
               line_width:  float = 1.0) -> None:
    """
    Draw the three axes (R, G, B) of a camera frame in a 3-D matplotlib axes.

    Parameters
    ----------
    ax          : matplotlib Axes3D
    R           : (3, 3) rotation matrix
    T           : (3, 1) or (3,) translation vector  (camera frame)
    axis_length : length of each axis line in world units
    line_width  : line width in points
    """
    C      = -R.T @ T.flatten()
    colors = ('r', 'g', 'b')
    for i in range(3):
        ax.plot([C[0], C[0] + axis_length * R[i, 0]],
                [C[1], C[1] + axis_length * R[i, 1]],
                [C[2], C[2] + axis_length * R[i, 2]],
                colors[i], linewidth=line_width)


def save_bootstrap_figure(out_path: str = 'prob2_bootstrap.png') -> None:
    """Save the currently active matplotlib figure as the bootstrap image."""
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")


def save_trajectory_map(T_W_history: list,
                        p_W_accum: np.ndarray,
                        out_path: str = 'prob2_map.png') -> None:
    """
    Save a two-panel trajectory map:  top-down (X-Z) and side view (X-Y).

    Parameters
    ----------
    T_W_history : list of (3,1) camera-centre positions
    p_W_accum   : (3, M) accumulated world-frame landmarks
    out_path    : output file path
    """
    traj = np.array([t.flatten() for t in T_W_history])
    lm   = p_W_accum

    # remove duplicate poses (frozen frames)
    if len(traj) > 1:
        diff = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        keep = np.insert(diff > 1e-4, 0, True)
        traj = traj[keep]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(lm[0, :], lm[2, :], s=1, c='gray', alpha=0.3, label='Landmarks')
    ax.plot(traj[:, 0], traj[:, 2], 'g-', lw=1.5, label='Trajectory')
    ax.scatter(traj[0,  0], traj[0,  2], s=80, c='blue', zorder=5, label='Start')
    ax.scatter(traj[-1, 0], traj[-1, 2], s=80, c='red',  zorder=5, label='End')
    ax.set_xlabel('X'); ax.set_ylabel('Z')
    ax.set_title('Top-down view (X-Z)')
    ax.set_aspect('equal'); ax.legend()

    ax2 = axes[1]
    ax2.scatter(lm[0, :], lm[1, :], s=1, c='gray', alpha=0.3)
    ax2.plot(traj[:, 0], traj[:, 1], 'g-', lw=1.5)
    ax2.scatter(traj[0,  0], traj[0,  1], s=80, c='blue', zorder=5)
    ax2.scatter(traj[-1, 0], traj[-1, 1], s=80, c='red',  zorder=5)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_title('Side view (X-Y)')
    ax2.set_aspect('equal')

    plt.suptitle('Visual Odometry — Complete Trajectory', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved {out_path}")
