"""
Shared helpers for lane perception (no ROS node in here, only maths).

Used by
  lane_detector.py              neural-network lane detector (runtime)
  training/collect_lane_dataset.py   auto-labelling of simulation images

Coordinate frames
  optical : z forward, x right, y down   (images, camera_info)
  base    : x forward, y left,  z up     (base_footprint, on the floor)
"""
import numpy as np

# The camera looks 5 deg down, so the horizon sits ~40 px above the image centre.
# Everything above this row is cave wall/ceiling: lane paint can only be below it.
HORIZON_ROW = 205


def white_mask(bgr, min_v=120, max_s=55, min_contrast=45, bright_v=150, min_area=40,
               min_extent=25):
    """
    Binary mask (uint8 0/255) of the white paint, from colour thresholds.

    White = bright, unsaturated *and* brighter than its surroundings (a top-hat
    filter, so far-away, dimmer paint still counts but bright rock does not).

    This is the *teacher* that labels the training images taken in the simulator;
    the neural network is what runs on the robot. Blobs smaller than `min_area`
    pixels or shorter than `min_extent` pixels are dropped (specular glints on the rock).
    """
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (21, 21)))
    white = hsv[..., 1] <= max_s
    # far paint: dim but brighter than its surroundings; near paint: simply very bright
    m = (white & (((v >= min_v) & (tophat >= min_contrast)) | (v >= bright_v))).astype(np.uint8) * 255
    m[:HORIZON_ROW, :] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        extent = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if stats[i, cv2.CC_STAT_AREA] >= min_area and extent >= min_extent:  # lines are long
            keep[labels == i] = 255
    return keep


def mask_to_polygons(mask, min_area=40, epsilon=1.5):
    """Contours of a binary mask as a list of Nx2 float arrays (pixels)."""
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, epsilon, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2).astype(np.float32))
    return polys


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pixels_to_ground(uv, K, R, t):
    """
    Project pixels onto the floor plane z = 0 of the base frame.

    uv : Nx2 pixel coordinates (u = column, v = row)
    K  : camera matrix [fx 0 cx; 0 fy cy; 0 0 1] as a flat list of 9
    R,t: rotation (3x3) and translation (3,) of optical -> base
    Returns an Nx2 array (x forward, y left) in metres, NaN where the ray does not
    hit the floor (pixel above the horizon).
    """
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    rays_cam = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))], axis=1)
    rays = rays_cam @ R.T                      # rays in the base frame
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -t[2] / rays[:, 2]                 # t_z + s * ray_z = 0
    s = np.where((rays[:, 2] < -1e-3) & (s > 0), s, np.nan)
    pts = t[None, :] + s[:, None] * rays
    return pts[:, :2]


def ground_to_pixels(xy, K, R, t):
    """Inverse of pixels_to_ground (for drawing the lane on the camera image)."""
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    p = np.hstack([xy, np.zeros((len(xy), 1))])
    pc = (p - t[None, :]) @ R                  # = R^T (p - t)
    ok = pc[:, 2] > 0.05
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * pc[:, 0] / pc[:, 2] + cx
        v = fy * pc[:, 1] / pc[:, 2] + cy
    return np.stack([u, v], axis=1), ok


def scanline_runs(mask, rows, gap=12):
    """
    For each requested image row: the list of white runs as (centre column, width).
    Two lines on the floor give two runs per row (left one first).
    """
    out = {}
    for r in rows:
        cols = np.flatnonzero(mask[r] > 0)
        if cols.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(cols) > gap)
        runs = np.split(cols, breaks + 1)
        out[r] = [(float(run.mean()), int(run[-1] - run[0] + 1)) for run in runs if run.size >= 2]
    return out


def lane_centre_from_runs(runs_by_row, K, R, t, lane_width, x_min=0.4, x_max=6.0,
                          prev_centre=None):
    """
    Turn white runs into lane geometry on the floor.

    Per row: two runs -> left/right line, midpoint = centre. One run -> assigned to
    the left or right line (by which side of the previous centre it falls on) and
    shifted by half a lane width along the line's own normal.

    Returns dict(left=Nx2, right=Nx2, centre=Mx2) in the base frame (x fwd, y left).
    """
    left, right, centre_pts = [], [], []
    singles = []
    for v, runs in runs_by_row.items():
        us = [r[0] for r in runs]
        g = pixels_to_ground([(u, v) for u in us], K, R, t)
        if np.isnan(g).any():
            continue
        in_range = [(x_min <= p[0] <= x_max) for p in g]
        if len(runs) >= 2:
            pl, pr = g[0], g[-1]
            sep = pl[1] - pr[1]
            if (in_range[0] and in_range[-1] and 0.55 * lane_width < sep < 1.6 * lane_width):
                left.append(pl)
                right.append(pr)
                centre_pts.append(0.5 * (pl + pr))
                continue
        for p, ok in zip(g, in_range):
            if ok:
                singles.append(p)

    left, right = list(left), list(right)
    for p in singles:
        cy = float(np.polyval(prev_centre, p[0])) if prev_centre is not None else 0.0
        (left if p[1] > cy else right).append(p)

    left = np.array(left).reshape(-1, 2)
    right = np.array(right).reshape(-1, 2)

    # rows where only one line was seen: shift along that line's normal
    centre_pts = [np.asarray(c) for c in centre_pts]
    n_two = len(centre_pts)
    for pts, sign in ((left, -1.0), (right, +1.0)):
        if len(pts) < 4:
            continue
        m = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
        n = np.array([-m, 1.0]) / np.hypot(1.0, m)   # left-pointing normal
        for p in pts:
            centre_pts.append(p + sign * n * lane_width / 2.0)
    centre = np.array(centre_pts).reshape(-1, 2)
    return dict(left=left, right=right, centre=centre, n_two=n_two)


def fit_centre(centre, min_points=6):
    """
    Polynomial y(x) through the centre points (x forward, y left).
    Quadratic when the points span >= 1.5 m, otherwise a straight line.
    Returns (coefficients, x_min, x_max) or None.
    """
    if len(centre) < min_points:
        return None
    x, y = centre[:, 0], centre[:, 1]
    span = x.max() - x.min()
    if span < 0.3:
        return None
    deg = 2 if span >= 1.5 else 1
    coef = np.polyfit(x, y, deg)
    return coef, float(x.min()), float(x.max())
