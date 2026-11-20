"""
measure_stems.py - stem diameter and internode length from the white-background images.

For every image (one stalk, cut into sections and laid side by side):
  1. each section is segmented, and a smooth centerline is built from its skeleton
     (longest path, end hooks trimmed, smoothed, ends extended to the cut faces);
  2. the reviewed node detections are placed on the centerline, the cut ends are added
     as internode boundaries, nodes closer than 2.0 cm are merged, and each internal
     node is moved to the nearest diameter peak (the node collar) within 0.5 cm;
  3. the diameter is measured perpendicular to the centerline at every node and at
     10, 30, 50, 70 and 90 % of the curved length of each internode (D1-D5);
  4. internode lengths are measured along the centerline (and as straight and vertical
     distances for comparison), and section lengths are added up to give positions on
     the original stalk;
  5. as a baseline, diameters are also measured every 5 cm along the stalk without
     using the nodes.

Scale: each section uses the pixel-per-mm value at its own position in the photo,
from <session>/calibration.csv (calibration.py).
Nodes: label files reviewed by hand (first line "# reviewed", from review_nodes.py) are
used as saved; for unreviewed files, detections closer than 250 px are merged first.

Results go to <session>/analysis_output_comparison/<image>/ (Excel tables and figures).
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import morphology, measure
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.worksheet.hyperlink import Hyperlink
import matplotlib.pyplot as plt
import os

from calibration import load_image_calibration, scale_at
from config import OUTPUT_DIR

# ── SETTINGS ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(OUTPUT_DIR)          # folder with the session (batch_*) folders

# Fallback scale, used only when an image has no row in calibration.csv
# or when USE_RULER_CALIBRATION = False.
DEFAULT_PX_PER_MM_VERT = 8.319
DEFAULT_PX_PER_MM_DIAM = 8.319
px_per_mm_vert = DEFAULT_PX_PER_MM_VERT
px_per_mm_diam = DEFAULT_PX_PER_MM_DIAM

# Each section gets the px/mm at its own position in the photo, because a slightly
# tilted camera makes one side of the photo look bigger.
USE_RULER_CALIBRATION = True
CAL = None   # calibration row of the image being processed (set in batch_process_stems)

MIN_NODE_GAP_CM = 2.0
MIN_VALID_ROWS_FOR_NODE_SNAP = 3
NODE_REFINE_CM = 0.5       # a YOLO node may be moved at most this far, to the nearest node-collar peak (0 = never move)
NODE_PEAK_MIN_PCT = 2.0    # a collar peak must stand out by this % of the local diameter

# Tangent settings
TANGENT_NEIGHBOR_RADIUS = 18
TANGENT_SAMPLE_HALF_WINDOW = 3
RAY_STEP = 0.5
MAX_RAY_STEPS = 2000

# Centerline settings: all points are placed ON the stem centerline
# (longest skeleton path, side branches removed, smoothed) and the tangent is
# taken from this smooth centerline, so points never fall outside a curved stem.
CENTERLINE_TRIM_RADII = 2.0   # skeleton end hooks (into cut-end corners) are trimmed: this many stem radii
CENTERLINE_TRIM_MIN_MM = 15.0 # ... and at least this length at each end
END_DIRECTION_MM = 10.0       # the trimmed ends are extended straight, direction from the last 10 mm
CENTERLINE_SMOOTH_MM = 8.0    # moving-average window along the centerline
END_MEASURE_MM = 4.0          # no cross-section closer than this to a cut end

# Five representative diameter positions are sampled within each internode.
INTERNODE_SAMPLE_FRACS = [0.10, 0.30, 0.50, 0.70, 0.90]

# Drawing settings (paper-friendly)
SHOW_SKELETON = False
SHOW_CONTOUR = True

# Mask smoothing: removes thin boundary spurs (residual background fringe,
# leaf-sheath fragments, adaptive-threshold texture) before the skeleton,
# centerline and diameters are computed. The drawn contour is the measured mask.
SMOOTH_MASK = True
SMOOTH_RADIUS_MM = 0.6     # opening radius; keep well below half the thinnest stem width

COLOR_CONTOUR = (0, 255, 0)        # green contour
COLOR_SKELETON = (255, 255, 0)      # cyan skeleton
COLOR_TANGENT_METHOD = (0, 215, 255) # bright yellow-green diameter lines
COLOR_NODE_TEXT = (0, 255, 0)       # green node markers
COLOR_DIAM_TEXT = (0, 0, 255)       # red diameter text

LINE_THICKNESS = 7
TEXT_SCALE = 2.3
TEXT_THICKNESS = 5
NODE_TEXT_SCALE = 2.4
NODE_TEXT_THICKNESS = 5
LABEL_X_OFFSET = 32

# ── NODE-GUIDED vs FIXED-SPACING SETTINGS ───────────────────────────────
# Method A = node-guided sampling: 5 points per internode.
# Method B2 = fixed-coordinate baseline: points every FIXED_SPACING_CM (5 cm) along the
#            curved stem centerline, measured from the ORIGINAL STALK BASE
#            (cut sections keep their stalk positions). Nodes are not used at
#            all, neither for the positions nor for the number of points, so
#            A and B2 can have different point counts (reported in the summary).
# Both methods use the same measure_robust_tangent_diameter(); only the
# sampling positions differ.
FIXED_SPACING_CM = 5.0    # Method B2 spacing along the stalk centerline (cm)
B2_LABEL = f"B2_fixed_{FIXED_SPACING_CM:g}cm"   # method name used in Excel and figures
# Extra fixed-spacing run at a density close to Method A (images only, no Excel):
# shows the node-guidance effect when both methods have a similar number of points.
EXTRA_FIXED_SPACING_CM = 2.5
FIGURE_DPI = 300                 # all saved figures carry this DPI
CENTERLINE_DRAW_THICKNESS = 12   # px, centerline in the centerline figure
COLOR_CENTERLINE_DRAW = (0, 0, 230)   # red (BGR)
INTERNODE_COLORS = [(0, 0, 220), (220, 110, 0), (160, 30, 160), (0, 140, 255), (0, 150, 0)]   # BGR, one per internode
SAMPLES_PER_INTERNODE = len(INTERNODE_SAMPLE_FRACS)
NODE_ZONE_CM = 1.0        # a point within this vertical distance of a node is "near node"
MIDPOINT_WINDOW = 0.10    # internode midpoint covered if a point has |r - 0.5| <= this

# B2 bookkeeping, written to Excel: where each cut section sits on the original
# stalk, and which fixed points were skipped because they fell next to a cut end.
B2_SECTION_LOG = []
NODE_REFINE_LOG = []   # what happened to each internal YOLO node
JOINT_END_MM = (8.0, 10.0, 12.0, 14.0)   # end diameter = median of cross-sections this far from the cut end
JOINT_TOL_PCT = 10.0                     # joint flagged when P(k) top and P(k+1) bottom differ by more
B2_SKIPPED_LOG = []
# Internode lengths by 3 methods, and every point's position on the piece and on the stalk.
INTERNODE_LENGTH_LOG = []
POSITION_LOG = []

COLOR_FIXED_METHOD = (0, 140, 255)     # orange (BGR) fixed-spacing lines
COLOR_FIXED_NEAR_NODE = (255, 0, 255)  # magenta: fixed points inside the node zone

COMPARISON_OUTPUT_SUBDIR = "analysis_output_comparison"   # results folder inside each session

# Sessions listed in this file are skipped on later runs (delete a line to redo a session).
comparison_log = os.path.join(BASE_DIR, "comparison_completed_batches.txt")

completed_comparison_batches = set()
if os.path.exists(comparison_log):
    with open(comparison_log, "r") as f:
        completed_comparison_batches = set(line.strip() for line in f)


# ── HELPERS ──────────────────────────────────────────────────────────────
def load_image_gray(path):
    """Load an image as normalized 8-bit grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def sharpen_unsharp_mask(img, radius=3, amount=1.5):
    """Apply mild unsharp masking before segmentation."""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def load_node_labels(label_file, img_shape, distance_thresh=250):
    """Load YOLO-format node labels and merge nearby duplicate detections.

    Labels checked by hand in review_nodes.py start with "# reviewed".
    Those nodes were confirmed one by one (added / deleted by hand), so the
    250-px clustering is skipped for them. (The 2.0-cm node-gap rule in
    process_single_stem still applies.)
    Within a cluster the detection with the tallest box (h) is kept.
    """
    img_h, img_w = img_shape
    nodes = []
    confs = []

    if not Path(label_file).exists():
        return np.zeros((0, 2), dtype=int)

    with open(label_file) as f:
        reviewed = f.readline().startswith("# reviewed")

    with open(label_file) as f:
        for line in f:
            parts = line.split()
            if line.startswith("#") or len(parts) < 5:
                continue
            cls, x_rel, y_rel, w_rel, h_rel = map(float, parts[:5])
            nodes.append((int(x_rel * img_w), int(y_rel * img_h)))
            confs.append(h_rel)

    if not nodes:
        return np.zeros((0, 2), dtype=int)

    nodes = np.array(nodes, int)
    confs = np.array(confs)
    if reviewed:
        print(f"   using hand-reviewed nodes ({len(nodes)}) from {Path(label_file).name}")
        return nodes

    filtered_nodes = []
    used = set()

    for i, pt in enumerate(nodes):
        if i in used:
            continue

        cluster = [i]
        for j, other in enumerate(nodes):
            if j != i and j not in used:
                if np.linalg.norm(pt - other) < distance_thresh:
                    cluster.append(j)

        best_idx = max(cluster, key=lambda idx: confs[idx])
        filtered_nodes.append(tuple(nodes[best_idx]))
        used.update(cluster)

    return np.array(filtered_nodes, int)


def skeletonize_region(region_mask):
    """Convert a binary stem region to a one-pixel-wide centerline."""
    try:
        from cv2.ximgproc import thinning
        return thinning(region_mask, cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        return morphology.skeletonize(region_mask > 0).astype(np.uint8) * 255


def connect_and_fill(binary_img, connect_dist=15):
    """Close small gaps and fill holes in the segmented stem mask."""
    binary = (binary_img > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (connect_dist, connect_dist))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    inv = cv2.bitwise_not(connected)
    h, w = inv.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = inv.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = connected | flood_inv

    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return filled


def extract_final_contours(binary_filled, min_area=500):
    """Extract smoothed external contours from the filled stem mask."""
    contours, _ = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    smooth_contours = [cv2.approxPolyDP(cnt, epsilon=1.0, closed=True) for cnt in contours]
    return smooth_contours


def snap_to_skeleton_ywise(x0, y0, skeleton_mask):
    """Move a point horizontally to the nearest skeleton pixel in the same row."""
    h, w = skeleton_mask.shape
    y0 = int(np.clip(y0, 0, h - 1))
    row = skeleton_mask[y0, :]
    x_coords = np.where(row > 0)[0]
    if len(x_coords) == 0:
        return int(x0), int(y0)
    nearest_x = x_coords[np.argmin(np.abs(x_coords - x0))]
    return int(nearest_x), int(y0)


def snap_to_skeleton_nearest(x0, y0, skeleton_mask, search_radius=20):
    """Move a point to the nearest skeleton pixel within a local search window."""
    h, w = skeleton_mask.shape
    x0 = int(np.clip(round(x0), 0, w - 1))
    y0 = int(np.clip(round(y0), 0, h - 1))

    y1 = max(0, y0 - search_radius)
    y2 = min(h, y0 + search_radius + 1)
    x1 = max(0, x0 - search_radius)
    x2 = min(w, x0 + search_radius + 1)

    patch = skeleton_mask[y1:y2, x1:x2]
    ys, xs = np.where(patch > 0)
    if len(xs) == 0:
        return int(x0), int(y0)

    xs = xs + x1
    ys = ys + y1
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    idx = np.argmin(d2)
    return int(xs[idx]), int(ys[idx])


# ── TANGENT-BASED DIAMETER MEASUREMENT ──────────────────────────────────
# Diameter is measured perpendicular to the local stem centerline.
# The local tangent direction is estimated from nearby skeleton pixels.
# A normal line is then traced to both stem boundaries, and the distance
# between the two boundary points is reported as the stem diameter.
def collect_local_skeleton_points(skeleton_mask, x0, y0, radius=TANGENT_NEIGHBOR_RADIUS):
    """Return skeleton pixels near the requested measurement point.

    These local centerline pixels are used to estimate stem orientation.
    A local neighborhood is used because stalks may curve along their length.
    """
    h, w = skeleton_mask.shape
    x0 = int(np.clip(round(x0), 0, w - 1))
    y0 = int(np.clip(round(y0), 0, h - 1))

    y1 = max(0, y0 - radius)
    y2 = min(h, y0 + radius + 1)
    x1 = max(0, x0 - radius)
    x2 = min(w, x0 + radius + 1)

    patch = skeleton_mask[y1:y2, x1:x2]
    ys, xs = np.where(patch > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    xs = xs + x1
    ys = ys + y1
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    keep = d2 <= radius * radius

    xs = xs[keep]
    ys = ys[keep]

    return np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])


def estimate_local_tangent_direction(skeleton_mask, x0, y0, radius=TANGENT_NEIGHBOR_RADIUS):
    """Estimate the local stem tangent using PCA on nearby skeleton pixels.

    The first principal component gives the dominant centerline direction
    around the measurement point.
    """
    x0, y0 = snap_to_skeleton_nearest(x0, y0, skeleton_mask, search_radius=20)
    pts = collect_local_skeleton_points(skeleton_mask, x0, y0, radius=radius)

    if len(pts) < 3:
        return np.array([0.0, 1.0], dtype=np.float32)

    pts_centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, np.argmax(eigvals)].astype(np.float32)

    norm = np.linalg.norm(tangent)
    if norm < 1e-8:
        return np.array([0.0, 1.0], dtype=np.float32)

    tangent = tangent / norm

    if tangent[1] < 0:
        tangent = -tangent

    return tangent.astype(np.float32)


def trace_to_stem_boundary(mask, x0, y0, dx, dy, step=RAY_STEP, max_steps=MAX_RAY_STEPS):
    """Trace a ray from the centerline until it leaves the stem mask.

    The last location inside the mask is used as the boundary point.
    """
    h, w = mask.shape
    x, y = float(x0), float(y0)
    last_inside = (x, y)

    for _ in range(max_steps):
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h or mask[yi, xi] == 0:
            break
        last_inside = (x, y)
        x += dx * step
        y += dy * step

    return last_inside


def measure_normal_diameter(region_mask_filled, skeleton, x0, y0):
    """Measure diameter along the normal to the local stem centerline.

    The tangent direction is rotated by 90 degrees to form the cross-section
    direction. The ray is traced in both directions to the stem boundary.
    """
    x0, y0 = snap_to_skeleton_nearest(x0, y0, skeleton, search_radius=20)
    tangent = estimate_local_tangent_direction(skeleton, x0, y0, radius=TANGENT_NEIGHBOR_RADIUS)
    tx, ty = float(tangent[0]), float(tangent[1])

    nx, ny = -ty, tx
    nrm = np.hypot(nx, ny)
    if nrm < 1e-8:
        nx, ny = 1.0, 0.0
    else:
        nx, ny = nx / nrm, ny / nrm

    boundary_positive = trace_to_stem_boundary(region_mask_filled, x0, y0, nx, ny)
    boundary_negative = trace_to_stem_boundary(region_mask_filled, x0, y0, -nx, -ny)

    diameter_px = float(np.hypot(boundary_positive[0] - boundary_negative[0], boundary_positive[1] - boundary_negative[1]))

    return {
        "diameter_px": diameter_px,
        "pt1": boundary_negative,
        "pt2": boundary_positive,
        "center": (float(x0), float(y0)),
        "tangent": (tx, ty),
        "normal": (nx, ny)
    }


def measure_robust_tangent_diameter(region_mask_filled, skeleton, x0, y0, half_window=TANGENT_SAMPLE_HALF_WINDOW, centerline=None):
    """Compute a stable tangent-based diameter near one point.

    Multiple nearby cross-sections are sampled along the local tangent. The
    cross-section closest to the median diameter is retained to reduce the
    effect of small mask irregularities.
    When a centerline is given (normal use), the point is placed on the smooth
    centerline and the tangent is taken from it (see measure_on_centerline).
    """
    if centerline is not None:
        return measure_on_centerline(region_mask_filled, centerline, x0, y0, half_window)
    x0, y0 = snap_to_skeleton_nearest(x0, y0, skeleton, search_radius=20)
    tangent = estimate_local_tangent_direction(skeleton, x0, y0, radius=TANGENT_NEIGHBOR_RADIUS)
    tx, ty = float(tangent[0]), float(tangent[1])

    samples = []
    for s in range(-half_window, half_window + 1):
        xs = x0 + s * tx
        ys = y0 + s * ty
        xs, ys = snap_to_skeleton_nearest(xs, ys, skeleton, search_radius=10)
        info = measure_normal_diameter(region_mask_filled, skeleton, xs, ys)
        if info["diameter_px"] > 0:
            samples.append(info)

    if not samples:
        return measure_normal_diameter(region_mask_filled, skeleton, x0, y0)

    diameters = np.array([s["diameter_px"] for s in samples], dtype=np.float32)
    med = float(np.median(diameters))
    idx = int(np.argmin(np.abs(diameters - med)))
    return samples[idx]




def smooth_region_mask(mask01, radius_mm=SMOOTH_RADIUS_MM):
    """Remove thin boundary spurs from one stem-section mask (0/1 uint8).

    Morphological opening with an elliptical kernel removes protrusions narrower
    than the kernel; a light Gaussian blur + re-threshold smooths the jagged
    boundary. The largest connected component is kept (opening can leave small
    fragments). Image borders are not eroded (OpenCV default border value), so
    sections that run out of the frame keep their cut ends.
    """
    r = max(1, int(round(radius_mm * px_per_mm_diam)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx((mask01 > 0).astype(np.uint8) * 255, cv2.MORPH_OPEN, k)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, r / 2))
    m = (m >= 128).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 2:
        m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    # never let smoothing delete the section
    return m if m.sum() > 0.8 * (mask01 > 0).sum() else (mask01 > 0).astype(np.uint8)


def fill_holes_touching_edge(mask):
    """Fill holes in a stem mask. Where the stem runs out of the image, the image
    edge is treated as part of the stem, so a gap that opens to the edge through
    the stem's own cut end is also filled."""
    from scipy import ndimage
    m = mask > 0
    p = np.pad(m, 1)
    p[0, 1:-1] = m[0]
    p[-1, 1:-1] = m[-1]
    p[1:-1, 0] = m[:, 0]
    p[1:-1, -1] = m[:, -1]
    # bridge the stem's own extent along each edge row/column
    for edge in (p[0], p[-1], p[:, 0], p[:, -1]):
        idx = np.flatnonzero(edge)
        if len(idx):
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 60) + 1)   # gaps < 60 px belong to one stem end
            for r in runs:
                edge[r[0]:r[-1] + 1] = True
    return ndimage.binary_fill_holes(p)[1:-1, 1:-1].astype(np.uint8)

# ── CENTERLINE (points placed on the stem centerline) ─────────────────────
def _skeleton_longest_path(skeleton):
    """Longest path through the skeleton (removes side branches/spurs). Returns (N, 2) x, y."""
    ys, xs = np.nonzero(skeleton)
    if len(xs) < 10:
        return None
    index = -np.ones(skeleton.shape, dtype=np.int64)
    index[ys, xs] = np.arange(len(xs))
    h, w = skeleton.shape
    nbrs = [[] for _ in range(len(xs))]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            j = np.full(len(xs), -1, dtype=np.int64)
            j[ok] = index[yy[ok], xx[ok]]
            for a in np.nonzero(j >= 0)[0]:
                nbrs[a].append(int(j[a]))

    def bfs(start):
        dist = np.full(len(xs), -1, dtype=np.int64)
        prev = np.full(len(xs), -1, dtype=np.int64)
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            p = queue[head]
            head += 1
            for q in nbrs[p]:
                if dist[q] < 0:
                    dist[q] = dist[p] + 1
                    prev[q] = p
                    queue.append(q)
        return int(np.argmax(dist)), prev

    a, _ = bfs(int(np.argmin(ys)))
    b, prev = bfs(a)
    path = []
    p = b
    while p >= 0:
        path.append(p)
        p = prev[p]
    pts = np.column_stack([xs[path], ys[path]]).astype(np.float64)
    if pts[0, 1] < pts[-1, 1]:
        pts = pts[::-1]            # bottom -> top
    return pts


def _arc(pts):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]


def build_centerline(region_mask_filled, skeleton):
    """Smooth stem centerline used for point placement and tangent direction.

    1. longest skeleton path (side branches removed)
    2. end hooks trimmed (the skeleton bends into the corners of a cut end)
    3. moving-average smoothing
    4. each end extended straight along its own direction until it leaves the stem
    Returns dict with pts (N, 2) x,y bottom->top, arc length s, unit tangents,
    the index range where cross-sections are allowed (i_lo..i_hi, END_MEASURE_MM
    from the ends) and the index range without the straight extensions (j_lo..j_hi).
    """
    raw = _skeleton_longest_path(skeleton)
    if raw is None or len(raw) < 20:
        return None
    mask = (region_mask_filled > 0).astype(np.uint8)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    radius = float(np.median(dt[raw[:, 1].astype(int), raw[:, 0].astype(int)]))
    s = _arc(raw)
    trim = max(CENTERLINE_TRIM_RADII * radius, CENTERLINE_TRIM_MIN_MM * px_per_mm_vert)
    keep = (s > trim) & (s < s[-1] - trim)
    if keep.sum() >= 20:
        raw = raw[keep]

    win = max(3, int(CENTERLINE_SMOOTH_MM * px_per_mm_vert) | 1)
    if len(raw) > win:
        pad = win // 2
        sm = np.empty_like(raw)
        for c in range(2):
            sm[:, c] = np.convolve(np.pad(raw[:, c], pad, mode="edge"), np.ones(win) / win, mode="valid")
        raw = sm

    h, w = mask.shape
    k = min(len(raw) - 1, max(2, int(END_DIRECTION_MM * px_per_mm_vert)))

    def extend(end_pt, direction):
        d = direction / (np.linalg.norm(direction) + 1e-9)
        out = []
        p = end_pt.copy()
        for _ in range(4000):
            p = p + d
            xi, yi = int(round(p[0])), int(round(p[1]))
            if not (0 <= xi < w and 0 <= yi < h) or mask[yi, xi] == 0:
                break
            out.append(p.copy())
        return out

    lower = extend(raw[0], raw[0] - raw[k])[::-1]
    upper = extend(raw[-1], raw[-1] - raw[-1 - k])
    pts = np.vstack([np.array(lower).reshape(-1, 2), raw, np.array(upper).reshape(-1, 2)])
    s = _arc(pts)

    r = max(1, TANGENT_NEIGHBOR_RADIUS)
    idx = np.arange(len(pts))
    tang = pts[np.minimum(idx + r, len(pts) - 1)] - pts[np.maximum(idx - r, 0)]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    tang[tang[:, 1] < 0] *= -1     # same sign convention as the skeleton PCA tangent

    margin = END_MEASURE_MM * px_per_mm_vert
    i_lo = int(np.searchsorted(s, margin))
    i_hi = int(np.searchsorted(s, s[-1] - margin)) - 1
    if i_hi <= i_lo:
        i_lo, i_hi = 0, len(pts) - 1
    # index range of the smoothed skeleton path, i.e. without the straight end extensions
    j_lo, j_hi = len(lower), len(pts) - len(upper) - 1
    return {"pts": pts, "s": s, "tang": tang, "i_lo": i_lo, "i_hi": i_hi, "j_lo": j_lo, "j_hi": j_hi}


def centerline_curvature_profile(C, stem_bottom_ref_y, smooth_mm=5.0):
    """Measured inclination and curvature along the smooth centerline.

    Inclination = angle of the local tangent from vertical (0 deg = upright).
    Curvature   = |d(theta)/ds|, the change of tangent direction per unit arc
                  length, reported in 1/m (radius of curvature = 1 / kappa).
    "len_cm" uses the same vertical reference as the "Stem Length (cm)" column,
    so the profile can be binned into the same 10 cm segments.
    """
    if C is None or len(C["pts"]) < 10:
        return None
    pts, s, tang = C["pts"], C["s"], C["tang"]
    keep = np.r_[True, np.diff(s) > 1e-6]          # drop duplicate arc positions
    pts, s, tang = pts[keep], s[keep], tang[keep]

    theta = np.unwrap(np.arctan2(tang[:, 0], tang[:, 1]))   # tangent angle from vertical
    win = max(3, int(smooth_mm * px_per_mm_vert) | 1)
    if len(theta) > win:
        pad = win // 2
        theta = np.convolve(np.pad(theta, pad, mode="edge"), np.ones(win) / win, mode="valid")
    kappa_per_px = np.abs(np.gradient(theta, s))

    # Use only the smoothed skeleton part of the centerline: the straight end
    # extensions (added after trimming the end hooks) are excluded, and so are the
    # END_MEASURE_MM next to each cut end.
    lo = max(C["i_lo"], C.get("j_lo", C["i_lo"]))
    hi = min(C["i_hi"], C.get("j_hi", C["i_hi"]))
    if hi <= lo:
        return None
    sel = np.zeros(len(pts), dtype=bool)
    sel[np.searchsorted(s, C["s"][lo]):np.searchsorted(s, C["s"][hi]) + 1] = True

    return pd.DataFrame({
        "len_cm": (stem_bottom_ref_y - pts[sel, 1]) / px_per_mm_vert / 10.0,
        "incl_deg": np.degrees(np.abs(theta[sel])),
        "kappa_per_m": kappa_per_px[sel] * px_per_mm_vert * 1000.0,
    })


def centerline_index(C, x, y):
    """Index of the centerline point nearest to (x, y)."""
    d2 = (C["pts"][:, 0] - x) ** 2 + (C["pts"][:, 1] - y) ** 2
    return int(np.argmin(d2))


def snap_to_centerline(C, x, y):
    i = centerline_index(C, x, y)
    return int(round(C["pts"][i, 0])), int(round(C["pts"][i, 1]))


def centerline_point_between(C, p_a, p_b, frac):
    """Point on the centerline at fraction frac of the CURVED length from p_a to p_b."""
    ia, ib = centerline_index(C, *p_a), centerline_index(C, *p_b)
    target = (1 - frac) * C["s"][ia] + frac * C["s"][ib]
    i = int(np.argmin(np.abs(C["s"] - target)))
    return int(round(C["pts"][i, 0])), int(round(C["pts"][i, 1]))


def measure_on_centerline(region_mask_filled, C, x0, y0, half_window=TANGENT_SAMPLE_HALF_WINDOW):
    """Tangent diameter at the centerline point nearest (x0, y0).

    The cross-section is perpendicular to the smooth centerline. Cross-sections
    at the neighbouring centerline points (+-half_window) are also measured and
    the one closest to their median is kept (same robust rule as before).
    Points within END_MEASURE_MM of the cut ends are moved inward, so the cut face
    itself is not measured.
    """
    i0 = int(np.clip(centerline_index(C, x0, y0), C["i_lo"], C["i_hi"]))
    samples = []
    for j in range(i0 - half_window, i0 + half_window + 1):
        j = int(np.clip(j, 0, len(C["pts"]) - 1))
        x, y = C["pts"][j]
        tx, ty = C["tang"][j]
        nx, ny = -ty, tx
        p_pos = trace_to_stem_boundary(region_mask_filled, x, y, nx, ny)
        p_neg = trace_to_stem_boundary(region_mask_filled, x, y, -nx, -ny)
        # Re-centre once: if the centerline point sits off-centre, restart from the
        # middle of the cross-section so both rays reach the two stem edges.
        mx, my = (p_pos[0] + p_neg[0]) / 2, (p_pos[1] + p_neg[1]) / 2
        p_pos = trace_to_stem_boundary(region_mask_filled, mx, my, nx, ny)
        p_neg = trace_to_stem_boundary(region_mask_filled, mx, my, -nx, -ny)
        d = float(np.hypot(p_pos[0] - p_neg[0], p_pos[1] - p_neg[1]))
        # Safety check: a cross-section perpendicular to the stem can never be wider
        # than the horizontal width through the same point. If it is, the direction
        # is wrong (cut end, mask bump) and the horizontal width is used instead.
        h_right = trace_to_stem_boundary(region_mask_filled, mx, my, 1.0, 0.0)
        h_left = trace_to_stem_boundary(region_mask_filled, mx, my, -1.0, 0.0)
        d_h = float(h_right[0] - h_left[0])
        if d_h > 0 and d > d_h + 2:
            p_pos, p_neg, d = h_right, h_left, d_h
            tx, ty, nx, ny = 0.0, 1.0, 1.0, 0.0
        if d > 0:
            samples.append({"diameter_px": d, "pt1": p_neg, "pt2": p_pos, "center": (float(x), float(y)),
                            "tangent": (float(tx), float(ty)), "normal": (float(nx), float(ny))})
    if not samples:
        x, y = C["pts"][i0]
        return {"diameter_px": 0.0, "pt1": (x, y), "pt2": (x, y), "center": (float(x), float(y)),
                "tangent": (0.0, 1.0), "normal": (1.0, 0.0)}
    diameters = np.array([s["diameter_px"] for s in samples])
    return samples[int(np.argmin(np.abs(diameters - np.median(diameters))))]


# ── DRAWING ──────────────────────────────────────────────────────────────
def draw_node_label(img, node_xy, node_label):
    """Draw a labeled node marker on the overlay image."""
    x, y = int(round(node_xy[0])), int(round(node_xy[1]))

    # Node marker used in the final overlay
    cv2.circle(img, (x, y), 34, (0, 180, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x, y), 34, (120, 255, 120), 4, lineType=cv2.LINE_AA)

    # Keep the label readable over both light and dark stem regions
    cv2.putText(
        img,
        node_label,
        (x - 28, y + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        3,
        lineType=cv2.LINE_AA
    )

def draw_measurement(img, tangent_measurement, diameter_mm, text_label):
    """Draw one tangent-based diameter line and its value on the overlay."""
    x1, y1 = int(round(tangent_measurement["pt1"][0])), int(round(tangent_measurement["pt1"][1]))
    x2, y2 = int(round(tangent_measurement["pt2"][0])), int(round(tangent_measurement["pt2"][1]))

    cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        COLOR_TANGENT_METHOD,
        LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    tx = int(round(max(x1, x2))) + LABEL_X_OFFSET
    ty = int(round((y1 + y2) / 2))
    text = f"{text_label}: {diameter_mm:.1f}"

    cv2.putText(
        img,
        text,
        (tx, ty + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        COLOR_DIAM_TEXT,
        TEXT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


# ── FIXED-SPACING BASELINE ───────────────────────────────────────────────
def fixed_spacing_positions(section_offset_px, spacing_px, centerline=None, mask=None, skeleton=None):
    """Method B2: fixed-coordinate points every spacing_px from the ORIGINAL STALK BASE.

    Positions are k * spacing (k = 1, 2, ...) along the stalk. A stalk cut into
    sections keeps its original positions: this section starts at
    section_offset_px (sum of the centerline lengths of the sections below it),
    so a point at stalk position 2k cm is placed at (2k cm - offset) along this
    section's curved centerline. Nodes are not used at all, neither for the
    positions nor for the number of points. Points closer than END_MEASURE_MM
    to a cut end are skipped (the cut end itself is not a valid cross-section).

    Returns (points, section_length_px, skipped); points = [(x, y, stalk_position_px)],
    skipped = [(stalk_position_px, distance_to_cut_end_px, "bottom"/"top")].
    """
    margin = END_MEASURE_MM * px_per_mm_vert
    if centerline is not None:
        C = centerline
        L = float(C["s"][-1])
        pts, skipped = [], []
        k = int(np.floor(section_offset_px / spacing_px)) + 1
        while k * spacing_px < section_offset_px + L:
            local = k * spacing_px - section_offset_px
            if margin <= local <= L - margin:
                i = int(np.argmin(np.abs(C["s"] - local)))
                pts.append((int(round(C["pts"][i, 0])), int(round(C["pts"][i, 1])), k * spacing_px))
            else:
                end = "bottom" if local < margin else "top"
                skipped.append((k * spacing_px, local if end == "bottom" else L - local, end))
            k += 1
        return pts, L, skipped
    # fallback without a centerline: vertical distance from the section bottom
    ys, xs = np.nonzero(mask)
    y_bot, y_top = int(ys.max()), int(ys.min())
    L = float(y_bot - y_top)
    pts, skipped = [], []
    k = int(np.floor(section_offset_px / spacing_px)) + 1
    while k * spacing_px < section_offset_px + L:
        local = k * spacing_px - section_offset_px
        if margin <= local <= L - margin:
            y = int(round(y_bot - local))
            x, y = snap_to_skeleton_ywise(int(np.median(xs[ys == y])) if np.any(ys == y) else int(np.median(xs)), y, skeleton)
            pts.append((x, y, k * spacing_px))
        else:
            end = "bottom" if local < margin else "top"
            skipped.append((k * spacing_px, local if end == "bottom" else L - local, end))
        k += 1
    return pts, L, skipped


def stalk_position_px(C, x, y, section_offset_px):
    """Position along the original stalk (centerline length from the stalk base) of a point."""
    if C is None:
        return np.nan
    return section_offset_px + float(C["s"][centerline_index(C, x, y)])


def locate_in_internodes(y, region_nodes):
    """Return (internode index from 0, relative position r, distance to nearest node in px).

    r = 0 at the lower node and 1 at the upper node of the internode.
    region_nodes must be sorted bottom to top (decreasing y).
    """
    node_ys = [float(p[1]) for p in region_nodes]
    dist_px = min(abs(float(y) - ny) for ny in node_ys)
    for k in range(len(node_ys) - 1):
        y_lo, y_hi = node_ys[k], node_ys[k + 1]
        if y_hi <= y <= y_lo and y_lo != y_hi:
            return k, (y_lo - float(y)) / (y_lo - y_hi), dist_px
    return None, None, dist_px


def draw_fixed_measurement(img, tangent_measurement, diameter_mm, text_label, near_node):
    """Draw one fixed-spacing diameter line; points in the node zone are magenta."""
    x1, y1 = int(round(tangent_measurement["pt1"][0])), int(round(tangent_measurement["pt1"][1]))
    x2, y2 = int(round(tangent_measurement["pt2"][0])), int(round(tangent_measurement["pt2"][1]))
    color = COLOR_FIXED_NEAR_NODE if near_node else COLOR_FIXED_METHOD

    cv2.line(img, (x1, y1), (x2, y2), color, LINE_THICKNESS, lineType=cv2.LINE_AA)

    tx = int(round(max(x1, x2))) + LABEL_X_OFFSET
    ty = int(round((y1 + y2) / 2))
    cv2.putText(
        img, f"{text_label}: {diameter_mm:.1f}", (tx, ty + 5),
        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, COLOR_DIAM_TEXT, TEXT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def draw_fixed_legend(img):
    """Legend for the fixed-spacing overlay."""
    lines = [
        ("Fixed-spacing point", COLOR_FIXED_METHOD),
        (f"Fixed point within {NODE_ZONE_CM:.1f} cm of a node", COLOR_FIXED_NEAR_NODE),
    ]
    scale, thick = TEXT_SCALE * 0.8, max(2, TEXT_THICKNESS - 1)
    (tw, th), _ = cv2.getTextSize(lines[1][0], cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad, swatch, gap = 20, 80, int(th * 1.8)
    cv2.rectangle(img, (10, 10), (10 + pad * 3 + swatch + tw, 10 + pad * 2 + gap * len(lines)),
                  (255, 255, 255), -1)
    for i, (text, color) in enumerate(lines):
        y = 10 + pad + gap * i + th
        cv2.line(img, (10 + pad, y - th // 2), (10 + pad + swatch, y - th // 2), color, LINE_THICKNESS)
        cv2.putText(img, text, (10 + pad * 2 + swatch, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thick, lineType=cv2.LINE_AA)


def summarize_methods(df_compare, group_cols):
    """Per-method sampling metrics used to compare node-guided vs fixed spacing.

    Reports positions and measured differences only; it does not claim either
    method is more accurate (that needs the manual caliper validation).
    """
    out = []
    for keys, g in df_compare.groupby(group_cols + ["Method"], sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols + ["Method"], keys))

        in_int = g[g["Internode Index"].notna()]
        per_int = in_int.groupby(["Image", "Stem", "Internode"]).size() if len(in_int) else pd.Series(dtype=int)

        # All internodes that exist for these stems (from Method A rows, which cover every internode).
        all_int = df_compare[(df_compare["Method"] == "A_node_guided")]
        for col, val in zip(group_cols, keys[:len(group_cols)]):
            all_int = all_int[all_int[col] == val]
        all_int_keys = set(map(tuple, all_int[["Image", "Stem", "Internode"]].drop_duplicates().values))

        mid = in_int[(in_int["Relative Position (0-1)"] - 0.5).abs() <= MIDPOINT_WINDOW]
        mid_keys = set(map(tuple, mid[["Image", "Stem", "Internode"]].drop_duplicates().values))
        counts_all = [int(per_int.get(k, 0)) for k in all_int_keys] if all_int_keys else [0]

        near = g[g["Within Node Zone"] == "Yes"]
        away = g[g["Within Node Zone"] == "No"]

        rec.update({
            "Points": len(g),
            f"Points within {NODE_ZONE_CM} cm of node": len(near),
            "Points near node (%)": round(100 * len(near) / len(g), 1) if len(g) else np.nan,
            "Mean distance to nearest node (cm)": round(g["Distance to Nearest Node (cm)"].mean(), 2),
            "Min distance to nearest node (cm)": round(g["Distance to Nearest Node (cm)"].min(), 2),
            "Internodes": len(all_int_keys),
            "Min points per internode": min(counts_all),
            "Max points per internode": max(counts_all),
            "Internodes with 0 points": sum(1 for c in counts_all if c == 0),
            f"Internodes with a point at midpoint (r 0.5 +/- {MIDPOINT_WINDOW}) (%)":
                round(100 * len(mid_keys & all_int_keys) / len(all_int_keys), 1) if all_int_keys else np.nan,
            "Mean diameter (mm)": round(g["Diameter Tangent (mm)"].mean(), 2),
            "Mean diff vs node-guided profile, near-node points (mm)":
                round(near["Diff vs Node-Guided Profile (mm)"].mean(), 2) if len(near) else np.nan,
            "Mean diff vs node-guided profile, other points (mm)":
                round(away["Diff vs Node-Guided Profile (mm)"].mean(), 2) if len(away) else np.nan,
        })
        out.append(rec)
    return pd.DataFrame(out)


def save_relative_position_plot(df_compare, out_path, title):
    """Two panels: where points fall inside internodes (r) for each method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bins = np.linspace(0, 1, 21)
    panels = [
        ("A_node_guided", "Method A: node-guided", "#2a9d4b"),
        (B2_LABEL, f"Method B2: fixed {FIXED_SPACING_CM:g} cm", "#e8861a"),
    ]
    for ax, (method, label, color) in zip(axes, panels):
        r = df_compare.loc[df_compare["Method"] == method, "Relative Position (0-1)"].dropna()
        ax.hist(r, bins=bins, color=color, edgecolor="white", linewidth=2)
        ax.set_title(f"{label} (n = {len(r)})")
        ax.set_xlabel("Relative position in internode (0 = lower node, 1 = upper node)")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Number of points")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_png(path, img_bgr):
    """Save a BGR image as PNG with FIGURE_DPI stored in the file (for journals)."""
    from PIL import Image
    Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(str(path), dpi=(FIGURE_DPI, FIGURE_DPI))


def save_side_by_side(img_a, img_b, out_path, title_a=None, title_b=None):
    """Method A overlay (left) next to Method B overlay (right) with a title band."""
    def titled(img, title, color):
        band_h = 140
        band = np.full((band_h, img.shape[1], 3), 255, np.uint8)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6)
        cv2.putText(band, title, ((img.shape[1] - tw) // 2, (band_h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 6, lineType=cv2.LINE_AA)
        cv2.rectangle(band, (0, band_h - 12), (img.shape[1], band_h), color, -1)
        return np.vstack([band, img])

    left = titled(img_a, title_a or "A: Node-guided (5 per internode)", COLOR_TANGENT_METHOD)
    right = titled(img_b, title_b or f"B2: Fixed coordinates (every {FIXED_SPACING_CM:g} cm from stalk base)", COLOR_FIXED_METHOD)
    gap = np.full((left.shape[0], 40, 3), 255, np.uint8)
    save_png(out_path, np.hstack([left, gap, right]))


def internode_d_profile(df):
    """D1-D5 profile per stalk internode number, with plants (images) as replicates.

    Each plant contributes one value per position: if a plant has several values
    for the same internode number and position, they are averaged first. Mean,
    SD and SE are then computed across plants, so internodes are never counted
    as independent replicates.
    """
    a = df[df["Method"] == "A_node_guided"].copy()
    if a.empty:
        return pd.DataFrame(), pd.DataFrame()
    frac_by_label = {f"D{i + 1}": f for i, f in enumerate(INTERNODE_SAMPLE_FRACS)}
    a["Nominal Position (%)"] = a["Sample Position"].map(lambda s: round(100 * frac_by_label.get(s, np.nan)))

    per_plant = (
        a.groupby(["Stalk Internode No.", "Sample Position", "Nominal Position (%)", "Image"], as_index=False)
        ["Diameter Tangent (mm)"].mean()
    )
    summary = (
        per_plant.groupby(["Stalk Internode No.", "Sample Position", "Nominal Position (%)"])
        ["Diameter Tangent (mm)"]
        .agg(n_plants="count", mean_mm="mean", sd_mm="std")
        .reset_index()
    )
    summary["se_mm"] = summary["sd_mm"] / np.sqrt(summary["n_plants"])
    summary = summary.rename(columns={
        "n_plants": "Plants (n)", "mean_mm": "Mean Diameter (mm)",
        "sd_mm": "SD (mm)", "se_mm": "SE (mm)",
    })
    for c in ["Mean Diameter (mm)", "SD (mm)", "SE (mm)"]:
        summary[c] = summary[c].round(3)
    return summary, per_plant


def save_d_profile_plot(summary, per_plant, out_path, title):
    """Small multiples: one panel per stalk internode number, mean +/- SE across plants."""
    if summary.empty:
        return
    internodes = sorted(summary["Stalk Internode No."].dropna().unique())
    ncols = min(4, len(internodes))
    nrows = int(np.ceil(len(internodes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), sharey=True, squeeze=False)
    color = "#2a9d4b"

    for ax, k in zip(axes.flat, internodes):
        s = summary[summary["Stalk Internode No."] == k].sort_values("Nominal Position (%)")
        p = per_plant[per_plant["Stalk Internode No."] == k]
        for _, g in p.groupby("Image"):          # each plant, faint
            g = g.sort_values("Nominal Position (%)")
            ax.plot(g["Nominal Position (%)"], g["Diameter Tangent (mm)"], color="#bbbbbb", linewidth=1)
        se = s["SE (mm)"].fillna(0)
        ax.fill_between(s["Nominal Position (%)"], s["Mean Diameter (mm)"] - se,
                        s["Mean Diameter (mm)"] + se, color=color, alpha=0.2, linewidth=0)
        ax.plot(s["Nominal Position (%)"], s["Mean Diameter (mm)"], color=color, linewidth=2,
                marker="o", markersize=6)
        n = int(s["Plants (n)"].max()) if len(s) else 0
        ax.set_title(f"Internode {int(k)} (n = {n} plant{'' if n == 1 else 's'})", fontsize=10)
        ax.set_xticks([round(100 * f) for f in INTERNODE_SAMPLE_FRACS])
        ax.set_xticklabels([f"D{i + 1}\n{round(100 * f)}%" for i, f in enumerate(INTERNODE_SAMPLE_FRACS)], fontsize=8)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(internodes):]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r, 0].set_ylabel("Diameter (mm)")
    fig.suptitle(f"{title}\nGreen: mean across plants, band: +/- SE; grey: individual plants", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def internode_midpoint_coverage(df):
    """Per stalk internode number: how often each method samples near the midpoint.

    Unit = plant (image). A plant counts as covered for an internode when at least
    one point of that method has |r - 0.5| <= MIDPOINT_WINDOW in that internode.
    """
    a_keys = df[df["Method"] == "A_node_guided"][["Image", "Stalk Internode No."]].drop_duplicates()
    out = []
    for k, plants in a_keys.groupby("Stalk Internode No."):
        rec = {"Stalk Internode No.": int(k), "Plants (n)": len(plants)}
        for method, tag in [("A_node_guided", "A"), (B2_LABEL, "B2")]:
            g = df[(df["Method"] == method) & (df["Stalk Internode No."] == k)]
            covered, counts, nearest = 0, [], []
            for img in plants["Image"]:
                gi = g[g["Image"] == img]
                counts.append(len(gi))
                r = gi["Relative Position (0-1)"].dropna()
                if len(r):
                    d = (r - 0.5).abs().min()
                    nearest.append(d)
                    covered += int(d <= MIDPOINT_WINDOW)
            rec[f"{tag}: plants with midpoint point (%)"] = round(100 * covered / len(plants), 1)
            rec[f"{tag}: mean |r - 0.5| of closest point"] = round(float(np.mean(nearest)), 3) if nearest else np.nan
            rec[f"{tag}: points per internode (min-max)"] = f"{min(counts)}-{max(counts)}"
        out.append(rec)
    return pd.DataFrame(out)


def single_location_emulation(df):
    """What a single midpoint (D3) measurement would miss, from the existing D1-D5 data.

    This is an emulation of single-location reporting on our own measurements,
    not a run of Maize-IAS. Plants (images) are the replicates: values are
    averaged within a plant first, then summarized across plants.

    Returns
      per_internode: one row per plant x stalk internode (D1-D5, range, CV,
                     D3 minus five-point mean)
      by_internode:  the same metrics summarized across plants per internode number
      per_plant:     along-stalk variation per plant (within- vs between-internode SD)
      overall:       one-row summary across plants
    """
    a = df[df["Method"] == "A_node_guided"]
    if a.empty:
        return (pd.DataFrame(),) * 4
    labels = [f"D{i + 1}" for i in range(len(INTERNODE_SAMPLE_FRACS))]

    wide = (
        a.groupby(["Image", "Stalk Internode No.", "Sample Position"])["Diameter Tangent (mm)"].mean()
        .unstack("Sample Position").reindex(columns=labels).reset_index()
    )
    ends = (
        a.groupby(["Image", "Stalk Internode No."])["Touches Section End"]
        .agg(lambda s: "Yes" if (s == "Yes").any() else "No").reset_index()
    )
    wide = wide.merge(ends, on=["Image", "Stalk Internode No."], how="left")
    d = wide[labels]
    wide["Five-Point Mean (mm)"] = d.mean(axis=1)
    wide["D1-D5 Range (max-min) (mm)"] = d.max(axis=1) - d.min(axis=1)
    wide["D1 minus D5 (mm)"] = wide["D1"] - wide[labels[-1]]
    wide["Within-Internode CV (%)"] = 100 * d.std(axis=1, ddof=1) / wide["Five-Point Mean (mm)"]
    wide["D3 minus Five-Point Mean (mm)"] = wide["D3"] - wide["Five-Point Mean (mm)"]
    wide["|D3 minus Five-Point Mean| (%)"] = 100 * wide["D3 minus Five-Point Mean (mm)"].abs() / wide["Five-Point Mean (mm)"]

    # Along-stalk: how much a single internode's value differs from the whole-stalk mean.
    stalk_mean = wide.groupby("Image")["Five-Point Mean (mm)"].transform("mean")
    wide["D3 minus Stalk Mean (mm)"] = wide["D3"] - stalk_mean
    wide["|D3 minus Stalk Mean| (%)"] = 100 * wide["D3 minus Stalk Mean (mm)"].abs() / stalk_mean

    metric_cols = [
        "D1-D5 Range (max-min) (mm)", "D1 minus D5 (mm)", "Within-Internode CV (%)",
        "D3 minus Five-Point Mean (mm)", "|D3 minus Five-Point Mean| (%)",
        "D3 minus Stalk Mean (mm)", "|D3 minus Stalk Mean| (%)",
    ]
    by_internode = wide.groupby("Stalk Internode No.")[metric_cols].agg(["mean", "std"])
    by_internode.columns = [f"{c} [{s}]" for c, s in by_internode.columns]
    by_internode.insert(0, "Plants (n)", wide.groupby("Stalk Internode No.").size())
    by_internode = by_internode.reset_index()

    plant_recs = []
    for img, g in wide.groupby("Image"):
        plant_recs.append({
            "Image": img,
            "Internodes": len(g),
            "Stalk Mean Diameter (mm)": g["Five-Point Mean (mm)"].mean(),
            "Along-Stalk Range of Internode Means (mm)": g["Five-Point Mean (mm)"].max() - g["Five-Point Mean (mm)"].min(),
            "Between-Internode SD (mm)": g["Five-Point Mean (mm)"].std(ddof=1),
            "Mean Within-Internode SD (mm)": g[labels].std(axis=1, ddof=1).mean(),
            "Mean D1-D5 Range (mm)": g["D1-D5 Range (max-min) (mm)"].mean(),
        })
    per_plant = pd.DataFrame(plant_recs)

    overall = pd.DataFrame([{
        "Plants (n)": per_plant["Image"].nunique(),
        "Plant x internode combinations": len(wide),
        "Mean D1-D5 Range (mm) [mean of plant means]": per_plant["Mean D1-D5 Range (mm)"].mean(),
        "Mean Within-Internode SD (mm) [mean of plant means]": per_plant["Mean Within-Internode SD (mm)"].mean(),
        "Between-Internode SD (mm) [mean of plant values]": per_plant["Between-Internode SD (mm)"].mean(),
        "Along-Stalk Range (mm) [mean of plant values]": per_plant["Along-Stalk Range of Internode Means (mm)"].mean(),
        "|D3 minus Five-Point Mean| (%) [mean of plant means]":
            wide.groupby("Image")["|D3 minus Five-Point Mean| (%)"].mean().mean(),
        "|D3 minus Stalk Mean| (%) [mean of plant means]":
            wide.groupby("Image")["|D3 minus Stalk Mean| (%)"].mean().mean(),
        "Note": "Emulation of single-location reporting on this dataset; not a run of Maize-IAS. "
                "Compare within-internode values with the measurement error from manual validation.",
    }])

    for t in (wide, by_internode, per_plant, overall):
        num = t.select_dtypes("number").columns
        t[num] = t[num].round(3)
    return wide, by_internode, per_plant, overall




def section_check_tables(image_name):
    """Per-piece table (with manual-length columns) and joint table for one image."""
    sec = pd.DataFrame([r for r in B2_SECTION_LOG if r["Image"] == image_name])
    if sec.empty:
        return sec, pd.DataFrame()
    sec = sec.sort_values("Stem (section, base = 1)").reset_index(drop=True)
    flags = []
    for _, r in sec.iterrows():
        b, t = r["Bottom end diameter (mm)"], r["Top end diameter (mm)"]
        if pd.notna(b) and pd.notna(t) and t > b * (1 + JOINT_TOL_PCT / 100):
            flags.append("CHECK: top wider than bottom (piece upside down?)")
        else:
            flags.append("ok")
    sec["Piece direction check"] = flags
    joints = []
    for k in range(len(sec) - 1):
        lo, hi = sec.iloc[k], sec.iloc[k + 1]
        top, bot = lo["Top end diameter (mm)"], hi["Bottom end diameter (mm)"]
        diff = round(100 * (bot - top) / top, 1) if pd.notna(top) and pd.notna(bot) and top > 0 else np.nan
        joints.append({
            "Image": image_name,
            "Joint": f"P{int(lo['Stem (section, base = 1)'])} top -> P{int(hi['Stem (section, base = 1)'])} bottom",
            "Joint position on stalk (cm)": lo["Section end on stalk (cm)"],
            "P(k) top diameter (mm)": top,
            "P(k+1) bottom diameter (mm)": bot,
            "Difference (%)": diff,
            "Order check": ("ok" if abs(diff) <= JOINT_TOL_PCT else
                            f"CHECK: > {JOINT_TOL_PCT:g}% jump (wrong order or flipped piece?)") if pd.notna(diff) else "no data",
        })
    return sec, pd.DataFrame(joints)


def write_section_check(xlsx_path, image_name):
    """Add 'Section_Check' (with manual length columns + Excel formulas) and 'Joint_Check' sheets."""
    sec, joints = section_check_tables(image_name)
    if sec.empty:
        return
    cols = ["Stem (section, base = 1)", "Section start on stalk (cm)", "Section end on stalk (cm)",
            "Section length along centerline (cm)", "Bottom end diameter (mm)", "Top end diameter (mm)",
            "Piece direction check", "Cut by image edge"]
    table = sec[[c for c in cols if c in sec.columns]].copy()
    table["Manual piece length, ruler (cm)"] = None
    table["Diff computer vs manual (%)"] = None
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        table.to_excel(w, sheet_name="Section_Check", index=False)
        joints.to_excel(w, sheet_name="Joint_Check", index=False)
    wb = load_workbook(xlsx_path)
    ws = wb["Section_Check"]
    hdr = {c.value: c.column_letter for c in ws[1]}
    L, M, D = hdr["Section length along centerline (cm)"], hdr["Manual piece length, ruler (cm)"], hdr["Diff computer vs manual (%)"]
    n = len(table)
    for r in range(2, n + 2):
        ws[f"{D}{r}"] = f'=IF({M}{r}="","",ROUND(100*({L}{r}-{M}{r})/{M}{r},2))'
    t = n + 3
    ws[f"A{t}"] = "Whole stalk"
    ws[f"{L}{t}"] = f"=SUM({L}2:{L}{n + 1})"
    ws[f"{M}{t - 0}"] = None
    ws[f"A{t + 1}"] = "Manual whole-stalk length before cutting (cm) ->"
    ws[f"{L}{t + 1}"] = "type here:"
    ws[f"{M}{t + 1}"] = None
    ws[f"{D}{t + 1}"] = f'=IF({M}{t + 1}="","",ROUND(100*({L}{t}-{M}{t + 1})/{M}{t + 1},2))'
    ws[f"A{t + 2}"] = ("Computer total is the sum of the pieces; saw cuts remove ~1-2 mm each, "
                       "so a slightly smaller computer total is expected.")
    wb.save(xlsx_path)


def save_length_workbooks(image_path, output_dir, main_xlsx):
    """Two easy-to-read workbooks for manual checking, plus an internode-length sheet.

    <image>_positions_continuous.xlsx : positions measured from the ORIGINAL STALK BASE
                                        (pieces joined in base-to-top order)
    <image>_positions_per_piece.xlsx  : positions measured from the bottom of EACH PIECE (0 cm)
    Both contain the same internode lengths (straight / curved / vertical) with empty
    columns for the manual ruler and thread readings.
    """
    name = image_path.stem
    pos = pd.DataFrame([r for r in POSITION_LOG if r["Image"] == name])
    ints = pd.DataFrame([r for r in INTERNODE_LENGTH_LOG if r["Image"] == name])
    if pos.empty:
        return
    order = {"Node": 0, "A_node_guided": 1, B2_LABEL: 2}
    lengths = ["Straight length (cm)", "Curved length, centerline (cm)", "Vertical length, y only (cm)",
               "Curved vs straight (%)", "Vertical vs curved (%)",
               "Manual straight, ruler (cm)", "Manual curved, thread (cm)", "Manual vertical (cm, optional)"]
    base = ["Image", "Stem (section, base = 1)", "Internode", "Stalk Internode No.", "Touches Section End"]
    for kind, col, ncols in [
        ("continuous", "Stalk Position (cm)", ["Lower node, stalk position (cm)", "Upper node, stalk position (cm)"]),
        ("per_piece", "Piece Position (cm)", ["Lower node, piece position (cm)", "Upper node, piece position (cm)"]),
    ]:
        p = pos[["Image", "Stem (section, base = 1)", "Method", "Point", "Internode / detail", col,
                 "Diameter Tangent (mm)", "X(px)", "Y(px)"]].copy()
        p["_o"] = p["Method"].map(order)
        p = p.sort_values(["Stem (section, base = 1)", col, "_o"]).drop(columns="_o")
        out = output_dir / f"{name}_positions_{kind}.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl", mode="w") as w:
            for method, sheet in [("Node", "Nodes"), ("A_node_guided", "A_node_guided"), (B2_LABEL, B2_LABEL[:31])]:
                p[p["Method"] == method].to_excel(w, sheet_name=sheet, index=False)
            p.to_excel(w, sheet_name="All_points_sorted", index=False)
            if not ints.empty:
                ints[base + ncols + lengths].to_excel(w, sheet_name="Internode_Lengths", index=False)
        write_section_check(out, name)
        print(f"Saved: {out}")
    write_section_check(main_xlsx, name)
    if not ints.empty:
        with pd.ExcelWriter(main_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            ints.to_excel(w, sheet_name="Internode_Lengths_3_methods", index=False)
    nlog = pd.DataFrame([r for r in NODE_REFINE_LOG if r["Image"] == name])
    if not nlog.empty:
        with pd.ExcelWriter(main_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            nlog.to_excel(w, sheet_name="Node_Refinement", index=False)


def save_batch_comparison(frames, output_dir, batch_name):
    """Combine all images of a batch: cross-stem comparison of both methods."""
    df = pd.concat(frames, ignore_index=True)
    profile, per_plant = internode_d_profile(df)
    coverage = internode_midpoint_coverage(df)
    sl_internode, sl_by_internode, sl_plant, sl_overall = single_location_emulation(df)

    out_xlsx = output_dir / f"{batch_name}_method_comparison_summary.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
        summarize_methods(df, []).to_excel(writer, sheet_name="Summary_Batch", index=False)
        summarize_methods(df, ["Image"]).to_excel(writer, sheet_name="Summary_by_Image", index=False)
        coverage.to_excel(writer, sheet_name="Midpoint_Coverage_by_Internode", index=False)
        profile.to_excel(writer, sheet_name="D1_D5_Profile_Summary", index=False)
        per_plant.to_excel(writer, sheet_name="D1_D5_Profile_per_Plant", index=False)
        sl_overall.to_excel(writer, sheet_name="SingleLoc_Overall", index=False)
        sl_by_internode.to_excel(writer, sheet_name="SingleLoc_by_Internode", index=False)
        sl_plant.to_excel(writer, sheet_name="SingleLoc_per_Plant", index=False)
        sl_internode.to_excel(writer, sheet_name="SingleLoc_per_Internode", index=False)
        df.to_excel(writer, sheet_name="All_Points_Both_Methods", index=False)
        imgs = set(df["Image"])
        jt = [section_check_tables(i)[1] for i in sorted(imgs)]
        jt = [j for j in jt if not j.empty]
        (pd.concat(jt, ignore_index=True) if jt else pd.DataFrame()).to_excel(
            writer, sheet_name="Joint_Check", index=False)
        pd.DataFrame([r for r in INTERNODE_LENGTH_LOG if r["Image"] in imgs]).to_excel(
            writer, sheet_name="Internode_Lengths_3_methods", index=False)
        pd.DataFrame([r for r in B2_SECTION_LOG if r["Image"] in imgs]).to_excel(
            writer, sheet_name="B2_Section_Positions", index=False)
        pd.DataFrame([r for r in B2_SKIPPED_LOG if r["Image"] in imgs],
                     columns=["Image", "Stem (section, base = 1)", "Stalk Position (cm)",
                              "Distance to cut end (mm)", "Cut end", "Reason"]).to_excel(
            writer, sheet_name="B2_Skipped_Points", index=False)
    save_relative_position_plot(
        df, output_dir / f"{batch_name}_relative_position_histogram.png",
        f"{batch_name}: where sampling points fall inside internodes",
    )
    save_d_profile_plot(
        profile, per_plant, output_dir / f"{batch_name}_D1_D5_profile.png",
        f"{batch_name}: D1-D5 diameter profile by stalk internode number",
    )
    print(f"Batch comparison saved: {out_xlsx}")


# ── MAIN BATCH ───────────────────────────────────────────────────────────
def batch_process_stems():
    """Process all batch folders and save tangent-only outputs."""
    global CAL
    base_batch_dir = BASE_DIR

    for batch_folder in sorted(d for d in base_batch_dir.iterdir() if d.is_dir()):
        batch_name = batch_folder.name

        if batch_name in completed_comparison_batches:
            print(f"Skipping completed batch: {batch_name}")
            continue

        input_dir = batch_folder / "whitebg" / "tif"
        labels_dir = batch_folder / "yolo_output" / "labels"
        output_dir = batch_folder / COMPARISON_OUTPUT_SUBDIR
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_compare_frames = []

        tif_images = sorted(list(input_dir.glob("*.tif")))
        if not tif_images:
            print(f"No TIF images found in {batch_name}; skipping.")
            continue

        processed_count = 0

        for image_path in tif_images:
            image_name = image_path.stem
            base_name = image_name.replace("_whitebg", "").replace("_blackbg", "")
            label_path = labels_dir / f"{base_name}_blackbg.txt"

            # Ruler calibration for this image (falls back to the default px/mm).
            CAL = None
            if USE_RULER_CALIBRATION:
                try:
                    CAL = load_image_calibration(batch_folder, base_name)
                except Exception as e:
                    print(f"   ruler calibration not available for {base_name} ({e})")
            if CAL is None:
                print(f"   using default px/mm = {DEFAULT_PX_PER_MM_VERT}")

            image_output_dir = output_dir / base_name
            image_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nProcessing {base_name} in batch {batch_name}")
            try:
                df_compare_img = process_single_stem(image_path, label_path, image_output_dir)
                if df_compare_img is not None and not df_compare_img.empty:
                    batch_compare_frames.append(df_compare_img)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {base_name}: {e}")

        if batch_compare_frames:
            save_batch_comparison(batch_compare_frames, output_dir, batch_name)

        if processed_count > 0:
            with open(comparison_log, "a") as f:
                f.write(batch_name + "\n")
            print(f"Batch {batch_name} completed and logged ({processed_count} images).")
        else:
            print(f"Batch {batch_name} had no valid images; not logged.")


def process_single_stem(image_path, label_path, output_dir, global_node_start=1):
    """Process one image and export tangent-based stem measurements."""
    global px_per_mm_vert, px_per_mm_diam
    # Start every image from the default, so a missing calibration never
    # inherits the scale of the previous image.
    px_per_mm_vert, px_per_mm_diam = DEFAULT_PX_PER_MM_VERT, DEFAULT_PX_PER_MM_DIAM

    gray = load_image_gray(image_path)
    sharp = sharpen_unsharp_mask(gray)
    blur = cv2.bilateralFilter(sharp, 7, 50, 50)

    is_white_bg = np.mean(blur) > 128
    thresh_type = cv2.THRESH_BINARY_INV if is_white_bg else cv2.THRESH_BINARY

    _, otsu = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 31, 2
    )
    binary = cv2.bitwise_or(otsu, adaptive)

    binary_filled = connect_and_fill(binary, connect_dist=15)
    contours = extract_final_contours(binary_filled)

    contour_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

    img_overlay = cv2.cvtColor(sharp.copy(), cv2.COLOR_GRAY2BGR)
    if SHOW_CONTOUR and not SMOOTH_MASK:
        # (with SMOOTH_MASK the contour is drawn per section from the measured mask, below)
        cv2.drawContours(img_overlay, contours, -1, COLOR_CONTOUR, thickness=2, lineType=cv2.LINE_AA)

    # Second overlay for the fixed-spacing baseline.
    img_overlay_fixed = img_overlay.copy()
    img_overlay_extra = img_overlay.copy()   # fixed spacing at EXTRA_FIXED_SPACING_CM (figure only)
    img_centerline = img_overlay.copy()      # contour + thick centerline + nodes (figure only)
    img_internode = img_overlay.copy()       # internode lengths: curved / straight / vertical (figure only)
    extra_point_count = 0
    a_point_count = 0
    compare_rows = []
    # One image = one stalk; sections are processed left to right, so this
    # running count numbers internodes along the whole stalk (base = 1).
    stalk_internode_offset = 0
    # Running centerline length of the sections already processed (stalk base = 0).
    stalk_offset_px = 0.0

    nodes_all = load_node_labels(label_path, gray.shape)

    merged = cv2.morphologyEx(binary_filled, cv2.MORPH_CLOSE, np.ones((15, 3), np.uint8), iterations=2)
    labeled = measure.label(merged)
    regions_raw = measure.regionprops(labeled)

    def region_bottom_x(region):
        ys, xs = np.nonzero(labeled == region.label)
        if len(ys) == 0:
            return 1e18
        yb = ys.max()
        xs_row = xs[ys == yb]
        if len(xs_row) == 0:
            return float(np.median(xs))
        return float(np.median(xs_row))

    regions = sorted(regions_raw, key=lambda r: region_bottom_x(r))

    rows = []
    curv_profiles = {}
    global_int = int(global_node_start)
    stem_counter = 0

    for region in regions:
        if region.area < 5000:
            continue

        # ── Local ruler calibration for this stem section ──
        if CAL is not None:
            cy, cx = region.centroid
            new_diam, new_vert = scale_at(CAL, cx, cy)
            # stalk_offset_px holds earlier sections' length in pixels at the old scale;
            # re-express it at the new scale so the running stalk length (and the
            # B2 fixed-spacing positions) stay correct in mm.
            stalk_offset_px *= new_vert / px_per_mm_vert
            px_per_mm_diam, px_per_mm_vert = new_diam, new_vert
            print(f"   section at x={cx:.0f}, y={cy:.0f}: "
                  f"px_per_mm_diam={px_per_mm_diam:.3f}, px_per_mm_vert={px_per_mm_vert:.3f}")

        region_mask = (labeled == region.label).astype(np.uint8)
        region_mask = morphology.binary_closing(region_mask, morphology.disk(3))
        region_mask = morphology.binary_dilation(region_mask, morphology.disk(1))
        region_mask = (region_mask * 255).astype(np.uint8)
        region_mask_filled = (region_mask > 0).astype(np.uint8)
        # Fill holes inside the stem, including holes that reach the image edge
        # through the cut end (a pale stem can be thresholded as two strips there).
        region_mask_filled = fill_holes_touching_edge(region_mask_filled)
        if SMOOTH_MASK:
            region_mask_filled = smooth_region_mask(region_mask_filled)
        region_mask = region_mask_filled * 255
        if SHOW_CONTOUR and SMOOTH_MASK:
            # Draw the contour of the exact mask used for the measurements.
            cs, _ = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for _im in (img_overlay, img_overlay_fixed, img_overlay_extra, img_centerline, img_internode):
                cv2.drawContours(_im, cs, -1, COLOR_CONTOUR, thickness=2, lineType=cv2.LINE_AA)

        skeleton = skeletonize_region(region_mask)
        if len(np.nonzero(skeleton)[0]) < 10:
            continue

        # Smooth centerline: every point below is placed on it (C = None falls back to the skeleton).
        C = build_centerline(region_mask_filled, skeleton)

        if SHOW_SKELETON:
            sk_y, sk_x = np.where(skeleton > 0)
            for x, y in zip(sk_x, sk_y):
                img_overlay[y, x] = COLOR_SKELETON

        ys, xs = np.nonzero(region_mask_filled)
        if len(ys) == 0:
            continue

        stem_counter += 1
        stem_id = stem_counter

        stem_bottom_ref_y = int(ys.max())
        stem_top_ref_y = int(ys.min())
        stem_total_len_cm = (stem_bottom_ref_y - stem_top_ref_y) / px_per_mm_vert / 10.0
        curv_profiles[stem_id] = centerline_curvature_profile(C, stem_bottom_ref_y)

        y_bottom_actual = ys.max()
        x_bottom_actual = int(np.median(xs[ys == y_bottom_actual]))

        y_bottom_offset = max(0, y_bottom_actual - 5)
        x_bottom_offset = int(np.median(xs[ys == y_bottom_offset])) if np.any(ys == y_bottom_offset) else x_bottom_actual

        bottom_actual = measure_robust_tangent_diameter(region_mask_filled, skeleton, x_bottom_actual, y_bottom_actual, centerline=C)
        bottom_offset = measure_robust_tangent_diameter(region_mask_filled, skeleton, x_bottom_offset, y_bottom_offset, centerline=C)

        if bottom_offset["diameter_px"] > bottom_actual["diameter_px"]:
            bottom_node = (x_bottom_offset, y_bottom_offset)
        else:
            bottom_node = (x_bottom_actual, y_bottom_actual)

        top_node = (int(np.median(xs[ys == ys.min()])), ys.min())

        y_shift = int(round(px_per_mm_vert))
        new_bottom = (bottom_node[0], bottom_node[1] - y_shift)
        new_top = (top_node[0], top_node[1] + y_shift)

        minr, minc, maxr, maxc = region.bbox

        def in_region(pt):
            return minc <= pt[0] <= maxc and minr <= pt[1] <= maxr

        region_nodes = [pt for pt in nodes_all if in_region(pt)] + [new_bottom]

        top_threshold_y = minr + (maxr - minr) * 0.15
        top_present = any(pt[1] <= top_threshold_y for pt in nodes_all if in_region(pt))
        if not top_present:
            region_nodes.append(new_top)

        region_nodes = sorted(region_nodes, key=lambda p: p[1], reverse=True)

        # Merge node detections that are too close to represent separate internodes.
        collapsed = []
        i = 0
        while i < len(region_nodes):
            if i == len(region_nodes) - 1:
                collapsed.append(region_nodes[i])
                break

            a = region_nodes[i]
            b = region_nodes[i + 1]
            gap_cm = abs(a[1] - b[1]) / px_per_mm_vert / 10.0

            if gap_cm < MIN_NODE_GAP_CM:
                fracs = [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0]
                best_pt = a
                best_d = -1.0
                for f in fracs:
                    x = int((1 - f) * a[0] + f * b[0])
                    y = int((1 - f) * a[1] + f * b[1])
                    x, y = snap_to_skeleton_nearest(x, y, skeleton, search_radius=20)
                    tangent_measurement = measure_robust_tangent_diameter(region_mask_filled, skeleton, x, y, centerline=C)
                    if tangent_measurement["diameter_px"] > best_d:
                        best_d = tangent_measurement["diameter_px"]
                        best_pt = (x, y)
                collapsed.append(best_pt)
                i += 2
            else:
                collapsed.append(a)
                i += 1

        region_nodes = collapsed
        # Nodes are moved onto the centerline (a YOLO box centre or the top fibre tip
        # can lie beside / above the stem).
        if C is not None:
            region_nodes = [snap_to_centerline(C, x, y) for (x, y) in region_nodes]
        else:
            region_nodes = [snap_to_skeleton_nearest(x, y, skeleton, search_radius=20) for (x, y) in region_nodes]

        # Refine internal node locations: YOLO is trusted; the node is only snapped to the
        # node collar, i.e. the diameter PEAK nearest to the YOLO detection, searched within
        # +-NODE_REFINE_CM (never past the midpoints to the neighbouring nodes). If there is
        # no peak (e.g. only a taper step, thicker below the node), the YOLO position is kept.
        if len(region_nodes) >= 3 and NODE_REFINE_CM > 0:
            rn = list(region_nodes)
            win_px = NODE_REFINE_CM * 10 * px_per_mm_vert
            for k in range(1, len(rn) - 1):
                prev_y = rn[k - 1][1]
                curr_x, curr_y = rn[k]
                next_y = rn[k + 1][1]

                h, w = contour_mask.shape[:2]
                y_a = int(np.clip(max(curr_y - win_px, min((prev_y + curr_y) / 2, (curr_y + next_y) / 2)), 0, h - 1))
                y_b = int(np.clip(min(curr_y + win_px, max((prev_y + curr_y) / 2, (curr_y + next_y) / 2)), 0, h - 1))

                prof = []
                for yy in range(y_a, y_b + 1, 2):
                    xx, yy2 = snap_to_skeleton_ywise(curr_x, yy, skeleton)
                    dpx = measure_robust_tangent_diameter(region_mask_filled, skeleton, xx, yy2, centerline=C)["diameter_px"]
                    if dpx > 0:
                        prof.append((dpx, yy2, xx))
                if len(prof) < MIN_VALID_ROWS_FOR_NODE_SNAP:
                    continue
                from scipy.signal import find_peaks
                d = np.array([p[0] for p in prof], float)
                ds = np.convolve(np.pad(d, 2, mode="edge"), np.ones(5) / 5, mode="valid")   # light smoothing
                peaks, _ = find_peaks(ds, prominence=np.median(ds) * NODE_PEAK_MIN_PCT / 100)
                is_peak = len(peaks) > 0
                if is_peak:
                    # node collar = the diameter peak NEAREST to the YOLO detection
                    j = int(peaks[np.argmin([abs(prof[p][1] - curr_y) for p in peaks])])
                    rn[k] = (int(prof[j][2]), int(prof[j][1]))
                NODE_REFINE_LOG.append({
                    "Image": image_path.stem, "Stem": stem_id,
                    "YOLO node y (px)": int(curr_y),
                    "Refined node y (px)": int(rn[k][1]),
                    "Moved (mm)": round(abs(rn[k][1] - curr_y) / px_per_mm_vert, 1),
                    "Decision": "moved to collar peak" if is_peak else "kept YOLO position (no diameter peak)",
                })

            region_nodes = sorted(rn, key=lambda p: p[1], reverse=True)

        if len(region_nodes) < 2:
            continue

        # Assign node names from the base toward the upper stem.
        node_names = []
        if stem_id == 1:
            node_names.append(f"node_{global_int}.0")
        else:
            node_names.append(f"node_{global_int}.1")

        for _ in range(1, len(region_nodes)):
            global_int += 1
            node_names.append(f"node_{global_int}.0")

        # Draw node labels in increasing numeric order.
        numeric_node_labels = []
        for nm in node_names:
            base = nm.split("_")[1]
            n_num = base.split(".")[0]
            numeric_node_labels.append(f"N{n_num}")

        for node_label, (nx, ny) in zip(numeric_node_labels, region_nodes):
            draw_node_label(img_overlay, (nx, ny), node_label)
            draw_node_label(img_overlay_fixed, (nx, ny), node_label)
            draw_node_label(img_overlay_extra, (nx, ny), node_label)

        # Centerline figure: thick centerline, then the nodes on top.
        if C is not None:
            cl_pts = np.round(C["pts"]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img_centerline, [cl_pts], False, COLOR_CENTERLINE_DRAW,
                          CENTERLINE_DRAW_THICKNESS, lineType=cv2.LINE_AA)
        for node_label, (nx, ny) in zip(numeric_node_labels, region_nodes):
            draw_node_label(img_centerline, (nx, ny), node_label)
        a_point_count += SAMPLES_PER_INTERNODE * (len(region_nodes) - 1)

        # Method A diameters per internode, used as the reference profile for Method B.
        node_guided_profile = {k: [] for k in range(len(region_nodes) - 1)}

        # Store rows in the same measurement order used for analysis: node -> D points -> next node.
        # This order is important because internode length is calculated from consecutive node rows.
        for i in range(len(region_nodes) - 1):
            node_a, node_b = region_nodes[i], region_nodes[i + 1]
            name_a, name_b = node_names[i], node_names[i + 1]

            def interpolate(p1, p2, frac):
                return (
                    int((1 - frac) * p1[0] + frac * p2[0]),
                    int((1 - frac) * p1[1] + frac * p2[1])
                )

            # Add the lower node of this internode. The upper node is added as
            # the lower node of the next internode, which prevents duplicate node rows.
            node_x, node_y = node_a
            node_point_len_cm = (stem_bottom_ref_y - node_y) / px_per_mm_vert / 10.0
            node_point_len_cm = max(0.0, node_point_len_cm)

            if i == 0:
                node_stem_len_cm = 0.0
            else:
                node_stem_len_cm = node_point_len_cm

            tangent_measurement_node = measure_robust_tangent_diameter(region_mask_filled, skeleton, node_x, node_y, centerline=C)
            node_diameter_mm = (
                tangent_measurement_node["diameter_px"] / px_per_mm_diam
                if tangent_measurement_node["diameter_px"] > 0 else 0.0
            )

            rows.append({
                "Stem": stem_id,
                "Type": name_a,
                "Y(px)": int(node_y),
                "X(px)": int(node_x),
                "Point Length (cm)": round(node_point_len_cm, 2),
                "Stem Length (cm)": round(node_stem_len_cm, 2),
                "Diameter Tangent (mm)": round(node_diameter_mm, 2),
            })

            # Measure tangent diameters at five fixed positions within this internode.
            # D1-D5 at 10/30/50/70/90 % of the CURVED centerline length between the
            # two nodes (straight-line interpolation could fall outside a bent stem).
            def place(frac):
                if C is not None:
                    return centerline_point_between(C, node_a, node_b, frac)
                return interpolate(node_a, node_b, frac)

            point_defs = [(f"D{k + 1}", place(f)) for k, f in enumerate(INTERNODE_SAMPLE_FRACS)]

            for label, (x0, y0) in point_defs:
                if C is None:
                    x0, y0 = snap_to_skeleton_nearest(x0, y0, skeleton, search_radius=20)

                tangent_measurement = measure_robust_tangent_diameter(region_mask_filled, skeleton, x0, y0, centerline=C)
                diameter_mm = (
                    tangent_measurement["diameter_px"] / px_per_mm_diam
                    if tangent_measurement["diameter_px"] > 0 else 0.0
                )

                center_x, center_y = tangent_measurement["center"]
                center_x, center_y = int(round(center_x)), int(round(center_y))

                point_len_cm = (stem_bottom_ref_y - center_y) / px_per_mm_vert / 10.0
                point_len_cm = max(0.0, point_len_cm)

                rows.append({
                    "Stem": stem_id,
                    "Type": f"{label}_between_{name_a}_{name_b}",
                    "Y(px)": int(center_y),
                    "X(px)": int(center_x),
                    "Point Length (cm)": round(point_len_cm, 2),
                    "Stem Length (cm)": round(point_len_cm, 2),
                    "Diameter Tangent (mm)": round(diameter_mm, 2),
                })

                draw_measurement(img_overlay, tangent_measurement, diameter_mm, label)

                # Record the Method A point for the comparison table.
                k_int, r_pos, d_node_px = locate_in_internodes(center_y, region_nodes)
                d_node_cm = d_node_px / px_per_mm_vert / 10.0
                node_guided_profile[i].append((point_len_cm, diameter_mm))
                compare_rows.append({
                    "Image": image_path.stem,
                    "Stem": stem_id,
                    "Method": "A_node_guided",
                    "Point": f"{label}_between_{name_a}_{name_b}",
                    "Internode": f"{name_a}_{name_b}",
                    "Internode Index": i + 1,
                    "Stalk Internode No.": stalk_internode_offset + i + 1,
                    "Touches Section End": "Yes" if i in (0, len(region_nodes) - 2) else "No",
                    "Sample Position": label,
                    "Y(px)": int(center_y),
                    "X(px)": int(center_x),
                    "Point Length (cm)": round(point_len_cm, 2),
                    "Stalk Position (cm)": round(stalk_position_px(C, center_x, center_y, stalk_offset_px) / px_per_mm_vert / 10.0, 2),
                    "Relative Position (0-1)": round(r_pos, 3) if r_pos is not None else np.nan,
                    "Distance to Nearest Node (cm)": round(d_node_cm, 2),
                    "Within Node Zone": "Yes" if d_node_cm <= NODE_ZONE_CM else "No",
                    "Diameter Tangent (mm)": round(diameter_mm, 2),
                    "Node-Guided Profile Diameter (mm)": np.nan,
                    "Diff vs Node-Guided Profile (mm)": np.nan,
                })

        # Add the final upper node once, after all internodes have been written.
        final_node = region_nodes[-1]
        final_name = node_names[-1]
        final_x, final_y = final_node

        final_point_len_cm = (stem_bottom_ref_y - final_y) / px_per_mm_vert / 10.0
        final_point_len_cm = max(0.0, final_point_len_cm)
        final_stem_len_cm = stem_total_len_cm

        final_measurement = measure_robust_tangent_diameter(region_mask_filled, skeleton, final_x, final_y, centerline=C)
        final_diameter_mm = (
            final_measurement["diameter_px"] / px_per_mm_diam
            if final_measurement["diameter_px"] > 0 else 0.0
        )

        rows.append({
            "Stem": stem_id,
            "Type": final_name,
            "Y(px)": int(final_y),
            "X(px)": int(final_x),
            "Point Length (cm)": round(final_point_len_cm, 2),
            "Stem Length (cm)": round(final_stem_len_cm, 2),
            "Diameter Tangent (mm)": round(final_diameter_mm, 2),
        })

        # ── Internode length by 3 methods + point positions (piece / stalk) ──
        # Straight = Euclidean node-to-node distance; Curved = centerline (skeleton)
        # arc length between the two nodes; Vertical = |y difference|.
        def s_of(x, y):
            return float(C["s"][centerline_index(C, x, y)]) if C is not None else float(stem_bottom_ref_y - y)
        cm = lambda px: round(px / px_per_mm_vert / 10.0, 2)
        for k in range(len(region_nodes) - 1):
            (xa, ya), (xb, yb) = region_nodes[k], region_nodes[k + 1]
            straight = float(np.hypot(xb - xa, yb - ya))
            curved = abs(s_of(xb, yb) - s_of(xa, ya))
            vertical = float(abs(yb - ya))
            # ── internode-length figure: curved part in colour, straight chord dashed,
            #    vertical extent as a bar on the left, values written beside the internode
            col = INTERNODE_COLORS[(stalk_internode_offset + k) % len(INTERNODE_COLORS)]
            if C is not None:
                ia, ib = centerline_index(C, xa, ya), centerline_index(C, xb, yb)
                seg = np.round(C["pts"][min(ia, ib):max(ia, ib) + 1]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img_internode, [seg], False, col, CENTERLINE_DRAW_THICKNESS, cv2.LINE_AA)
            n_dash = max(2, int(straight / 25))
            for q in range(0, n_dash, 2):
                p1 = (int(xa + (xb - xa) * q / n_dash), int(ya + (yb - ya) * q / n_dash))
                p2 = (int(xa + (xb - xa) * (q + 1) / n_dash), int(ya + (yb - ya) * (q + 1) / n_dash))
                cv2.line(img_internode, p1, p2, (40, 40, 40), 5, cv2.LINE_AA)
            ys_m, xs_m = np.nonzero(region_mask_filled)
            bar_x = int(xs_m.max()) + 45                       # vertical bar just right of the stem
            cv2.line(img_internode, (bar_x, int(ya)), (bar_x, int(yb)), col, 10, cv2.LINE_AA)
            for yy in (int(ya), int(yb)):
                cv2.line(img_internode, (bar_x - 22, yy), (bar_x + 22, yy), col, 8, cv2.LINE_AA)
            name_a = numeric_node_labels[k]; name_b = numeric_node_labels[k + 1]
            tx = bar_x + 40
            ty = int((ya + yb) / 2)
            lines_txt = [f"{name_a}-{name_b}", f"C {cm(curved):.2f}", f"S {cm(straight):.2f}", f"V {cm(vertical):.2f}"]
            for li, t in enumerate(lines_txt):
                cv2.putText(img_internode, t, (tx, ty - 70 + 50 * li), cv2.FONT_HERSHEY_SIMPLEX, 1.35,
                            col if li == 0 else (30, 30, 30), 4 if li == 0 else 3, cv2.LINE_AA)
            INTERNODE_LENGTH_LOG.append({
                "Image": image_path.stem, "Stem (section, base = 1)": stem_id,
                "Internode": f"{node_names[k]}_{node_names[k + 1]}",
                "Stalk Internode No.": stalk_internode_offset + k + 1,
                "Touches Section End": "Yes" if k in (0, len(region_nodes) - 2) else "No",
                "Lower node, piece position (cm)": cm(s_of(xa, ya)),
                "Upper node, piece position (cm)": cm(s_of(xb, yb)),
                "Lower node, stalk position (cm)": cm(stalk_offset_px + s_of(xa, ya)),
                "Upper node, stalk position (cm)": cm(stalk_offset_px + s_of(xb, yb)),
                "Straight length (cm)": cm(straight),
                "Curved length, centerline (cm)": cm(curved),
                "Vertical length, y only (cm)": cm(vertical),
                "Curved vs straight (%)": round(100 * (curved / straight - 1), 2) if straight > 0 else np.nan,
                "Vertical vs curved (%)": round(100 * (vertical / curved - 1), 2) if curved > 0 else np.nan,
                "Manual straight, ruler (cm)": "",
                "Manual curved, thread (cm)": "",
                "Manual vertical (cm, optional)": "",
            })
        for node_label, (nx_, ny_) in zip(numeric_node_labels, region_nodes):
            draw_node_label(img_internode, (nx_, ny_), node_label)
            xs_l = int(np.nonzero(region_mask_filled[int(np.clip(ny_, 0, region_mask_filled.shape[0] - 1))])[0].min()) \
                if region_mask_filled[int(np.clip(ny_, 0, region_mask_filled.shape[0] - 1))].any() else int(nx_)
            label = f"{cm(s_of(nx_, ny_)):.1f} cm"
            (tw_, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.25, 3)
            cv2.putText(img_internode, label, (xs_l - tw_ - 25, int(ny_) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 110, 0), 3, cv2.LINE_AA)
        for r in rows:
            if r["Stem"] != stem_id:
                continue
            is_node = str(r["Type"]).startswith("node_")
            sp = s_of(r["X(px)"], r["Y(px)"])
            POSITION_LOG.append({
                "Image": image_path.stem, "Stem (section, base = 1)": stem_id,
                "Method": "Node" if is_node else "A_node_guided",
                "Point": r["Type"] if is_node else str(r["Type"]).split("_")[0],
                "Internode / detail": "" if is_node else str(r["Type"]).split("_between_")[-1],
                "Piece Position (cm)": cm(sp),
                "Stalk Position (cm)": cm(stalk_offset_px + sp),
                "Diameter Tangent (mm)": r["Diameter Tangent (mm)"],
                "X(px)": r["X(px)"], "Y(px)": r["Y(px)"],
            })

        # ── Method B2: fixed-coordinate baseline ─────────────────────────
        # Every FIXED_SPACING_CM from the original stalk base along the curved
        # centerline; nodes are ignored. Same diameter function as Method A.
        n_internodes = len(region_nodes) - 1
        fixed_spacing_cm = FIXED_SPACING_CM
        fixed_pts, section_len_px, fixed_skipped = fixed_spacing_positions(
            stalk_offset_px, FIXED_SPACING_CM * 10 * px_per_mm_vert,
            centerline=C, mask=region_mask_filled, skeleton=skeleton)
        to_cm = lambda px: round(px / px_per_mm_vert / 10.0, 2)

        def end_diameter(from_top):
            """Median diameter a little inside one cut end (the cut face itself is irregular)."""
            if C is None:
                return np.nan
            L = float(C["s"][-1])
            vals = []
            for mm in JOINT_END_MM:
                local = L - mm * px_per_mm_vert if from_top else mm * px_per_mm_vert
                i = int(np.argmin(np.abs(C["s"] - local)))
                r = measure_on_centerline(region_mask_filled, C, *C["pts"][i])
                if r["diameter_px"] > 0:
                    vals.append(r["diameter_px"] / px_per_mm_diam)
            return round(float(np.median(vals)), 2) if vals else np.nan

        B2_SECTION_LOG.append({
            "Image": image_path.stem, "Stem (section, base = 1)": stem_id,
            "Section start on stalk (cm)": to_cm(stalk_offset_px),
            "Section end on stalk (cm)": to_cm(stalk_offset_px + section_len_px),
            "Section length along centerline (cm)": to_cm(section_len_px),
            "Bottom end diameter (mm)": end_diameter(False),
            "Top end diameter (mm)": end_diameter(True),
            "px/mm vertical (calibration)": round(px_per_mm_vert, 4),
            "px/mm diameter (calibration)": round(px_per_mm_diam, 4),
            "B2 points measured": len(fixed_pts),
            f"B2 points skipped (< {END_MEASURE_MM:g} mm from a cut end)": len(fixed_skipped),
            # A section running out of the photo is longer than measured, so every
            # section above it is shifted on the stalk: check these before using B2.
            "Cut by image edge": ", ".join(e for e, hit in (
                ("bottom", bool(region_mask_filled[-1].any())), ("top", bool(region_mask_filled[0].any())))
                if hit) or "no",
        })
        for sp_px, d_px, end in fixed_skipped:
            B2_SKIPPED_LOG.append({
                "Image": image_path.stem, "Stem (section, base = 1)": stem_id,
                "Stalk Position (cm)": to_cm(sp_px),
                "Distance to cut end (mm)": round(d_px / px_per_mm_vert, 1),
                "Cut end": end,
                "Reason": f"within {END_MEASURE_MM:g} mm of the section's cut end (irregular contour)",
            })

        for j, (fx, fy, f_stalk_px) in enumerate(fixed_pts, start=1):
            # Continuous numbering along the whole stalk: F<k> sits at k x FIXED_SPACING_CM
            # from the stalk base (a skipped point leaves a gap in the numbers).
            f_label = f"F{int(round(f_stalk_px / (FIXED_SPACING_CM * 10 * px_per_mm_vert)))}"
            fixed_measurement = measure_robust_tangent_diameter(region_mask_filled, skeleton, fx, fy, centerline=C)
            fixed_diameter_mm = (
                fixed_measurement["diameter_px"] / px_per_mm_diam
                if fixed_measurement["diameter_px"] > 0 else 0.0
            )
            cx, cy = fixed_measurement["center"]
            cx, cy = int(round(cx)), int(round(cy))

            f_len_cm = max(0.0, (stem_bottom_ref_y - cy) / px_per_mm_vert / 10.0)
            k_int, r_pos, d_node_px = locate_in_internodes(cy, region_nodes)
            d_node_cm = d_node_px / px_per_mm_vert / 10.0
            near_node = d_node_cm <= NODE_ZONE_CM

            # Reference: Method A diameter profile of the same internode, interpolated
            # at this height (clamped to D1/D5 outside the 10-90% range).
            ref_mm = np.nan
            if k_int is not None and node_guided_profile.get(k_int):
                prof = sorted(node_guided_profile[k_int])
                ref_mm = float(np.interp(f_len_cm, [p[0] for p in prof], [p[1] for p in prof]))

            compare_rows.append({
                "Image": image_path.stem,
                "Stem": stem_id,
                "Method": B2_LABEL,
                "Point": f_label,
                "Internode": f"{node_names[k_int]}_{node_names[k_int + 1]}" if k_int is not None else "outside",
                "Internode Index": k_int + 1 if k_int is not None else np.nan,
                "Stalk Internode No.": stalk_internode_offset + k_int + 1 if k_int is not None else np.nan,
                "Touches Section End": ("Yes" if k_int in (0, n_internodes - 1) else "No") if k_int is not None else "",
                "Sample Position": f_label,
                "Fixed Spacing (cm)": round(fixed_spacing_cm, 2),
                "Stalk Position (cm)": round(f_stalk_px / px_per_mm_vert / 10.0, 2),
                "Y(px)": cy,
                "X(px)": cx,
                "Point Length (cm)": round(f_len_cm, 2),
                "Relative Position (0-1)": round(r_pos, 3) if r_pos is not None else np.nan,
                "Distance to Nearest Node (cm)": round(d_node_cm, 2),
                "Within Node Zone": "Yes" if near_node else "No",
                "Diameter Tangent (mm)": round(fixed_diameter_mm, 2),
                "Node-Guided Profile Diameter (mm)": round(ref_mm, 2) if not np.isnan(ref_mm) else np.nan,
                "Diff vs Node-Guided Profile (mm)":
                    round(fixed_diameter_mm - ref_mm, 2) if not np.isnan(ref_mm) else np.nan,
            })

            draw_fixed_measurement(img_overlay_fixed, fixed_measurement, fixed_diameter_mm, f_label, near_node)
            POSITION_LOG.append({
                "Image": image_path.stem, "Stem (section, base = 1)": stem_id,
                "Method": B2_LABEL, "Point": f_label,
                "Internode / detail": "near node (< 1 cm)" if near_node else "",
                "Piece Position (cm)": round((f_stalk_px - stalk_offset_px) / px_per_mm_vert / 10.0, 2),
                "Stalk Position (cm)": round(f_stalk_px / px_per_mm_vert / 10.0, 2),
                "Diameter Tangent (mm)": round(fixed_diameter_mm, 2),
                "X(px)": cx, "Y(px)": cy,
            })

        # Extra fixed spacing (EXTRA_FIXED_SPACING_CM), drawn on its own overlay only.
        extra_pts, _, _ = fixed_spacing_positions(
            stalk_offset_px, EXTRA_FIXED_SPACING_CM * 10 * px_per_mm_vert,
            centerline=C, mask=region_mask_filled, skeleton=skeleton)
        for (ex, ey, e_stalk_px) in extra_pts:
            em = measure_robust_tangent_diameter(region_mask_filled, skeleton, ex, ey, centerline=C)
            e_mm = em["diameter_px"] / px_per_mm_diam if em["diameter_px"] > 0 else 0.0
            _, _, e_dnode = locate_in_internodes(int(round(em["center"][1])), region_nodes)
            e_label = f"F{int(round(e_stalk_px / (EXTRA_FIXED_SPACING_CM * 10 * px_per_mm_vert)))}"
            draw_fixed_measurement(img_overlay_extra, em, e_mm, e_label,
                                   e_dnode / px_per_mm_vert / 10.0 <= NODE_ZONE_CM)
            extra_point_count += 1

        stalk_internode_offset += n_internodes
        stalk_offset_px += section_len_px

    # Save final overlay image.
    overlay_path = output_dir / f"{image_path.stem}_tangent_overlay_5points.png"
    save_png(overlay_path, img_overlay)
    print(f"OVERLAY SAVED: {overlay_path}")

    # Save the fixed-spacing overlay image.
    draw_fixed_legend(img_overlay_fixed)
    overlay_fixed_path = output_dir / f"{image_path.stem}_fixed_spacing_overlay.png"
    save_png(overlay_fixed_path, img_overlay_fixed)
    print(f"FIXED-SPACING OVERLAY SAVED: {overlay_fixed_path}")

    # Side-by-side figure: Method A (left) and Method B (right), same image and nodes.
    side_path = output_dir / f"{image_path.stem}_side_by_side_A_vs_B.png"
    save_side_by_side(img_overlay, img_overlay_fixed, side_path)
    print(f"SIDE-BY-SIDE SAVED: {side_path}")

    # Extra comparison at a similar point density (figure only, no Excel).
    draw_fixed_legend(img_overlay_extra)
    tag = f"{EXTRA_FIXED_SPACING_CM:g}".replace(".", "p")
    save_png(output_dir / f"{image_path.stem}_fixed_spacing_{tag}cm_overlay.png", img_overlay_extra)
    save_side_by_side(
        img_overlay, img_overlay_extra,
        output_dir / f"{image_path.stem}_side_by_side_A_vs_fixed_{tag}cm.png",
        title_a=f"A: Node-guided (5 per internode, n = {a_point_count})",
        title_b=f"Fixed coordinates every {EXTRA_FIXED_SPACING_CM:g} cm (n = {extra_point_count})")
    print(f"EXTRA {EXTRA_FIXED_SPACING_CM:g} cm FIGURES SAVED")

    # Centerline figure: shows that the centerline follows the stem curvature.
    lg_scale, lg_th = TEXT_SCALE * 0.8, max(2, TEXT_THICKNESS - 1)
    cv2.rectangle(img_centerline, (10, 10), (900, 130), (255, 255, 255), -1)
    cv2.line(img_centerline, (30, 70), (130, 70), COLOR_CENTERLINE_DRAW, CENTERLINE_DRAW_THICKNESS, cv2.LINE_AA)
    cv2.putText(img_centerline, "Stem centerline", (150, 88), cv2.FONT_HERSHEY_SIMPLEX, lg_scale, (0, 0, 0), lg_th, cv2.LINE_AA)
    save_png(output_dir / f"{image_path.stem}_centerline_overlay.png", img_centerline)
    print("CENTERLINE FIGURE SAVED")

    # Internode-length figure (for manual ruler / thread checks).
    lg = [("C = curved length along the centerline (coloured line), cm", (0, 0, 220)),
          ("S = straight node-to-node length (dashed line), cm", (40, 40, 40)),
          ("V = vertical length, height only (bar right of the stem), cm", (220, 110, 0)),
          ("Green number left of a node = node position on the piece from its bottom, cm", (0, 110, 0))]
    cv2.rectangle(img_internode, (10, 10), (2050, 30 + 60 * len(lg)), (255, 255, 255), -1)
    for li, (t, c_) in enumerate(lg):
        y_ = 60 + 60 * li
        cv2.line(img_internode, (30, y_ - 12), (130, y_ - 12), c_, 10, cv2.LINE_AA)
        cv2.putText(img_internode, t, (150, y_), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
    save_png(output_dir / f"{image_path.stem}_internode_lengths.png", img_internode)
    print("INTERNODE LENGTH FIGURE SAVED")
    df_compare = pd.DataFrame(compare_rows)

    # Save measurement table, per-stem sheets, and summary plots.
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        print(f"No valid tangent measurements found for {image_path.stem}.")
        return

    internode_lengths = []
    last_node_point_len = None
    current_stem = None

    for _, r in df_all.iterrows():
        row_type = str(r["Type"])
        stem_id = r["Stem"]

        if stem_id != current_stem:
            last_node_point_len = None
            current_stem = stem_id

        if row_type.lower().startswith("node_"):
            current_len = float(r["Point Length (cm)"])
            if last_node_point_len is not None:
                internode_lengths.append(round(current_len - last_node_point_len, 2))
            else:
                internode_lengths.append("")
            last_node_point_len = current_len
        else:
            internode_lengths.append("")

    df_all["Internode Length(cms)"] = internode_lengths

    df_all = df_all[[
        "Stem", "Type", "Y(px)", "X(px)",
        "Point Length (cm)", "Stem Length (cm)",
        "Internode Length(cms)",
        "Diameter Tangent (mm)"
    ]]

    df_all["Stem Length (cm)"] = pd.to_numeric(df_all["Stem Length (cm)"], errors="coerce")
    df_all["Diameter Tangent (mm)"] = pd.to_numeric(df_all["Diameter Tangent (mm)"], errors="coerce")

    global_xmin = 0.0
    global_xmax = float(df_all["Stem Length (cm)"].max()) if df_all["Stem Length (cm)"].notna().any() else 0.0

    diameter_values = df_all["Diameter Tangent (mm)"].dropna()
    diameter_values = diameter_values[(diameter_values > 0) & (diameter_values < 200)]
    global_ymin = float(diameter_values.min()) if len(diameter_values) else 0.0
    global_ymax = float(diameter_values.max()) if len(diameter_values) else 1.0
    pad = 0.05 * (global_ymax - global_ymin + 1e-6)
    global_ymin -= pad
    global_ymax += pad

    excel_output_path = output_dir / f"{image_path.stem}_tangent_5point_measurements.xlsx"
    with pd.ExcelWriter(excel_output_path, engine="openpyxl", mode="w") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="All_Points", index=False)

    wb = load_workbook(excel_output_path)
    ws_all = wb["All_Points"]
    current_row = ws_all.max_row + 2

    for stem_id in sorted(df_all["Stem"].unique()):
        df_stem = df_all[df_all["Stem"] == stem_id].copy()
        if df_stem.empty:
            continue

        header_cell = ws_all.cell(row=current_row, column=2, value=f"Stem {stem_id} - Tangent Analysis")
        header_cell.font = Font(bold=True, color="800080")

        total_len = float(df_stem["Stem Length (cm)"].max())
        length_cell = ws_all.cell(row=current_row, column=6, value=f"Total Length = {round(total_len, 2)}cms")
        length_cell.font = Font(bold=True)

        sheet_name = f"Stem_{stem_id}"
        if sheet_name not in wb.sheetnames:
            wb.create_sheet(sheet_name)
        header_cell.hyperlink = Hyperlink(ref="", location=f"'{sheet_name}'!A1", display=header_cell.value)

        for col_idx, col_name in enumerate(df_all.columns, start=1):
            ws_all.cell(row=current_row + 1, column=col_idx, value=col_name).font = Font(bold=True)

        data_start_row = current_row + 2
        for i, row_data in enumerate(df_stem.itertuples(index=False)):
            for col_idx, value in enumerate(row_data, start=1):
                ws_all.cell(row=data_start_row + i, column=col_idx, value=value)
        current_row = data_start_row + len(df_stem) + 4

        ws_stem = wb[sheet_name]
        max_length = int(np.ceil(df_stem["Stem Length (cm)"].max()))
        bins = list(range(0, max_length + 10, 10))
        if len(bins) < 2:
            bins = [0, 10]

        df_stem_summary = df_stem.copy()
        df_stem_summary["Segment"] = pd.cut(
            df_stem_summary["Stem Length (cm)"],
            bins=bins,
            right=False,
            include_lowest=True
        ).astype(str)

        segment_order = df_stem_summary["Segment"].dropna().unique()
        node_counts = (
            df_stem_summary[df_stem_summary["Type"].str.contains("^node_", regex=True, case=False)]
            .groupby("Segment", observed=False)
            .size()
        )
        internode_length_mean = (
            df_stem_summary[df_stem_summary["Type"].str.contains("^node_", regex=True, case=False)]
            .copy()
        )
        internode_length_mean["Internode Length(cms)"] = pd.to_numeric(
            internode_length_mean["Internode Length(cms)"], errors="coerce"
        )
        internode_length_mean = (
            internode_length_mean
            .groupby("Segment", observed=False)["Internode Length(cms)"]
            .mean()
        )
        avg_diameter = df_stem_summary.groupby("Segment", observed=False)["Diameter Tangent (mm)"].mean()

        # Inclination and curvature measured along the smooth centerline.
        prof = curv_profiles.get(stem_id)
        if prof is not None and not prof.empty:
            prof = prof.copy()
            prof["Segment"] = pd.cut(prof["len_cm"], bins=bins, right=False, include_lowest=True).astype(str)
            seg_incl = prof.groupby("Segment")["incl_deg"].mean()
            seg_kappa = prof.groupby("Segment")["kappa_per_m"].mean()
        else:
            seg_incl = pd.Series(dtype=float)
            seg_kappa = pd.Series(dtype=float)

        summary = pd.DataFrame({
            "Segment": segment_order.astype(str),
            "Node Count": node_counts.reindex(segment_order).fillna(0).astype(int).values,
            "Avg Internode Length (cm)": internode_length_mean.reindex(segment_order).values,
            "Avg Tangent Diameter (mm)": avg_diameter.reindex(segment_order).values,
            "Mean Inclination (deg)": seg_incl.reindex(segment_order).round(2).values,
            "Mean Curvature (1/m)": seg_kappa.reindex(segment_order).round(3).values,
        })

        for col_idx, header in enumerate(summary.columns, start=1):
            ws_stem.cell(row=1, column=col_idx, value=header).font = Font(bold=True)
        for row_idx, row in enumerate(summary.itertuples(index=False), start=2):
            for col_idx, value in enumerate(row, start=1):
                ws_stem.cell(row=row_idx, column=col_idx, value=value)

        offset = len(summary) + 2
        ws_stem.cell(row=offset, column=6, value=f"Total Length = {round(total_len, 2)}cms").font = Font(bold=True)

        for col_idx, col_name in enumerate(df_all.columns, start=1):
            ws_stem.cell(row=offset + 2, column=col_idx, value=col_name).font = Font(bold=True)
        for i, row_data in enumerate(df_stem_summary.itertuples(index=False)):
            for col_idx, value in enumerate(row_data, start=1):
                ws_stem.cell(row=offset + 3 + i, column=col_idx, value=value)

        # Segment-wise summary plot (10-cm segments).
        summary_plot = summary.copy()
        summary_plot["Segment Mid"] = summary_plot["Segment"].str.extract(r"(\d+)").astype(float)
        summary_plot = summary_plot.sort_values("Segment Mid")

        fig, ax = plt.subplots(figsize=(12, 6))
        x_labels = summary_plot["Segment"].tolist()
        x_pos = np.arange(len(x_labels))
        ax.plot(x_pos, summary_plot["Node Count"], marker="o", label="Node Count")
        ax.plot(x_pos, summary_plot["Avg Tangent Diameter (mm)"], marker="s", label="Avg Tangent Diameter (mm)")
        # ax.plot(x_pos, summary_plot["Avg Internode Length (cm)"], marker="^", label="Avg Internode Length (cm)")
        ax.plot(x_pos, summary_plot["Mean Inclination (deg)"], marker="d", label="Mean Inclination (deg)")
        ax.set_title(f"Stem {stem_id} - Segment-wise Tangent Analysis")
        ax.set_xlabel("Segment (cm)")
        ax.set_ylabel("Value")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"{image_path.stem}_stem_{stem_id}_segments.png", dpi=FIGURE_DPI)
        plt.close(fig)

        # Diameter profile along stem axis.
        fig_profile, ax_profile = plt.subplots(figsize=(10, 5))
        raw_curve = df_stem[["Stem Length (cm)", "Diameter Tangent (mm)"]].copy()
        raw_curve = raw_curve.dropna().sort_values("Stem Length (cm)")
        raw_curve = raw_curve.drop_duplicates(subset=["Stem Length (cm)"], keep="first")

        if raw_curve.shape[0] >= 2:
            x_raw = raw_curve["Stem Length (cm)"].values
            y_raw = raw_curve["Diameter Tangent (mm)"].values
            x_interp = np.arange(x_raw.min(), x_raw.max() + 1e-6, 0.5)
            y_interp = np.interp(x_interp, x_raw, y_raw)

            ax_profile.plot(x_interp, y_interp, marker="o", linestyle="-", label="Tangent Diameter (mm)")
            ax_profile.set_title(f"Stem {stem_id} - Tangent Diameter vs Stem Length")
            ax_profile.set_xlabel("Stem Length from Bottom (cm)")
            ax_profile.set_ylabel("Diameter Tangent (mm)")
            ax_profile.grid(True)
            ax_profile.legend()
            ax_profile.set_xlim(global_xmin, global_xmax)
            ax_profile.set_ylim(global_ymin, global_ymax)
            plt.tight_layout()
            plt.savefig(output_dir / f"{image_path.stem}_stem_{stem_id}_diameter_vs_length.png", dpi=FIGURE_DPI)
            plt.close(fig_profile)
        else:
            print(f"Skipped diameter vs length plot for Stem {stem_id}; insufficient points.")
            plt.close(fig_profile)

    wb.save(excel_output_path)

    # Add both methods' results to the same workbook (new sheets only).
    if not df_compare.empty:
        per_int = (
            df_compare[df_compare["Internode Index"].notna()]
            .groupby(["Stem", "Internode", "Method"]).size()
            .unstack("Method", fill_value=0)
            .reset_index()
        )
        with pd.ExcelWriter(excel_output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            summarize_methods(df_compare, ["Image", "Stem"]).to_excel(
                writer, sheet_name="Method_Comparison_Summary", index=False)
            df_compare.to_excel(writer, sheet_name="Method_Comparison_Points", index=False)
            per_int.to_excel(writer, sheet_name="Points_per_Internode", index=False)
            img_sec = [r for r in B2_SECTION_LOG if r["Image"] == image_path.stem]
            img_skip = [r for r in B2_SKIPPED_LOG if r["Image"] == image_path.stem]
            pd.DataFrame(img_sec).to_excel(writer, sheet_name="B2_Section_Positions", index=False)
            pd.DataFrame(img_skip, columns=["Image", "Stem (section, base = 1)", "Stalk Position (cm)",
                                            "Distance to cut end (mm)", "Cut end", "Reason"]).to_excel(
                writer, sheet_name="B2_Skipped_Points", index=False)

    save_length_workbooks(image_path, output_dir, excel_output_path)
    print(f"Finished: {excel_output_path}")
    return df_compare


if __name__ == "__main__":
    batch_process_stems()