"""
background_removal.py - RAW photos of cut maize stalks -> stalk on a black and on a white background.

The black-background image is used for node detection, the white-background image for
segmentation and measurement. The stalk mask itself is not used for measuring.

Input  : <input_dir>/<batch>/*.CR3            (one sub-folder per imaging session)
Output : <output_dir>/<batch>_<YYYYmmdd_HHMMSS>/
            raw_to_jpg/<name>.jpg                  converted photo (6000 x 4000)
            blackbg/jpg/<name>_blackbg.jpg         stalks on black
            blackbg/tif/<name>_blackbg.tif
            whitebg/jpg/<name>_whitebg.jpg         stalks on white
            whitebg/tif/<name>_whitebg.tif
            mask/<name>_mask.png                   binary stalk mask
            alpha/<name>_alpha.png                 soft-edged matte
            processing_log.csv, failed_log.csv, params.json
         <output_dir>/converted_batches_log.txt    batches fully completed (skipped next run)

Steps per image:
  0. RAW conversion : rawpy, camera white balance, no auto-brightness, brightness x4.5,
                      oriented to 6000 x 4000 landscape.
  1. Colour seed    : the board colour is estimated locally in CIELAB in every image; pixels
                      whose colour difference from the board (lightness weighted down)
                      exceeds an Otsu threshold are stalk candidates. This works for green,
                      yellow and brown stalks and for changes in lighting. Ruler bands and
                      objects that are not long and thin are removed.
  2. Refinement     : GrabCut (Rother et al., 2004) initialised from the seed mask.
  3. Sheath tissue  : attached non-green tissue added where the stalk lies against the board.
  4. Panicles       : stalks with a panicle are cut at the panicle base (not needed for
                      cut maize sections, but harmless).
  5. Soft edges     : contour smoothing + guided filter (He et al., 2013) alpha matte.
  6. Composites     : culms placed on black (0) and white (255) backgrounds.

Pixel-size parameters refer to --ref_width (2576 px) and are scaled to each image's width.
A batch is logged as done only when every image succeeded; running again resumes unfinished
batches and skips images whose outputs already exist.

Run:  python background_removal.py            (folders from config.py)
      python background_removal.py --input_dir <folder> --output_dir <folder>
"""
import argparse
import csv
import glob
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from config import INPUT_DIR, OUTPUT_DIR


# ============================================================================= SEGMENTATION
# ----------------------------------------------------------------------------- helpers
def odd(n):
    """Nearest odd integer >= 1 (kernel sizes must be odd)."""
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


def scaled(p, scale):
    """Copy of parameter dict with pixel-size parameters scaled to the current image."""
    q = dict(p)
    for k in ("close_k", "open_k", "blur_k", "bg_window", "width_median_window", "bg_kernel", "edge_close"):
        q[k] = odd(p[k] * scale)
    for k in ("ruler_min_width", "gc_band", "tissue_band", "gf_radius", "taper_rows", "min_height",
              "cluster_gap", "cluster_min_rows", "cut_margin", "contrast_radius", "bite_window", "bridge_gap"):
        q[k] = max(1, int(round(p[k] * scale)))
    q["smooth_sigma"] = p["smooth_sigma"] * scale
    q["exclude"] = [[int(round(c * scale)) for c in box] for box in p["exclude"]]
    return q


def fill_holes(m):
    """Fill enclosed holes. The mask is padded with a 1-px background border first, so
    culms running edge-to-edge cannot cut off part of the background from the flood."""
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    hgt, wid = padded.shape
    ff = np.zeros((hgt + 2, wid + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return m | holes


# ----------------------------------------------------------------------------- 1. colour seed
def exg_mask(img_bgr, p):
    """Return a uint8 {0,255} seed mask using ExG + Otsu + HSV gate + morphology."""
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    s = b + g + r + 1e-6
    exg = 2 * g / s - r / s - b / s                      # Excess Green on normalised RGB
    exg8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    exg8 = cv2.GaussianBlur(exg8, (p["blur_k"], p["blur_k"]), 0)
    otsu_t, _ = cv2.threshold(exg8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = exg8 > otsu_t + p["exg_offset"]

    # HSV gate: rejects the white board, glare (low saturation), dark objects (low value)
    # and red/orange rulers (hue outside the green-yellow range). OpenCV hue range is 0-179.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m &= (h >= p["hue_min"]) & (h <= p["hue_max"]) & (sat >= p["sat_min"]) & (val >= p["val_min"])
    m = m.astype(np.uint8) * 255

    return clean_seed(m, p), float(otsu_t)


def sat_mask(img_bgr, p):
    """Seed mask for culms of any colour (green, yellow, brown, purple) on a pale board.
    The board and metal rulers have low HSV saturation (~10-70) while culms are saturated
    (~95-150), so saturation is thresholded with Otsu's method, with a floor of sat_floor
    so that a near-empty image cannot pull the threshold into the board."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[..., 1], (p["blur_k"], p["blur_k"]), 0)
    otsu_t, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(otsu_t), float(p["sat_floor"]))
    m = ((s > t) & (hsv[..., 2] >= p["val_min"])).astype(np.uint8) * 255
    return clean_seed(m, p), t


def culm_hue(img_bgr, p):
    """Median hue of saturated (non-board) pixels in the central part of the image.
    Green culms ~27-35, dry yellow/brown/purple culms ~10-23 (OpenCV hue scale)."""
    H, W = img_bgr.shape[:2]
    c = img_bgr[int(0.15 * H):int(0.85 * H), int(0.15 * W):int(0.85 * W)]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    sel = hsv[..., 1] > p["sat_floor"]
    return float(np.median(hsv[..., 0][sel])) if sel.sum() > 1000 else 0.0


def estimate_board(lab, p):
    """Per-pixel estimate of the board colour in CIELAB, so that uneven lighting, vignetting
    and the board's own colour are measured in every image rather than assumed.
    Two passes at reduced resolution: a large median filter gives a first estimate; pixels
    that differ strongly from it (culms, rulers) are then excluded and the board colour is
    re-estimated from the remaining pixels only (normalised box average), then upsampled."""
    H, W = lab.shape[:2]
    f = p["bg_scale"]
    small = cv2.resize(lab, (max(8, W // f), max(8, H // f)), interpolation=cv2.INTER_AREA).astype(np.float32)
    k = odd(p["bg_kernel"] / f)
    first = np.dstack([cv2.medianBlur(np.ascontiguousarray(small[..., c]).astype(np.uint8), min(k, 255))
                       for c in range(3)]).astype(np.float32)
    d = color_distance(small, first, p)
    board = (d < np.percentile(d, 60)).astype(np.float32)
    num = np.dstack([cv2.boxFilter(small[..., c] * board, -1, (k, k), normalize=False) for c in range(3)])
    den = cv2.boxFilter(board, -1, (k, k), normalize=False)[..., None]
    est = np.where(den > 1, num / np.maximum(den, 1e-6), first)
    return cv2.resize(est, (W, H), interpolation=cv2.INTER_LINEAR)


def color_distance(lab, ref, p):
    """Weighted CIELAB distance. Lightness differences are down-weighted (l_weight) so that
    shadows and lighting gradients count less than real colour differences; a dark culm still
    differs strongly in lightness and is detected."""
    lab = lab.astype(np.float32)
    dL = (lab[..., 0] - ref[..., 0]) * (100.0 / 255.0)
    da = lab[..., 1] - ref[..., 1]
    db = lab[..., 2] - ref[..., 2]
    # shadows are darker than the board, highlights brighter: only darkening beyond the
    # shadow allowance is counted fully
    dL = np.where(dL < 0, np.maximum(0, -dL - p["shadow_dl"]), dL)
    return np.sqrt((p["l_weight"] * dL) ** 2 + da ** 2 + db ** 2)


def board_distance(img_bgr, p):
    """Weighted CIELAB distance of every pixel from the locally estimated board colour."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (p["blur_k"], p["blur_k"]), 0), cv2.COLOR_BGR2LAB)
    return color_distance(lab, estimate_board(lab, p), p)


def contrast_limit(mask, d, t, p, img_bgr=None):
    """Keep only pixels that differ from the board by at least contrast_ratio times the
    strongest difference nearby (window gc_band + 2 px), and by at least 0.6 x the global
    threshold. This removes faint halos, colour casts and shadow fringes that GrabCut or the
    tissue step can attach to a culm, while the culm's own edge (where the difference rises
    steeply towards the culm interior) is kept. Interior holes are refilled afterwards."""
    if p["contrast_ratio"] <= 0:
        return mask
    r = p["gc_band"] + 2
    rx = p["contrast_radius"] if p.get("contrast_radius", 0) > 0 else r
    # wide sideways (halos lie beside the upright culms), short vertically (so a dark node
    # above or below a pale internode does not make the internode look like a halo)
    dmax = cv2.dilate(d.astype(np.float32),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * r + 1)))
    keep = (d > p["contrast_ratio"] * dmax) & (d > 0.6 * t)
    if img_bgr is not None and p.get("tissue_sat_keep", 0) > 0:
        # strongly saturated pixels are culm tissue, never a pale halo or shadow
        sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1]
        keep |= (sat >= p["tissue_sat_keep"]) & (d > 0.6 * t)
    out = ((mask > 0) & keep).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = fill_holes(out)
    # put back removed pieces that the culm mostly surrounds (pale spots or reflections on
    # the culm surface); a halo or shadow lies beside the culm and touches it on one side only
    removed = ((mask > 0) & (out == 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(removed, connectivity=8)
    ring_k = np.ones((3, 3), np.uint8)
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(mask.shape[1], x + w + 2), min(mask.shape[0], y + h + 2)
        comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(comp, ring_k) - comp
        if ring.sum() and (out[y0:y1, x0:x1][ring > 0] > 0).mean() >= p["enclosed_frac"]:
            out[y0:y1, x0:x1][comp > 0] = 255
    return restore_bites(mask, out, p, d)


def restore_bites(mask, out, p, d):
    """Put back removed pixels that are bites out of the culm rather than halos, row by row.
    For each culm, its normal width at every row is the running median of its width over
    bite_window rows above and below. Where the culm is narrower than bite_ratio x normal, removed
    pixels touching it in that row are put back, most culm-like first (largest colour difference
    from the board), until the row reaches its normal width; halo pixels beyond that stay out."""
    r = p.get("bite_ratio", 0)
    if r <= 0:
        return out
    removed = (mask > 0) & (out == 0)
    if not removed.any():
        return out
    H, W = out.shape
    win = p["bite_window"]
    n, lab, st, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if h < 3 * win // 2:
            continue
        pad = max(4, w // 2)
        x0, x1 = max(0, x - pad), min(W, x + w + pad)
        c = lab[y:y + h, x0:x1] == i
        rem = removed[y:y + h, x0:x1]
        dd = d[y:y + h, x0:x1]
        now = c.sum(axis=1).astype(np.float32)
        valid = now > 0
        if valid.sum() < 10:
            continue
        idx = np.where(valid)[0]
        sub = out[y:y + h, x0:x1]
        for k in idx:
            lo, hi = np.searchsorted(idx, [k - win, k + win])
            ref = float(np.median(now[idx[lo:hi]]))
            if now[k] >= r * ref:
                continue
            cols = np.where(c[k])[0]
            a0, a1 = cols.min(), cols.max()
            # removed runs directly left and right of the culm in this row
            cand = []
            j = a0 - 1
            while j >= 0 and rem[k, j]:
                cand.append(j); j -= 1
            j = a1 + 1
            while j < c.shape[1] and rem[k, j]:
                cand.append(j); j += 1
            need = int(round(ref - now[k]))
            if need <= 0 or not cand:
                continue
            # grow outwards from the culm, taking the side whose next pixel is more culm-like
            left = sorted([j for j in cand if j < a0], reverse=True)
            right = sorted([j for j in cand if j > a1])
            li = ri = 0
            for _ in range(need):
                lv = dd[k, left[li]] if li < len(left) else -1
                rv = dd[k, right[ri]] if ri < len(right) else -1
                if lv < 0 and rv < 0:
                    break
                if lv >= rv:
                    sub[k, left[li]] = 255; li += 1
                else:
                    sub[k, right[ri]] = 255; ri += 1
    return out

def board_mask(img_bgr, p):
    """Seed mask for culms of any colour or lightness: pixels whose weighted CIELAB distance
    from the locally estimated board colour exceeds an Otsu threshold (floor diff_floor)."""
    d = board_distance(img_bgr, p)
    d8 = np.clip(d * p["diff_gain"], 0, 255).astype(np.uint8)
    otsu_t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(otsu_t / p["diff_gain"], p["diff_floor"])
    m = (d > t).astype(np.uint8) * 255
    return clean_seed(m, p), float(t)


def seed_mask(img_bgr, p):
    """Seed mask. Default 'board': colour difference from the locally estimated board, which
    works for any culm colour, mixed images and changed lighting or board colour.
    'exg' and 'sat' are kept as alternatives. Returns (mask, threshold, mode, culm_hue)."""
    hue = culm_hue(img_bgr, p)
    mode = p["seed_mode"]
    if mode in ("board", "auto"):
        m, t = board_mask(img_bgr, p)
        mode = "board"
    elif mode == "exg":
        m, t = exg_mask(img_bgr, p)
    else:
        m, t = sat_mask(img_bgr, p)
    return m, t, mode, hue


def remove_ruler_bands(m, frac, min_width, edge=0.12):
    """Rulers lie along the image edges. A side ruler is a run of columns within `edge` of the
    left/right border whose mask fraction exceeds `frac`; it is removed. A bottom/top ruler is
    a run of rows within `edge` of the top/bottom border whose fraction exceeds `frac`; inside
    it only pixels continuing a culm from just outside the band are kept. Column runs must be
    at least min_width px wide, so a thin culm lying next to a side ruler is not removed.
    Finally, in the outer edge band at the bottom and top of the image only pixels that
    continue a culm row by row are kept, which removes thin horizontal ruler edges."""
    if frac <= 0:
        return m
    H, W = m.shape
    out = m.copy()
    colfrac = (m > 0).mean(axis=0)
    cols = np.where(colfrac > frac)[0]
    cols = cols[(cols < edge * W) | (cols >= (1 - edge) * W)]
    if len(cols):
        for g in np.split(cols, np.where(np.diff(cols) > 1)[0] + 1):
            x0, x1 = g[0], g[-1] + 1
            # extend the run a little while columns stay mostly foreground (ruler edges),
            # but never far enough to swallow a culm lying next to the ruler
            lim = max(3, min_width // 6)
            x0_min, x1_max = max(0, x0 - lim), min(W, x1 + lim)
            while x0 > x0_min and colfrac[x0 - 1] > 0.5 * frac:
                x0 -= 1
            while x1 < x1_max and colfrac[x1] > 0.5 * frac:
                x1 += 1
            if (x0 < edge * W or x1 > (1 - edge) * W) and (x1 - x0) >= min_width:
                out[:, x0:x1] = 0
    rowfrac = (out > 0).mean(axis=1)
    rows = np.where(rowfrac > frac)[0]
    rows = rows[(rows < edge * H) | (rows >= (1 - edge) * H)]
    if len(rows):
        for g in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1):
            y0, y1 = g[0], g[-1] + 1
            while y0 > 0 and rowfrac[y0 - 1] > 0.5 * frac:
                y0 -= 1
            while y1 < H and rowfrac[y1] > 0.5 * frac:
                y1 += 1
            y0, y1 = max(0, y0 - 3), min(H, y1 + 3)
            ref_y = y0 - 3 if y0 >= 3 else min(H - 1, y1 + 3)
            ref = cv2.dilate((out[ref_y:ref_y + 1] > 0).astype(np.uint8), np.ones((1, 9), np.uint8))
            out[y0:y1] = out[y0:y1] * ref
    # vertical-continuity filter in the bottom and top edge bands
    band = int(edge * H)
    k = np.ones((1, 3), np.uint8)                      # culm may shift at most 1 px per row
    for rng in (range(H - band, H), range(band - 1, -1, -1)):
        rows_ = list(rng)
        prev_y = rows_[0] - 1 if rows_[0] > 0 and rng.step == 1 else min(H - 1, rows_[0] + 1)
        prev = (out[prev_y] > 0).astype(np.uint8)
        for y in rows_:
            keep = (out[y] > 0).astype(np.uint8) & cv2.dilate(prev[None, :], k)[0]
            out[y] = keep * 255
            prev = keep
    return out


def clean_seed(m, p):
    """Shared clean-up: exclusion boxes, ruler bands, morphology, holes, small objects."""
    for (x0, y0, x1, y1) in p["exclude"]:                # optional exclusion boxes
        m[y0:y1, x0:x1] = 0
    m = remove_ruler_bands(m, p["ruler_frac"], p["ruler_min_width"])

    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["close_k"], p["close_k"]))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["open_k"], p["open_k"]))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
    m = fill_holes(m)

    min_area = p["min_area_frac"] * m.shape[0] * m.shape[1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


# ----------------------------------------------------------------------------- 2. GrabCut
def grabcut_refine(img_bgr, mask, band, iters, seed, margin_factor=4):
    """Refine the seed mask with GrabCut. Eroded mask = definite foreground, outside the
    dilated mask = definite background; the band in between is decided by the colour models
    and graph cut. GrabCut is run separately on a crop around each culm (culm + band + a
    margin of margin_factor x band of surrounding background), which keeps memory and time
    low at full camera resolution. The RNG is re-seeded per crop so results do not depend on
    file order."""
    if band <= 0 or iters <= 0 or mask.max() == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    sure_fg = cv2.erode(mask, k)
    maybe = cv2.dilate(mask, k)
    gc_full = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    gc_full[maybe > 0] = cv2.GC_PR_BGD
    gc_full[mask > 0] = cv2.GC_PR_FGD
    gc_full[sure_fg > 0] = cv2.GC_FGD

    out = np.zeros_like(mask)
    H, W = mask.shape
    margin = margin_factor * band
    n, lab, st, _ = cv2.connectedComponentsWithStats(maybe, connectivity=8)
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
        gc = gc_full[y0:y1, x0:x1].copy()
        if not ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).any() or not (gc == cv2.GC_BGD).any():
            continue
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(seed)
        cv2.grabCut(np.ascontiguousarray(img_bgr[y0:y1, x0:x1]), gc, None, bgd, fgd, iters,
                    cv2.GC_INIT_WITH_MASK)
        fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        own = lab[y0:y1, x0:x1] == i                     # write back only this culm's region
        out[y0:y1, x0:x1][fg & own] = 255
    return fill_holes(out)


# ----------------------------------------------------------------------------- 3. sheath tissue
def add_nongreen_tissue(img_bgr, mask, p):
    """Recover attached non-green tissue (senescent / pale leaf sheath at nodes and edges).
    Within tissue_band px of the culm, pixels with saturation >= tissue_sat_min, hue in
    [tissue_hue_min, tissue_hue_max] and value in [tissue_val_min, tissue_val_max] are added,
    but only where the local background (bg_window, culm excluded) has mean saturation
    <= board_sat_max, i.e. the culm lies against the board rather than a ruler."""
    band = p["tissue_band"]
    if band <= 0 or mask.max() == 0:
        return mask
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    tissue = ((s >= p["tissue_sat_min"]) & (v >= p["tissue_val_min"]) & (v <= p["tissue_val_max"]) &
              (h >= p["tissue_hue_min"]) & (h <= p["tissue_hue_max"])).astype(np.uint8) * 255
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2))

    bg = (near == 0).astype(np.float32)
    ksz = (p["bg_window"], p["bg_window"])
    local_bg_sat = cv2.boxFilter(s.astype(np.float32) * bg, -1, ksz, normalize=False) / \
        (cv2.boxFilter(bg, -1, ksz, normalize=False) + 1e-6)
    on_board = ((local_bg_sat <= p["board_sat_max"]) &
                (cv2.boxFilter(bg, -1, ksz) > 0.05)).astype(np.uint8) * 255
    cand = cv2.bitwise_and(cv2.bitwise_and(tissue, near), on_board)
    merged = cv2.bitwise_or(mask, cand)
    n, lab = cv2.connectedComponents(merged, connectivity=8)
    touching = np.unique(lab[mask > 0])
    out = np.isin(lab, touching[touching > 0]).astype(np.uint8) * 255
    return fill_holes(out)


# ----------------------------------------------------------------------------- 4. panicles
def bridge_gaps(mask, max_gap, n_end=6):
    """Join culm pieces that line up end to end across a short gap. Very pale culm sections can
    be almost the colour of the board and drop out of the mask, cutting a culm in two. When the
    bottom end of one piece and the top end of another are at most max_gap rows apart and their
    centres are no further apart than the culm width, the gap is filled with a strip of the
    narrower end width, its centre interpolated linearly between the two ends."""
    if max_gap <= 0:
        return mask
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    ends = {}
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if h < 4 * n_end:
            continue
        comp = lab[y:y + h, x:x + w] == i
        def end_info(rows):
            cs = [np.where(comp[r])[0] for r in rows if comp[r].any()]
            if not cs:
                return None
            return (float(np.mean([c.mean() for c in cs])) + x, float(np.median([c.size for c in cs])))
        top = end_info(range(0, n_end))
        bot = end_info(range(h - n_end, h))
        if top and bot:
            ends[i] = (y, y + h - 1, top, bot)
    out = mask.copy()
    for a, (ya0, ya1, _, bota) in ends.items():
        best = None
        for b, (yb0, yb1, topb, _) in ends.items():
            gap = yb0 - ya1
            if b == a or gap <= 1 or gap > max_gap:
                continue
            if abs(bota[0] - topb[0]) > max(bota[1], topb[1]):
                continue
            if best is None or gap < best[0]:
                best = (gap, b, topb)
        if best is None:
            continue
        gap, b, topb = best
        wdt = min(bota[1], topb[1])
        for k in range(1, gap):
            t = k / gap
            cx = (1 - t) * bota[0] + t * topb[0]
            x0, x1 = int(round(cx - wdt / 2)), int(round(cx + wdt / 2))
            out[ya1 + k, max(0, x0):min(out.shape[1], x1 + 1)] = 255
    return out


def keep_elongated(mask, min_elong, min_area_frac):
    """Remove objects that are not culm-shaped: culms are long and thin, so components whose
    minimum-area rectangle is less than min_elong times longer than wide (e.g. the edge of a
    pot or a stray patch) or smaller than min_area_frac of the image are dropped."""
    if min_elong <= 0:
        return mask
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    min_area = min_area_frac * mask.shape[0] * mask.shape[1]
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.nonzero(lab[st[i, 1]:st[i, 1] + st[i, 3], st[i, 0]:st[i, 0] + st[i, 2]] == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        if max(rw, rh) >= min_elong * max(1.0, min(rw, rh)):
            out[lab == i] = 255
    return out


def _row_profile(comp, x, y, w, h):
    widths = np.zeros(h)
    runs = np.zeros(h, int)
    for k in range(h):
        row = comp[y + k, x:x + w].astype(np.int8)
        d = np.diff(np.r_[0, row, 0])
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        runs[k] = len(st)
        widths[k] = (en - st).sum()
    return widths, runs


def _rolling_median(v, size):
    half = size // 2
    pad = np.pad(v, half, mode="edge")
    return np.array([np.median(pad[i:i + size]) for i in range(len(v))])


def trim_panicles(mask, p):
    """Cut panicle-bearing culms flat at the panicle base.
    A culm is panicle-bearing when its apex tapers (mean width of the top taper_rows rows <
    taper_max x median width), or, for culms running out of the top of the frame, when side
    branches are found. The panicle base is the lowest branch
    point in the upper panicle_search fraction of the culm: clusters of rows with a detached
    piece or width > branch_ratio x local median (width_median_window rows); if none, the
    widest row there (the fork). The culm is cut where width returns to <= return_ratio x
    normal width, plus cut_margin rows."""
    if p["taper_max"] <= 0:
        return mask, 0
    out = mask.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    trimmed = 0
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if h < p["min_height"]:                   # debris
            continue
        comp = lab == i
        wd, rn = _row_profile(comp, x, y, w, h)
        med = np.median(wd)
        at_top = y == 0                           # culm/panicle continues out of the frame:
        if not at_top and wd[:p["taper_rows"]].mean() >= p["taper_max"] * med:   # flat cut end
            continue
        upper = max(12, int(p["panicle_search"] * h))
        loc = _rolling_median(wd, p["width_median_window"])
        flag = (wd > p["branch_ratio"] * loc) | (rn > 1)
        flag[:6] = False
        idx = np.where(flag[:upper])[0]
        base = None
        if len(idx):
            groups = np.split(idx, np.where(np.diff(idx) > p["cluster_gap"])[0] + 1)
            groups = [g for g in groups if len(g) >= p["cluster_min_rows"]]
            # a real branch cluster must contain rows where the mask widens sideways
            # (branches leaving the rachis); split fibres and ragged nodes only create
            # parallel runs without widening and are ignored
            wide = wd > p["branch_ratio"] * loc
            groups = [g for g in groups if wide[g].sum() >= max(2, p["cluster_min_rows"] // 2)]
            if groups:
                base = int(groups[-1][-1])
        if base is None and at_top:               # out of frame: only clear branching counts
            continue
        if base is None:                          # no branch cluster: look for a fork
            cand = int(np.argmax(wd[10:upper]) + 10)
            below = np.median(wd[cand + 20:cand + 170]) if cand + 30 < h else wd[cand]
            above = np.median(wd[10:max(11, cand - 10)])
            # a fork needs a widening at the fork AND a thinner rachis above it; a node on a
            # cut culm widens too, but the internode above is as wide as the one below
            if wd[cand] < p["fork_ratio"] * below or above > p["rachis_ratio"] * below:
                continue
            base = cand
        normal = np.median(wd[base:base + 150])
        cut = base
        while cut < min(h - 1, upper + 60) and wd[cut] > p["return_ratio"] * normal:
            cut += 1
        cut += p["cut_margin"]
        out[y:y + cut][comp[y:y + cut]] = 0
        trimmed += 1
    return out, trimmed


# ----------------------------------------------------------------------------- 5. soft edges
def smooth_contour(mask, sigma):
    """Remove stair-step boundaries: Gaussian-blur the binary mask, re-threshold at 0.5."""
    if sigma <= 0:
        return mask
    b = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigma)
    return (b >= 0.5).astype(np.uint8) * 255


def guided_filter(guide_gray, src, radius, eps):
    """Grey-scale guided filter (He, Sun & Tang, 2013). Inputs float32 in [0,1]."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide_gray, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    cov_Ip = cv2.boxFilter(guide_gray * src, -1, k) - mean_I * mean_p
    var_I = cv2.boxFilter(guide_gray * guide_gray, -1, k) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, k) * guide_gray + cv2.boxFilter(b, -1, k)


def refine_alpha(img_bgr, mask, sigma, radius, eps):
    """Binary mask -> smooth, edge-aware soft alpha matte in [0,1]."""
    m = smooth_contour(mask, sigma)
    if radius <= 0:
        return m.astype(np.float32) / 255.0, m
    guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    alpha = np.clip(guided_filter(guide, m.astype(np.float32) / 255.0, radius, eps), 0, 1)
    k = np.ones((2 * radius + 1,) * 2, np.uint8)
    band = cv2.dilate(m, k) - cv2.erode(m, k)              # soft only near the boundary
    out = m.astype(np.float32) / 255.0
    out[band > 0] = alpha[band > 0]
    return out, m


# ----------------------------------------------------------------------------- 6. composite
def composite(img_bgr, alpha, bg_value, feather):
    a = alpha.astype(np.float32)
    if a.max() > 1:
        a = a / 255.0
    if feather > 0:
        k = feather * 2 + 1
        a = cv2.GaussianBlur(a, (k, k), 0)
    a = a[..., None]
    bg = np.full_like(img_bgr, bg_value, dtype=np.uint8)
    return np.clip(np.round(a * img_bgr + (1 - a) * bg), 0, 255).astype(np.uint8)



# ============================================================================= PIPELINE
# CR3 batch folders -> converted JPG -> culm mask -> black / white background JPG + TIF


# ----------------------------------------------------------------------------- CR3 -> image
def convert_cr3(input_path):
    """Decode a Canon CR3 and turn it to 6000 x 4000 landscape.
    Returns (RGB uint8 array, note) or raises on failure."""
    import rawpy
    with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, bright=4.5)
    img = Image.fromarray(rgb)
    note = f"original {img.size[0]}x{img.size[1]}"

    if img.size == (6000, 4000):
        pass
    elif img.size == (4000, 6000):
        img = img.rotate(90, expand=True)
        img = img.rotate(180)
        note += ", rotated"
    else:
        if img.height > img.width:
            img = img.rotate(90, expand=True)
        src_ratio = img.width / img.height
        if abs(src_ratio - 1.5) > 0.01:
            note += f", WARNING aspect {src_ratio:.3f} stretched to 1.5"
        img = img.resize((6000, 4000))
        img = img.rotate(180)
        note += ", resized"
    return np.asarray(img.convert("RGB")), note


def load_image(path):
    """CR3 -> converted array; JPG/PNG/TIF are read as they are (already converted)."""
    if path.lower().endswith(".cr3"):
        return convert_cr3(path)
    return np.asarray(Image.open(path).convert("RGB")), "already converted"


# ----------------------------------------------------------------------------- saving
def save_rgb(arr_rgb, path, jpg_quality):
    img = Image.fromarray(arr_rgb)
    if path.lower().endswith((".jpg", ".jpeg")):
        img.save(path, quality=jpg_quality, subsampling=0)
    elif path.lower().endswith((".tif", ".tiff")):
        img.save(path, compression="tiff_lzw")          # lossless
    else:
        img.save(path)


# ----------------------------------------------------------------------------- one image
def process_one(task):
    """Worker: convert, segment, composite and save one image. Returns a log row."""
    (src, dirs, name, p, seed, jpg_quality, save_tif, save_masks) = task
    t0 = time.time()
    row = {"Filename": os.path.basename(src)}
    try:
        rgb, note = load_image(src)
        row["Conversion"] = note
        save_rgb(rgb, os.path.join(dirs["converted"], name + ".jpg"), jpg_quality)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        # segment at work_width (faster; the parameters were tuned there), then scale the mask
        # back up; soft edges and composites are made at full resolution
        ww = p.get("work_width", 0)
        if ww and W > ww:
            seg = cv2.resize(bgr, (ww, int(round(H * ww / W))), interpolation=cv2.INTER_AREA)
        else:
            seg = bgr
        scale = seg.shape[1] / p["ref_width"]
        q = scaled(p, scale)
        mask, otsu_t, mode, hue = seed_mask(seg, q)
        mask = grabcut_refine(seg, mask, q["gc_band"], q["gc_iters"], seed)
        mask = add_nongreen_tissue(seg, mask, q)
        if mode == "board":
            mask = contrast_limit(mask, board_distance(seg, q), otsu_t, q, seg)
        if q["edge_close"] > 1:
            # fill short bites along the culm edges (a vertical closing: only gaps shorter than
            # edge_close px between mask pixels in the same column are filled, so halos that run
            # alongside the culm are not added back)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (1, q["edge_close"])))
        mask = bridge_gaps(mask, q["bridge_gap"])
        mask = keep_elongated(mask, q["min_elong"], q["min_area_frac"])
        mask, n_trim = trim_panicles(mask, q)
        if seg is not bgr:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = (mask >= 128).astype(np.uint8) * 255
        qf = scaled(p, W / p["ref_width"])
        alpha, mask = refine_alpha(bgr, mask, qf["smooth_sigma"], qf["gf_radius"], qf["gf_eps"])
        if mask.max() == 0:
            raise RuntimeError("no culm detected")

        black = cv2.cvtColor(composite(bgr, alpha, 0, qf["feather"]), cv2.COLOR_BGR2RGB)
        white = cv2.cvtColor(composite(bgr, alpha, 255, qf["feather"]), cv2.COLOR_BGR2RGB)
        save_rgb(black, os.path.join(dirs["black_jpg"], name + "_blackbg.jpg"), jpg_quality)
        save_rgb(white, os.path.join(dirs["white_jpg"], name + "_whitebg.jpg"), jpg_quality)
        if save_tif:
            save_rgb(black, os.path.join(dirs["black_tif"], name + "_blackbg.tif"), jpg_quality)
            save_rgb(white, os.path.join(dirs["white_tif"], name + "_whitebg.tif"), jpg_quality)
        if save_masks:
            cv2.imwrite(os.path.join(dirs["mask"], name + "_mask.png"), mask)
            cv2.imwrite(os.path.join(dirs["alpha"], name + "_alpha.png"),
                        (alpha * 255).round().astype(np.uint8))

        row.update({"Status": "ok", "Width": bgr.shape[1], "Height": bgr.shape[0],
                    "Scale": round(scale, 4), "WorkWidth": seg.shape[1], "SeedMode": mode, "CulmHue": round(hue, 1),
                    "OtsuThreshold": otsu_t,
                    "PaniclesTrimmed": n_trim, "Seconds": round(time.time() - t0, 1)})
    except Exception as e:
        row.update({"Status": "failed", "Error": f"{type(e).__name__}: {e}",
                    "Seconds": round(time.time() - t0, 1)})
    return row


def outputs_exist(dirs, name, save_tif):
    need = [os.path.join(dirs["black_jpg"], name + "_blackbg.jpg"),
            os.path.join(dirs["white_jpg"], name + "_whitebg.jpg")]
    if save_tif:
        need += [os.path.join(dirs["black_tif"], name + "_blackbg.tif"),
                 os.path.join(dirs["white_tif"], name + "_whitebg.tif")]
    return all(os.path.exists(f) for f in need)


# ----------------------------------------------------------------------------- batches
def make_dirs(session_dir, save_tif, save_masks):
    black, white = "blackbg", "whitebg"
    dirs = {"converted": os.path.join(session_dir, "raw_to_jpg"),
            "black_jpg": os.path.join(session_dir, black, "jpg"),
            "white_jpg": os.path.join(session_dir, white, "jpg"),
            "black_tif": os.path.join(session_dir, black, "tif"),
            "white_tif": os.path.join(session_dir, white, "tif"),
            "mask": os.path.join(session_dir, "mask"),
            "alpha": os.path.join(session_dir, "alpha")}
    for k, d in dirs.items():
        if (k.endswith("_tif") and not save_tif) or (k in ("mask", "alpha") and not save_masks):
            continue
        os.makedirs(d, exist_ok=True)
    return dirs


def write_log(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def find_session(base_output_dir, batch):
    """Most recent unfinished session folder for this batch (to resume), else None."""
    cands = sorted(d for d in glob.glob(os.path.join(base_output_dir, batch + "_*"))
                   if os.path.isdir(d) and os.path.exists(os.path.join(d, "params.json")))
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default=str(INPUT_DIR),
                    help="folder containing one sub-folder per batch of CR3 images")
    ap.add_argument("--output_dir", default=str(OUTPUT_DIR))
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="images processed in parallel (each needs ~1.5 GB RAM at 6000x4000)")
    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--no_tif", action="store_true", help="do not write TIF copies")
    ap.add_argument("--no_masks", action="store_true", help="do not write mask / alpha PNGs")
    ap.add_argument("--include_converted", action="store_true",
                    help="also process JPG/PNG/TIF files found in batch folders (no CR3 conversion)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for GrabCut")
    ap.add_argument("--seed_mode", choices=["board", "exg", "sat"], default="board",
                    help="board: colour difference from the locally estimated board (any stalk "
                         "colour); exg: Excess Green (green stalks only); sat: saturation")
    ap.add_argument("--work_width", type=int, default=2576,
                    help="segment at this width (mask is scaled back up; outputs stay full size). "
                         "0 = segment at full resolution (slower)")
    ap.add_argument("--ref_width", type=int, default=2576,
                    help="image width the pixel-size parameters refer to; scaled per image")
    g = ap.add_argument_group("1. colour seed")
    g.add_argument("--blur_k", type=int, default=5)
    g.add_argument("--sat_floor", type=float, default=75, help="minimum saturation threshold (sat mode)")
    g.add_argument("--l_weight", type=float, default=0.35, help="board mode: weight of lightness difference")
    g.add_argument("--shadow_dl", type=float, default=8, help="board mode: darkening (L*) tolerated as shadow")
    g.add_argument("--contrast_ratio", type=float, default=0.55,
                   help="board mode: pixel must differ from the board by this fraction of the nearby culm's difference (0 = off)")
    g.add_argument("--contrast_radius", type=int, default=45,
                   help="board mode: search distance (px at ref_width) for the strongest nearby culm "
                        "difference in the halo filter; must exceed the width of shadows/halos (0 = gc_band+2)")
    g.add_argument("--tissue_sat_keep", type=int, default=100,
                   help="board mode: pixels with HSV saturation >= this are never removed by the halo "
                        "filter (culm tissue; halos are pale, saturation ~50-90). 0 = off")
    g.add_argument("--bite_ratio", type=float, default=0.85,
                   help="halo filter: where a culm is narrower than this x its normal width, removed pixels next "
                        "to it are put back up to the normal width (bites out of a shaded culm side). 0 = off")
    g.add_argument("--bite_window", type=int, default=120,
                   help="rows above/below a removed piece used to measure the culm's normal width (px at ref_width)")
    g.add_argument("--bridge_gap", type=int, default=80,
                   help="join culm pieces lined up end to end across a gap of at most this many rows "
                        "(px at ref_width; very pale sections that match the board). 0 = off")
    g.add_argument("--edge_close", type=int, default=31,
                   help="fill bites along culm edges shorter than this (px at ref_width, vertical); 0 = off")
    g.add_argument("--enclosed_frac", type=float, default=0.6,
                   help="board mode: pixels removed by the halo filter are restored when at least this "
                        "fraction of their outline touches the culm (pale spots on the culm surface)")
    g.add_argument("--diff_floor", type=float, default=10, help="board mode: minimum colour-difference threshold")
    g.add_argument("--diff_gain", type=float, default=4, help="board mode: scaling of distance before Otsu")
    g.add_argument("--bg_scale", type=int, default=8, help="board mode: downsampling for board estimate")
    g.add_argument("--bg_kernel", type=int, default=400, help="board mode: board-estimate window (px)")
    g.add_argument("--ruler_min_width", type=int, default=40, help="minimum ruler band width (px)")
    g.add_argument("--ruler_frac", type=float, default=0.5,
                   help="row/column fraction of mask above which a band is treated as a ruler (0 = off)")
    g.add_argument("--exg_offset", type=float, default=0.0)
    g.add_argument("--hue_min", type=int, default=20)
    g.add_argument("--hue_max", type=int, default=95)
    g.add_argument("--sat_min", type=int, default=40)
    g.add_argument("--val_min", type=int, default=40)
    g.add_argument("--close_k", type=int, default=7)
    g.add_argument("--open_k", type=int, default=3)
    g.add_argument("--min_area_frac", type=float, default=5e-4)
    g.add_argument("--min_elong", type=float, default=3.0,
                   help="objects less than this many times longer than wide are removed (0 = off)")
    g.add_argument("--exclude", default="[]", help="JSON list of [x0,y0,x1,y1] boxes at ref_width")
    g = ap.add_argument_group("2. GrabCut")
    g.add_argument("--gc_band", type=int, default=12)
    g.add_argument("--gc_iters", type=int, default=5)
    g = ap.add_argument_group("3. sheath tissue")
    g.add_argument("--tissue_band", type=int, default=20)
    g.add_argument("--tissue_sat_min", type=int, default=80)
    g.add_argument("--tissue_hue_min", type=int, default=12)
    g.add_argument("--tissue_hue_max", type=int, default=45)
    g.add_argument("--tissue_val_min", type=int, default=100)
    g.add_argument("--tissue_val_max", type=int, default=255)
    g.add_argument("--board_sat_max", type=int, default=70)
    g.add_argument("--bg_window", type=int, default=121)
    g = ap.add_argument_group("4. panicle trimming")
    g.add_argument("--taper_max", type=float, default=0.75, help="0 = off")
    g.add_argument("--taper_rows", type=int, default=40)
    g.add_argument("--panicle_search", type=float, default=0.4)
    g.add_argument("--branch_ratio", type=float, default=1.5)
    g.add_argument("--rachis_ratio", type=float, default=0.85,
                   help="fork fallback: width above the fork must be below this x width below it")
    g.add_argument("--fork_ratio", type=float, default=1.12,
                   help="without a branch cluster, trim only if the widest upper row is this much wider than below")
    g.add_argument("--width_median_window", type=int, default=151)
    g.add_argument("--cluster_gap", type=int, default=15)
    g.add_argument("--cluster_min_rows", type=int, default=4)
    g.add_argument("--return_ratio", type=float, default=1.08)
    g.add_argument("--cut_margin", type=int, default=3)
    g.add_argument("--min_height", type=int, default=300)
    g = ap.add_argument_group("5. soft edges / 6. composite")
    g.add_argument("--smooth_sigma", type=float, default=3.0)
    g.add_argument("--gf_radius", type=int, default=4)
    g.add_argument("--gf_eps", type=float, default=1e-3)
    g.add_argument("--feather", type=int, default=0)
    a = ap.parse_args()

    seg_keys = [k for k in vars(a) if k not in {
        "input_dir", "output_dir", "workers", "jpg_quality", "no_tif", "no_masks",
        "include_converted", "seed"}]
    p = {k: getattr(a, k) for k in seg_keys}
    p["exclude"] = json.loads(a.exclude)
    save_tif, save_masks = not a.no_tif, not a.no_masks

    print(f"\nInput Directory: {a.input_dir}\nOutput Directory: {a.output_dir}")
    os.makedirs(a.output_dir, exist_ok=True)
    done_log = os.path.join(a.output_dir, "converted_batches_log.txt")
    done = set()
    if os.path.exists(done_log):
        with open(done_log) as fh:
            done = {line.strip() for line in fh if line.strip()}

    exts = (".cr3",) + ((".jpg", ".jpeg", ".png", ".tif", ".tiff") if a.include_converted else ())
    batches = sorted(d for d in os.listdir(a.input_dir) if os.path.isdir(os.path.join(a.input_dir, d)))

    for batch in batches:
        if batch in done:
            print(f"Skipping already processed batch: {batch}")
            continue
        in_dir = os.path.join(a.input_dir, batch)
        files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(exts))
        if not files:
            print(f"No images in batch {batch}, skipping.")
            continue

        # resume an unfinished session for this batch, otherwise start a new one
        session_dir = find_session(a.output_dir, batch)
        if session_dir is None:
            session_dir = os.path.join(a.output_dir, f"{batch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            print(f"Resuming unfinished session: {os.path.basename(session_dir)}")
        dirs = make_dirs(session_dir, save_tif, save_masks)
        with open(os.path.join(session_dir, "params.json"), "w") as fh:
            json.dump({"parameters": p, "seed": a.seed, "jpg_quality": a.jpg_quality,
                       "raw_conversion": {"library": "rawpy", "use_camera_wb": True,
                                          "no_auto_bright": True, "bright": 4.5,
                                          "output_size": [6000, 4000]},
                       "versions": {"python": platform.python_version(), "opencv": cv2.__version__,
                                    "numpy": np.__version__, "platform": platform.platform()}},
                      fh, indent=2)

        previous = {}                                   # keep log rows from an earlier run
        prev_log = os.path.join(session_dir, "processing_log.csv")
        if os.path.exists(prev_log):
            with open(prev_log, newline="") as fh:
                previous = {r["Filename"]: r for r in csv.DictReader(fh)}

        tasks, skipped = [], []
        for f in files:
            name = os.path.splitext(f)[0]
            if outputs_exist(dirs, name, save_tif):
                skipped.append(previous.get(f, {"Filename": f, "Status": "ok",
                                                "Conversion": "done in earlier run"}))
                continue
            tasks.append((os.path.join(in_dir, f), dirs, name, p, a.seed,
                          a.jpg_quality, save_tif, save_masks))

        print(f"\nBatch {batch}: {len(files)} images ({len(skipped)} already done), "
              f"{a.workers} worker(s) -> {session_dir}")
        rows = list(skipped)
        if a.workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(process_one, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futs), 1):
                    r = fut.result()
                    rows.append(r)
                    print(f"[{i}/{len(tasks)}] {r['Status']:6s} {r['Filename']} "
                          f"{r.get('Error', '')}{'' if 'Error' in r else str(r.get('Seconds')) + 's'}")
        else:
            for i, t in enumerate(tasks, 1):
                r = process_one(t)
                rows.append(r)
                print(f"[{i}/{len(tasks)}] {r['Status']:6s} {r['Filename']} "
                      f"{r.get('Error', '')}{'' if 'Error' in r else str(r.get('Seconds')) + 's'}")

        rows.sort(key=lambda r: r["Filename"])
        ok = [r for r in rows if r["Status"] == "ok"]
        failed = [r for r in rows if r["Status"] != "ok"]
        write_log(os.path.join(session_dir, "processing_log.csv"), ok,
                  ["Filename", "Status", "Conversion", "Width", "Height", "Scale", "WorkWidth", "SeedMode", "CulmHue",
                   "OtsuThreshold", "PaniclesTrimmed", "Seconds"])
        write_log(os.path.join(session_dir, "failed_log.csv"), failed, ["Filename", "Error"])

        if not failed:
            with open(done_log, "a") as fh:
                fh.write(batch + "\n")
            print(f"Batch {batch} complete ({len(ok)} images) and logged.")
        else:
            print(f"Batch {batch}: {len(ok)} ok, {len(failed)} failed - NOT logged as done; "
                  f"re-run to retry the failed images (see failed_log.csv).")

    print("\nAll batches processed.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)       # safe with rawpy/OpenMP on all platforms
    main()