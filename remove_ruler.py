"""
remove_ruler.py - remove the rulers that background removal kept.

The background step removes the board but keeps the steel rulers lying on it, and the
bottom ruler touches every stalk section. Left in place, it would join all sections
into one object in the measurement step.

For each image:
    1. Ruler bands are found as rows (bottom ruler) or columns (side ruler) that are
       mostly foreground over a strip at least MIN_RULER_WIDTH_MM wide. A stalk section
       is too thin to look like this.
    2. Inside a band, only pixels that look like stalk are kept: coloured
       (saturation >= STEM_MIN_SATURATION; the steel ruler is grey) and connected to a
       stalk outside the band. This also removes the ruler numbers and ticks.
    3. Everything else in the band is set to background (black or white).

The black- and white-background images (jpg and tif) are changed in place. The
originals are copied to <session>/with_ruler/ first, and a check picture with the
removed area in red is written to <session>/ruler_removal_check/.
The calibration step reads the rulers from raw_to_jpg, which is not changed.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

from config import OUTPUT_DIR

# ───── CONFIG ─────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR = Path(OUTPUT_DIR)
REMOVE_RULERS = True
PX_PER_MM_GUESS = 8.0          # rough scale, only used to turn mm settings into pixels
MIN_RULER_WIDTH_MM = 15        # a ruler band is at least this wide (stems are thinner)
BAND_FILL = 0.55               # row/column counts as ruler if this share of it is foreground
BAND_MARGIN_MM = 2             # widen the band a little (ruler edge, slight tilt)
STEM_MIN_SATURATION = 80       # stalk pixels are tan or green (S ~110-150), the steel ruler is grey (S ~25-55)
STEM_GAP_FILL_MM = 3           # close gaps up to ~3 mm inside a stem lying on the ruler
FOREGROUND_MIN = 25            # blackbg pixel brighter than this = foreground (JPEG noise below)
DONE_MARKER = "ruler_removed.txt"


def _runs(flags):
    """[(start, end)] of consecutive True values."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def ruler_mask(img_bgr, px_per_mm=PX_PER_MM_GUESS):
    """Return (remove_mask, band_mask) as bool arrays."""
    H, W = img_bgr.shape[:2]
    fg = img_bgr.max(axis=2) > FOREGROUND_MIN
    min_w = int(MIN_RULER_WIDTH_MM * px_per_mm)
    margin = int(BAND_MARGIN_MM * px_per_mm)

    band = np.zeros((H, W), bool)
    for a, b in _runs(fg.mean(axis=1) > BAND_FILL):            # horizontal rulers (rows)
        if b - a >= min_w:
            band[max(0, a - margin):min(H, b + margin), :] = True
    # vertical rulers: judge columns only above/below horizontal bands, so the
    # bottom ruler does not make every column look full
    rows_free = ~band.any(axis=1)
    if rows_free.sum() > 0:
        col_fill = fg[rows_free].mean(axis=0)
        for a, b in _runs(col_fill > BAND_FILL):
            if b - a >= min_w:
                band[:, max(0, a - margin):min(W, b + margin)] = True
    if not band.any():
        return np.zeros((H, W), bool), band

    # inside the band keep only coloured pixels that connect to a stem outside it
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    stemlike = fg & ((~band) | (hsv[..., 1] >= STEM_MIN_SATURATION))
    stemlike = cv2.morphologyEx(stemlike.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, lab = cv2.connectedComponents(stemlike.astype(np.uint8), connectivity=8)
    outside_labels = np.unique(lab[stemlike & ~band])
    keep = np.isin(lab, outside_labels[outside_labels > 0])
    # fill small gaps in the stem over the ruler (dark ink labels, shadows), but only
    # with foreground pixels and never wider than about STEM_GAP_FILL_MM
    k = max(3, int(STEM_GAP_FILL_MM * px_per_mm) | 1)
    closed = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    keep |= closed & fg & band
    remove = band & ~keep          # everything in the band that is not stem -> background
    return remove, band


BLACK, WHITE = "blackbg", "whitebg"     # folder names written by background_removal.py


def _has_images(session):
    d = session / BLACK / "jpg"
    return d.is_dir() and any(d.glob("*_blackbg.jpg"))


def _paths(session, base, black, white):
    return {
        "black_jpg": session / black / "jpg" / f"{base}_blackbg.jpg",
        "black_tif": session / black / "tif" / f"{base}_blackbg.tif",
        "white_jpg": session / white / "jpg" / f"{base}_whitebg.jpg",
        "white_tif": session / white / "tif" / f"{base}_whitebg.tif",
    }


def process_session(session):
    if not _has_images(session):
        return                                   # nothing to do (yet)
    black, white = BLACK, WHITE
    black_dir = session / black / "jpg"
    if (session / DONE_MARKER).exists():
        print(f"Rulers already removed: {session.name}")
        return
    backup = session / "with_ruler"
    check = session / "ruler_removal_check"
    check.mkdir(exist_ok=True)

    images = sorted(black_dir.glob("*_blackbg.jpg"))
    print(f" Removing rulers in {session.name} ({len(images)} images)")
    for bj in images:
        base = bj.stem[: -len("_blackbg")]
        img = cv2.imread(str(bj))
        if img is None:
            continue
        remove, band = ruler_mask(img)
        share = 100 * remove.mean()
        if not remove.any():
            print(f"   {base}: no ruler found, unchanged")
            continue

        for key, p in _paths(session, base, black, white).items():
            if not p.exists():
                continue
            dst = backup / p.parent.parent.name / p.parent.name / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(p, dst)
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None or im.shape[:2] != remove.shape:
                print(f"   {p.name}: size differs, skipped")
                continue
            im[remove] = 0 if key.startswith("black") else (255 if im.dtype == np.uint8 else np.iinfo(im.dtype).max)
            cv2.imwrite(str(p), im)

        # check picture: removed area in red, ruler band outlined in yellow
        vis = img.copy()
        vis[remove] = (0.4 * vis[remove] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cnts, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 6)
        cv2.imwrite(str(check / f"{base}_ruler_removed.jpg"),
                    cv2.resize(vis, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA))
        print(f"   {base}: ruler removed ({share:.1f}% of the image)")
    (session / DONE_MARKER).write_text(f"rulers removed from {black} and {white}\n")


if __name__ == "__main__":
    if not REMOVE_RULERS:
        print("Ruler removal switched off (REMOVE_RULERS = False).")
    else:
        for session in sorted(d for d in BASE_OUTPUT_DIR.iterdir() if d.is_dir()):
            process_session(session)
        print("\nRuler removal finished.")