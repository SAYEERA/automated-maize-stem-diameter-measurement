"""
calibration.py - pixel-per-mm scale from the rulers in each photo.

For every image in  <batch>/raw_to_jpg/*.jpg  it finds
    * the horizontal ruler along the bottom  -> scale along x (diameters)
    * the vertical ruler along the right     -> scale along y (lengths)
and writes
    <batch>/calibration.csv            one row per image
    <batch>/calibration_debug/*.jpg    overlay + local-scale plot, for checking

WHY A SCALE *FIELD* AND NOT ONE NUMBER
    If the camera is not exactly parallel to the board, parts of the board that are
    farther away look smaller. In our photos the bottom ruler read about 8.26 px/mm at
    the left edge and 7.79 px/mm at the right edge.
    For a tilted flat board the scale changes linearly across the image, so each
    image gets
        px_per_mm_x(x, y) = ax + b*x + c*y        (diameters, horizontal distances)
        px_per_mm_y(x, y) = ay + b*x + c*y        (lengths, vertical distances)
    b comes from how the scale changes along the bottom ruler, c from the right ruler.
    The CSV also stores the values at the image centre and the % variation over
    the image, so you can see how much the camera tilt matters for each photo.

HOW A RULER IS MEASURED
    1. The tick spacing is found with a Fourier transform of the brightness profile
       across the tick strip (averages hundreds of ticks -> sub-pixel precision).
    2. The tick strip is tracked chunk by chunk so a tilted ruler is followed.
    3. The local spacing is measured in sliding windows along the ruler, and a
       straight line is fitted to those local values (outliers, e.g. where a stem
       crosses the ruler, are rejected).

Only the raw_to_jpg images are used, because the background-removed images no longer
contain the rulers. Both have the same pixel grid, so the values apply directly.

QUALITY CHECKS
    An image gets the batch median scale (and a note in the CSV) when no ruler was found,
    when its scale is more than 3 % from the batch median, or when its x and y scales
    differ by more than 3 % (pixels are square, so this means a ruler was misread).

Run:  python calibration.py            all sessions in config.OUTPUT_DIR
      python calibration.py --force    redo sessions that already have calibration.csv
      python calibration.py IMG_1.jpg  quick test on single images
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from config import OUTPUT_DIR

# ───── CONFIG ─────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR = Path(OUTPUT_DIR)

# One entry per ruler.
#   axis     "x" = ruler runs left-right, "y" = ruler runs top-bottom
#   units    graduations the ruler might have; the one with the clearest tick pattern
#            whose scale falls inside PX_PER_MM_RANGE is used (so wood cm/inch rulers
#            and metal 1/8"-1/16" rulers all work without changing anything)
#   band     part of the image to search, measured ACROSS the ruler
#            ("x": rows, 0 = top, 1 = bottom; "y": columns, 0 = left, 1 = right)
RULERS = [
    {"name": "bottom", "axis": "x", "units": "auto", "band": (0.85, 1.00)},
    {"name": "right",  "axis": "y", "units": "auto", "band": (0.80, 1.00)},
]

# Candidate graduations (mm). "auto" tries all of them.
GRADUATIONS = {"1 mm": 1.0, "1/8 in": 25.4 / 8, "1/16 in": 25.4 / 16}

# Plausible scale for this rig. Keep it within about +-25 % of the real value:
# a wider range lets a cm ruler be mistaken for a 1/16 in ruler (ratio 1.59).
# Update it if you change camera height or zoom a lot.
PX_PER_MM_RANGE = (6.2, 10.4)
TRACK_CHUNK_PX = 400            # chunk length used to follow a tilted ruler
STRIP_HEIGHT_PX = 14            # rows averaged across the ticks
LOCAL_WIN_PX = 800              # window for local px/mm along the ruler
LOCAL_STEP_PX = 200
MIN_PEAK_SNR = 8.0              # weaker than this -> ruler "not_found"
MAX_FIT_RESID_PCT = 0.6         # RMS residual of local scale fit above this -> "check"
MAX_IMAGE_VARIATION_PCT = 1.0   # scale changes more than this over the image -> "check" (camera tilt)
MAX_BATCH_DEV_PCT = 1.5         # centre value differs from batch median by more -> "check" (value kept)
MAX_BATCH_REPLACE_PCT = 3.0     # ... by more than this -> ruler misread: batch median used instead
MAX_XY_DIFF_PCT = 3.0           # pixels are square, so x and y scales must agree; larger gap -> batch median used
DEFAULT_PX_PER_MM = 8.319       # last resort, used only when no image in a batch has a readable ruler
SAVE_DEBUG = True

CSV_NAME = "calibration.csv"
CSV_FIELDS = [
    "image", "width", "height",
    "px_per_mm_x", "px_per_mm_y",              # at image centre
    "ax", "ay", "b", "c",                      # scale field coefficients
    "variation_pct",                           # (max-min)/mean of scale over the image
    "x_left", "x_right", "y_top", "y_bottom",  # px/mm at the four image edges (centre lines)
    "angle_x_deg", "angle_y_deg", "snr_x", "snr_y", "fit_resid_x_pct", "fit_resid_y_pct",
    "source_x", "source_y", "unit_x", "unit_y", "status", "note",
]


# ───── SIGNAL HELPERS ────────────────────────────────────────────────────
def _highpass(profile, win=41):
    """Remove slow brightness changes so only the tick pattern remains."""
    k = np.ones(win) / win
    pad = np.pad(profile, win, mode="edge")
    return profile - np.convolve(pad, k, mode="same")[win:-win]


def _dense_peak(sig, f_lo, f_hi, n=2001):
    """Exact DFT magnitude on a dense frequency grid (cycles/px). Returns (freq, mag)."""
    freqs = np.linspace(f_lo, f_hi, n)
    nidx = np.arange(len(sig))
    mags = np.empty(n)
    for i in range(0, n, 250):
        f = freqs[i:i + 250]
        mags[i:i + 250] = np.abs(np.exp(-2j * np.pi * np.outer(f, nidx)) @ sig)
    k = int(np.argmax(mags))
    if 0 < k < n - 1:
        a, b0, c = mags[k - 1], mags[k], mags[k + 1]
        den = a - 2 * b0 + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
        return freqs[k] + d * (freqs[1] - freqs[0]), b0
    return freqs[k], mags[k]


def _snr(sig, f_lo, f_hi):
    """Height of the strongest periodic peak in [f_lo, f_hi] relative to the spectrum floor."""
    nfft = 1 << int(np.ceil(np.log2(len(sig) * 4)))
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), nfft))
    fr = np.fft.rfftfreq(nfft)
    inb = (fr >= f_lo) & (fr <= f_hi)
    ref = (fr >= f_lo * 0.5) & (fr <= min(0.5, f_hi * 2))
    k = np.argmax(np.where(inb, spec, 0))
    return spec[k] / (np.median(spec[ref]) + 1e-9), fr[k]


def _robust_line(x, y, w=None, n_iter=5, k=2.5):
    """Weighted straight-line fit with iterative outlier rejection. Returns (slope, intercept, mask)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    keep = np.ones(len(x), bool)
    slope, icpt = 0.0, float(np.median(y))
    for _ in range(n_iter):
        if keep.sum() < 3:
            break
        slope, icpt = np.polyfit(x[keep], y[keep], 1, w=w[keep])
        r = y - (slope * x + icpt)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) * 1.4826 + 1e-9
        new = np.abs(r) < k * mad
        if (new == keep).all():
            break
        keep = new
    return slope, icpt, keep


# ───── RULER MEASUREMENT ─────────────────────────────────────────────────
def measure_ruler(gray, axis, units, band_frac):
    """
    Measure one ruler. Returns None if not found, else a dict with
        px_per_mm      mean scale along the ruler
        local_pos      positions along the ruler (image x for axis x, image y for axis y)
        local_across   matching positions across the ruler (image y / image x)
        local_ppmm     local px/mm at those positions (outliers removed)
        angle_deg, snr, fit_resid_pct, line (for drawing)
    """
    img = gray if axis == "x" else gray.T            # work as if the ruler runs left-right
    H, W = img.shape
    r0, r1 = int(band_frac[0] * H), int(band_frac[1] * H)
    band = img[r0:r1].astype(np.float32)
    if units == "auto":
        cands = dict(GRADUATIONS)
    else:
        cands = {u: GRADUATIONS[u] for u in ([units] if isinstance(units, str) else units)}

    # 1) profiles of thin strips (every 3 rows) over the whole band
    csum = np.vstack([np.zeros((1, W), np.float64), np.cumsum(band, axis=0, dtype=np.float64)])
    starts = np.arange(0, band.shape[0] - STRIP_HEIGHT_PX, 3)
    profs = (csum[starts + STRIP_HEIGHT_PX] - csum[starts]) / STRIP_HEIGHT_PX
    hp = np.apply_along_axis(_highpass, 1, profs)

    # 2) pick the graduation and strip with the clearest tick pattern
    best = (0.0, None, None)
    for uname, umm in cands.items():
        lo, hi = 1.0 / (PX_PER_MM_RANGE[1] * umm), 1.0 / (PX_PER_MM_RANGE[0] * umm)
        snrs = np.array([_snr(p, lo, hi)[0] for p in hp])
        if snrs.max() > best[0]:
            best = (snrs.max(), uname, int(np.argmax(snrs)))
    if best[0] < MIN_PEAK_SNR:
        return None
    unit_name, i_best = best[1], best[2]
    unit_mm = cands[unit_name]
    f_lo = 1.0 / (PX_PER_MM_RANGE[1] * unit_mm)
    f_hi = 1.0 / (PX_PER_MM_RANGE[0] * unit_mm)
    f0, _ = _dense_peak(hp[i_best] * np.hanning(W), f_lo, f_hi, 4001)

    # 3) follow the tick strip chunk by chunk (handles tilt)
    freqs = f0 * np.linspace(0.94, 1.06, 13)
    chunk_c, chunk_r, chunk_w = [], [], []
    for c in range(0, W - TRACK_CHUNK_PX + 1, TRACK_CHUNK_PX // 2):
        seg = hp[:, c:c + TRACK_CHUNK_PX] * np.hanning(TRACK_CHUNK_PX)
        basis = np.exp(-2j * np.pi * np.outer(freqs, np.arange(c, c + TRACK_CHUNK_PX)))
        power = np.abs(seg @ basis.T).max(axis=1)          # per strip
        j = int(np.argmax(power))
        chunk_c.append(c + TRACK_CHUNK_PX / 2)
        chunk_r.append(starts[j] + STRIP_HEIGHT_PX / 2)
        chunk_w.append(power[j])
    chunk_w = np.array(chunk_w)
    good = chunk_w > 0.25 * np.percentile(chunk_w, 90)
    slope, icpt, keep = _robust_line(np.array(chunk_c)[good], np.array(chunk_r)[good],
                                     chunk_w[good], k=3.0)

    # 4) sample a strip along the fitted line
    cols = np.arange(W, dtype=np.float32)
    centre = (r0 + icpt + slope * cols).astype(np.float32)
    offs = np.arange(STRIP_HEIGHT_PX, dtype=np.float32) - STRIP_HEIGHT_PX / 2
    map_x = np.tile(cols, (STRIP_HEIGHT_PX, 1))
    map_y = (centre[None, :] + offs[:, None]).astype(np.float32)
    strip = cv2.remap(img.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
    sig = _highpass(strip.mean(axis=0))
    snr, _ = _snr(sig, f_lo, f_hi)
    if snr < MIN_PEAK_SNR:
        return None

    # 5) local scale in sliding windows along the ruler
    loc_c, loc_s, loc_w = [], [], []
    hw = np.hanning(LOCAL_WIN_PX)
    for c in range(0, W - LOCAL_WIN_PX + 1, LOCAL_STEP_PX):
        s = sig[c:c + LOCAL_WIN_PX]
        if np.std(s) < 1e-6:
            continue
        fl, mag = _dense_peak(s * hw, f0 * 0.92, f0 * 1.08, 801)
        loc_c.append(c + LOCAL_WIN_PX / 2)
        loc_s.append(1.0 / (fl * unit_mm))
        loc_w.append(mag / (np.sqrt(np.mean(s ** 2)) * LOCAL_WIN_PX + 1e-9))  # peak purity
    loc_c, loc_s, loc_w = map(np.asarray, (loc_c, loc_s, loc_w))
    ok = loc_w > 0.4 * np.percentile(loc_w, 90)            # drop windows hidden by stems etc.
    if ok.sum() < 4:
        return None
    g, s0, keep = _robust_line(loc_c[ok], loc_s[ok], loc_w[ok])
    pos, val = loc_c[ok][keep], loc_s[ok][keep]
    resid = val - (g * pos + s0)
    resid_pct = float(np.sqrt(np.mean(resid ** 2)) / np.mean(val) * 100)

    across = r0 + icpt + slope * pos
    angle = float(np.degrees(np.arctan(slope)))
    return {
        "px_per_mm": float(np.mean(g * pos + s0)),
        "local_pos": pos, "local_across": across, "local_ppmm": val,
        "angle_deg": angle if axis == "x" else -angle,
        "snr": float(snr), "fit_resid_pct": resid_pct, "unit": unit_name,
        "line": (r0 + icpt, slope, W), "axis": axis, "gradient": g, "s0": s0,
        "all_pos": loc_c, "all_ppmm": loc_s, "used": (ok, keep),
    }


# ───── SCALE FIELD ───────────────────────────────────────────────────────
def fit_scale_field(rx, ry):
    """
    rx: bottom ruler result (or None), ry: right ruler result (or None).
    Model:  sx(x,y) = ax + b*x + c*y,   sy(x,y) = ay + b*x + c*y
    b = d(scale)/dx from the bottom ruler, c = d(scale)/dy from the right ruler.
    """
    b = rx["gradient"] if rx else 0.0
    c = ry["gradient"] if ry else 0.0
    ax = ay = None
    if rx:   # along the bottom ruler:  s = ax + b*x + c*y_ruler
        ax = float(np.mean(rx["local_ppmm"] - b * rx["local_pos"] - c * rx["local_across"]))
    if ry:   # along the right ruler:   s = ay + b*x_ruler + c*y
        ay = float(np.mean(ry["local_ppmm"] - b * ry["local_across"] - c * ry["local_pos"]))
    return ax, ay, float(b), float(c)


def scale_at(cal, x, y):
    """Local (px_per_mm_x, px_per_mm_y) at image point (x, y). `cal` is a calibration row/dict."""
    b, c = float(cal["b"]), float(cal["c"])
    return float(cal["ax"]) + b * x + c * y, float(cal["ay"]) + b * x + c * y


def dist_mm(cal, p1, p2):
    """Distance in mm between two image points, using the local scale at their midpoint."""
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    sx, sy = scale_at(cal, mx, my)
    return float(np.hypot((p2[0] - p1[0]) / sx, (p2[1] - p1[1]) / sy))


def path_length_mm(cal, pts):
    """Length in mm of a polyline (e.g. a stem centreline) given as [(x, y), ...]."""
    pts = np.asarray(pts, float)
    return float(sum(dist_mm(cal, pts[i], pts[i + 1]) for i in range(len(pts) - 1)))


# ───── DEBUG FIGURE ──────────────────────────────────────────────────────
def draw_debug(img_bgr, results, row, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = img_bgr.copy()
    cols = {"x": (0, 200, 0), "y": (255, 0, 255)}
    for r in results.values():
        if r is None:
            continue
        a0, slope, W = r["line"]
        p = [(0, a0), (W, a0 + slope * W)]
        if r["axis"] == "y":
            p = [(q[1], q[0]) for q in p]
        cv2.line(out, tuple(map(int, p[0])), tuple(map(int, p[1])), cols[r["axis"]], 6)
    small = cv2.cvtColor(cv2.resize(out, (out.shape[1] // 4, out.shape[0] // 4)), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(14, 9))
    ax0 = fig.add_subplot(2, 2, (1, 2))
    ax0.imshow(small, extent=(0, img_bgr.shape[1], img_bgr.shape[0], 0))
    # scale-field contours
    if row.get("ax") not in (None, ""):
        X, Y = np.meshgrid(np.linspace(0, img_bgr.shape[1], 60), np.linspace(0, img_bgr.shape[0], 40))
        S = float(row["ax"]) + float(row["b"]) * X + float(row["c"]) * Y
        cs = ax0.contour(X, Y, S, levels=8, colors="cyan", linewidths=0.8)
        ax0.clabel(cs, fmt="%.2f", fontsize=7)
    ax0.set_title(f"{row['image']}:  centre {row['px_per_mm_x']} (x) / {row['px_per_mm_y']} (y) px/mm, "
                  f"variation {row['variation_pct']}%   [{row['status']}]  (cyan = px/mm along x)",
                  fontsize=10)
    ax0.axis("off")
    for k, (name, r) in enumerate(results.items()):
        ax = fig.add_subplot(2, 2, 3 + k)
        if r is None:
            ax.set_title(f"{name}: not found")
            continue
        ok, keep = r["used"]
        ax.plot(r["all_pos"], r["all_ppmm"], ".", color="0.7", label="all windows")
        ax.plot(r["local_pos"], r["local_ppmm"], "o", ms=4, label="used")
        xx = np.array([0, r["line"][2]])
        ax.plot(xx, r["gradient"] * xx + r["s0"], "r-", label="linear fit")
        ax.set_xlabel("image x (px)" if r["axis"] == "x" else "image y (px)")
        ax.set_ylabel("local px/mm")
        lo, hi = np.percentile(r["local_ppmm"], [0, 100])
        ax.set_ylim(lo - 0.15, hi + 0.15)
        ax.set_title(f"{name} ({r['unit']} ticks): mean {r['px_per_mm']:.3f}, tilt {r['angle_deg']:+.2f} deg, "
                     f"resid {r['fit_resid_pct']:.2f}%", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)


# ───── PER IMAGE / PER BATCH ─────────────────────────────────────────────
def calibrate_image(img_path, debug_dir=None):
    img = cv2.imread(str(img_path))
    if img is None:
        raise IOError(f"cannot read {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Hh, Ww = gray.shape

    results = {}
    for ruler in RULERS:
        try:
            results[ruler["name"]] = measure_ruler(gray, ruler["axis"], ruler["units"], ruler["band"])
        except Exception as e:
            print(f"   {ruler['name']}: error {e}")
            results[ruler["name"]] = None
    by_axis = {}
    for ruler in RULERS:
        r = results[ruler["name"]]
        if r is not None and ruler["axis"] not in by_axis:
            by_axis[ruler["axis"]] = (ruler["name"], r)
    rx = by_axis.get("x", (None, None))[1]
    ry = by_axis.get("y", (None, None))[1]

    row = {"image": img_path.stem, "width": Ww, "height": Hh}
    if rx is None and ry is None:
        row.update({"ax": None, "ay": None, "b": 0.0, "c": 0.0,
                    "source_x": "not_found", "source_y": "not_found"})
    else:
        ax, ay, b, c = fit_scale_field(rx, ry)
        # one ruler missing: assume the same scale in both directions
        row.update({"ax": ax if ax is not None else ay, "ay": ay if ay is not None else ax,
                    "b": b, "c": c,
                    "source_x": by_axis["x"][0] if rx else "copied_from_y",
                    "source_y": by_axis["y"][0] if ry else "copied_from_x"})
    for axis, r in (("x", rx), ("y", ry)):
        if r is not None:
            row[f"angle_{axis}_deg"] = r["angle_deg"]
            row[f"snr_{axis}"] = r["snr"]
            row[f"fit_resid_{axis}_pct"] = r["fit_resid_pct"]
            row[f"unit_{axis}"] = r["unit"]
    row["_results"] = results
    return row, img


def _summarise(row):
    W, H = row["width"], row["height"]
    sx_c, sy_c = scale_at(row, W / 2, H / 2)
    corners = [scale_at(row, x, y)[0] for x in (0, W) for y in (0, H)]
    row["px_per_mm_x"], row["px_per_mm_y"] = sx_c, sy_c
    row["variation_pct"] = (max(corners) - min(corners)) / sx_c * 100
    row["x_left"] = scale_at(row, 0, H / 2)[0]
    row["x_right"] = scale_at(row, W, H / 2)[0]
    row["y_top"] = scale_at(row, W / 2, 0)[1]
    row["y_bottom"] = scale_at(row, W / 2, H)[1]


def finalize_rows(rows):
    """Fill missing or misread images from the batch median, add QA status, round numbers.

    An image gets the batch-median scale field (ax, ay, b, c) when
      * no ruler was found, or
      * its centre scale differs from the batch median by more than MAX_BATCH_REPLACE_PCT
        (x or y), or
      * its x and y centre scales differ by more than MAX_XY_DIFF_PCT (pixels are square,
        so a large gap means one ruler was misread, e.g. a wrong graduation).
    Smaller deviations (MAX_BATCH_DEV_PCT .. MAX_BATCH_REPLACE_PCT) are only flagged "check".
    The reason is written to the "note" column.
    """
    keys = ("ax", "ay", "b", "c")
    found = [r for r in rows if r["ax"] is not None]
    for r in found:
        _summarise(r)

    # median of the images whose own rulers were read and whose x and y agree
    def xy_gap(r):
        return abs(r["px_per_mm_x"] - r["px_per_mm_y"]) / r["px_per_mm_y"] * 100
    good = [r for r in found if xy_gap(r) <= MAX_XY_DIFF_PCT] or found
    if good:
        med = {k: float(np.median([float(r[k]) for r in good])) for k in keys}
    else:
        med = {"ax": DEFAULT_PX_PER_MM, "ay": DEFAULT_PX_PER_MM, "b": 0.0, "c": 0.0}
    med_row = dict(med, width=rows[0]["width"], height=rows[0]["height"]) if rows else None
    if med_row:
        _summarise(med_row)

    for r in rows:
        notes = [n for n in r.get("note", "").split("; ") if n]
        replaced = None
        if r["ax"] is None:
            replaced = "no ruler found"
            r["source_x"] = r["source_y"] = "batch_median" if found else "default"
        else:
            dev_x = abs(r["px_per_mm_x"] - med_row["px_per_mm_x"]) / med_row["px_per_mm_x"] * 100
            dev_y = abs(r["px_per_mm_y"] - med_row["px_per_mm_y"]) / med_row["px_per_mm_y"] * 100
            gap = xy_gap(r)
            if gap > MAX_XY_DIFF_PCT:
                replaced = (f"x and y scales differ by {gap:.1f}% (x {r['px_per_mm_x']:.3f}, "
                            f"y {r['px_per_mm_y']:.3f}); ruler misread, batch median used")
            elif max(dev_x, dev_y) > MAX_BATCH_REPLACE_PCT:
                replaced = (f"{max(dev_x, dev_y):.1f}% from batch median "
                            f"(x {r['px_per_mm_x']:.3f}, y {r['px_per_mm_y']:.3f}); batch median used")
            elif max(dev_x, dev_y) > MAX_BATCH_DEV_PCT:
                notes.append(f"{max(dev_x, dev_y):.2f}% from batch median (value kept)")
            if replaced:
                r["source_x"] = r["source_y"] = "batch_median" if found else "default"
        if replaced:
            r.update(med)
            notes.append(replaced)
        _summarise(r)

        status = "check" if replaced or len(notes) else "ok"
        for axis in ("x", "y"):
            if str(r.get(f"source_{axis}", "")).startswith("copied"):
                status = "check"; notes.append(f"{axis} ruler not found, copied from other axis")
            res = r.get(f"fit_resid_{axis}_pct")
            if not replaced and res is not None and res > MAX_FIT_RESID_PCT:
                status = "check"; notes.append(f"{axis} local scale noisy ({res:.2f}%)")
        if not replaced and r["variation_pct"] > MAX_IMAGE_VARIATION_PCT:
            status = "check"; notes.append(f"scale varies {r['variation_pct']:.1f}% over image (camera tilt?)")
        r["status"], r["note"] = status, "; ".join(dict.fromkeys(notes))
        for k, v in list(r.items()):
            if isinstance(v, (float, np.floating)):
                r[k] = round(float(v), 8 if k in ("b", "c") else 4)
    return rows


def calibrate_batch(session_path, force=False):
    src = session_path / "raw_to_jpg"
    out_csv = session_path / CSV_NAME
    if not src.exists():
        print(f"No raw_to_jpg in {session_path.name}; skipping.")
        return
    if out_csv.exists() and not force:
        print(f"Calibration already done for {session_path.name}")
        return
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif"))
    if not images:
        print(f"No images in {src}")
        return
    debug_dir = session_path / "calibration_debug" if SAVE_DEBUG else None
    if debug_dir:
        debug_dir.mkdir(exist_ok=True)

    print(f"Calibrating {session_path.name} ({len(images)} images)")
    rows = _run(images, debug_dir)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_check = sum(r["status"] != "ok" for r in rows)
    print(f"{out_csv}  ({n_check} image(s) flagged 'check')")


def _run(images, debug_dir):
    rows, imgs = [], []
    for p in images:
        row, img = calibrate_image(p)
        rows.append(row)
        imgs.append((p, img))
    finalize_rows(rows)
    for row, (p, img) in zip(rows, imgs):
        print(f"   {row['image']}: centre x={row['px_per_mm_x']} y={row['px_per_mm_y']} px/mm, "
              f"variation {row['variation_pct']}%  [{row['status']}] {row['note']}")
        if debug_dir is not None:
            draw_debug(img, row["_results"], row, debug_dir / f"{p.stem}_calib.png")
        row.pop("_results", None)
    return rows


# ───── READER USED BY measure_stems.py ───────────────────────────────────
def load_image_calibration(batch_folder, base_name):
    """
    Return the calibration row (dict) for one image, e.g. base_name="IMG_3348".
    Use with scale_at(cal, x, y), dist_mm(cal, p1, p2), path_length_mm(cal, pts).
    Falls back to a uniform DEFAULT_PX_PER_MM if the image is not in calibration.csv.
    """
    path = Path(batch_folder) / CSV_NAME
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["image"] == base_name:
                    return r
    print(f"{base_name}: no calibration row, using default {DEFAULT_PX_PER_MM} px/mm")
    return {"image": base_name, "ax": DEFAULT_PX_PER_MM, "ay": DEFAULT_PX_PER_MM,
            "b": 0.0, "c": 0.0, "status": "default"}


# ───── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    force = "--force" in sys.argv
    files = [a for a in sys.argv[1:] if a != "--force"]

    if files:   # quick test on single images:  python calibration.py IMG_1.jpg IMG_2.jpg
        out = Path("calibration_debug")
        out.mkdir(exist_ok=True)
        rows = _run([Path(a) for a in files], out)
        with open(out / CSV_NAME, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {out / CSV_NAME}")
    else:
        for batch in sorted(d for d in BASE_OUTPUT_DIR.iterdir() if d.is_dir()):
            calibrate_batch(batch, force=force)
        print("\nCalibration complete.")