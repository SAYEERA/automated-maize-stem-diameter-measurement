"""
review_nodes.py - check the detected nodes by hand before measuring.

For every image in <session>/blackbg/jpg a window opens showing the detected nodes
(green dots).
    LEFT click ........ add a missing node (magenta dot)
    RIGHT click ....... remove the nearest node (a false detection)
    Mouse wheel ....... zoom in / out at the mouse
    + / - ............. zoom in / out
    W A S D ........... move the view (when zoomed in)
    R ................. reset view (whole image)
    U ................. undo last change
    ENTER or SPACE .... save this image and go to the next one
    N ................. next image WITHOUT saving changes (keep YOLO as it is)
    Q ................. stop reviewing (remaining images keep YOLO nodes)

Saved label files start with the line "# reviewed". This script does not open them again,
and measure_stems.py skips its 250-px duplicate merging for them, so nodes added by hand
are kept (the 2.0-cm minimum node gap in measure_stems.py still applies).
The original YOLO files are kept in yolo_output/labels_yolo_original/, and every added or
removed node is listed in yolo_output/node_review_log.csv.
"""

import csv
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import OUTPUT_DIR

# ───── CONFIG ─────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR = Path(OUTPUT_DIR)
REVIEW_NODES = True             # False = skip this step completely
REVIEW_AGAIN = False            # True  = also open images already reviewed
MERGE_DISTANCE_PX = 250         # detections closer than this are shown as one node
                                # (same rule measure_stems.py uses for unreviewed files)
WINDOW_W, WINDOW_H = None, None # review window size in pixels; None = fit your screen automatically
SCREEN_FILL = 0.64              # automatic size uses this share of the screen (leave room for taskbar)
DEFAULT_BOX_REL = 0.02          # box size written for hand-added nodes (YOLO format needs one)

REVIEWED_HEADER = "# reviewed"
WIN = "Node review"


def _fit_window_to_screen():
    """Pick a window size that fits the screen, also with Windows display scaling (125 %, 150 %...)."""
    global WINDOW_W, WINDOW_H
    if WINDOW_W and WINDOW_H:
        return
    try:                                   # draw in real screen pixels, not scaled ones (Windows)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    sw, sh = 1600, 900
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except Exception:
        pass
    # keep the photo's 3:2 shape and leave room for the title bar and taskbar
    WINDOW_W = int(min(sw * SCREEN_FILL, sh * SCREEN_FILL * 1.5))
    WINDOW_H = int(WINDOW_W / 1.5)
    print(f"Review window: {WINDOW_W} x {WINDOW_H} px (screen {sw} x {sh})")


# ───── LABEL FILES ───────────────────────────────────────────────────────
def read_labels(path, W, H):
    """Return list of dicts {x, y, w, h, src} in pixels, plus whether file was reviewed."""
    nodes, reviewed = [], False
    if not path.exists():
        return nodes, reviewed
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith(REVIEWED_HEADER):
                reviewed = True
                continue
            p = line.split()
            if len(p) < 5 or line.startswith("#"):
                continue
            _, xr, yr, wr, hr = map(float, p[:5])
            nodes.append({"x": xr * W, "y": yr * H, "w": wr, "h": hr, "src": "yolo"})
    return nodes, reviewed


def merge_duplicates(nodes):
    """Same duplicate rule as the pipeline: within MERGE_DISTANCE_PX keep the tallest box."""
    out, used = [], set()
    for i, a in enumerate(nodes):
        if i in used:
            continue
        cluster = [j for j, b in enumerate(nodes)
                   if j not in used and np.hypot(a["x"] - b["x"], a["y"] - b["y"]) < MERGE_DISTANCE_PX]
        best = max(cluster, key=lambda j: nodes[j]["h"])
        out.append(nodes[best])
        used.update(cluster)
    return out


def write_labels(path, nodes, W, H):
    with open(path, "w") as f:
        f.write(f"{REVIEWED_HEADER} by hand {datetime.now():%Y-%m-%d %H:%M}\n")
        for n in sorted(nodes, key=lambda n: (n["x"], n["y"])):
            f.write(f"0 {n['x'] / W:.6f} {n['y'] / H:.6f} {n['w']:.6f} {n['h']:.6f}\n")


# ───── INTERACTIVE VIEWER ────────────────────────────────────────────────
class Reviewer:
    def __init__(self, img, nodes, title):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.nodes = [dict(n) for n in nodes]
        self.history = []
        self.title = title
        self.events = []                       # (action, x, y) for the log
        boxes = [n["w"] for n in nodes] or [DEFAULT_BOX_REL]
        self.box_w = float(np.median(boxes))
        self.box_h = float(np.median([n["h"] for n in nodes] or [DEFAULT_BOX_REL]))
        self.reset_view()

    # view = image rectangle (x0, y0, vw, vh) shown in the window
    def reset_view(self):
        self.zoom = 1.0
        self.cx, self.cy = self.W / 2, self.H / 2

    def view_rect(self):
        fit = max(self.W / WINDOW_W, self.H / WINDOW_H)      # image px per screen px at zoom 1
        s = fit / self.zoom
        vw, vh = WINDOW_W * s, WINDOW_H * s
        x0 = float(np.clip(self.cx - vw / 2, 0, max(0, self.W - vw)))
        y0 = float(np.clip(self.cy - vh / 2, 0, max(0, self.H - vh)))
        return x0, y0, s

    def to_img(self, mx, my):
        x0, y0, s = self.view_rect()
        return x0 + mx * s, y0 + my * s

    def zoom_at(self, mx, my, factor):
        ix, iy = self.to_img(mx, my)
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 12.0))
        x0, y0, s = self.view_rect()
        # keep the image point under the mouse
        self.cx += ix - (x0 + mx * s)
        self.cy += iy - (y0 + my * s)

    def on_mouse(self, event, mx, my, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self.to_img(mx, my)
            self.history.append([dict(n) for n in self.nodes])
            self.nodes.append({"x": ix, "y": iy, "w": self.box_w, "h": self.box_h, "src": "hand"})
            self.events.append(("added", ix, iy))
        elif event == cv2.EVENT_RBUTTONDOWN and self.nodes:
            ix, iy = self.to_img(mx, my)
            d = [np.hypot(n["x"] - ix, n["y"] - iy) for n in self.nodes]
            k = int(np.argmin(d))
            _, _, s = self.view_rect()
            if d[k] / s < 40:                          # within 40 screen pixels
                self.history.append([dict(n) for n in self.nodes])
                n = self.nodes.pop(k)
                self.events.append(("removed", n["x"], n["y"]))
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom_at(mx, my, 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8)

    def render(self):
        x0, y0, s = self.view_rect()
        vw, vh = WINDOW_W * s, WINDOW_H * s
        crop = self.img[int(y0):int(min(self.H, y0 + vh)), int(x0):int(min(self.W, x0 + vw))]
        view = np.zeros((WINDOW_H, WINDOW_W, 3), np.uint8)
        ch, cw = int(crop.shape[0] / s), int(crop.shape[1] / s)
        view[:ch, :cw] = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA if s > 1 else cv2.INTER_LINEAR)
        for n in self.nodes:
            px, py = int((n["x"] - x0) / s), int((n["y"] - y0) / s)
            if 0 <= px < WINDOW_W and 0 <= py < WINDOW_H:
                col = (0, 220, 0) if n["src"] == "yolo" else (255, 0, 255)
                cv2.circle(view, (px, py), 11, (0, 0, 0), -1)
                cv2.circle(view, (px, py), 9, col, -1)
        n_y = sum(n["src"] == "yolo" for n in self.nodes)
        n_h = len(self.nodes) - n_y
        lines = [
            f"{self.title}   nodes: {len(self.nodes)} (YOLO {n_y}, added {n_h})   zoom x{self.zoom:.1f}",
            "LEFT add | RIGHT remove | wheel/+/- zoom | WASD move | R reset | U undo",
            "ENTER/SPACE save & next | N next (no save) | Q stop",
        ]
        fs = max(0.45, min(0.7, WINDOW_W / 2100))
        for i, t in enumerate(lines):
            y = int(12 + 40 * fs * (i + 1))
            cv2.putText(view, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(view, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
        return view

    def run(self):
        """Returns 'save', 'next' or 'quit'."""
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKey(30)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:   # window closed with X
                return "quit"
            if k == -1:
                continue
            k &= 0xFF
            step = 0.15 * max(self.W, self.H) / self.zoom
            if k in (13, 32):
                return "save"
            if k in (ord("n"), ord("N")):
                return "next"
            if k in (ord("q"), ord("Q"), 27):
                return "quit"
            if k in (ord("u"), ord("U")) and self.history:
                self.nodes = self.history.pop()
                self.events.append(("undo", np.nan, np.nan))
            elif k in (ord("r"), ord("R")):
                self.reset_view()
            elif k in (ord("+"), ord("=")):
                self.zoom_at(WINDOW_W / 2, WINDOW_H / 2, 1.25)
            elif k in (ord("-"), ord("_")):
                self.zoom_at(WINDOW_W / 2, WINDOW_H / 2, 0.8)
            elif k in (ord("a"), ord("A")):
                self.cx -= step
            elif k in (ord("d"), ord("D")):
                self.cx += step
            elif k in (ord("w"), ord("W")):
                self.cy -= step
            elif k in (ord("s"), ord("S")):
                self.cy += step


# ───── BATCH LOOP ────────────────────────────────────────────────────────
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def find_blackbg(session):
    """The black-background image folder of a session (the images YOLO looked at), or None."""
    d = session / "blackbg" / "jpg"
    if d.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir()):
        return d
    return None


def review_batch(session):
    img_dir = find_blackbg(session)
    lab_dir = session / "yolo_output" / "labels"
    if img_dir is None or not (session / "yolo_output").exists():
        return "next"
    lab_dir.mkdir(parents=True, exist_ok=True)
    backup = session / "yolo_output" / "labels_yolo_original"
    log_path = session / "yolo_output" / "node_review_log.csv"

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for img_path in images:
        lab_path = lab_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"cannot read {img_path.name}")
            continue
        H, W = img.shape[:2]
        nodes, reviewed = read_labels(lab_path, W, H)
        if reviewed and not REVIEW_AGAIN:
            print(f"{session.name}/{img_path.stem}: already reviewed")
            continue
        if not reviewed:
            nodes = merge_duplicates(nodes)

        print(f" Reviewing {session.name}/{img_path.stem} ({len(nodes)} nodes from YOLO)")
        rv = Reviewer(img, nodes, f"{session.name} / {img_path.stem}")
        action = rv.run()

        if action == "save":
            backup.mkdir(exist_ok=True)
            if lab_path.exists() and not (backup / lab_path.name).exists():
                shutil.copy2(lab_path, backup / lab_path.name)
            write_labels(lab_path, rv.nodes, W, H)
            new_log = not log_path.exists()
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                if new_log:
                    w.writerow(["time", "image", "action", "x_px", "y_px"])
                for a, x, y in rv.events:
                    w.writerow([f"{datetime.now():%Y-%m-%d %H:%M:%S}", img_path.stem, a,
                                "" if np.isnan(x) else int(x), "" if np.isnan(y) else int(y)])
            added = sum(e[0] == "added" for e in rv.events)
            removed = sum(e[0] == "removed" for e in rv.events)
            print(f"   saved: {len(rv.nodes)} nodes ({added} added, {removed} removed)")
        elif action == "next":
            print("   not saved, YOLO nodes kept")
        else:
            print("   review stopped; the remaining images keep their YOLO nodes")
            return "quit"
    return "next"


if __name__ == "__main__":
    if not REVIEW_NODES:
        print("Node review switched off (REVIEW_NODES = False).")
    else:
        _fit_window_to_screen()
        for session in sorted(d for d in BASE_OUTPUT_DIR.iterdir() if d.is_dir()):
            if review_batch(session) == "quit":
                break
        cv2.destroyAllWindows()
        print("\nNode review finished.")