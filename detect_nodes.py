"""
detect_nodes.py - run the trained YOLOv8n node detector on the black-background images.

For every session folder in config.OUTPUT_DIR the images in <session>/blackbg/jpg are
passed to the detector, and one label file per image is written to
<session>/yolo_output/labels/ (YOLO format: class, x, y, width, height, all relative).
Sessions that already have a yolo_output folder are skipped.

Needs the Ultralytics package (pip install ultralytics) and the trained weights
(config.MODEL_PATH).
"""
import subprocess
from pathlib import Path

from config import OUTPUT_DIR, MODEL_PATH

CONF_THRESHOLD = 0.25          # Ultralytics default; F1 changed little between 0.20 and 0.50
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def blackbg_images(session):
    d = session / "blackbg" / "jpg"
    if d.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir()):
        return d
    return None


def main():
    out_dir, model = Path(OUTPUT_DIR), Path(MODEL_PATH)
    if not model.exists():
        raise SystemExit(f"Model weights not found: {model}  (set MODEL_PATH in config.py)")
    print(f"Sessions: {out_dir}\nModel:    {model}")

    for session in sorted(d for d in out_dir.iterdir() if d.is_dir()):
        if (session / "yolo_output").exists():
            print(f"already done: {session.name}")
            continue
        src = blackbg_images(session)
        if src is None:
            print(f"no black-background images in {session.name}, skipped")
            continue

        print(f"detecting nodes: {session.name}")
        cmd = (f'yolo task=detect mode=predict model="{model}" source="{src}" '
               f'conf={CONF_THRESHOLD} save_txt=True project="{session}" name="yolo_output"')
        try:
            subprocess.run(cmd, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"YOLO failed for {session.name}: {e}")

    print("\nNode detection finished.")


if __name__ == "__main__":
    main()
