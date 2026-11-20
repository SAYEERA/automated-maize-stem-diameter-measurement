"""
Runs the whole pipeline, one step after the other.

    python run_pipeline.py

Each step only processes sessions it has not finished before, so the script can be
started again after an error or after adding new photos. Folder settings are in config.py.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    "background_removal.py",   # RAW photos -> stalk on black and on white background
    "remove_ruler.py",         # take out the ruler pixels that background removal kept
    "calibration.py",          # pixel-per-mm scale from the rulers in each photo
    "detect_nodes.py",         # YOLOv8n node detection
    "review_nodes.py",         # check the nodes by hand (opens a window)
    "measure_stems.py",        # diameters, internode lengths, tables and figures
]


def run(script):
    print(f"\n=== {script} ===\n")
    try:
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)
    except subprocess.CalledProcessError as e:
        sys.exit(f"{script} stopped with an error ({e.returncode}); fix it and run again.")


if __name__ == "__main__":
    for step in STEPS:
        run(step)
    print("\nAll steps finished.")
