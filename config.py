"""
Folder and model settings shared by all scripts.

Change these three paths to match your computer. Relative paths are taken from
the folder this file is in.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "data" / "input_sessions"     # one sub-folder of CR3 photos per imaging session
OUTPUT_DIR = ROOT / "data" / "output_sessions"   # results, one folder per session
MODEL_PATH = ROOT / "models" / "best.pt"         # trained YOLOv8n node detector
