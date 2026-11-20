# Maize stem diameter profiling

Code for the paper *A Deep Learning and Computer Vision Approach for Measuring Maize Stem Diameter* (Shaik et al.).

A maize stalk is cut into sections, the sections are laid side by side on a white board
with two steel rulers, and the board is photographed from above. From that photo the code
finds the nodes with a YOLOv8n detector and measures the stem diameter perpendicular to the
stem centerline at every node and at 10, 30, 50, 70 and 90 % of each internode. Internode
lengths are measured along the curved centerline, and each section is placed back at its
position on the original stalk.

## Setup

Python 3.10 was used.

```
pip install -r requirements.txt
```

Open `config.py` and set three paths:

| Setting | What it is |
|---|---|
| `INPUT_DIR` | folder with your photos, one sub-folder per imaging session |
| `OUTPUT_DIR` | where results are written, one folder per session |
| `MODEL_PATH` | trained YOLOv8n weights (`best.pt`) |

The trained weights, the annotated images and labels, and the train/validation/test
lists are on Zenodo: https://doi.org/10.5281/zenodo.24385780

Input layout:

```
data/input_sessions/
    batch_2026_11_03/
        set1_11032026.CR3
        set2_11032026.CR3
        ...
```

## Running

Everything at once:

```
python run_pipeline.py
```

or step by step:

| Step | Script | What it does | Written to (inside each session folder) |
|---|---|---|---|
| 1 | `background_removal.py` | RAW conversion; stalk separated from the board (local board colour in CIELAB + Otsu threshold, GrabCut, guided-filter edge); stalk placed on black and on white | `raw_to_jpg/`, `blackbg/`, `whitebg/` |
| 2 | `remove_ruler.py` | removes ruler pixels that step 1 kept | images changed in place, originals in `with_ruler/` |
| 3 | `calibration.py` | pixel-per-mm scale from the two rulers in every photo, as a linear scale field over the image | `calibration.csv`, `calibration_debug/` |
| 4 | `detect_nodes.py` | YOLOv8n node detection on the black-background images | `yolo_output/labels/` |
| 5 | `review_nodes.py` | window for checking the nodes by hand: left click adds, right click removes, Enter saves | `yolo_output/labels/`, `node_review_log.csv` |
| 6 | `measure_stems.py` | centerline, node placement, diameters, internode lengths, fixed-spacing comparison | `analysis_output_comparison/` |

Each step skips sessions it has already finished, so you can stop and start again.
Settings for each step are at the top of its script; the values there are the ones
used in the paper.

## Main outputs

For every photo, `analysis_output_comparison/<image>/` contains:

- `*_tangent_5point_measurements.xlsx`: node and D1–D5 diameters for each section
- `*_positions_continuous.xlsx`: every point with its position on the original stalk, and internode lengths (curved, straight and vertical)
- `*_tangent_overlay_5points.png`: the photo with nodes and measured diameters drawn on it
- `*_centerline_overlay.png`, `*_internode_lengths.png`, `*_fixed_spacing_overlay.png`: figures for checking

and `<session>_method_comparison_summary.xlsx` combines all photos of a session.

## Notes

- Photos are expected at 6000 × 4000 pixels with the stalk sections roughly upright and the
  base at the bottom. Sections are read from left to right as base to apex.
- Diameters are the width of the stem as seen from above (projected width), not a
  caliper reading in a chosen direction.

## Other folders

- `test_images/`: example input photos and example outputs.
- `old_version_v1/`: the code used in the original submission (v1.0.1, archived at
  https://doi.org/10.5281/zenodo.20834860). It is kept for reference; use the scripts
  in the top folder.

## Citation

Please cite the paper and the archived code:
Shaik, S. (2026). Automated maize stem diameter profiling pipeline (Version 2.0.0). Zenodo. https://doi.org/______

## License

MIT, see `LICENSE`.
