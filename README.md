# Automated Maize Stem Diameter Measurement Pipeline

This repository contains the source code accompanying the manuscript:

**"Automated Measurement of Maize Stem Diameter and Internode Structure from RGB Images"**

The pipeline performs image preprocessing, node localization, stem segmentation, skeleton extraction, and automated stem diameter measurement from RGB images. Two independent diameter measurement approaches are implemented and compared:

* **Tangent-Based Diameter Measurement**
* **Horizontal Scanline Diameter Measurement**

The framework automatically generates annotated overlay images, Excel reports, and graphical summaries of stem morphology.

---

# Repository Structure

```text
CORN_DIAMETER/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── background_preprocessing.py
├── corn_images_preprocessing.py
├── corn_yolo_detection.py
├── tangent_diameter_measurement.py
├── horizontal_scanline_measurement.py
├── main_pipeline.py
│
├── weights/
│   └── best.pt
│
├── test_images/
│   ├── sample_inputs/
│   ├── sample_labels/
│
└── sample_outputs/
    ├── overlays/
    ├── excel_reports/
    └── plots/
```

---

# Overview

Maize stem architecture, including node position, internode length, and stem diameter, plays an important role in crop development, mechanical stability, and lodging resistance. This repository provides a reproducible image-based workflow for extracting these structural traits from RGB stem images.

The pipeline integrates image preprocessing, object detection, skeleton-based analysis, and diameter measurement to generate high-density stem morphology data suitable for phenotyping and plant science research.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/CORN_DIAMETER.git
cd CORN_DIAMETER
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

# YOLO Weights

The trained YOLO model used for node detection is stored in:

```text
weights/best.pt
```

The detection script automatically loads the model from this location.

If a custom model is used, update the model path in:

```text
corn_yolo_detection.py
```

---

# Pipeline Workflow

```text
Raw RGB Images
        │
        ▼
Image Preprocessing
        │
        ▼
Foreground Extraction
        │
        ▼
Stem Segmentation
        │
        ▼
YOLO Node Detection
        │
        ▼
Skeleton Extraction
        │
        ▼
Diameter Measurement
   ┌──────────────────┬──────────────────┐
   │                  │
   ▼                  ▼
Tangent Method   Horizontal Scanline Method
   │                  │
   └────────┬─────────┘
            ▼
Annotated Overlays
Excel Reports
Morphological Plots
```

---

# Methodology

## 1. Image Preprocessing

Raw Canon CR3 images are converted to standardized RGB images suitable for analysis.

Processing steps include:

* CR3 to JPG conversion
* Orientation correction
* Image standardization
* Background processing
* Export to JPG and TIFF formats

Scripts:

```text
corn_images_preprocessing.py
background_preprocessing.py
```

---

## 2. Stem Segmentation

Stem regions are separated from the background using thresholding, contour extraction, and morphological refinement.

Outputs include cleaned stem masks suitable for skeleton extraction and diameter measurement.

---

## 3. Node Detection

Stem node locations are detected using a YOLO-based object detection model.

Processing steps include:

* Node localization
* Duplicate detection filtering
* Coordinate extraction

Script:

```text
corn_yolo_detection.py
```

---

## 4. Skeleton Extraction

A one-pixel-wide stem centerline is generated from the segmented stem mask. The centerline serves as the reference axis for diameter measurements and internode analysis.

---

## 5. Diameter Measurement

### Tangent-Based Diameter Measurement

The local stem orientation is estimated using neighboring skeleton pixels. Diameter is measured perpendicular to the local stem tangent by tracing rays to opposite stem boundaries.

Outputs:

* Tangent diameter measurements
* Annotated overlays
* Excel reports
* Diameter profiles
* Segment summaries

Script:

```text
tangent_diameter_measurement.py
```

---

### Horizontal Scanline Diameter Measurement

Diameter is measured using a horizontal scanline passing through the measurement location. The left and right stem boundaries are identified and used to calculate stem width.

Outputs:

* Horizontal scanline diameter measurements
* Annotated overlays
* Excel reports
* Diameter profiles
* Segment summaries

Script:

```text
horizontal_scanline_measurement.py
```

---

# Input and Output Locations

Sample input images:

```text
test_images/sample_inputs/
```

Sample YOLO annotation files:

```text
test_images/sample_labels/
```

Generated outputs:

```text
sample_outputs/
```

---

# Running the Pipeline

Run the complete workflow:

```bash
python main_pipeline.py
```

Individual modules may also be executed independently for preprocessing, node detection, or diameter measurement.

---

# Example Outputs

The pipeline generates:

## Overlay Images

* Detected node positions
* Diameter measurement locations
* Diameter measurement lines
* Diameter annotation labels

## Excel Reports

* Node coordinates
* Stem length measurements
* Internode length measurements
* Diameter measurements
* Segment-wise summaries

## Graphical Outputs

* Diameter versus stem length profiles
* Segment-wise structural summaries
* Morphological trend visualizations

---

# Sample Data

A small set of representative maize stem images and corresponding node annotation files are provided for demonstration and testing purposes.

The complete experimental dataset used in the study is not included in this repository.

---

# Requirements

The software has been tested using:

* Python 3.10+

Major dependencies include:

* OpenCV
* NumPy
* Pandas
* SciPy
* scikit-image
* OpenPyXL
* Pillow
* Matplotlib
* RawPy
* Requests
* Ultralytics YOLO

All dependencies are listed in:

```text
requirements.txt
```

---

# Citation

If you use this repository in your research, please cite the associated publication.

Citation information will be updated upon publication of the manuscript.

---

# License

This repository is provided for academic and research purposes.

---

# Contact

For questions regarding the methodology, implementation, or associated publication, please contact the corresponding author.
