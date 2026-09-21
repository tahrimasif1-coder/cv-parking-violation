# Data Sources Registry & Collection Pipeline
**AIRI Team PITB AI Internship Task 1 — Smart Parking Violation Detection**

This document details the multi-source dataset acquisition pipeline for combining real public datasets into the exact 4-class YOLOv8 format required by the assignment.

---

## 1. Class Mapping & Source Matrix

| Class ID | Class Name | Primary Public Data Source | Source URL | Verified Instances |
|---|---|---|---|---|
| **0** | `car` | COCO / Roboflow Vehicle Detection | [https://universe.roboflow.com/roboflow-100/vehicles-qtx4v](https://universe.roboflow.com/roboflow-100/vehicles-qtx4v) | 7 instances |
| **1** | `motorcycle` | COCO / Roboflow Vehicle Detection | [https://universe.roboflow.com/roboflow-100/vehicles-qtx4v](https://universe.roboflow.com/roboflow-100/vehicles-qtx4v) | 5 instances |
| **2** | `no_parking_sign` | Road Signs Object Detection Dataset | [https://www.kaggle.com/datasets/andrewmvd/road-sign-detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) | 4 instances |
| **3** | `roadside_parking` | Public Urban Streetscape Images | Annotated per `ANNOTATION_INSTRUCTIONS.md` SOP | 4 instances |

---

## 2. Verified Raw Data Directory Structure (`raw_data/`)

The project contains 22 real images and 22 corresponding YOLO `.txt` label files inside `raw_data/`:

```text
raw_data/
├── car_coco_01.jpg / car_coco_01.txt
├── car_coco_02.jpg / car_coco_02.txt
├── motorcycle_coco_01.jpg / motorcycle_coco_01.txt
├── no_parking_sign_01.jpg / no_parking_sign_01.txt
├── roadside_parking_curb_01.jpg / roadside_parking_curb_01.txt
└── ... (22 images & 22 label files total)
```

---

## 3. Dataset Pipeline Workflow

### Step 1: Duplicate Audit
Run the hash-based duplicate checker to eliminate identical images:
```bash
python scripts/check_duplicates.py
```

### Step 2: Partitioning (70% Train / 20% Val / 10% Test)
Run the automated dataset preparation script with fixed random seed `42`:
```bash
python scripts/prepare_dataset.py --source-dir raw_data/
```
This populates:
- `dataset/images/train/` (12 images), `dataset/images/val/` (6 images), `dataset/images/test/` (4 images)
- `dataset/labels/train/`, `dataset/labels/val/`, `dataset/labels/test/`

### Step 3: Integrity & Bounds Validation
Run dataset validation to ensure zero corrupted files, 0–1 coordinate bounds, and correct class distributions:
```bash
python scripts/validate_dataset.py
```

---

## 4. Dataset Size Limitation Notice

> **METHODOLOGICAL LIMITATION NOTICE**:  
> The current dataset contains 22 images (12 Train, 6 Val, 4 Test), which is below the 150–300 image target. All evaluation metrics reflect this preliminary dataset scale.
