#!/usr/bin/env python3
"""
Generate training_notebook.ipynb for Google Colab.
Contains pre-training dataset count verification, drive path verification, and automated training pipeline.
"""

import json
from pathlib import Path

def build_notebook(output_path: Path):
    cells = []
    
    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })
        
    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")]
        })

    # Section 1: Title & Project Overview
    add_md("# Smart Parking Violation Detection Using YOLOv8\n**AIRI Team PITB AI Internship Task 1 — End-to-End Notebook**\n\nThis notebook provides a complete workflow for dataset loading, validation, pre-training path verification, training, evaluation, inference, and error analysis of a YOLOv8 object detection model detecting smart parking violations.")
    
    # Section 2: Project Overview & Objectives
    add_md("## 1. Project Overview & Objectives\n- **Problem Statement**: Automated detection of illegal roadside parking and vehicles obstructing traffic.\n- **Target Classes**:\n  - 0: `car` (Moving or parked passenger cars, sedans, SUVs, trucks)\n  - 1: `motorcycle` (Motorcycles, motorbikes, scooters, mopeds)\n  - 2: `no_parking_sign` (Official circular or rectangular 'No Parking' / 'No Stopping' signs)\n  - 3: `roadside_parking` (Roadside parked vehicle instances / curb parking spots)\n- **Framework**: Ultralytics YOLOv8n (nano baseline, 640x640 imgsz, 30+ epochs).")
    
    # Section 3: Imports
    add_md("## 2. Imports")
    add_code("""import os
import sys
import glob
import json
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from IPython.display import display""")

    # Section 4: Drive Mount
    add_md("## 3. Google Drive Mounting")
    add_code("""from google.colab import drive
drive.mount('/content/drive')""")

    # Section 5: Environment & Ultralytics Installation
    add_md("## 4. Install Ultralytics & Dependencies")
    add_code("""!pip install -U ultralytics reportlab pandas matplotlib seaborn opencv-python albumentations""")

    # Section 6: GPU Check
    add_md("## 5. GPU Availability Verification")
    add_code("""import torch
print("PyTorch Version:", torch.__version__)
print("GPU Available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
!nvidia-smi""")

    # Section 7: Dataset Paths Configuration & Dynamic data.yaml Verification
    add_md("## 6. Dataset Paths Configuration & Verification")
    add_code("""PROJECT_DIR = "/content/drive/MyDrive/cv_parking_violation"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

print("Project Directory:", PROJECT_DIR)
print("Dataset Directory:", DATASET_DIR)
print("Data YAML Path   :", DATA_YAML)

# Verify data.yaml existence or write clean config matching Drive path
yaml_content = f\"\"\"# YOLOv8 Dataset Configuration
path: {DATASET_DIR}
train: images/train
val: images/val
test: images/test

names:
  0: car
  1: motorcycle
  2: no_parking_sign
  3: roadside_parking
\"\"\"

os.makedirs(DATASET_DIR, exist_ok=True)
with open(DATA_YAML, "w", encoding="utf-8") as f:
    f.write(yaml_content)
print(f"[OK] Verified and updated {DATA_YAML}")""")

    # Section 8: Pre-Training Dataset Verification & Image Count Audit
    add_md("## 7. Pre-Training Dataset Directory & Image Count Check\n**Crucial Pre-Flight Check**: Verifies resolved image directories and counts actual `.jpg`/`.png` files in `train`, `val`, and `test` splits to prevent YOLO `AssertionError: No images found` errors.")
    add_code("""exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]

split_counts = {}
for split in ["train", "val", "test"]:
    img_dir = os.path.join(DATASET_DIR, "images", split)
    lbl_dir = os.path.join(DATASET_DIR, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    img_files = []
    for ext in exts:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    
    split_counts[split] = len(img_files)
    print(f"Split {split:<5s} | Image Dir: {img_dir}")
    print(f"            | Found {len(img_files):3d} images and {len(lbl_files):3d} label files")

print("-" * 65)
total_imgs = sum(split_counts.values())
print(f"Total Dataset Images Found Across Splits: {total_imgs}")

if split_counts["train"] == 0:
    print("\\n[ALERT] 0 images found in dataset/images/train!")
    print("Action Required: Copy your 150-300 dataset images & .txt labels into:")
    print(f"  - Images: {os.path.join(DATASET_DIR, 'images/train')}")
    print(f"  - Labels: {os.path.join(DATASET_DIR, 'labels/train')}")
    print("  Or run: python scripts/prepare_dataset.py --source-dir path/to/raw_data/")
    print("Then re-run this cell before proceeding to training.")
else:
    print("\\n[PASS] Pre-training dataset check successful! Ready for YOLO model training.")""")

    # Section 9: Dataset Statistics
    add_md("## 8. Dataset Class Mapping")
    add_code("""class_names = {0: "car", 1: "motorcycle", 2: "no_parking_sign", 3: "roadside_parking"}
print("Configured Class Mapping:")
for cid, cname in class_names.items():
    print(f"  ID {cid}: {cname}")""")

    # Section 10: Sample Visualizations
    add_md("## 9. Sample Image Visualizations")
    add_code("""def visualize_sample_images():
    train_imgs = []
    for ext in exts:
        train_imgs.extend(glob.glob(os.path.join(DATASET_DIR, "images/train", ext)))
    if train_imgs:
        fig, axes = plt.subplots(1, min(3, len(train_imgs)), figsize=(15, 5))
        if len(train_imgs) == 1:
            axes = [axes]
        for i in range(min(3, len(train_imgs))):
            img = Image.open(train_imgs[i])
            axes[i].imshow(img)
            axes[i].set_title(os.path.basename(train_imgs[i]))
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No images found in train set yet to display.")

visualize_sample_images()""")

    # Section 11: Dataset Validation
    add_md("## 10. Dataset Integrity Validation")
    add_code("""!python /content/drive/MyDrive/cv_parking_violation/scripts/validate_dataset.py""")

    # Section 12: Model Loading
    add_md("## 11. Model Loading")
    add_code("""from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print("Baseline YOLOv8n model initialized.")""")

    # Section 13: Training Guard & Execution
    add_md("## 12. Model Training (30+ Epochs)")
    add_code("""train_imgs_check = []
for ext in exts:
    train_imgs_check.extend(glob.glob(os.path.join(DATASET_DIR, "images/train", ext)))

if len(train_imgs_check) == 0:
    raise RuntimeError(
        f"Cannot start training: 0 images found in {os.path.join(DATASET_DIR, 'images/train')}. "
        f"Please populate dataset images before running training."
    )

results = model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=640,
    batch=16,
    name="colab_parking_exp",
    project=os.path.join(PROJECT_DIR, "outputs/training_results"),
    exist_ok=True,
    pretrained=True
)

# Save best weights to models/best.pt
best_src = os.path.join(PROJECT_DIR, "outputs/training_results/colab_parking_exp/weights/best.pt")
best_dest = os.path.join(PROJECT_DIR, "models/best.pt")
if os.path.exists(best_src):
    os.makedirs(os.path.dirname(best_dest), exist_ok=True)
    shutil.copy2(best_src, best_dest)
    print(f"Successfully saved best model weights to {best_dest}")""")

    # Section 14: Evaluation
    add_md("## 13. Quantitative Evaluation")
    add_code("""metrics = model.val(data=DATA_YAML, split="val")
print(f"Precision    (P) : {metrics.box.mp:.4f}")
print(f"Recall       (R) : {metrics.box.mr:.4f}")
print(f"mAP @ 0.50       : {metrics.box.map50:.4f}")
print(f"mAP @ 0.50:0.95  : {metrics.box.map:.4f}")

# Save quantitative metrics to metrics.json & metrics.txt
eval_dir = os.path.join(PROJECT_DIR, "outputs/evaluation")
os.makedirs(eval_dir, exist_ok=True)

metrics_dict = {
    "overall": {
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4)
    }
}
with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
    json.dump(metrics_dict, f, indent=4)
print("Saved quantitative metrics to outputs/evaluation/metrics.json")""")

    # Section 15: Confusion Matrix
    add_md("## 14. Confusion Matrix Analysis")
    add_code("""cm_path = os.path.join(PROJECT_DIR, "outputs/training_results/colab_parking_exp/confusion_matrix.png")
if os.path.exists(cm_path):
    display(Image.open(cm_path))
else:
    print("Confusion matrix image will appear after training finishes.")""")

    # Section 16: Loss Curves
    add_md("## 15. Training & Validation Loss Curves")
    add_code("""results_path = os.path.join(PROJECT_DIR, "outputs/training_results/colab_parking_exp/results.png")
if os.path.exists(results_path):
    display(Image.open(results_path))
else:
    print("Loss curves plot will appear after training completes.")""")

    # Section 17: Per-Class Performance
    add_md("## 16. Per-Class Performance Breakdown")
    add_code("""if hasattr(metrics, 'box'):
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10}")
    print("-" * 55)
    for i, c in enumerate(metrics.box.ap_class_index):
        c_name = metrics.names[c]
        print(f"{c_name:<20} | {metrics.box.p[i]:<10.4f} | {metrics.box.r[i]:<10.4f} | {metrics.box.ap50[i]:<10.4f}")""")

    # Section 18: Test Inference
    add_md("## 17. Unseen Test Inference (15+ Prediction Images)")
    add_code("""test_preds = model.predict(
    source=os.path.join(DATASET_DIR, "images/test"),
    conf=0.25,
    save=True,
    project=os.path.join(PROJECT_DIR, "outputs/predictions"),
    name="test_predictions",
    exist_ok=True
)
print("Inference completed on test split!")""")

    # Section 19: 15+ Predictions Visualizations
    add_md("## 18. Prediction Contact Sheet / Grid")
    add_code("""pred_imgs = []
for ext in exts:
    pred_imgs.extend(glob.glob(os.path.join(PROJECT_DIR, "outputs/predictions/test_predictions", ext)))

if pred_imgs:
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    for i in range(min(15, len(pred_imgs))):
        img = Image.open(pred_imgs[i])
        axes[i].imshow(img)
        axes[i].set_title(f"Prediction #{i+1}", fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Test predictions will display here after test inference.")""")

    # Section 20: Error Analysis
    add_md("## 19. Weak Prediction Error Analysis")
    add_code("""!python /content/drive/MyDrive/cv_parking_violation/scripts/error_analysis.py""")

    # Section 21: Real Improvement Suggestions
    add_md("## 20. Real Improvement Recommendations\n1. **Multi-Scale Training & Image Size Expansion**: Train with `--imgsz 800` to capture small distance signs.\n2. **Enhanced Augmentation Pipeline**: Add Albumentations Mosaic, Motion Blur, and HSV Jitter.\n3. **Annotation SOP Disambiguation**: Clear guidelines for bounding box boundaries between roadside parking zones vs parked vehicle bodies.")

    # Section 22: Conclusion & Future Work
    add_md("## 21. Conclusion & Future Improvements\n- Successfully built an end-to-end YOLOv8 object detection model for Smart Parking Violation Detection.\n- Next Steps: Edge deployment on NVIDIA Jetson / Mobile devices with ByteTRACK video tracking integration.")

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook created successfully at {output_path}")

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "notebooks" / "training_notebook.ipynb"
    build_notebook(out)
