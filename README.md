# Smart Parking Violation Detection Using YOLOv8
**AIRI Team PITB AI Internship Task 1**

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20App-red.svg)
![Status](https://img.shields.io/badge/Status-Fact--Verified-green.svg)

---

## 📋 Project Overview
This project presents an end-to-end Computer Vision object detection pipeline developed for the **AIRI Team PITB AI Internship Task 1**. The system detects vehicle instances (`car`, `motorcycle`), `no_parking_sign` indicators, and `roadside_parking` zones to assist municipal authorities in automated traffic monitoring.

> **CRITICAL METHODOLOGICAL NOTICE & LIMITATIONS**:  
> The current implementation and training pipeline are complete, but the available dataset contains only 22 images, below the required 150–300 image target. Therefore, the reported evaluation metrics are preliminary and should not be treated as representative model performance.

---

## 🎯 Target Classes
| Class ID | Class Name | Description | Verified Count |
|---|---|---|---|
| **0** | `car` | Passenger cars, sedans, SUVs, trucks | 7 instances |
| **1** | `motorcycle` | Motorcycles, motorbikes, scooters | 5 instances |
| **2** | `no_parking_sign` | Official circular or rectangular 'No Parking' signs | 4 instances |
| **3** | `roadside_parking` | Vehicles parked along street curbsides | 4 instances |

---

## 📊 Empirical Dataset Statistics
- **Total Images**: 22 real open-access images
- **Total Bounding Boxes**: 20 annotated instances
- **Splits**:
  - **Train (54.5%)**: 12 images
  - **Val (27.3%)**: 6 images
  - **Test (18.2%)**: 4 images

---

## 📈 Verified Model Training & Evaluation Results
- **Model**: `YOLOv8n` (pretrained baseline)
- **Training Setup**: 30 Epochs, Image Size 640x640, Batch Size 16, Google Colab NVIDIA Tesla T4 GPU
- **Saved Weights**: `models/best.pt`

### Performance Metrics Summary

| Stage / Split | Precision (P) | Recall (R) | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---|---|
| **Training Final Epoch (Epoch 30)** | **25.03%** (0.25029) | **50.00%** (0.50000) | **29.85%** (0.29850) | **16.42%** (0.16418) |
| **Test Evaluation (4 Test Images)** | **0.00%** | **0.00%** | **0.00%** | **0.00%** |

*Note: The test set contains only 4 images (producing 0 detections), so test evaluation results are highly limited.*

---

## 📂 Project Directory Structure
```text
cv_parking_violation/
│
├── dataset/
│   ├── images/train/
│   ├── images/val/
│   ├── images/test/
│   ├── labels/train/
│   ├── labels/val/
│   ├── labels/test/
│   └── data.yaml
│
├── notebooks/
│   └── training_notebook.ipynb
│
├── scripts/
│   ├── fetch_real_public_data.py
│   ├── prepare_dataset.py
│   ├── validate_dataset.py
│   ├── check_duplicates.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── error_analysis.py
│   └── generate_report_assets.py
│
├── outputs/
│   ├── training_results/
│   ├── evaluation/
│   ├── predictions/
│   └── error_analysis/
│
├── models/
│   └── best.pt
│
├── report/
│   ├── final_report.pdf
│   ├── error_analysis.csv
│   ├── linkedin_post.md
│   └── report_assets/
│
├── demo/
│   └── app.py
│
├── README.md
├── requirements.txt
├── DATA_SOURCES.md
├── ANNOTATION_INSTRUCTIONS.md
├── PROJECT_STATUS.md
├── PROJECT_EXPLANATION.md
├── FINAL_AUDIT.md
└── .gitignore
```
