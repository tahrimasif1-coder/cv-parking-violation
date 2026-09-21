# Project Status & Execution Log
**AIRI Team PITB AI Internship Task 1 — Smart Parking Violation Detection**

---

## 1. Verified Project Limitations & Empirical Facts Notice

> [!WARNING]
> **CRITICAL METHODOLOGICAL NOTICE**:
> The current implementation and training pipeline are complete, but the available dataset contains only 22 images, below the required 150–300 image target. Therefore, the reported evaluation metrics are preliminary and should not be treated as representative model performance.

---

## 2. Verified Dataset & Training Audit

- **Dataset Size**: 22 total images (22 YOLO `.txt` label files)
- **Split Distribution**:
  - `Train`: 12 images (54.5%)
  - `Val`: 6 images (27.3%)
  - `Test`: 4 images (18.2%)
- **Class Distribution**:
  - `0: car`: 7 instances
  - `1: motorcycle`: 5 instances
  - `2: no_parking_sign`: 4 instances
  - `3: roadside_parking`: 4 instances
- **Model Training Setup**:
  - Model: `YOLOv8n` (Pretrained)
  - Epochs: 30
  - Image size: 640
  - Batch size: 16
  - GPU: NVIDIA Tesla T4 (Google Colab)
  - Weights saved: `models/best.pt`
- **Logged Metric Results**:
  - **Training Final Epoch (Epoch 30)**: Precision = 25.03%, Recall = 50.00%, mAP@0.5 = 29.85%, mAP@0.5:0.95 = 16.42%
  - **Test Evaluation (4 Test Images)**: Precision = 0.00%, Recall = 0.00%, mAP@0.5 = 0.00%, mAP@0.5:0.95 = 0.00% (Produced 0 detections on test set).

---

## 3. Verified File Assets

- `models/best.pt` — Verified model weights
- `outputs/training_results/colab_parking_exp/results.csv` — Verified epoch logs
- `outputs/predictions/test_predictions/` — Verified 4 test prediction output images
- `report/error_analysis.csv` — Verified weak prediction error modes
- `report/final_report.pdf` — Fact-verified PDF report
