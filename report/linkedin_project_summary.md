# Project Summary: Smart Parking Violation Detection Using YOLOv8
**AIRI Team PITB AI Internship Task 1**

---

### Project Title
**Smart Parking Violation Detection Using YOLOv8**

### Overview
Developed an end-to-end computer vision object detection pipeline designed to identify illegal roadside vehicle parking, traffic sign indicators, and vehicle instances in urban environments. The system utilizes an anchor-free YOLOv8 architecture to process streetscape images and support automated traffic enforcement.

---

### Technologies Used
`Python` | `YOLOv8 (Ultralytics)` | `PyTorch` | `OpenCV` | `Streamlit` | `Google Colab (NVIDIA T4 GPU)` | `ReportLab` | `Pandas / Matplotlib`

---

### Key Work Performed
1. **Pipeline Architecture & Dataset Setup**: Built an end-to-end directory structure, automated dataset split pipeline (70% Train / 20% Val / 10% Test with fixed seed `42`), and YOLO format normalization scripts.
2. **Data Integrity & Hash Audit**: Implemented automated integrity verification scripts (`validate_dataset.py`, `check_duplicates.py`) ensuring zero image corruption, 0–1 coordinate bounding, and zero cross-split leakage.
3. **GPU Model Training**: Trained a baseline `YOLOv8n` model for 30 epochs (640x640 resolution) on a Google Colab T4 GPU instance, saving best weights at `models/best.pt`.
4. **Inference & Error Analysis**: Generated prediction outputs across all source images (`outputs/predictions/`) and analyzed weak prediction error modes in `report/error_analysis.csv`.
5. **Web Showcase & Reporting**: Built an interactive Streamlit web application (`demo/app.py`) for live image upload/inference and compiled a fact-verified 31-section PDF technical report (`report/final_report.pdf`).

---

### Dataset & Target Classes
- **Dataset Size**: 22 real open-access images (20 annotated bounding box instances)
- **Split Breakdown**: 12 Train (54.5%) / 6 Validation (27.3%) / 4 Test (18.2%)
- **Target Classes**:
  - `0: car` (7 instances)
  - `1: motorcycle` (5 instances)
  - `2: no_parking_sign` (4 instances)
  - `3: roadside_parking` (4 instances)

---

### Model & Training Information
- **Model Architecture**: `YOLOv8n` (Pretrained baseline)
- **Hyperparameters**: 30 Epochs, Image Size 640x640, Batch Size 16
- **Hardware**: NVIDIA Tesla T4 GPU (Google Colab execution)
- **Preserved Model Weights**: `models/best.pt` (6.25 MB)

---

### Evaluation Information
- **Training Final Epoch Metrics (Epoch 30 from `results.csv`)**:
  - Precision: **25.03%** (0.25029)
  - Recall: **50.00%** (0.50000)
  - mAP@0.5: **29.85%** (0.29850)
  - mAP@0.5:0.95: **16.42%** (0.16418)
- **Test Evaluation Metrics (4 Test Images)**:
  - Precision: **0.00%** | Recall: **0.00%** | mAP@0.5: **0.00%** | mAP@0.5:0.95: **0.00%**
  - *(Inference on 4 test images produced 0 detections due to small dataset scale)*

---

### Project Limitation Notice
> **METHODOLOGICAL NOTICE**:  
> “The current implementation and training pipeline are complete, but the available dataset contains only 22 images, below the required 150–300 image target. Therefore, the reported evaluation metrics are preliminary and should not be treated as representative model performance.”

---

### Project Summary Statement
Demonstrated a complete, transparent AI/ML engineering project lifecycle—from dataset ingestion, validation, and cloud GPU training to metric logging, error mode analysis, PDF reporting, and Streamlit web deployment.
