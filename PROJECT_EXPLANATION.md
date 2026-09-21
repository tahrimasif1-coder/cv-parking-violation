# Technical Interview Preparation Guide
**AIRI Team PITB AI Internship Task 1 — Smart Parking Violation Detection**

This document provides clear, concise, and honest interview-ready answers based on the actual verified project execution.

---

### Q1: What problem did you solve?
**Answer**: I built an end-to-end Computer Vision system using YOLOv8 to automatically detect illegal roadside parking violations, vehicle instances (`car`, `motorcycle`), and official `no_parking_sign` indicators.

### Q2: What are the exact classes in your dataset?
**Answer**: The 4 target classes are strictly: `0: car`, `1: motorcycle`, `2: no_parking_sign`, and `3: roadside_parking`.

### Q3: What is the current size of your dataset and split distribution?
**Answer**: The current verified dataset contains 22 real open-access images (20 annotated bounding box instances), partitioned into **12 Train (54.5%)**, **6 Validation (27.3%)**, and **4 Test (18.2%)**.

### Q4: What were your actual model training setup and logged results?
**Answer**: 
- **Model & Hardware**: YOLOv8n (pretrained baseline) trained for 30 epochs on an NVIDIA Tesla T4 GPU in Google Colab (batch size 16, imgsz 640). Model weights saved at `models/best.pt`.
- **Training Final Epoch (Epoch 30)**:
  - Precision: 25.03% (0.25029)
  - Recall: 50.00% (0.50000)
  - mAP@0.5: 29.85% (0.29850)
  - mAP@0.5:0.95: 16.42% (0.16418)
- **Test Evaluation (4 Test Images)**: Precision, Recall, mAP@0.5, and mAP@0.5:0.95 were 0.00% because prediction inference on the 4 test images produced 0 detections due to small dataset scale.

### Q5: What is the primary limitation of this project?
**Answer**: 
> “The current implementation and training pipeline are complete, but the available dataset contains only 22 images, below the required 150–300 image target. Therefore, the reported evaluation metrics are preliminary and should not be treated as representative model performance.”

### Q6: What is Precision and Recall?
**Answer**: 
- **Precision**: Measures the proportion of positive detections that were correct: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$.
- **Recall**: Measures the proportion of actual ground truth instances detected: $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$.

### Q7: What is mAP (Mean Average Precision)?
**Answer**: mAP is the mean of Average Precision across all object classes. `mAP@0.5` calculates precision-recall AUC at Intersection-over-Union (IoU) threshold 0.50. `mAP@0.5:0.95` averages mAP across IoU thresholds from 0.50 to 0.95.

### Q8: What would you improve in future iterations?
**Answer**: 
1. Expand the dataset to 300+ images.
2. Implement Albumentations Mosaic, Motion Blur, and HSV Jitter.
3. Integrate ByteTRACK multi-object video tracking.
