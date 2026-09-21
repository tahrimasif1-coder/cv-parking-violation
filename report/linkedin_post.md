# LinkedIn Showcase Post
**AIRI Team PITB AI Internship Task 1**

---

🚀 **Sharing my Computer Vision Project: Smart Parking Violation Detection Using YOLOv8!** 🚗📸

As part of the **AIRI Team PITB AI Internship Task 1**, I built an end-to-end Computer Vision object detection system designed for automated traffic monitoring and smart parking violation detection.

### 🔍 **Project Workflow & Empirical Summary**:
1. **Problem Definition & Use Case**: Designed a vision detection system targeting illegal roadside parking and traffic sign indicators.
2. **Data Curation & Split**: Curated an initial dataset of 22 real open-source images (20 annotated bounding box instances), partitioned into **12 Train (54.5%) / 6 Val (27.3%) / 4 Test (18.2%)**.
3. **YOLO Annotation Schema**: Standardized YOLO format bounding box annotations across 4 target classes: `0: car`, `1: motorcycle`, `2: no_parking_sign`, and `3: roadside_parking`.
4. **Data Integrity & Validation**: Developed automated validation and hash-based duplicate auditing scripts to verify image health, coordinate bounds (`[0, 1]`), and zero cross-split leakage.
5. **Model Training & Quantitative Evaluation**: Trained a baseline **YOLOv8n** model for 30 epochs (640x640 resolution) using GPU acceleration on Google Colab (NVIDIA Tesla T4 GPU). Model weights saved at `models/best.pt`.
   - **Training Final Epoch (Epoch 30)**: Precision = 25.03%, Recall = 50.00%, mAP@0.5 = 29.85%, mAP@0.5:0.95 = 16.42%.
   - **Test Evaluation (4 Test Images)**: Precision = 0.00%, Recall = 0.00%, mAP@0.5 = 0.00%, mAP@0.5:0.95 = 0.00% (Produced 0 detections due to scale limitations on the 4 test images).
6. **Error Analysis & Web Showcase**: Generated `report/error_analysis.csv` outlining false negative modes and built an interactive Streamlit web application (`demo/app.py`).

### 📌 **Methodological Limitation Notice**:
> “The current implementation and training pipeline are complete, but the available dataset contains only 22 images, below the required 150–300 image target. Therefore, the reported evaluation metrics are preliminary and should not be treated as representative model performance.”

### 🛠️ **Tools & Technologies**:
`PyTorch` | `YOLOv8 (Ultralytics)` | `OpenCV` | `Python` | `Streamlit` | `Google Colab GPU` | `ReportLab` | `Matplotlib / Seaborn`

Check out the full repository and project documentation!

#ArtificialIntelligence #ComputerVision #YOLOv8 #DeepLearning #MachineLearning #Python #PITB #AIRI #Streamlit #ObjectDetection #TechInternship
