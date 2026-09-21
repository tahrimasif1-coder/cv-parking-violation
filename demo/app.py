import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="Smart Parking Violation Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Smart Parking Violation Detection App")
st.subheader("AIRI Team PITB AI Internship Task 1 — Computer Vision Showcase")

st.markdown("""
Upload a street scene or parking area image to automatically detect vehicle instances, no-parking signs, and roadside parking violations using **YOLOv8**.
""")

# Sidebar controls
st.sidebar.header("Model Configuration")
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
model_path_input = st.sidebar.text_input("Path to Weights", value="models/best.pt")

@st.cache_resource
def load_model(weights_path):
    try:
        from ultralytics import YOLO
        if Path(weights_path).exists():
            return YOLO(weights_path)
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

model = load_model(model_path_input)

if model is None:
    st.warning(f"⚠️ Model weights not found at `{model_path_input}`. Place trained `best.pt` file in `models/` directory to run live inference.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Original Input Image")
        st.image(image, use_column_width=True)
        
    if model is not None:
        # Save temp file for YOLO
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name
            
        results = model.predict(source=tmp_path, conf=conf_thresh)
        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted[:, :, ::-1])
        
        with col2:
            st.subheader("Detected Parking Violations")
            st.image(res_image, use_column_width=True)
            
        st.markdown("### Detection Summary")
        boxes = results[0].boxes
        if len(boxes) > 0:
            det_data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                det_data.append({"Class": cls_name, "Confidence Score": f"{conf:.2%}"})
            st.dataframe(det_data)
        else:
            st.info("No parking violations or targets detected above confidence threshold.")
else:
    st.info("Please upload an image file to trigger automated detection.")
