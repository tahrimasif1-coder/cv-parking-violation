#!/usr/bin/env python3
"""
AIRI Team PITB AI Internship Task 1
Script: predict.py
Description: Inference pipeline running detection on unseen/test images (15+ images minimum),
overlaying bounding boxes, class labels, and confidence scores, saving to outputs/predictions/.
"""

import os
import sys
from pathlib import Path

def run_inference(model_path: Path, source_dir: Path, output_dir: Path, conf_threshold: float = 0.25):
    """Run inference on test images and save visualization outputs."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics not installed. Run pip install ultralytics")
        return False
        
    print("=" * 60)
    print("         YOLOv8 UNSEEN TEST INFERENCE PIPELINE         ")
    print("=" * 60)
    print(f"Model File     : {model_path}")
    print(f"Source Directory: {source_dir}")
    print(f"Confidence Thresh: {conf_threshold}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60 + "\n")
    
    if not model_path.exists():
        print(f"[ERROR] Model weights {model_path} not found.")
        print("INFERENCE STATUS: PENDING USER ACTION (Requires trained best.pt)")
        return False
        
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(source_dir.glob(ext)))
        
    print(f"[INFERENCE] Found {len(test_images)} images for testing.")
    if len(test_images) < 15:
        print(f"[NOTE] Assignment requires at least 15 test predictions. Found {len(test_images)}.")
        
    model = YOLO(str(model_path))
    
    results = model.predict(
        source=str(source_dir),
        conf=conf_threshold,
        save=True,
        project=str(output_dir),
        name="test_predictions",
        exist_ok=True
    )
    
    save_path = output_dir / "test_predictions"
    print(f"\n[SUCCESS] Inference completed! Annotated predictions saved to: {save_path}")
    return True

def main():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "best.pt"
    source_dir = base_dir / "dataset" / "images" / "test"
    output_dir = base_dir / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print("\n=======================================================")
        print("INFERENCE STATUS: PENDING USER ACTION")
        print("=======================================================")
        print("No trained weights file found at models/best.pt.")
        print("Steps:")
        print("1. Train model on Google Colab or locally.")
        print("2. Place best.pt in models/ directory.")
        print("3. Run python scripts/predict.py")
        print("=======================================================\n")
        sys.exit(0)
        
    run_inference(model_path, source_dir, output_dir)

if __name__ == "__main__":
    main()
