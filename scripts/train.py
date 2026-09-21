#!/usr/bin/env python3
"""
AIRI Team PITB AI Internship Task 1
Script: train.py
Description: Model training script for Smart Parking Violation Detection using YOLOv8n.
Trains for minimum 30 epochs, logs metrics, saves best.pt and training assets.
"""

import os
import sys
import shutil
from pathlib import Path

def train_yolov8(
    data_yaml: str,
    epochs: int = 30,
    imgsz: int = 640,
    batch_size: int = 16,
    model_name: str = "yolov8n.pt",
    output_dir: Path = None,
    models_dir: Path = None
):
    """
    Train YOLOv8 model using Ultralytics framework.
    """
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("\n=======================================================")
        print("ERROR: PyTorch or Ultralytics library is not installed.")
        print("Install dependencies using: pip install -r requirements.txt")
        print("=======================================================\n")
        return False
        
    print("=" * 60)
    print("            YOLOv8 MODEL TRAINING INITIATED           ")
    print("=" * 60)
    print(f"Model Baseline : {model_name}")
    print(f"Dataset YAML   : {data_yaml}")
    print(f"Epochs         : {epochs}")
    print(f"Image Size     : {imgsz}")
    print(f"Batch Size     : {batch_size}")
    print(f"CUDA Available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not detected. Training on CPU will be slower.")
    print("=" * 60 + "\n")

    # Load baseline model
    model = YOLO(model_name)
    
    # Run training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name="parking_violation_exp",
        project=str(output_dir / "runs"),
        exist_ok=True,
        pretrained=True,
        save=True,
        plots=True,
        verbose=True
    )
    
    # Save best weights to models/best.pt
    run_dir = output_dir / "runs" / "parking_violation_exp"
    best_weights = run_dir / "weights" / "best.pt"
    
    if best_weights.exists() and models_dir:
        dest_best = models_dir / "best.pt"
        shutil.copy2(best_weights, dest_best)
        print(f"\n[SUCCESS] Best model saved to: {dest_best}")
        
    print(f"[SUCCESS] Training artifacts saved to: {run_dir}")
    return True

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_yaml = str(base_dir / "dataset" / "data.yaml")
    output_dir = base_dir / "outputs" / "training_results"
    models_dir = base_dir / "models"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset images exist
    train_imgs = list((base_dir / "dataset" / "images" / "train").glob("*.jpg")) + \
                 list((base_dir / "dataset" / "images" / "train").glob("*.png"))
                 
    if len(train_imgs) == 0:
        print("\n=======================================================")
        print("TRAINING STATUS: PENDING USER ACTION")
        print("=======================================================")
        print("No training images found in dataset/images/train/.")
        print("Steps to complete training:")
        print("1. Populate dataset/ using prepare_dataset.py or Google Colab.")
        print("2. Run python scripts/train.py --epochs 30")
        print("=======================================================\n")
        sys.exit(0)
        
    train_yolov8(
        data_yaml=data_yaml,
        epochs=30,
        imgsz=640,
        batch_size=16,
        output_dir=output_dir,
        models_dir=models_dir
    )

if __name__ == "__main__":
    main()
