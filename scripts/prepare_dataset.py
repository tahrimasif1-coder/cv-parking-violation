#!/usr/bin/env python3
"""
AIRI Team PITB AI Internship Task 1
Script: prepare_dataset.py
Description: Download, structure, clean, and reproducibly split (70% train / 20% val / 10% test)
the Smart Parking Violation Dataset into YOLOv8 layout.
"""

import os
import sys
import glob
import shutil
import random
import argparse
from pathlib import Path

# Fixed seed for reproducible split
SEED = 42

CLASSES = {
    0: "car",
    1: "motorcycle",
    2: "no_parking_sign",
    3: "roadside_parking"  # or parking_meter
}

def setup_directory_structure(base_dir: Path):
    """Create YOLO standard directory structure."""
    splits = ["train", "val", "test"]
    subdirs = ["images", "labels"]
    
    for split in splits:
        for sd in subdirs:
            p = base_dir / "dataset" / sd / split
            p.mkdir(parents=True, exist_ok=True)
            print(f"[INIT] Created directory: {p}")

def split_data(image_files, base_dir: Path, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    """
    Split dataset reproducibly into train/val/test splits.
    Train: 70%, Val: 20%, Test: 10%
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    random.seed(SEED)
    random.shuffle(image_files)
    
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    print(f"\n[SPLIT] Total images found: {n_total}")
    print(f"  - Train ({train_ratio*100:.0f}%): {len(train_files)}")
    print(f"  - Val   ({val_ratio*100:.0f}%): {len(val_files)}")
    print(f"  - Test  ({test_ratio*100:.0f}%): {len(test_files)}")
    
    for split_name, files in splits.items():
        img_dest = base_dir / "dataset" / "images" / split_name
        lbl_dest = base_dir / "dataset" / "labels" / split_name
        
        for img_path in files:
            lbl_path = img_path.with_suffix(".txt")
            
            # Copy image
            shutil.copy2(img_path, img_dest / img_path.name)
            
            # Copy label if exists, else create empty label file
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dest / lbl_path.name)
            else:
                (lbl_dest / lbl_path.name).touch()
                
    print("[SPLIT] Dataset successfully partitioned and copied.")

def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int, base_dir: Path):
    """Download dataset from Roboflow Universe using official Roboflow API."""
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=str(base_dir / "raw_downloads"))
        print(f"[ROBOFLOW] Download completed to {dataset.location}")
        return Path(dataset.location)
    except Exception as e:
        print(f"[ERROR] Roboflow download failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Prepare Smart Parking Violation Dataset")
    parser.add_argument("--source-dir", type=str, default=None, help="Directory containing raw images & labels")
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow API key for direct download")
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent
    setup_directory_structure(base_dir)
    
    if args.source_dir and os.path.exists(args.source_dir):
        raw_path = Path(args.source_dir)
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(raw_path.glob(ext)))
            
        if image_files:
            split_data(image_files, base_dir)
        else:
            print(f"[WARNING] No image files found in {args.source_dir}")
    elif args.roboflow_key:
        download_path = download_roboflow_dataset(
            api_key=args.roboflow_key,
            workspace="parking-violation",
            project="smart-parking-detection",
            version=1,
            base_dir=base_dir
        )
    else:
        print("\n=======================================================")
        print("DATASET PREPARATION STATUS: PENDING USER ACTION")
        print("=======================================================")
        print("No raw dataset directory or Roboflow API key supplied.")
        print("To acquire and prepare the dataset:")
        print("1. Download 150-300 images of parking violations (cars, motorcycles, no-parking signs, meters).")
        print("2. Place images and YOLO format label .txt files in a raw directory e.g., 'data_raw/'.")
        print("3. Run: python scripts/prepare_dataset.py --source-dir data_raw/")
        print("=======================================================\n")

if __name__ == "__main__":
    main()
