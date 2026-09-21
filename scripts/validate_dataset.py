#!/usr/bin/env python3
"""
AIRI Team PITB AI Internship Task 1
Script: validate_dataset.py
Description: Full validation suite checking corrupted images, missing labels, YOLO coordinate bounds (0-1),
valid class IDs, empty files, image/label alignment, and printing class distribution statistics.
"""

import os
import sys
import glob
from pathlib import Path
from collections import Counter

CLASSES = {
    0: "car",
    1: "motorcycle",
    2: "no_parking_sign",
    3: "roadside_parking"
}

def validate_image(img_path: Path):
    """Check if image file is non-empty and readable."""
    if img_path.stat().st_size == 0:
        return False, "Empty (0 byte) file"
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            img.verify()
        return True, "Valid"
    except Exception as e:
        return False, f"Corrupted image: {e}"

def validate_yolo_label(lbl_path: Path, max_class_id=3):
    """Validate YOLO label format: <class_id> <x_center> <y_center> <width> <height>."""
    if not lbl_path.exists():
        return False, "Missing label file", []
    
    if lbl_path.stat().st_size == 0:
        return True, "Empty label file (background image)", []
        
    errors = []
    annotations = []
    
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {i}: Expected 5 values, got {len(parts)} ({line})")
            continue
            
        try:
            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            
            if cls_id < 0 or cls_id > max_class_id:
                errors.append(f"Line {i}: Invalid class ID {cls_id} (allowed 0-{max_class_id})")
                
            for val_name, val in [("x_center", x), ("y_center", y), ("width", w), ("height", h)]:
                if not (0.0 <= val <= 1.0):
                    errors.append(f"Line {i}: {val_name} {val} outside [0, 1] bounds")
                    
            annotations.append((cls_id, x, y, w, h))
        except ValueError as e:
            errors.append(f"Line {i}: Non-numeric value in line '{line}'")
            
    is_valid = len(errors) == 0
    return is_valid, "; ".join(errors) if errors else "Valid", annotations

def run_validation(base_dir: Path):
    """Run full verification on dataset/ directory."""
    print("=" * 60)
    print("      SMART PARKING DATASET VALIDATION & AUDIT REPORT     ")
    print("=" * 60)
    
    dataset_dir = base_dir / "dataset"
    splits = ["train", "val", "test"]
    
    total_images = 0
    total_labels = 0
    corrupted_images = []
    invalid_labels = []
    unmatched_pairs = []
    class_counts = Counter()
    split_counts = {}
    
    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        
        if not img_dir.exists():
            print(f"[WARNING] Directory missing: {img_dir}")
            continue
            
        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        split_counts[split] = len(image_files)
        total_images += len(image_files)
        
        for img_path in image_files:
            # 1. Validate image integrity
            valid_img, img_err = validate_image(img_path)
            if not valid_img:
                corrupted_images.append((img_path, img_err))
                
            # 2. Check corresponding label
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                unmatched_pairs.append((img_path, "Missing corresponding .txt label"))
            else:
                total_labels += 1
                valid_lbl, lbl_err, annos = validate_yolo_label(lbl_path)
                if not valid_lbl:
                    invalid_labels.append((lbl_path, lbl_err))
                for cls_id, x, y, w, h in annos:
                    class_counts[cls_id] += 1

    print(f"\n1. DATASET OVERVIEW & SPLITS")
    print(f"   - Total Images Found : {total_images}")
    print(f"   - Total Labels Found : {total_labels}")
    for split, count in split_counts.items():
        pct = (count / total_images * 100) if total_images > 0 else 0
        print(f"   - {split.capitalize():<5} split count  : {count} images ({pct:.1f}%)")
        
    print(f"\n2. INTEGRITY CHECKS")
    print(f"   - Corrupted Images   : {len(corrupted_images)}")
    for p, err in corrupted_images:
        print(f"     [FAIL] {p.name}: {err}")
        
    print(f"   - Invalid Labels     : {len(invalid_labels)}")
    for p, err in invalid_labels:
        print(f"     [FAIL] {p.name}: {err}")
        
    print(f"   - Mismatched Files   : {len(unmatched_pairs)}")
    for p, err in unmatched_pairs:
        print(f"     [FAIL] {p.name}: {err}")
        
    print(f"\n3. CLASS DISTRIBUTION & ANNOTATION COUNTS")
    total_objects = sum(class_counts.values())
    print(f"   - Total Bounding Box Objects Annotated: {total_objects}")
    for cls_id, cls_name in CLASSES.items():
        cnt = class_counts[cls_id]
        pct = (cnt / total_objects * 100) if total_objects > 0 else 0
        print(f"   - Class {cls_id} ({cls_name:<18}): {cnt} instances ({pct:.1f}%)")
        
    print("\n" + "=" * 60)
    if total_images == 0:
        print("RESULT: PENDING USER ACTION (No images in dataset directory)")
    elif len(corrupted_images) == 0 and len(invalid_labels) == 0 and len(unmatched_pairs) == 0:
        print("RESULT: PASS - Dataset is clean and fully compliant!")
    else:
        print("RESULT: NEEDS ACTION - Issues found above. Fix labels/images before training.")
    print("=" * 60 + "\n")
    
    return {
        "total_images": total_images,
        "total_labels": total_labels,
        "split_counts": split_counts,
        "corrupted": len(corrupted_images),
        "invalid_labels": len(invalid_labels),
        "unmatched": len(unmatched_pairs),
        "class_counts": dict(class_counts)
    }

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    run_validation(base_dir)
