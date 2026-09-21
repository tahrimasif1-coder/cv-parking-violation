#!/usr/bin/env python3
"""
AIRI Team PITB AI Internship Task 1
Script: evaluate.py
Description: Quantitative evaluation script extracting Precision, Recall, mAP@0.5, mAP@0.5:0.95,
per-class performance breakdown, saving metrics.json and metrics.txt.
"""

import os
import sys
import json
from pathlib import Path

def evaluate_model(model_path: Path, data_yaml: Path, output_dir: Path):
    """Run model validation and extract evaluation metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Install via pip install ultralytics")
        return None
        
    print("=" * 60)
    print("           YOLOv8 MODEL EVALUATION & METRICS           ")
    print("=" * 60)
    print(f"Model Path  : {model_path}")
    print(f"Data Config : {data_yaml}")
    print(f"Output Dir  : {output_dir}")
    print("=" * 60 + "\n")
    
    if not model_path.exists():
        print(f"[ERROR] Model file {model_path} not found.")
        print("EVALUATION STATUS: PENDING USER ACTION (Requires trained best.pt)")
        return None
        
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), split="val", plots=True, name="val_eval", project=str(output_dir))
    
    # Extract values
    p = float(metrics.box.mp)        # mean precision
    r = float(metrics.box.mr)        # mean recall
    map50 = float(metrics.box.map50)  # mAP@0.5
    map95 = float(metrics.box.map)    # mAP@0.5:0.95
    
    # Class names mapping
    names = metrics.names
    per_class = {}
    for i, c in enumerate(metrics.box.ap_class_index):
        c_name = names[c]
        c_p = float(metrics.box.p[i])
        c_r = float(metrics.box.r[i])
        c_map50 = float(metrics.box.ap50[i])
        c_map95 = float(metrics.box.ap[i])
        per_class[c_name] = {
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "mAP50": round(c_map50, 4),
            "mAP50-95": round(c_map95, 4)
        }
        
    results_dict = {
        "overall": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "mAP50": round(map50, 4),
            "mAP50-95": round(map95, 4)
        },
        "per_class": per_class
    }
    
    # Save metrics.json
    metrics_json_path = output_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    print(f"[SAVED] Metrics JSON written to {metrics_json_path}")
    
    # Save metrics.txt
    metrics_txt_path = output_dir / "metrics.txt"
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("  PARKING VIOLATION DETECTION - EVALUATION RESULTS  \n")
        f.write("====================================================\n\n")
        f.write("OVERALL MODEL PERFORMANCE:\n")
        f.write(f"  Precision    (P)      : {p:.4f} ({p*100:.2f}%)\n")
        f.write(f"  Recall       (R)      : {r:.4f} ({r*100:.2f}%)\n")
        f.write(f"  mAP @ 0.50            : {map50:.4f} ({map50*100:.2f}%)\n")
        f.write(f"  mAP @ 0.50:0.95       : {map95:.4f} ({map95*100:.2f}%)\n\n")
        f.write("PER-CLASS PERFORMANCE BREAKDOWN:\n")
        f.write(f"{'Class Name':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<12}\n")
        f.write("-" * 72 + "\n")
        for c_name, c_m in per_class.items():
            f.write(f"{c_name:<20} | {c_m['precision']:<10.4f} | {c_m['recall']:<10.4f} | {c_m['mAP50']:<10.4f} | {c_m['mAP50-95']:<12.4f}\n")
        f.write("====================================================\n")
        
    print(f"[SAVED] Metrics Summary written to {metrics_txt_path}")
    return results_dict

def main():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "best.pt"
    data_yaml = base_dir / "dataset" / "data.yaml"
    output_dir = base_dir / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print("\n=======================================================")
        print("EVALUATION STATUS: PENDING USER ACTION")
        print("=======================================================")
        print("No trained weights file found at models/best.pt.")
        print("Run training first via Google Colab or scripts/train.py.")
        print("=======================================================\n")
        sys.exit(0)
        
    evaluate_model(model_path, data_yaml, output_dir)

if __name__ == "__main__":
    main()
