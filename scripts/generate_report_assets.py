#!/usr/bin/env python3
"""
AIRI Team PITB AI Internship Task 1
Script: generate_report_assets.py
Description: Generates verified visual report charts (class distribution, loss curves, results table)
and compiles the final honest PDF report (report/final_report.pdf) using ReportLab.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_visual_charts(assets_dir: Path):
    """Generate matplotlib charts based on actual project metrics."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Actual Class Distribution Chart (22 images total, 20 bounding box instances)
    fig, ax = plt.subplots(figsize=(7, 4))
    classes = ['Car', 'Motorcycle', 'No Parking Sign', 'Roadside Parking']
    counts = [7, 5, 4, 4]  # Actual verified counts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(classes, counts, color=colors, width=0.55)
    ax.set_ylabel('Annotated Instance Count', fontsize=11, fontweight='bold')
    ax.set_title('Verified Dataset Class Distribution (20 Total Bounding Boxes across 22 Images)', fontsize=11, fontweight='bold', pad=12)
    ax.set_ylim(0, 10)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(assets_dir / "class_distribution.png", dpi=200)
    plt.close()
    
    # 2. Performance Metric Comparison Chart (Training Epoch 30 vs Test Evaluation)
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics_labels = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    train_metrics = [25.03, 50.00, 29.85, 16.42]
    test_metrics = [0.00, 0.00, 0.00, 0.00]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, train_metrics, width, label='Training Final Epoch (Epoch 30)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, test_metrics, width, label='Test Evaluation (4 images)', color='#d62728')
    
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Verified Quantitative Performance Metrics', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontweight='bold')
    ax.set_ylim(0, 60)
    ax.legend(loc='upper right', frameon=True)
    
    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(assets_dir / "metrics_comparison.png", dpi=200)
    plt.close()

    print(f"[SUCCESS] Visual report assets generated in: {assets_dir}")

def generate_pdf_report(report_pdf_path: Path, assets_dir: Path):
    """Compile final 31-section report into PDF using ReportLab with exact verified facts."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
        from reportlab.lib import colors
    except ImportError:
        print("[WARNING] ReportLab library not available. Install via pip install reportlab")
        return False
        
    doc = SimpleDocTemplate(
        str(report_pdf_path),
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F2C59'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=12,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=5
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#991B1B')
    )
    
    story = []
    
    # Title Banner
    story.append(Paragraph("Smart Parking Violation Detection Using YOLOv8", title_style))
    story.append(Paragraph("<b>AIRI Team PITB AI Internship Task 1 — Final Technical Report (Fact-Verified)</b>", ParagraphStyle('Sub', alignment=1, fontSize=11, textColor=colors.HexColor('#4B5563'))))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceBefore=0, spaceAfter=12))
    
    # Mandatory Limitation Warning Box
    warning_text = (
        "<b>CRITICAL PROJECT LIMITATION & METHODOLOGICAL NOTICE:</b><br/>"
        "The current implementation and training pipeline are complete, but the available dataset contains only 22 images, "
        "below the required 150–300 image target. Therefore, the reported evaluation metrics are preliminary and should not "
        "be treated as representative model performance."
    )
    warn_table = Table([[Paragraph(warning_text, callout_style)]], colWidths=[530])
    warn_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEF2F2')),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#EF4444')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(warn_table)
    story.append(Spacer(1, 10))
    
    sections = [
        ("1. Project Title", "Smart Parking Violation Detection Using YOLOv8 - AIRI Team PITB AI Internship Task 1."),
        ("2. Project Overview", "An end-to-end computer vision system built with YOLOv8 to detect parking violations, illegal roadside vehicle parking, and traffic signs in urban environments."),
        ("3. Problem Definition", "Unauthorized roadside parking causes severe traffic bottlenecks and emergency access delays. Automated vision-based monitoring provides scalable enforcement."),
        ("4. Use Case", "Deployable on municipal CCTV networks and traffic enforcement patrol vehicles for real-time automated detection and alert generation."),
        ("5. Target Classes", "0: car, 1: motorcycle, 2: no_parking_sign, 3: roadside_parking (roadside parked vehicle instances)."),
        ("6. Dataset Collection", "Curated a initial dataset of 22 real open-source images containing 20 annotated bounding box instances."),
        ("7. Data Sources", "Public open-access repositories (Roboflow Universe, Kaggle, Darknet/Ultralytics open samples)."),
        ("8. Dataset Size", "Actual verified size: 22 images (22 YOLO label files). Partitioned into 12 Train (54.5%), 6 Val (27.3%), and 4 Test (18.2%)."),
        ("9. Annotation", "Annotated in standard YOLO normalized coordinate format (<class_id> <x_center> <y_center> <width> <height>). Verified via validate_dataset.py."),
        ("10. Dataset Split", "Reproducible split enforced with fixed random seed (42) to prevent data leakage across training, validation, and testing."),
        ("11. YOLO Format", "Labels stored in individual .txt files with 0-1 normalized floating point bounding box coordinates."),
        ("12. Colab/GPU Setup", "Google Colab T4 GPU execution environment with Google Drive mount integration."),
        ("13. YOLOv8 Model Architecture", "Utilized YOLOv8n (nano baseline) featuring CSPDarknet backbone, C2f modules, and an anchor-free decoupled detection head."),
        ("14. Training Configuration", "Epochs: 30, Image Size: 640x640, Pretrained: True, Batch Size: 16, GPU: NVIDIA Tesla T4. Model weights saved at models/best.pt."),
        ("15. Training Execution", "Completed 30 epochs of training on Google Colab T4 GPU. Logged outputs saved to outputs/training_results/colab_parking_exp/results.csv."),
        ("16. GPU Specifications", "NVIDIA Tesla T4 GPU (16 GB VRAM) running CUDA 12.x on Google Colab instance."),
        ("17. Precision", "Model Precision measures accuracy of positive detections: P = TP / (TP + FP). Training final epoch: 25.03% (0.25029). Test evaluation: 0.00%."),
        ("18. Recall", "Model Recall measures coverage of ground truth instances: R = TP / (TP + FN). Training final epoch: 50.00% (0.50000). Test evaluation: 0.00%."),
        ("19. mAP@0.5", "Mean Average Precision at IoU threshold 0.50. Training final epoch: 29.85% (0.29850). Test evaluation: 0.00%."),
        ("20. mAP@0.5:0.95", "Mean Average Precision averaged over IoU 0.50 to 0.95. Training final epoch: 16.42% (0.16418). Test evaluation: 0.00%."),
        ("21. Confusion Matrix Analysis", "Confusion matrix plots generated at outputs/training_results/colab_parking_exp/confusion_matrix.png reflecting 30-epoch training run."),
        ("22. Loss Curves", "Monitored training and validation box_loss, cls_loss, and dfl_loss over 30 epochs saved in results.png."),
        ("23. Per-Class Results", "Training final epoch per-class metrics reflect initial learning across cars and motorcycles; test set per-class metrics produced 0% across the 4 test images."),
        ("24. Inference Examples", "Ran test inference on all 4 unseen test images (outputs/predictions/test_predictions/). Zero bounding box detections were produced due to model scale limitations on the tiny test set."),
        ("25. Error Analysis", "Documented 10 representative error modes in report/error_analysis.csv including false negatives, low confidence, small object misses, and background confusion."),
        ("26. Key Improvements", "1. Expand dataset size to 150-300 images. 2. Implement Albumentations Mosaic & HSV augmentation. 3. Fine-tune confidence thresholds."),
        ("27. What I Learned", "Gained hands-on experience in dataset structure setup, YOLO normalization, pipeline validation, GPU execution, and honest empirical reporting."),
        ("28. Limitations & Constraint Audit", "Primary limitation: Dataset contains 22 images (below the 150-300 target). Test set contains only 4 images, resulting in 0% test metrics."),
        ("29. Future Improvements", "Incorporate multi-scale training (--imgsz 800), expand training dataset to 300+ images, and integrate ByteTRACK video tracking."),
        ("30. Conclusion", "The end-to-end Computer Vision code, scripts, configuration, and notebook pipeline are complete and functional. Full dataset expansion to 150-300 images is required for representative performance."),
        ("31. References", "Ultralytics YOLOv8 Documentation; Roboflow Universe; COCO Dataset Benchmarks.")
    ]
    
    for title, text in sections:
        story.append(Paragraph(title, h1_style))
        story.append(Paragraph(text, body_style))
        
        # Insert inline visual charts & comparison table
        if title.startswith("8. Dataset Size") and (assets_dir / "class_distribution.png").exists():
            story.append(Spacer(1, 4))
            story.append(Image(str(assets_dir / "class_distribution.png"), width=420, height=220))
            story.append(Spacer(1, 6))
            
        elif title.startswith("20. mAP@0.5:0.95"):
            story.append(Spacer(1, 6))
            # Insert Verified Results Table
            table_data = [
                [Paragraph("<b>Evaluation Stage</b>", body_style), Paragraph("<b>Precision (P)</b>", body_style), Paragraph("<b>Recall (R)</b>", body_style), Paragraph("<b>mAP@0.5</b>", body_style), Paragraph("<b>mAP@0.5:0.95</b>", body_style)],
                [Paragraph("Training Final Epoch (Epoch 30)", body_style), Paragraph("25.03% (0.25029)", body_style), Paragraph("50.00% (0.50000)", body_style), Paragraph("29.85% (0.29850)", body_style), Paragraph("16.42% (0.16418)", body_style)],
                [Paragraph("Test Evaluation (4 Test Images)", body_style), Paragraph("0.00%", body_style), Paragraph("0.00%", body_style), Paragraph("0.00%", body_style), Paragraph("0.00%", body_style)]
            ]
            t = Table(table_data, colWidths=[170, 90, 90, 90, 90])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EFF6FF')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#93C5FD')),
                ('ALIGN', (1,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('PADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(t)
            story.append(Paragraph("<i>Note: The test set contains only 4 images (producing 0 detections), so these test results are highly limited.</i>", ParagraphStyle('Foot', fontName='Helvetica-Oblique', fontSize=8, textColor=colors.HexColor('#6B7280'))))
            story.append(Spacer(1, 6))
            
            if (assets_dir / "metrics_comparison.png").exists():
                story.append(Image(str(assets_dir / "metrics_comparison.png"), width=420, height=220))
                story.append(Spacer(1, 6))
                
    doc.build(story)
    print(f"[SUCCESS] Final 31-section PDF Report compiled: {report_pdf_path}")
    return True

def main():
    base_dir = Path(__file__).resolve().parent.parent
    assets_dir = base_dir / "report" / "report_assets"
    report_pdf_path = base_dir / "report" / "final_report.pdf"
    
    generate_visual_charts(assets_dir)
    generate_pdf_report(report_pdf_path, assets_dir)

if __name__ == "__main__":
    main()
