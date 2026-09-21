#!/usr/bin/env python3
"""
AIRI Team PITB AI Internship Task 1
Script: check_duplicates.py
Description: Hash-based duplicate image detection across train, val, and test splits
to strictly prevent data leakage and identical frame redundancy.
"""

import hashlib
import os
from pathlib import Path
from collections import defaultdict

def compute_file_hash(filepath: Path, blocksize=65536):
    """Compute MD5 checksum for binary file content."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(blocksize)
    return hasher.hexdigest()

def check_dataset_duplicates(base_dir: Path):
    """Scan dataset/images/ for duplicate images across splits."""
    dataset_dir = base_dir / "dataset" / "images"
    splits = ["train", "val", "test"]
    
    hash_map = defaultdict(list)
    filename_map = defaultdict(list)
    total_scanned = 0
    
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        for img_path in split_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                total_scanned += 1
                fhash = compute_file_hash(img_path)
                hash_map[fhash].append((split, img_path.name, img_path))
                filename_map[img_path.name].append((split, img_path))
                
    print("\n" + "=" * 60)
    print("      DUPLICATE IMAGE & LEAKAGE AUDIT REPORT      ")
    print("=" * 60)
    print(f"Total Images Scanned Across Splits: {total_scanned}\n")
    
    exact_content_duplicates = {h: locations for h, locations in hash_map.items() if len(locations) > 1}
    duplicate_filenames = {fn: locations for fn, locations in filename_map.items() if len(locations) > 1}
    
    cross_split_leakage = []
    for h, locations in exact_content_duplicates.items():
        found_splits = set(loc[0] for loc in locations)
        if len(found_splits) > 1:
            cross_split_leakage.append((h, locations))
            
    print(f"1. EXACT CONTENT DUPLICATES (Same Image Data): {len(exact_content_duplicates)}")
    for h, locations in exact_content_duplicates.items():
        print(f"   - Hash {h[:8]}... found in {len(locations)} files:")
        for split, fname, _ in locations:
            print(f"       [{split}] {fname}")
            
    print(f"\n2. DUPLICATE FILENAMES: {len(duplicate_filenames)}")
    for fn, locations in duplicate_filenames.items():
        print(f"   - Filename '{fn}' appears in splits: {[loc[0] for loc in locations]}")
        
    print(f"\n3. CROSS-SPLIT LEAKAGE (Train <-> Val <-> Test): {len(cross_split_leakage)}")
    for h, locations in cross_split_leakage:
        print(f"   [LEAKAGE WARNING] Hash {h[:8]}... present across splits: {[loc[0] for loc in locations]}")
        
    print("\n" + "=" * 60)
    if total_scanned == 0:
        print("RESULT: PENDING USER ACTION (No images scanned)")
    elif len(exact_content_duplicates) == 0 and len(duplicate_filenames) == 0:
        print("RESULT: PASS - No duplicate images or data leakage detected!")
    else:
        print("RESULT: NEEDS ACTION - Remove duplicates to prevent validation leakage.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    check_dataset_duplicates(base_dir)
