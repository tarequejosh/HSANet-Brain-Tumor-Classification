"""
Convert Figshare Brain Tumor Dataset from .mat to .jpg format
==============================================================
This script converts MATLAB .mat files to JPG images organized by class.

Dataset: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
Classes: 1 = Meningioma (708), 2 = Glioma (1426), 3 = Pituitary (930)

Author: HSANet Team
Date: January 2026
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py  # For MATLAB v7.3 files (HDF5 format)
from PIL import Image


def convert_mat_to_jpg(input_dir: str, output_dir: str):
    """
    Convert .mat files to .jpg images organized by class folders.
    
    Args:
        input_dir: Directory containing .mat files
        output_dir: Output directory for organized images
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Class mapping based on Figshare documentation
    # Label: 1 = Meningioma, 2 = Glioma, 3 = Pituitary
    class_names = {
        1: 'meningioma',
        2: 'glioma', 
        3: 'pituitary'
    }
    
    # Create output directories
    for class_name in class_names.values():
        (output_path / class_name).mkdir(parents=True, exist_ok=True)
    
    # Get all .mat files
    mat_files = sorted(input_path.glob('*.mat'))
    print(f"Found {len(mat_files)} .mat files")
    
    # Statistics
    class_counts = {1: 0, 2: 0, 3: 0}
    errors = []
    
    # Process each file
    for mat_file in tqdm(mat_files, desc="Converting"):
        try:
            # Load .mat file using h5py (MATLAB v7.3 format)
            with h5py.File(str(mat_file), 'r') as f:
                # Get cjdata structure
                cjdata = f['cjdata']
                
                # Get label (1=meningioma, 2=glioma, 3=pituitary)
                label = int(cjdata['label'][0, 0])
                
                # Get image data (need to transpose for correct orientation)
                image_data = np.array(cjdata['image']).T
                
                # Normalize to 0-255
                img_float = image_data.astype(np.float64)
                img_min = img_float.min()
                img_max = img_float.max()
                
                if img_max > img_min:
                    img_normalized = (255.0 * (img_float - img_min) / (img_max - img_min))
                else:
                    img_normalized = np.zeros_like(img_float)
                
                img_uint8 = img_normalized.astype(np.uint8)
                
                # Create PIL image
                img = Image.fromarray(img_uint8, mode='L')  # 'L' for grayscale
                
                # Save to appropriate class folder
                class_name = class_names[label]
                output_file = output_path / class_name / f"{mat_file.stem}.jpg"
                img.save(output_file, quality=95)
                
                class_counts[label] += 1
            
        except Exception as e:
            errors.append((mat_file.name, str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"\nClass Distribution:")
    for label, name in class_names.items():
        print(f"  {name}: {class_counts[label]} images")
    print(f"\nTotal: {sum(class_counts.values())} images")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for fname, err in errors[:10]:  # Show first 10 errors
            print(f"  {fname}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"\nOutput saved to: {output_path}")
    
    return class_counts, errors


def verify_output(output_dir: str):
    """Verify the converted dataset"""
    output_path = Path(output_dir)
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    total = 0
    for class_dir in sorted(output_path.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')))
            print(f"  {class_dir.name}: {count} images")
            total += count
    
    print(f"\nTotal images: {total}")
    
    # Expected counts from documentation
    expected = {
        'meningioma': 708,
        'glioma': 1426,
        'pituitary': 930
    }
    
    print("\nExpected vs Actual:")
    for class_name, exp_count in expected.items():
        class_path = output_path / class_name
        actual_count = len(list(class_path.glob('*.jpg'))) if class_path.exists() else 0
        status = "✓" if actual_count == exp_count else "✗"
        print(f"  {class_name}: Expected {exp_count}, Got {actual_count} {status}")


if __name__ == "__main__":
    # Paths
    INPUT_DIR = "/Users/tarequejosh/Downloads/files_updated/FigshareBrats/extracted_data"
    OUTPUT_DIR = "/Users/tarequejosh/Downloads/files_updated/FigshareBrats/figshare_images"
    
    # Convert
    print("Converting Figshare Brain Tumor Dataset...")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    class_counts, errors = convert_mat_to_jpg(INPUT_DIR, OUTPUT_DIR)
    
    # Verify
    verify_output(OUTPUT_DIR)
