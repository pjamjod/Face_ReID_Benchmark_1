#!/usr/bin/env python3
"""
Test script for 300W dataset loading and validation
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))  # Add current directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add src

from data_loaders.data_loader_300w import Dataset300W
import numpy as np

def test_300w_dataset():
    """Comprehensive test of 300W dataset loading"""
    print("🧪 Testing 300W Dataset Loading")
    print("=" * 50)

    # Initialize dataset
    dataset = Dataset300W()

    # Basic checks
    num_images = len(dataset)
    print(f"📊 Dataset size: {num_images} images")

    if num_images == 0:
        print("❌ No images loaded!")
        return False

    # Analyze dataset composition
    print("\n📂 Dataset Composition:")
    subdirs = {}
    for img_path in dataset.image_paths:
        # Extract subdirectory path
        rel_path = os.path.relpath(img_path, "datasets/300w")
        parts = rel_path.split(os.sep)

        if len(parts) >= 2 and parts[0] == "ibug_300W_large_face_landmark_dataset":
            # For ibug dataset, use the specific subfolder name
            subdir_key = parts[1]
            if len(parts) > 2 and parts[2] in ["testset", "trainset"]:
                subdir_key = f"{parts[1]}_{parts[2]}"
        else:
            subdir_key = parts[0]

        if subdir_key not in subdirs:
            subdirs[subdir_key] = 0
        subdirs[subdir_key] += 1

    for subdir, count in sorted(subdirs.items()):
        print(f"  {subdir}: {count} images")

    # Test first few samples
    print("\n🔍 Testing sample loading...")
    for i in range(min(5, num_images)):
        try:
            item = dataset[i]
            img_shape = item['image'].shape
            lm68_shape = item['landmarks_68'].shape
            lm5_shape = item['landmarks_5'].shape

            print(f"  Sample {i}: Image {img_shape}, LM68 {lm68_shape}, LM5 {lm5_shape}")

            # Validate shapes
            assert img_shape[2] == 3, f"Expected RGB image, got {img_shape}"
            assert lm68_shape == (68, 2), f"Expected (68, 2) landmarks, got {lm68_shape}"
            assert lm5_shape == (5, 2), f"Expected (5, 2) landmarks, got {lm5_shape}"

            # Validate landmark ranges
            h, w = img_shape[:2]
            lm68 = item['landmarks_68']
            lm5 = item['landmarks_5']

            assert np.all(lm68 >= 0), "Landmarks 68 contain negative coordinates"
            assert np.all(lm68[:, 0] < w), "Landmarks 68 X out of bounds"
            assert np.all(lm68[:, 1] < h), "Landmarks 68 Y out of bounds"

            assert np.all(lm5 >= 0), "Landmarks 5 contain negative coordinates"
            assert np.all(lm5[:, 0] < w), "Landmarks 5 X out of bounds"
            assert np.all(lm5[:, 1] < h), "Landmarks 5 Y out of bounds"

        except Exception as e:
            print(f"❌ Error loading sample {i}: {e}")
            return False

    # Test landmark extraction consistency
    print("\n🔍 Testing landmark extraction...")
    item = dataset[0]
    lm68 = item['landmarks_68']
    lm5 = item['landmarks_5']

    # Manual extraction for verification
    left_eye_manual = np.mean(lm68[36:42], axis=0)
    right_eye_manual = np.mean(lm68[42:48], axis=0)
    nose_manual = lm68[30]
    left_mouth_manual = lm68[48]
    right_mouth_manual = lm68[54]

    lm5_manual = np.array([left_eye_manual, right_eye_manual, nose_manual, left_mouth_manual, right_mouth_manual])

    if not np.allclose(lm5, lm5_manual, atol=1e-6):
        print("❌ Landmark extraction mismatch!")
        print(f"Auto: {lm5}")
        print(f"Manual: {lm5_manual}")
        return False

    print("✅ Landmark extraction verified")

    # Statistics
    print("\n📈 Dataset Statistics:")
    all_lm68 = np.array(dataset.landmarks_68)  # (N, 68, 2)
    all_lm5 = np.array(dataset.landmarks_5)    # (N, 5, 2)

    print(f"  Landmarks 68 range: [{all_lm68.min():.1f}, {all_lm68.max():.1f}]")
    print(f"  Landmarks 5 range: [{all_lm5.min():.1f}, {all_lm5.max():.1f}]")

    # Test image loading
    print("\n🖼️  Testing image loading...")
    item = dataset[0]
    image = item['image']
    assert isinstance(image, np.ndarray), "Image should be numpy array"
    assert image.dtype == np.uint8, f"Expected uint8 image, got {image.dtype}"
    print(f"  Sample image: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")

    print("\n✅ All 300W dataset tests passed!")
    return True

if __name__ == "__main__":
    success = test_300w_dataset()
    if not success:
        sys.exit(1)