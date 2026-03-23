"""
Complete 300W Dataset Loader for Face Alignment Benchmark

Loads the complete 300W dataset including:
- Main 300W dataset (indoor/outdoor images)
- Additional test sets: afw, helen, ibug, lfpw

Supports both 68-point and 5-point landmark formats.
Total: ~1626 images with 68-point facial landmarks.
"""

import os
import glob
import numpy as np
from PIL import Image

# Import the extraction function
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from landmark_utils import extract_5_points_from_68

class Dataset300W:
    """
    Complete 300W Dataset Loader

    Loads all 300W datasets including main dataset and additional test sets:
    - 300W: Main indoor/outdoor images (600 images)
    - afw: Annotated Faces in the Wild (337 images)
    - helen: HELEN dataset test set (330 images)
    - ibug: iBUG dataset (135 images)
    - lfpw: LFPW dataset test set (224 images)

    Total: ~1626 images with 68-point facial landmarks
    """
    def __init__(self, root_dir="datasets/300w"):
        self.root_dir = root_dir
        self.image_paths = []
        self.landmarks_68 = []
        self.landmarks_5 = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading 300W dataset from {self.root_dir}...")

        # Load indoor images
        indoor_dir = os.path.join(self.root_dir, "300W", "01_Indoor")
        if os.path.exists(indoor_dir):
            self._load_subdirectory(indoor_dir)

        # Load outdoor images
        outdoor_dir = os.path.join(self.root_dir, "300W", "02_Outdoor")
        if os.path.exists(outdoor_dir):
            self._load_subdirectory(outdoor_dir)

        # Load additional test sets from ibug_300W_large_face_landmark_dataset
        ibug_dataset_dir = os.path.join(self.root_dir, "ibug_300W_large_face_landmark_dataset")

        # Load ibug test set
        ibug_dir = os.path.join(ibug_dataset_dir, "ibug")
        if os.path.exists(ibug_dir):
            self._load_subdirectory(ibug_dir, exclude_mirror=True)

        # Load afw test set
        afw_dir = os.path.join(ibug_dataset_dir, "afw")
        if os.path.exists(afw_dir):
            self._load_subdirectory(afw_dir, exclude_mirror=True)

        # Load helen test set
        helen_test_dir = os.path.join(ibug_dataset_dir, "helen", "testset")
        if os.path.exists(helen_test_dir):
            self._load_subdirectory(helen_test_dir, exclude_mirror=True)

        # Load lfpw test set
        lfpw_test_dir = os.path.join(ibug_dataset_dir, "lfpw", "testset")
        if os.path.exists(lfpw_test_dir):
            self._load_subdirectory(lfpw_test_dir, exclude_mirror=True)

        print(f"✅ Loaded {len(self.image_paths)} images with landmarks")

    def _load_subdirectory(self, sub_dir, exclude_mirror=False):
        pts_files = glob.glob(os.path.join(sub_dir, "*.pts"))

        for pts_file in pts_files:
            # Skip mirror images if requested
            if exclude_mirror and "_mirror" in pts_file:
                continue

            # Get corresponding image file (try multiple extensions)
            base_name = os.path.splitext(pts_file)[0]
            img_file = None

            for ext in ['.jpg', '.png']:
                candidate = base_name + ext
                if os.path.exists(candidate):
                    img_file = candidate
                    break

            if img_file is None:
                continue

            landmarks_68 = self._load_pts_file(pts_file)
            if landmarks_68 is None:
                continue

            landmarks_5 = extract_5_points_from_68(landmarks_68)

            self.image_paths.append(img_file)
            self.landmarks_68.append(landmarks_68)
            self.landmarks_5.append(landmarks_5)

    def _load_pts_file(self, pts_file):
        try:
            with open(pts_file, 'r') as f:
                lines = f.readlines()

            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() == "{":
                    data_start = i + 1
                    break

            landmarks = []
            for line in lines[data_start:]:
                line = line.strip()
                if line == "}":
                    break
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        landmarks.append([x, y])

            landmarks = np.array(landmarks)
            if landmarks.shape[0] != 68:
                return None
            return landmarks
        except:
            return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        landmarks_68 = self.landmarks_68[idx]
        landmarks_5 = self.landmarks_5[idx]

        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        return {
            'image_path': img_path,
            'image': image,
            'landmarks_68': landmarks_68,
            'landmarks_5': landmarks_5
        }