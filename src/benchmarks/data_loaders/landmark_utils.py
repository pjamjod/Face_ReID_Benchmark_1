"""
5-Point Landmark Extraction from 68-Point Annotations

Extracts standard 5 facial landmarks from 300W's 68-point annotations:
- Left eye center: average of points 36-41
- Right eye center: average of points 42-47
- Nose tip: point 30
- Left mouth corner: point 48
- Right mouth corner: point 54

Input: 68-point landmarks as numpy array (68, 2)
Output: 5-point landmarks as numpy array (5, 2)
"""

import numpy as np

def extract_5_points_from_68(landmarks_68):
    """
    Extract 5 standard facial landmarks from 68-point annotations.

    Args:
        landmarks_68: numpy array of shape (68, 2) with (x, y) coordinates

    Returns:
        landmarks_5: numpy array of shape (5, 2) with 5-point landmarks
                     Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    if landmarks_68.shape != (68, 2):
        raise ValueError(f"Expected (68, 2) landmarks, got {landmarks_68.shape}")

    # Left eye: average of points 36-41 (eye contour)
    left_eye = np.mean(landmarks_68[36:42], axis=0)

    # Right eye: average of points 42-47 (eye contour)
    right_eye = np.mean(landmarks_68[42:48], axis=0)

    # Nose tip: point 30
    nose = landmarks_68[30]

    # Left mouth corner: point 48
    left_mouth = landmarks_68[48]

    # Right mouth corner: point 54
    right_mouth = landmarks_68[54]

    landmarks_5 = np.array([left_eye, right_eye, nose, left_mouth, right_mouth])

    return landmarks_5

def validate_5_point_extraction():
    """Test the 5-point extraction with sample data"""
    # Create mock 68-point landmarks (just for testing)
    mock_68 = np.random.rand(68, 2) * 100  # Random points

    landmarks_5 = extract_5_points_from_68(mock_68)

    assert landmarks_5.shape == (5, 2), f"Expected (5, 2), got {landmarks_5.shape}"

    # Check that points are within reasonable bounds
    assert np.all(landmarks_5 >= 0) and np.all(landmarks_5 <= 100), "Points out of bounds"

    print("✅ 5-point extraction validation passed")
    print(f"Sample 5-point landmarks:\n{landmarks_5}")

if __name__ == "__main__":
    validate_5_point_extraction()