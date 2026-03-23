import cv2
import numpy as np
from mtcnn import MTCNN
from skimage import transform as trans

def get_alignment_matrix(src_pts):
    """
    Computes the similarity transform matrix to map 5 points 
    to standard 112x112 coordinates.
    """
    # Standard reference points for a 112x112 face image
    reference_5pts = np.array([
        [30.2946, 51.6963],  # Left Eye
        [65.5318, 51.5014],  # Right Eye
        [48.0252, 71.7366],  # Nose Tip
        [33.5493, 92.3655],  # Left Mouth Corner
        [62.7299, 92.2041]   # Right Mouth Corner
    ], dtype=np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(src_pts, reference_5pts)
    return tform.params[0:2, :]

# 1. Initialize Detector
detector = MTCNN()

# 2. Load Your Image
img_path = "meen.jpg" # Change this to your file
image = cv2.imread(img_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Detect Face and Landmarks
results = detector.detect_faces(rgb_image)

if results:
    # Use the first face detected
    face_info = results[0]
    keypoints = face_info['keypoints']
    
    # Extract the 5 points in correct order
    src_pts = np.array([
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose'],
        keypoints['mouth_left'],
        keypoints['mouth_right']
    ], dtype=np.float32)

    # 4. Perform Alignment
    M = get_alignment_matrix(src_pts)
    aligned_face = cv2.warpAffine(image, M, (112, 112))

    # Show results
    cv2.imshow("Original", image)
    cv2.imshow("Aligned (5-Point)", aligned_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected!")