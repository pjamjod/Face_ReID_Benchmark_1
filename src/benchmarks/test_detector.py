# src/face_reid.py
import os
import cv2
import numpy as np
from face_detection.yunet import YuNetDetector
from face_detection.scrfd import SCRFDDetector

# --- Helper Function for Drawing (Used for Single Image Test) ---
def _plot_boxes(image: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Draws bounding boxes and scores onto the image."""
    out = image.copy()
    for f in faces:
        # Detections format: [x, y, w, h, ..., score] (N, 15)
        x, y, w, h = f[:4].astype(int)
        score = f[14]
        
        # Draw green rectangle
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare score text
        text = f"{score:.4f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Put score above the box
        cv2.putText(out, text, (x, max(0, y - 5)), 
                    font, font_scale, (0, 255, 0), font_thickness)
    return out


def main():
    # --- Params ---
    params = {
        "score_threshold": 0.5,
        "nms_threshold": 0.4,
        "top_k": 5000,
        "divisor": 32,
        "draw_pred": True,
        "save_pred_image": True,
        "save_txt": False,
        "save_json": False,
    }

    # ---- Configure paths ----
    SAVE_PATH = os.path.join("results", "test_detector")
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Define output path for the single image test
    SINGLE_TEST_OUTPUT = os.path.join(SAVE_PATH, "single_image_test_output.jpg")

    # --- Choose detector ---
    detector = SCRFDDetector(params=params)

    # --- Load model ---
    model_name = "scrfd10gkps.onnx"
    model_path = os.path.join(os.path.dirname(__file__), "../..", "models/face_detection", model_name)
    model_path = os.path.abspath(model_path)
    print("Using model:", model_path)
    detector.load_model(model_path)

    # --- Load sample data ---
    data_path = os.path.abspath("meen.jpg")
    assert os.path.exists(data_path), f"Image {data_path} not found"
    img = cv2.imread(data_path)
    assert img is not None, "Image failed to load"
    print("Loaded test image:", img.shape)

    # --- Run BATCH predict (This already handles drawing/saving via BaseDetector) ---
    print("\n--- Running Batch Prediction (Saves to results/test_detector/inference/images) ---")
    detections_batch = detector.predict(data_path, SAVE_PATH)

    print("Inference metrics:", detector.get_inference_metrics())
    print("Number of images processed:", len(detections_batch))

    # --- Example SINGLE image test: Detect, Draw, and Save ---
    print("\n--- Running Single Image Test: Detect, Draw, and Save ---")
    faces = detector.detect(img)
    print("Detections for one image:", faces.shape)
    for f in faces:
        print("Box:", f[:4], "Score:", f[14])
        
    # Draw the results onto the original image
    vis_img = _plot_boxes(img, faces)
    
    # Save the visualized image
    cv2.imwrite(SINGLE_TEST_OUTPUT, vis_img)
    print(f"Single test result saved to: {SINGLE_TEST_OUTPUT}")


if __name__ == "__main__":
    main()
