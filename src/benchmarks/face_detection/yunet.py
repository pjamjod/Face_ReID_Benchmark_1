# mlops/detectors/yunet.py
"""
YuNetDetector
-------------
Minimal YuNet ONNX face detector built on the refactored BaseDetector.
Only YuNet-specific parts remain: load_model, prepare_input, process_output, NMS.
"""

import numpy as np
import cv2
import onnxruntime as ort
import os.path as osp

from inference.onnx_inference import ONNXInference
from face_detection.base_detector import BaseDetector


class YuNetDetector( BaseDetector, ONNXInference):
    def __init__(self, params=None):
        super().__init__(params)
        self._padding_info = (0, 0, 1.0)
        self._inference_size = (0, 0)
        self.dynamic_input = bool(self.params.get("dynamic_input", False))

    # ----------------------------
    # Model loading
    # ----------------------------
    def load_model(self, model_path, device_id=None, execution_provider="cuda"):
        ort.set_default_logger_severity(3)
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        
        available = ort.get_available_providers()
        providers = ['CPUExecutionProvider']
        provider_key = (execution_provider or "cuda").lower()
        
        if provider_key == "dml":
            if 'DmlExecutionProvider' not in available:
                raise RuntimeError("DmlExecutionProvider is not available.")
            providers.insert(0, ('DmlExecutionProvider', {'device_id': int(device_id or 0)}))
            
        elif provider_key == "cuda":
            if 'CUDAExecutionProvider' not in available and device_id is not None:
                raise RuntimeError("CUDAExecutionProvider is not available.")
            
            if 'CUDAExecutionProvider' in available:

                cuda_options = {
                    'device_id': int(device_id) if device_id is not None else 0
                }
                providers.insert(0, ('CUDAExecutionProvider', cuda_options))

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_details["input_names"] = [i.name for i in model_inputs]
        self.input_details["input_shape"] = [list(i.shape) for i in model_inputs]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_details["output_names"] = [o.name for o in model_outputs]
        self.output_details["output_shape"] = [list(o.shape) for o in model_outputs]

    # ----------------------------
    # Pre/Post processing
    # ----------------------------
    def _blob(self, image, scalefactor=1.0, mean=(0, 0, 0), swapRB=False):
        image = np.array(image).astype(np.float32)
        if swapRB:
            image = image[:, :, [2, 1, 0]]
        image -= np.array(mean, dtype=np.float32)
        image *= scalefactor
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(image, axis=0)

    def prepare_input(self, image: np.ndarray):
        """Prepare input for either dynamic padded size or fixed letterbox size."""
        self.orig_h, self.orig_w = image.shape[:2]

        if self.dynamic_input:
            # Dynamic mode: keep native scale, only pad to divisor.
            h, w = image.shape[:2]
            divisor = int(self.params.get("divisor", 32))
            new_w = ((w - 1) // divisor + 1) * divisor
            new_h = ((h - 1) // divisor + 1) * divisor
            bottom_pad = new_h - h
            right_pad = new_w - w
            padded = cv2.copyMakeBorder(
                image, 0, bottom_pad, 0, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            blob = self._blob(padded)
            # Scale is 1.0 because we didn't resize, only padded
            self._padding_info = (bottom_pad, right_pad, 1.0)
            self._inference_size = (new_h, new_w)
            return blob

        # Fixed mode: letterbox to configured input size (e.g., 640x640).
        input_size = self.params.get("input_size", (640, 640))
        target_w, target_h = (input_size, input_size) if isinstance(input_size, int) else input_size

        scale = min(target_w / self.orig_w, target_h / self.orig_h)
        new_w = int(self.orig_w * scale)
        new_h = int(self.orig_h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        bottom_pad = target_h - new_h
        right_pad = target_w - new_w
        padded = cv2.copyMakeBorder(
            resized_image, 0, bottom_pad, 0, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        blob = self._blob(padded)
        self._padding_info = (bottom_pad, right_pad, scale)
        self._inference_size = (target_h, target_w)
        return blob


    def process_output(self, merged_output: list):
        """Decode YuNet outputs into [N,15] and map back to original image size."""
        H, W = self._inference_size
        _, _, scale = self._padding_info 
        
        score_thresh = float(self.params.get("score_threshold", 0.5))
        nms_thresh = float(self.params.get("nms_threshold", 0.45))
        topK = int(self.params.get("top_k", 5000))

        output = merged_output[0]
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
            
        # Dynamically check columns to support your 16-col model safely
        num_cols = output.shape[-1] if output.ndim == 2 else 16
        output = output.reshape(-1, num_cols)

        faces = []
        for det in output:
            # Your original model uses index 0 for the score
            score = det[0] if num_cols == 16 else det[14]
            
            if score < score_thresh:
                continue
                
            face = np.zeros((1, 15), dtype=np.float32)
            
            if num_cols == 16:
                # --- YOUR ORIGINAL 16-COLUMN LOGIC ---
                # 1. Un-normalize from inference size
                x1 = det[2] * W
                y1 = det[3] * H
                x2 = det[4] * W
                y2 = det[5] * H
                
                # 2. Reverse the scale to get original image coordinates
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale
                w, h = x2 - x1, y2 - y1

                face[0, 0:4] = [x1, y1, w, h]
                
                for n in range(5):
                    # Un-normalize and reverse scale in one step
                    face[0, 4 + 2 * n] = (det[6 + 2 * n] * W) / scale
                    face[0, 4 + 2 * n + 1] = (det[6 + 2 * n + 1] * H) / scale
                    
                face[0, 14] = score
                faces.append(face)
                
            elif num_cols == 15:
                # --- FALLBACK 15-COLUMN LOGIC (Just in case) ---
                x, y, w, h = det[0:4] / scale
                face[0, 0:4] = [x, y, w, h]
                
                for n in range(5):
                    face[0, 4 + 2 * n] = det[4 + 2 * n] / scale
                    face[0, 4 + 2 * n + 1] = det[5 + 2 * n] / scale
                    
                face[0, 14] = score
                faces.append(face)

        if not faces:
            return np.zeros((0, 15), dtype=np.float32)
            
        faces = np.vstack(faces)

        if faces.shape[0] > 1:
            boxes = [(f[0], f[1], f[0] + f[2], f[1] + f[3]) for f in faces]
            scores = [f[14] for f in faces]
            indices = self._nms(boxes, scores, nms_thresh, top_k=topK)
            faces = np.array([faces[i] for i in indices], dtype=np.float32)
            
        return faces
    # ----------------------------
    # NMS helper
    # ----------------------------
    @staticmethod
    def _nms(boxes, scores, nms_threshold, top_k):
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = np.argsort(scores)[::-1]
        keep = []
        while len(indices) > 0 and len(keep) < top_k:
            current = indices[0]
            keep.append(current)
            if len(indices) == 1:
                break
            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]
            x1 = np.maximum(current_box[0], other_boxes[:, 0])
            y1 = np.maximum(current_box[1], other_boxes[:, 1])
            x2 = np.minimum(current_box[2], other_boxes[:, 2])
            y2 = np.minimum(current_box[3], other_boxes[:, 3])
            inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            box_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (
                other_boxes[:, 3] - other_boxes[:, 1]
            )
            current_area = (current_box[2] - current_box[0]) * (
                current_box[3] - current_box[1]
            )
            union_area = current_area + box_area - inter_area
            iou = inter_area / np.maximum(union_area, 1e-6)
            indices = indices[1:][iou <= nms_threshold]
        return keep
