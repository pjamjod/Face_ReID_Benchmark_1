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
        # Use an explicit ONNX Runtime execution provider when requested.
        # 3 = ERROR (quiet); 0 = VERBOSE
        ort.set_default_logger_severity(3)
        # On Windows, importing torch first helps ORT find CUDA/cuDNN DLLs.
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        available = ort.get_available_providers()
        providers = ['CPUExecutionProvider']
        provider_key = (execution_provider or "cuda").lower()

        if provider_key == "dml":
            if 'DmlExecutionProvider' not in available:
                raise RuntimeError(
                    "DirectML device was selected, but ONNX Runtime DmlExecutionProvider is not available. "
                    f"Available providers: {available}. "
                    "Install the DirectML-enabled ONNX Runtime build and ensure the adapter supports DirectML."
                )
            provider_options = {'device_id': int(device_id or 0)}
            providers.insert(0, ('DmlExecutionProvider', provider_options))
        elif device_id is not None:
            if 'CUDAExecutionProvider' not in available:
                raise RuntimeError(
                    "CUDA device was selected, but ONNX Runtime CUDAExecutionProvider is not available. "
                    f"Available providers: {available}. "
                    "Install ONNX Runtime GPU build (onnxruntime-gpu) and ensure CUDA/cuDNN match the wheel."
                )
            providers.insert(0, ('CUDAExecutionProvider', {'device_id': int(device_id)}))
        elif 'CUDAExecutionProvider' in available:
            providers.insert(0, 'CUDAExecutionProvider')
        elif 'DmlExecutionProvider' in available and provider_key == "dml":
            providers.insert(0, ('DmlExecutionProvider', {'device_id': 0}))
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        active = self.session.get_providers()
        expected_provider = 'DmlExecutionProvider' if provider_key == "dml" else 'CUDAExecutionProvider'
        if expected_provider not in active and device_id is not None:
            raise RuntimeError(
                f"Requested {expected_provider}, but ONNX Runtime session did not activate it. "
                f"Active providers: {active}."
            )
        print(f"ONNX Model loaded with providers: {self.session.get_providers()}")
        
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_details["input_names"] = [i.name for i in model_inputs]
        self.input_details["input_shape"] = [list(i.shape) for i in model_inputs]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_details["output_names"] = [o.name for o in model_outputs]

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
                image,
                0, bottom_pad, 0, right_pad,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            blob = self._blob(padded)
            self._padding_info = (bottom_pad, right_pad, 1.0)
            self._inference_size = (new_h, new_w)
            return blob

        # Fixed mode: letterbox to configured input size (default 640x640).
        input_size = self.params.get("input_size", (640, 640))
        if isinstance(input_size, int):
            target_w = target_h = int(input_size)
        else:
            target_w, target_h = int(input_size[0]), int(input_size[1])

        scale = min(target_w / self.orig_w, target_h / self.orig_h)
        new_w = int(self.orig_w * scale)
        new_h = int(self.orig_h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        bottom_pad = target_h - new_h
        right_pad = target_w - new_w
        padded = cv2.copyMakeBorder(
            resized_image,
            0, bottom_pad, 0, right_pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        blob = self._blob(padded)
        self._padding_info = (bottom_pad, right_pad, scale)
        self._inference_size = (target_h, target_w)
        return blob

    def process_output(self, merged_output: list):
        """Decode YuNet outputs into [N,15] and map back to original image size."""
        H, W = self._inference_size # This is now 640, 640
        
        # 1. Unpack the scale we saved in prepare_input
        _, _, scale = self._padding_info 
        
        score_thresh = float(self.params.get("score_threshold", 0.5))
        nms_thresh = float(self.params.get("nms_threshold", 0.4))
        topK = int(self.params.get("top_k", 5000))

        output = merged_output[0]
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        output = output.reshape(-1, 16)

        faces = []
        for det in output:
            score = det[0]
            if score < score_thresh:
                continue
                
            # 2. Calculate coordinates in the 640x640 space
            x1, y1, x2, y2 = det[2:6] * np.array([W, H, W, H], dtype=np.float32)
            
            # 3. REVERSE THE SCALE! (Map back to original image dimensions)
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale
            
            w = x2 - x1
            h = y2 - y1
            
            face = np.zeros((1, 15), dtype=np.float32)
            face[0, 0:4] = [x1, y1, w, h]
            
            # 4. Reverse the scale for all 5 facial landmarks too
            for n in range(5):
                lm_x = det[6 + 2 * n] * W
                lm_y = det[6 + 2 * n + 1] * H
                face[0, 4 + 2 * n] = lm_x / scale
                face[0, 4 + 2 * n + 1] = lm_y / scale
                
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

    # dynamic-only alternate implementation removed; logic is now unified in prepare_input
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
