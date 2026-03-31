# mlops/detectors/scrfd.py
"""
SCRFDDetector
-------------
SCRFD face detector built on the refactored BaseDetector.
Adapts the multi-level FPN decoding logic from nn.py.
"""

import numpy as np
import cv2
import onnxruntime as ort
import os.path as osp

from inference.onnx_inference import ONNXInference
from face_detection.base_detector import BaseDetector

class SCRFDDetector(BaseDetector, ONNXInference):
    def __init__(self, params=None):
        super().__init__(params)
        self.center_cache = {}
        self._feat_stride_fpn = [8, 16, 32]
        self.use_kps = False
        self._det_scale = 1.0

    def load_model(self, model_path, device_id=None, execution_provider="cuda"):
        # Build providers list with an explicit execution provider when selected.
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
        self._init_model_type()

    def _init_model_type(self):
        """Identify if the model uses keypoints or has specific anchor counts."""
        outputs = self.session.get_outputs()
        num_outputs = len(outputs)
        
        # Logic adapted from nn.py to determine model structure
        if num_outputs == 6:
            self.use_kps = False
        elif num_outputs == 9:
            self.use_kps = True
        elif num_outputs == 10:
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self.use_kps = False
        elif num_outputs == 15:
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self.use_kps = True

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_details["input_names"] = [i.name for i in model_inputs]
        self.input_details["input_shapes"] = [i.shape for i in model_inputs]
        self.input_name = model_inputs[0].name

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_details["output_names"] = [o.name for o in model_outputs]
        self.output_details["output_shapes"] = [o.shape for o in model_outputs]
    def prepare_input(self, image: np.ndarray):
        """Handle resizing and padding while maintaining aspect ratio (from nn.py)."""
        input_size = self.params.get("input_size", (640, 640))
        h_orig, w_orig = image.shape[:2]
        
        # Scaling logic from nn.py
        im_ratio = float(h_orig) / w_orig
        model_ratio = float(input_size[1]) / input_size[0]
        
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
            
        self._det_scale = float(new_height) / h_orig
        
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        
        # Blob preprocessing (from nn.py: 127.5 mean and 1/128 scale)
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        return blob

    def process_output(self, net_outs: list):
        """Decode multi-level FPN outputs."""
        score_thresh = float(self.params.get("score_threshold", 0.3))
        nms_thresh = float(self.params.get("nms_threshold", 0.4))
        
        scores_list = []
        bboxes_list = []
        kps_list = []
        
        input_height, input_width = self.params.get("input_size", (640, 640))
        fmc = len(self._feat_stride_fpn)

        for idx, stride in enumerate(self._feat_stride_fpn):
            # SCRFD outputs are grouped: [scores_f1, scores_f2..., boxes_f1, boxes_f2...]
            scores = net_outs[idx]
            if scores.ndim == 3: # Handle batch dimension if present
                scores = scores[0]
            
            boxes = net_outs[idx + fmc]
            if boxes.ndim == 3:
                boxes = boxes[0]
            boxes = boxes * stride
            
            # Anchor generation
            height, width = input_height // stride, input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                # Note: Some SCRFD models use 2 anchors per scale, 
                # but modern ones often use 1. Using nn.py logic:
                if scores.shape[0] // anchor_centers.shape[0] > 1:
                    num_anchors = scores.shape[0] // anchor_centers.shape[0]
                    anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
                
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_index = np.where(scores >= score_thresh)[0]
            
            # Decode Bounding Boxes
            x1 = anchor_centers[:, 0] - boxes[:, 0]
            y1 = anchor_centers[:, 1] - boxes[:, 1]
            x2 = anchor_centers[:, 0] + boxes[:, 2]
            y2 = anchor_centers[:, 1] + boxes[:, 3]
            bboxes = np.stack([x1, y1, x2, y2], axis=-1)
            
            scores_list.append(scores[pos_index])
            bboxes_list.append(bboxes[pos_index])
            
            if self.use_kps:
                kps_out = net_outs[idx + fmc * 2]
                if kps_out.ndim == 3: kps_out = kps_out[0]
                points = kps_out * stride
                
                # Decode Keypoints
                kpss = []
                for i in range(0, points.shape[1], 2):
                    px = anchor_centers[:, i % 2] + points[:, i]
                    py = anchor_centers[:, i % 2 + 1] + points[:, i + 1]
                    kpss.append(px)
                    kpss.append(py)
                kpss = np.stack(kpss, axis=-1).reshape((points.shape[0], -1, 2))
                kps_list.append(kpss[pos_index])

        if not scores_list or len(np.vstack(scores_list)) == 0:
            return np.zeros((0, 15), dtype=np.float32)

        # Combine all scales
        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list) / self._det_scale
        
        # NMS format: [x1, y1, x2, y2, score]
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        keep = self._nms(pre_det, nms_thresh)
        det = pre_det[keep, :]
        
        # Convert to BaseDetector (YuNet) format: [x, y, w, h, kp1_x, kp1_y... score]
        final_faces = np.zeros((det.shape[0], 15), dtype=np.float32)
        final_faces[:, 0] = det[:, 0] # x1
        final_faces[:, 1] = det[:, 1] # y1
        final_faces[:, 2] = det[:, 2] - det[:, 0] # w
        final_faces[:, 3] = det[:, 3] - det[:, 1] # h
        final_faces[:, 14] = det[:, 4] # score
        
        if self.use_kps and kps_list:
            kpss = np.vstack(kps_list) / self._det_scale
            kpss = kpss[keep]
            for i in range(5):
                final_faces[:, 4 + i*2] = kpss[:, i, 0]
                final_faces[:, 5 + i*2] = kpss[:, i, 1]
                
        return final_faces

    @staticmethod
    def _nms(dets, thresh):
        """Standard NMS implementation from nn.py."""
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= thresh)[0] + 1]
        return keep