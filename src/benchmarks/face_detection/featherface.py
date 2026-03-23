import numpy as np
import torch
import cv2
import onnxruntime as ort

# Import your legacy PyTorch utilities
from face_detection.featherface_arch.utils.box_utils import decode, decode_landm
from face_detection.featherface_arch.utils.nms.py_cpu_nms import py_cpu_nms
from face_detection.featherface_arch.layers.functions.prior_box import PriorBox
from face_detection.base_detector import BaseDetector

class RetinaFaceDetector(BaseDetector):
    def __init__(self, params=None):
        super().__init__(params)
        self.mean = (104, 117, 123)
        self.variance = [0.1, 0.2]
        
        # Configuration matches cfg_mnet
        self.cfg = {
            'name': 'mobile0.25',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
        }
        self.anchor_cache = {}
        
        # New Params for Dynamic Sizing
        self.dynamic_input = self.params.get("dynamic_input", False)
        self.origin_size = self.params.get("origin_size", True)
        input_size = self.params.get("input_size", (640, 640))
        if isinstance(input_size, int):
            self.input_size = (int(input_size), int(input_size))
        else:
            self.input_size = (int(input_size[0]), int(input_size[1]))

    def load_model(self, model_path: str, device_id=None, execution_provider="cuda"):
        # Initialize ONNX session with an explicit execution provider when requested.
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
        
        self.input_details["input_names"] = [i.name for i in self.session.get_inputs()]
        self.output_details["output_names"] = [o.name for o in self.session.get_outputs()]

        # Detect model input metadata (used only when input mode is not explicitly forced)
        input_shape = self.session.get_inputs()[0].shape
        model_is_fixed = isinstance(input_shape[2], int) and isinstance(input_shape[3], int)

        # Respect explicit user config from params first.
        if "dynamic_input" not in self.params:
            # No explicit mode requested: follow model metadata.
            if model_is_fixed:
                self.dynamic_input = False
                self.input_size = (int(input_shape[2]), int(input_shape[3]))
            else:
                self.dynamic_input = True
        elif not self.dynamic_input and model_is_fixed and "input_size" not in self.params:
            # Explicit fixed mode but no input_size provided: use model fixed size.
            self.input_size = (int(input_shape[2]), int(input_shape[3]))

    def prepare_input(self, image: np.ndarray):
        """Handles both dynamic PyTorch-style scaling and fixed Letterbox padding."""
        h_orig, w_orig = image.shape[:2]

        if self.dynamic_input:
            # --- DYNAMIC INPUT MODE (Original PyTorch Logic) ---
            target_size = 1600
            max_size = 2150
            im_size_min = min(h_orig, w_orig)
            im_size_max = max(h_orig, w_orig)
            
            scale = float(target_size) / float(im_size_min)
            if np.round(scale * im_size_max) > max_size:
                scale = float(max_size) / float(im_size_max)
                
            if self.origin_size:
                scale = 1.0

            if scale != 1.0:
                resized_img = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            else:
                resized_img = image.copy()
            
            det_img = np.float32(resized_img)
            self._infer_h, self._infer_w = det_img.shape[:2]
            self._resize_scale = scale
            
        else:
            # --- FIXED INPUT MODE (Letterbox Padding) ---
            target_h, target_w = self.input_size
            scale = min(target_w / w_orig, target_h / h_orig)
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            
            resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            det_img = np.zeros((target_h, target_w, 3), dtype=np.float32)
            det_img[:new_h, :new_w, :] = resized_img
            
            self._infer_h, self._infer_w = target_h, target_w
            self._resize_scale = scale

        # Exact normalization
        det_img -= self.mean
        blob = det_img.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        
        return np.ascontiguousarray(blob, dtype=np.float32)

    def process_output(self, outputs):
        loc, conf, landms = outputs
        
        # Use the actual dimensions the model processed this frame
        infer_h, infer_w = self._infer_h, self._infer_w
        
        loc_t = torch.from_numpy(loc[0])
        conf_t = torch.from_numpy(conf[0])
        landms_t = torch.from_numpy(landms[0])

        # Apply softmax if the ONNX model outputs raw logits
        # (Uncomment the line below if your Max Confidence is > 1.0 or extremely weird)
        # conf_t = torch.nn.functional.softmax(conf_t, dim=-1)

        # 1. PriorBox Generation
        cache_key = (infer_h, infer_w)
        if cache_key not in self.anchor_cache:
            priorbox = PriorBox(self.cfg, image_size=cache_key)
            self.anchor_cache[cache_key] = priorbox.forward()
        priors = self.anchor_cache[cache_key]

        # 2. Decode Bounding Boxes
        boxes = decode(loc_t, priors, self.variance)
        scale = torch.Tensor([infer_w, infer_h, infer_w, infer_h])
        boxes = boxes * scale / self._resize_scale
        boxes = boxes.cpu().numpy()

        # 3. Decode Landmarks
        ldms = decode_landm(landms_t, priors, self.variance)
        scale1 = torch.Tensor([
            infer_w, infer_h, infer_w, infer_h,
            infer_w, infer_h, infer_w, infer_h,
            infer_w, infer_h
        ])
        ldms = ldms * scale1 / self._resize_scale
        ldms = ldms.cpu().numpy()

        # 4. Confidence Scores
        scores = conf_t[:, 1].cpu().numpy()

        # 5. Ignore Low Scores
        conf_thresh = self.params.get("confidence_threshold", 0.5)
        inds = np.where(scores > conf_thresh)[0]
        
        if len(inds) == 0:
            return np.zeros((0, 15), dtype=np.float32)

        boxes = boxes[inds]
        ldms = ldms[inds]
        scores = scores[inds]

        # 6. Keep Top-K Before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        ldms = ldms[order]
        scores = scores[order]

        # 7. Do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        nms_thresh = self.params.get("nms_threshold", 0.4)
        keep = py_cpu_nms(dets, nms_thresh)

        dets = dets[keep, :]
        ldms = ldms[keep]

        # 8. Convert to BaseDetector Format: [x, y, w, h, kp1_x, kp1_y..., score]
        final_faces = np.zeros((dets.shape[0], 15), dtype=np.float32)
        final_faces[:, 0] = dets[:, 0]                     # x1
        final_faces[:, 1] = dets[:, 1]                     # y1
        final_faces[:, 2] = dets[:, 2] - dets[:, 0]        # w
        final_faces[:, 3] = dets[:, 3] - dets[:, 1]        # h
        final_faces[:, 4:14] = ldms                        # 5 landmarks (x, y)
        final_faces[:, 14] = dets[:, 4]                    # score

        return final_faces