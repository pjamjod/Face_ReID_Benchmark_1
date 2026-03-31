# mlops/detectors/base.py
"""
BaseDetector
------------
Abstract base class for detectors.
Provides shared pipeline: data loading, inference loop, saving outputs, metrics.
Concrete detectors only implement: load_model(), prepare_input(), process_output().
"""

from abc import ABC, abstractmethod
import os
import os.path as osp
import time
import json
import numpy as np
import cv2

from inference.utils.utils import clear_dir
from inference.utils.timer import timeit

class BaseDetector(ABC):

    def __init__(self, params=None):
        self.params = params or {}
        self.session = None
        self.input_details = {}
        self.output_details = {}
        self.data = []
        self.inference_time = []
        self.gpu_metrics = None

        try:
            from inference.utils.gpu_monitor import GPUMonitor
            self.gpu_monitor = GPUMonitor()
        except ImportError:
            self.gpu_monitor = None

        self.run_path = "runs"
        self.inference_path = "inference"

    # ------------------------------------------------------------------
    # Mandatory overrides for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def load_model(self, model_path: str):
        """Load detector model from path and set self.session."""
        raise NotImplementedError

    @abstractmethod
    def prepare_input(self, image: np.ndarray):
        """Convert raw BGR image into model-ready tensor."""
        raise NotImplementedError

    @abstractmethod
    def process_output(self, outputs):
        """Decode raw model outputs to (N, 15)."""
        raise NotImplementedError

    def inference(self, input_tensor: np.ndarray):
        """Default ONNXRuntime inference step."""
        inputs = {self.input_details["input_names"][0]: input_tensor}
        return self.session.run(self.output_details["output_names"], inputs)
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Run one-image detection using subclass hooks, recording phase times."""
        import time
        
        # 1. Preprocess
        t0 = time.perf_counter()
        blob = self.prepare_input(image)
        
        # 2. Inference
        t1 = time.perf_counter()
        outputs = self.inference(blob)
        
        # 3. Postprocess
        t2 = time.perf_counter()
        faces = self.process_output(outputs)
        t3 = time.perf_counter()

        # Save timings (in milliseconds) for the benchmark script to collect
        self.last_timings = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
            "total_ms": (t3 - t0) * 1000.0
        }
        
        return faces
    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def load_params(self, params: dict):
        self.params.update(params)

    def preprocess_data(self, data_path):
        """Expand input path(s) into a list of image files."""
        if isinstance(data_path, list):
            files = data_path
        elif osp.isdir(data_path):
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
            files = [
                osp.join(data_path, f)
                for f in os.listdir(data_path)
                if f.lower().endswith(exts)
            ]
        elif osp.isfile(data_path):
            files = [data_path]
        else:
            raise ValueError(f"Unsupported data_path: {data_path}")
        self.data = sorted(files)

    def postprocess_data(self, save_dir):
        """Hook for subclasses; default does nothing."""
        pass

    def clear(self):
        """Reset state after prediction."""
        self.data = []
        self.inference_time = []

    def warmup_inference(self, runs=5, default_hw=(640, 640)):
        """Generic warmup using first model input shape."""
        if self.session is None:
            return
        info = self.session.get_inputs()[0]
        name = info.name
        shape = [s if isinstance(s, int) and s > 0 else 1 for s in info.shape]
        # if dynamic H/W: replace with default_hw
        if len(shape) >= 4:
            shape[-2:] = default_hw
        dummy = np.random.rand(*shape).astype(np.float32)
        for _ in range(runs):
            self.session.run([o.name for o in self.session.get_outputs()], {name: dummy})

    def get_inference_metrics(self):
        if len(self.inference_time) == 0:
            return {}
        arr = np.array(self.inference_time)
        return {
            "avg_ms": float(np.mean(arr) * 1000),
            "min_ms": float(np.min(arr) * 1000),
            "max_ms": float(np.max(arr) * 1000),
            "fps": float(1.0 / np.mean(arr)),
        }

    # ------------------------------------------------------------------
    # High-level batch prediction pipeline
    # ------------------------------------------------------------------
    def predict(self, data_path, save_path):
        """
        Full inference loop:
          - preprocess_data()
          - warmup
          - run detect() on each file
          - save optional outputs (images/json/txt)
        """
        self.save_path = save_path
        clear_dir(self.save_path)
        clear_dir(self.run_path)

        self.preprocess_data(data_path)

        pred_frames_path = osp.join(self.run_path, self.inference_path)
        pred_frames_images_path = osp.join(pred_frames_path, "images")
        pred_frames_labels_path = osp.join(pred_frames_path, "labels")
        clear_dir(pred_frames_path)
        clear_dir(pred_frames_images_path)
        clear_dir(pred_frames_labels_path)

        try:
            self.warmup_inference()
        except Exception as e:
            print("[WARN] Warmup skipped:", e)

        if self.gpu_monitor:
            try:
                self.gpu_monitor.start()
            except Exception as e:
                print("[WARN] GPU monitor disabled:", e)

        all_faces = []
        json_records = []

        for data_file in self.data:
            fname, _ = os.path.splitext(data_file)
            image = cv2.imread(data_file)
            if image is None:
                raise ValueError(f"Could not read image {data_file}")

            start = time.perf_counter()
            faces = self.detect(image)
            self.inference_time.append(time.perf_counter() - start)
            all_faces.append(faces)

            if self.params.get("draw_pred") or self.params.get("save_pred_image"):
                vis = self._plot_boxes(image, faces)
                if self.params.get("save_pred_image"):
                    cv2.imwrite(
                        osp.join(pred_frames_images_path, osp.basename(fname) + ".png"),
                        vis,
                    )

            if self.params.get("save_txt"):
                txt_path = osp.join(pred_frames_labels_path, osp.basename(fname) + ".txt")
                with open(txt_path, "w") as f:
                    for det in faces:
                        xywh = det[0:4].tolist()
                        landmarks = det[4:14].tolist()
                        score = float(det[14])
                        f.write(" ".join([f"{v:.6f}" for v in xywh + [score] + landmarks]) + "\n")

            if self.params.get("save_json"):
                rec = {
                    "image": osp.basename(data_file),
                    "detections": [
                        {
                            "bbox_xywh": det[0:4].tolist(),
                            "score": float(det[14]),
                            "landmarks": [
                                {"x": float(det[4 + 2 * i]), "y": float(det[5 + 2 * i])}
                                for i in range(5)
                            ],
                        }
                        for det in faces
                    ],
                }
                json_records.append(rec)

        if self.params.get("save_json") and json_records:
            with open(osp.join(pred_frames_path, "detections.json"), "w") as jf:
                json.dump(json_records, jf, indent=2)

        self.postprocess_data(pred_frames_images_path)
        self.clear()

        if self.gpu_monitor:
            try:
                self.gpu_monitor.stop()
                self.gpu_metrics = self.gpu_monitor.get_data()
            except Exception:
                pass

        return all_faces

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _plot_boxes(self, image, faces, padding_info=None):
        out = image.copy()
        for f in faces:
            x, y, w, h = f[:4].astype(int)
            score = f[14]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(out, f"{score:.4f}", (x, max(0, y - 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return out
