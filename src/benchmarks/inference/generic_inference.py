# mlops/core/generic_inference.py
"""
GenericInference
----------------
A clean, reusable inference scaffold that:
 - handles dataset discovery (image / folder / video)
 - manages run/save paths and basic postprocessing (move outputs, create video)
 - provides visualization helpers and metrics
 - exposes abstract hooks `load_model` and `predict` for concrete implementations

Intended to be lightweight and framework-agnostic. Keep ONNX/torch specifics in subclasses.
"""
from abc import ABC, abstractmethod
import os
import os.path as osp
import shutil
import json
import numpy as np
import cv2
from typing import Tuple, Union

# NOTE: adapt these imports to your package structure. They assume you have utilities in mlops.utils.*
from inference.utils.utils import clear_dir, extract_video_frames, natural_keys, update_video_frames, video_from_frames
from inference.utils.params import Params
from inference.utils.gpu_monitor import GpuMonitor


class GenericInference(ABC):
    def __init__(self):
        # model & data
        self.model_path: str = None
        self.data: list = None
        self.data_path: str = None
        self.data_type: str = None  # 'image' | 'folder' | 'video'

        # params container (user can pass dict or Params instance)
        self.params = Params()

        # output/run paths (subclasses may override)
        self.save_path = None
        self.run_path = "runs"
        self.inference_path = "inference"
        self.txt_path = "labels"
        self.json_path = "detections.json"

        # file extension helpers
        self.video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        # visualization palette (BGR)
        self.color_pallete = [
            (242, 113, 51),
            (152, 253, 0),
            (106, 23, 255),
            (21, 249, 249),
            (249, 21, 211),
            (43, 0, 255),
        ]

        # metrics & monitoring
        self.inference_time = []
        self.gpu_metrics = {}
        self.gpu_monitor = GpuMonitor()

        # optional mapping for class ids -> text
        self.classes_dict = None

        # file where inference metrics are saved
        self.inference_metrics_file = "inference_metrics.json"

    # -------------------------
    # Abstract methods
    # -------------------------
    @abstractmethod
    def load_model(self, model_path: str):
        """Load model from disk. Concrete classes must implement."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_path: str, save_path: str) -> list:
        """Run inference on data_path and save results to save_path. Return per-sample results."""
        raise NotImplementedError

    # -------------------------
    # Params & data helpers
    # -------------------------
    def load_params(self, params: Union[dict, Params]):
        """Load parameters from a dict or Params object (shallow assign)."""
        if isinstance(params, Params):
            self.params = params
        elif isinstance(params, dict):
            for k, v in params.items():
                setattr(self.params, k, v)
        else:
            raise TypeError("params must be a dict or Params instance")

    
    def preprocess_data(self, data_path: str):
        """Populate self.data with a sorted list of frame/image paths."""
        self.data_path = data_path
        if osp.isdir(data_path):
            print("Inference data is a folder")
            data = [osp.join(data_path, f) for f in os.listdir(data_path)]
            data.sort(key=natural_keys)
            self.data_type = "folder"
        else:
            _, ext = osp.splitext(data_path)
            ext = ext.lower()
            if ext in self.video_extensions:
                print("Inference data is a video")
                frames_path = osp.join(self.run_path, "frames")
                clear_dir(frames_path)
                extract_video_frames(data_path, frames_path)
                data = [osp.join(frames_path, f) for f in os.listdir(frames_path)]
                data.sort(key=natural_keys)
                self.data_type = "video"
            elif ext in self.image_extensions:
                print("Inference data is an image")
                data = [data_path]
                self.data_type = "image"
            else:
                raise ValueError(
                    f"Data should be a video, image, or directory of images. "
                    f"Accepted video extensions: {self.video_extensions} "
                    f"and image extensions: {self.image_extensions}"
                )
        self.data = data

    def postprocess_data(self, frames_path: str):
        """
        Move/transform run outputs into final save_path. Handles:
         - video assembly if params.video_out
         - moving predictions & labels to save_path
        """
        if self.params.video_out:
            if self.data_type == "video":
                update_video_frames(self.data_path, frames_path, self.save_path)
            elif self.data_type == "folder":
                video_name = "pred_" + osp.basename(self.data_path) + ".mp4"
                video_from_frames(frames_path, self.save_path, self.params.fps, video_name=video_name)

        # move results from run_path to save_path if requested
        if self.run_path != self.save_path:
            if self.params.save_pred:
                try:
                    shutil.move(frames_path, self.save_path)
                except Exception:
                    pass
            if self.params.save_txt:
                try:
                    shutil.move(osp.join(self.run_path, self.inference_path, self.txt_path), self.save_path)
                except Exception:
                    pass
            if self.params.save_json:
                try:
                    # json_path is a filename under inference_path
                    src = osp.join(self.run_path, self.inference_path, self.json_path)
                    shutil.move(src, self.save_path)
                except Exception:
                    pass

    # -------------------------
    # Visualization helpers
    # -------------------------
    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Return consistent BGR color for class_id (extend palette deterministicly if needed)."""
        if class_id < 0:
            class_id = 0
        if len(self.color_pallete) <= class_id:
            rng = np.random.default_rng(3)
            extra = [(int(x[0]), int(x[1]), int(x[2])) for x in rng.uniform(0, 255, size=(class_id - len(self.color_pallete) + 1, 3))]
            self.color_pallete.extend(extra)
        return self.color_pallete[class_id]

    def plot_prediction(self, image: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Draw boxes on image. Expects `predictions` rows as:
          [x_center, y_center, w, h, class_id, conf]
        Generic and used by detectors that produce COCO-like xywh results.
        """
        if predictions is None or len(predictions) == 0:
            return image
        out = image.copy()
        for pred in predictions:
            # safe unpack (some detectors might add extra fields; we only use the first 6)
            x, y, w, h = pred[0], pred[1], pred[2], pred[3]
            class_id = int(pred[4]) if len(pred) > 4 else 0
            conf = float(pred[5]) if len(pred) > 5 else 1.0

            x1 = int(x - w // 2)
            y1 = int(y - h // 2)
            x2 = int(x + w // 2)
            y2 = int(y + h // 2)

            color = self.get_color(class_id)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.params.box_thickness)
            label = ""
            if self.params.draw_cls:
                if self.classes_dict and class_id in self.classes_dict:
                    label = self.classes_dict[class_id]
                else:
                    label = str(class_id)
            if self.params.draw_conf:
                label = (label + " - " if label else "") + f"{conf*100:.2f}%"
            if label:
                cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return out

    # -------------------------
    # Metrics
    # -------------------------
    def construct_inference_metrics(self) -> dict:
        metrics = {"inference_time": float(np.mean(self.inference_time)) if len(self.inference_time) else 0.0}
        if isinstance(self.gpu_metrics, dict):
            metrics.update(self.gpu_metrics)
        return metrics

    def save_inference_metrics(self, save_path: str):
        metrics = self.construct_inference_metrics()
        try:
            with open(osp.join(save_path, self.inference_metrics_file), "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print("Warning: failed to save inference metrics:", e)

    def get_inference_metrics(self) -> dict:
        return self.construct_inference_metrics()

    # -------------------------
    # Cleanup
    # -------------------------
    def clear(self):
        """Remove run_path unless it's the same as save_path (safety)."""
        if not self.save_path or not self.run_path:
            return
        try:
            if not osp.abspath(self.save_path).startswith(osp.abspath(self.run_path)):
                shutil.rmtree(self.run_path)
        except Exception:
            pass
