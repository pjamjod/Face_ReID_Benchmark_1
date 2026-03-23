# mlops/core/onnx_inference.py
"""
ONNXInference
-------------
Extends GenericInference with ONNX Runtime support.
Handles:
 - model loading
 - input/output introspection
 - warmup inference
 - actual inference calls

Subclasses must implement:
 - get_input_details()
 - get_output_details()
 - predict()
and usually also provide preprocess / postprocess methods.
"""
import os.path as osp
import numpy as np
import onnxruntime
from abc import abstractmethod

from .generic_inference import GenericInference


class ONNXInference(GenericInference):
    def __init__(self, providers=None):
        super().__init__()
        self.session: onnxruntime.InferenceSession = None
        self.input_details = {}
        self.output_details = {}
        self.providers = providers or [("DmlExecutionProvider",{'device_id': 0}),"CPUExecutionProvider"]

    # -------------------------
    # Model loading
    # -------------------------
    def load_model(self, model_path: str):
        assert osp.exists(model_path), f"Model {model_path} does not exist"
        self.model_path = model_path
        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        self.get_input_details()
        self.get_output_details()

    # -------------------------
    # Warmup
    # -------------------------

    def warmup_inference(self, runs: int = 5):
        """Warm up ONNX Runtime with a correctly sized dummy input."""
        if self.session is None:
            raise RuntimeError("Model not loaded before warmup")

        # --- Detect model input shape automatically ---
        input_info = self.session.get_inputs()[0]
        name = input_info.name
        shape = list(input_info.shape)  # e.g. [1, 3, None, None]

        # If there are dynamic dims, replace them with self.input_size if you have it
        # or fall back to something reasonable (640x640)
        for idx, s in enumerate(shape):
            if s is None or isinstance(s, str):
                # For batch dimension just use 1
                if idx == 0:
                    shape[idx] = 1
                # For H and W, use your detector's configured input_size if available
                elif hasattr(self, "input_size"):
                    # Many codebases store self.input_size as (h, w)
                    h, w = self.input_size
                    # This idx points to height or width: use them appropriately
                    if idx == 2:
                        shape[idx] = h
                    elif idx == 3:
                        shape[idx] = w
                else:
                    # Fallback to 640
                    shape[idx] = 640

        print(f"Warmup using shape {shape}")

        dummy = np.random.rand(*shape).astype(np.float32)

        for _ in range(runs):
            self.session.run(
                [o.name for o in self.session.get_outputs()],
                {name: dummy}
            )


    # -------------------------
    # Inference call
    # -------------------------
    def infer(self, inputs: dict):#execute or run
        """Run inference with a mapping name -> np.ndarray."""
        if self.session is None:
            raise RuntimeError("Model not loaded before inference")
        return self.session.run(self.output_details["output_names"], inputs)

    # -------------------------
    # Abstract hooks
    # -------------------------
    @abstractmethod
    def get_input_details(self):
        """Populate self.input_details dict with keys: input_names, input_shape, etc."""
        raise NotImplementedError

    @abstractmethod
    def get_output_details(self):
        """Populate self.output_details dict with keys: output_names, etc."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_path: str, save_path: str) -> list:
        """Run inference pipeline and return results (format defined by subclass)."""
        raise NotImplementedError

