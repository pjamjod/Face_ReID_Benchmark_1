import cv2
import tqdm
import numpy as np
import sys
import os
import time
import csv
import io
import contextlib
import json
import re
import shutil
from datetime import datetime
import onnxruntime as ort

from face_detection.scrfd import SCRFDDetector
from face_detection.yunet import YuNetDetector
from face_detection.featherface import RetinaFaceDetector

# 1. Get Project Root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# 2. Path to the WiderFace folder
widerface_path = os.path.join(project_root, "datasets", "widerface")

# 3. Add both to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if widerface_path not in sys.path:
    sys.path.insert(0, widerface_path)

from datasets.widerface.evaluation import evaluation

# ─── Shared base params ────────────────────────────────────────────────────────
_BASE = {"score_threshold": 0.5, "nms_threshold": 0.4, "divisor": 32}

# ─── Model registry ────────────────────────────────────────────────────────────
# label          : unique run name (used for output dirs / log filenames)
# cls            : detector class to instantiate
# model_file     : path relative to project root
# params         : full params dict passed to the detector
MODELS = [
    {
        "label":      "scrfd_640x640",
        "cls":        SCRFDDetector,
        "model_file": "models/face_detection/scrfd10gkps.onnx",
        "params":     {**_BASE, "dynamic_input": False, "input_size": (640, 640)},
    },
    {
        "label":      "yunet_dynamic",
        "cls":        YuNetDetector,
        "model_file": "models/face_detection/face_detection_yunet_2023mar_raven.onnx",
        "params":     {**_BASE, "dynamic_input": True, "origin_size": True},
    },
    {
        "label":      "yunet_640x640",
        "cls":        YuNetDetector,
        "model_file": "models/face_detection/face_detection_yunet_2023mar_raven.onnx",
        "params":     {**_BASE, "dynamic_input": False, "input_size": (640, 640)},
    },


]


WIDER_VAL_DIR = "datasets/widerface/WIDER_val/images"
GT_DIR        = "datasets/widerface/wider_face_eval_tools/eval_tools/ground_truth"
RESULTS_ROOT  = "src/benchmarks/results"
METRICS_JSON  = os.path.join(RESULTS_ROOT, "widerface_eval.json")
EVAL_TXT      = os.path.join(RESULTS_ROOT, "widerface_eval.txt")


def collect_runtime_info(device_id=None):
    """Collect runtime provider details for traceability."""
    available = ort.get_available_providers()
    return {
        "selected_device_id": device_id,
        "selected_device_name": f"DmlExecutionProvider(device_id={device_id})" if device_id is not None else None,
        "ort_available_providers": available,
        "directml_available": "DmlExecutionProvider" in available,
    }


def get_onnx_device_info(session):
    """Return ONNX Runtime provider-based device info for result metadata."""
    if session is None:
        return {
            "onnx_device_name": "unknown",
            "ort_providers": [],
            "ort_provider_options": {},
        }

    providers = session.get_providers()
    provider_options = session.get_provider_options()

    # Build a provider-derived device name without using torch device APIs.
    if "CUDAExecutionProvider" in providers:
        cuda_opts = provider_options.get("CUDAExecutionProvider", {})
        device_id = cuda_opts.get("device_id", "0")
        device_name = f"CUDAExecutionProvider(device_id={device_id})"
    elif "DmlExecutionProvider" in providers:
        dml_opts = provider_options.get("DmlExecutionProvider", {})
        device_id = dml_opts.get("device_id", "0")
        device_name = f"DmlExecutionProvider(device_id={device_id})"
    elif "CPUExecutionProvider" in providers:
        device_name = "CPUExecutionProvider"
    else:
        device_name = providers[0] if providers else "unknown"

    return {
        "onnx_device_name": device_name,
        "ort_providers": providers,
        "ort_provider_options": provider_options,
    }


def verify_onnx_runtime(device_id=None):
    """Verify ORT DirectML provider availability before the long benchmark loop."""
    available = ort.get_available_providers()

    print("\nONNX Runtime Info:")
    print(f"  Available Providers : {available}")

    if device_id is not None and "DmlExecutionProvider" not in available:
        raise RuntimeError(
            "DirectML device was selected, but ONNX Runtime DmlExecutionProvider is unavailable. "
            f"Available providers: {available}. "
            "Install the DirectML-enabled ONNX Runtime build and ensure the adapter supports DirectML."
        )


def select_device():
    """Interactive prompt to select a DirectML adapter id."""
    print("\n" + "=" * 40)
    print("DIRECTML CONFIGURATION")
    print("=" * 40)

    available = ort.get_available_providers()
    if "DmlExecutionProvider" not in available:
        raise RuntimeError(
            "DmlExecutionProvider is not available in this environment. "
            f"Available providers: {available}"
        )

    print("DirectML adapters are selected by numeric device id.")
    print("If you have a single GPU, use 0.")
    print("=" * 40)

    while True:
        try:
            choice = input("Select a DirectML device ID [default 0]: ").strip()
            device_id = 0 if choice == "" else int(choice)
            if device_id >= 0:
                print(f"\nSelected DirectML device ID: {device_id}")
                return device_id
            print("Invalid ID. Please enter 0 or a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")


def save_widerface_results(detector, image_path, save_path):
    """Runs detection on one image; saves WiderFace-format txt; returns inference ms."""
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, 0

    start_time = time.perf_counter()
    faces = detector.detect(img)
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(f"{os.path.basename(image_path)}\n")
        f.write(f"{len(faces)}\n")
        for face in faces:
            x, y, w, h = face[:4]
            score = face[14]
            f.write(f"{x} {y} {w} {h} {score}\n")

    return inference_time_ms, len(faces)


def parse_eval_output(eval_output):
    """Extract Easy/Medium/Hard AP values from WiderFace eval stdout."""
    scores = {}
    for line in eval_output.splitlines():
        m = re.search(r"(Easy|Medium|Hard)\s+Val\s+AP:\s*([0-9.]+)", line)
        if m:
            key = m.group(1).lower()
            scores[key] = float(m.group(2)) * 100.0
    return scores


def compute_timing_metrics(times_ms, det_counts):
    """Compute latency/FPS and detection-count summary metrics."""
    if not times_ms:
        return {
            "num_images": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "fps": 0.0,
            "mean_detections": 0.0,
        }

    arr_t = np.array(times_ms, dtype=np.float64)
    arr_d = np.array(det_counts, dtype=np.float64) if det_counts else np.array([0.0])
    return {
        "num_images": int(arr_t.size),
        "mean_ms": float(np.mean(arr_t)),
        "median_ms": float(np.median(arr_t)),
        "p95_ms": float(np.percentile(arr_t, 95)),
        "p99_ms": float(np.percentile(arr_t, 99)),
        "fps": float(1000.0 / np.mean(arr_t)),
        "mean_detections": float(np.mean(arr_d)),
    }


def run_single_model(model_cfg, device_id):
    """Run the full WiderFace-val benchmark for one model configuration."""
    label      = model_cfg["label"]
    DetClass   = model_cfg["cls"]
    model_path = os.path.abspath(model_cfg["model_file"])
    params     = model_cfg["params"]

    pred_dir   = os.path.join(RESULTS_ROOT, f"widerface_preds_{label}")
    log_path   = os.path.join(RESULTS_ROOT, f"inference_log_{label}.csv")
    # Ensure stale predictions from older/partial runs never contaminate eval.
    if os.path.isdir(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model : {label}")
    print(f"  File  : {model_path}")
    print(f"{'='*60}")

    # --- Load model ---
    detector = DetClass(params=params)
    detector.load_model(model_path, device_id=device_id, execution_provider="dml")
    detector.warmup_inference()
    onnx_runtime = get_onnx_device_info(detector.session)
    providers = onnx_runtime["ort_providers"]
    if providers and providers[0] == "DmlExecutionProvider":
        active_backend = "DirectML"
    elif providers and providers[0] == "CPUExecutionProvider":
        active_backend = "CPU"
    else:
        active_backend = providers[0] if providers else "unknown"
    print(f"  ORT Providers: {onnx_runtime['ort_providers']}")
    print(f"  ONNX Device  : {onnx_runtime['onnx_device_name']}")
    print(f"  Active EP    : {active_backend}")

    # --- Inference loop ---
    events = sorted(
        d for d in os.listdir(WIDER_VAL_DIR)
        if os.path.isdir(os.path.join(WIDER_VAL_DIR, d))
    )

    with open(log_path, mode="w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Event", "Image_Name", "Inference_Time_ms", "Num_Detections"])

        inf_times_ms = []
        det_counts = []

        for event in tqdm.tqdm(events, desc=f"{label}", unit="event"):
            event_path = os.path.join(WIDER_VAL_DIR, event)
            images = sorted(f for f in os.listdir(event_path) if f.endswith(".jpg"))

            for img_name in images:
                img_path  = os.path.join(event_path, img_name)
                save_path = os.path.join(pred_dir, event, img_name.replace(".jpg", ".txt"))

                inf_ms, n_dets = save_widerface_results(detector, img_path, save_path)

                if inf_ms > 0:
                    writer.writerow([event, img_name, round(inf_ms, 3), n_dets])
                    inf_times_ms.append(float(inf_ms))
                    det_counts.append(int(n_dets))

    print(f"  Log saved  : {log_path}")

    # --- WiderFace evaluation (capture output → widerface_eval.txt) ---
    print(f"  Running WiderFace evaluation...")
    eval_buf = io.StringIO()
    with contextlib.redirect_stdout(eval_buf):
        evaluation(pred_dir, GT_DIR)
    eval_output = eval_buf.getvalue()
    print(eval_output)  # echo to terminal

    with open(EVAL_TXT, "a", encoding="utf-8") as ef:
        ef.write(f"{label}\n")
        ef.write(eval_output)
        ef.write("\n")

    ap_scores = parse_eval_output(eval_output)
    timing_metrics = compute_timing_metrics(inf_times_ms, det_counts)

    return {
        "label": label,
        "class": DetClass.__name__,
        "model_file": model_cfg["model_file"],
        "params": params,
        "paths": {
            "predictions_dir": pred_dir,
            "inference_log_csv": log_path,
        },
        "runtime": onnx_runtime,
        "timing": timing_metrics,
        "ap": {
            "easy": ap_scores.get("easy", None),
            "medium": ap_scores.get("medium", None),
            "hard": ap_scores.get("hard", None),
        },
    }


def run_benchmark(device_id=None):
    """Sequentially benchmark every model in the registry."""
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # Clear the eval log so each full run starts fresh
    open(EVAL_TXT, "w").close()

    # Early provider check prevents long CPU fallback runs when DirectML was requested.
    verify_onnx_runtime(device_id)

    runtime_info = collect_runtime_info(device_id)
    print("\nRuntime Info:")
    print(f"  Selected Device ID   : {runtime_info['selected_device_id']}")
    print(f"  Selected Device Name : {runtime_info['selected_device_name']}")
    print(f"  DirectML Available   : {runtime_info['directml_available']}")
    print(f"  ORT Providers        : {runtime_info['ort_available_providers']}")

    print(f"\nBenchmarking {len(MODELS)} model configurations against WiderFace-val...")

    results = []
    for model_cfg in MODELS:
        results.append(run_single_model(model_cfg, device_id))

    payload = {
        "generated_at": datetime.now().isoformat(),
        "device": runtime_info,
        "dataset": "WiderFace val",
        "results": results,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nMetrics JSON saved to {METRICS_JSON}")

    print(f"\nAll models complete. Results saved under {RESULTS_ROOT}/")


if __name__ == "__main__":
    selected_device_id = select_device()
    run_benchmark(selected_device_id)