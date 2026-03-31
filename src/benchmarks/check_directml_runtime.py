import glob
import json
import os
from collections import defaultdict

import numpy as np
import onnxruntime as ort


MODEL_ROOT_CANDIDATES = [
    os.path.join("models", "face_detection"),
    os.path.join("models", "facedetection"),
]
TARGET_PROVIDER = "DmlExecutionProvider"
TARGET_LABEL = "DirectML"


def _resolve_dim(dim, axis_index):
    """Convert dynamic/invalid dims into safe defaults for one inference pass."""
    if isinstance(dim, int) and dim > 0:
        return dim
    if axis_index == 0:
        return 1
    if axis_index == 1:
        return 3
    return 640


def _build_feed_dict(session):
    feed = {}
    for inp in session.get_inputs():
        shape = [_resolve_dim(d, i) for i, d in enumerate(inp.shape)]
        if not shape:
            shape = [1]

        np_type = np.float32
        t = inp.type
        if "int64" in t:
            np_type = np.int64
        elif "int32" in t:
            np_type = np.int32
        elif "float16" in t:
            np_type = np.float16
        elif "double" in t or "float64" in t:
            np_type = np.float64
        elif "bool" in t:
            np_type = np.bool_

        if np_type == np.bool_:
            feed[inp.name] = np.zeros(shape, dtype=np_type)
        elif np.issubdtype(np_type, np.integer):
            feed[inp.name] = np.zeros(shape, dtype=np_type)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np_type)
    return feed


def _extract_node_provider_stats(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    node_to_provider = {}
    node_to_op = {}
    provider_to_ops = defaultdict(set)

    for ev in events:
        if ev.get("cat") != "Node":
            continue

        args = ev.get("args", {})
        provider = args.get("provider")
        op_name = args.get("op_name", "UnknownOp")

        node_name = args.get("node_name")
        if not node_name:
            node_name = ev.get("name", "")
        if node_name.endswith("_kernel_time"):
            node_name = node_name[: -len("_kernel_time")]
        if not node_name:
            continue

        if provider:
            node_to_provider[node_name] = provider
        node_to_op[node_name] = op_name
        if provider:
            provider_to_ops[provider].add(op_name)

    total_nodes = len(node_to_provider)
    target_nodes = sum(1 for p in node_to_provider.values() if p == TARGET_PROVIDER)
    cpu_nodes = sum(1 for p in node_to_provider.values() if p == "CPUExecutionProvider")
    other_nodes = total_nodes - target_nodes - cpu_nodes

    total_ops = len(set(node_to_op.values()))
    target_ops = len(provider_to_ops.get(TARGET_PROVIDER, set()))
    cpu_ops = len(provider_to_ops.get("CPUExecutionProvider", set()))

    return {
        "total_nodes": total_nodes,
        "target_nodes": target_nodes,
        "cpu_nodes": cpu_nodes,
        "other_nodes": other_nodes,
        "target_node_pct": (target_nodes / total_nodes * 100.0) if total_nodes else 0.0,
        "total_ops": total_ops,
        "target_ops": target_ops,
        "cpu_ops": cpu_ops,
        "target_provider_ops": sorted(provider_to_ops.get(TARGET_PROVIDER, set())),
        "cpu_provider_ops": sorted(provider_to_ops.get("CPUExecutionProvider", set())),
    }


def diagnose_model_directml_support(model_path):
    print(f"\nInspecting model: {model_path}")
    if not os.path.exists(model_path):
        print("  ERROR: model file not found")
        return None

    so = ort.SessionOptions()
    so.enable_profiling = True

    providers = ort.get_available_providers()
    if TARGET_PROVIDER not in providers:
        print(
            f"  ERROR: {TARGET_PROVIDER} is not available. Install/use onnxruntime-directml and run on Windows."
        )
        return None

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=[TARGET_PROVIDER, "CPUExecutionProvider"],
        )
    except Exception as ex:
        print(f"  ERROR: failed to create session: {ex}")
        return None

    try:
        feeds = _build_feed_dict(session)
        session.run(None, feeds)
        profile_path = session.end_profiling()
    except Exception as ex:
        print(f"  ERROR: inference/profiling failed: {ex}")
        return None

    if not os.path.exists(profile_path):
        print("  ERROR: profile output not found")
        return None

    stats = _extract_node_provider_stats(profile_path)
    try:
        os.remove(profile_path)
    except OSError:
        pass

    print(
        "  Nodes -> total: {total_nodes}, DirectML: {target_nodes}, CPU fallback: {cpu_nodes}, other: {other_nodes}, DirectML%: {target_node_pct:.2f}".format(
            **stats
        )
    )
    print(
        "  Ops   -> unique total: {total_ops}, DirectML ops: {target_ops}, CPU ops: {cpu_ops}".format(
            **stats
        )
    )

    return {"model_path": model_path, **stats}


def _find_models_root():
    for candidate in MODEL_ROOT_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return None


def list_onnx_models():
    models_root = _find_models_root()
    if models_root is None:
        return [], None
    pattern = os.path.join(models_root, "**", "*.onnx")
    return sorted(glob.glob(pattern, recursive=True)), models_root


def main():
    models, models_root = list_onnx_models()
    if models_root is None:
        print(
            "No face-detection model folder found. Expected one of: "
            + ", ".join(MODEL_ROOT_CANDIDATES)
        )
        return

    if not models:
        print(f"No ONNX model files found under: {models_root}")
        return

    print(f"Using model folder: {models_root}")
    print(f"Found {len(models)} ONNX model(s). Running one profiled inference each...")

    rows = []
    for model_path in models:
        result = diagnose_model_directml_support(model_path)
        if result is not None:
            rows.append(result)

    if not rows:
        print("No model could be profiled successfully.")
        return

    rows_sorted = sorted(rows, key=lambda x: x["target_node_pct"], reverse=True)

    print(f"\nSummary (sorted by {TARGET_LABEL}-supported node percentage):")
    print("-" * 118)
    print(
        f"{'Model':65} {'DML Nodes':>10} {'Total Nodes':>11} {'DML %':>8} {'DML Ops':>9} {'Total Ops':>9}"
    )
    print("-" * 118)
    for r in rows_sorted:
        print(
            f"{r['model_path'][:65]:65} {r['target_nodes']:10d} {r['total_nodes']:11d} {r['target_node_pct']:8.2f} {r['target_ops']:9d} {r['total_ops']:9d}"
        )


if __name__ == "__main__":
    main()