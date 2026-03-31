"""
plot_benchmark_results.py
─────────────────────────
Reads all inference_log_*.csv files from the benchmark results directory and
the widerface_eval.txt AP scores, then produces a full suite of plots saved
to src/benchmarks/results/plots/.

Usage (from project root):
    python src/benchmarks/plot_benchmark_results.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results/face_detection/results_realworld")  # <-- CHANGE THIS to match your benchmark run's output dir
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
METRICS_JSON = os.path.join(RESULTS_DIR, "widerface_eval.json")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
PALETTE = sns.color_palette("tab10")
TOTAL_CORRECT_FACES = 31_958

# ─── Pretty label map ─────────────────────────────────────────────────────────
# ─── Pretty base names for the classes ─────────────────────────────────────────
# We will dynamically build the full labels later using the JSON metadata
BASE_NAMES = {
    "YuNetDetector": "YuNet",
    "RetinaFaceDetector": "RetinaFace",
    "SCRFDDetector": "SCRFD"
}
LABEL_MAP = {} # Leave empty; we populate this in __main__
# ──────────────────────────────────────────────────────────────────────────────
# 1. Load all CSVs
# ──────────────────────────────────────────────────────────────────────────────
def load_inference_logs() -> dict[str, pd.DataFrame]:
    """Returns {label: DataFrame} for every inference_log_*.csv found."""
    logs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "inference_log_*.csv"))):
        label = os.path.basename(path).replace("inference_log_", "").replace(".csv", "")
        df = pd.read_csv(path)
        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # Keep only rows with valid timing
        df = df[df["total_ms"] > 0].reset_index(drop=True)
        logs[label] = df
    return logs


# ──────────────────────────────────────────────────────────────────────────────
# 2. Load benchmark metrics JSON
# ──────────────────────────────────────────────────────────────────────────────
def load_metrics_json(path: str) -> dict:
    """Load structured benchmark output from benchmark_detector.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Build summary stats table
# ──────────────────────────────────────────────────────────────────────────────
def build_summary(logs: dict, metrics_payload: dict) -> pd.DataFrame:
    metrics_by_label = {m["label"]: m for m in metrics_payload.get("results", [])}
    rows = []
    for label, df in logs.items():
        m = metrics_by_label.get(label, {})
        timing = m.get("timing", {})
        ap = m.get("ap", {})

        t = df["total_ms"]
        row = {
            "model":       label,
            "pretty":      LABEL_MAP.get(label, label),
            "n_images":    int(timing.get("num_images", len(df))),
            "mean_ms":     float(timing.get("mean_ms", round(t.mean(), 2))),
            "median_ms":   float(timing.get("median_ms", round(t.median(), 2))),
            "p95_ms":      float(timing.get("p95_ms", round(t.quantile(0.95), 2))),
            "p99_ms":      float(timing.get("p99_ms", round(t.quantile(0.99), 2))),
            "fps":         float(timing.get("fps", round(1000 / t.mean(), 2))),
        }
        row["mean_dets"] = float(timing.get("mean_detections", float("nan")))
        row["ap_easy"]   = ap.get("easy", float("nan"))
        row["ap_medium"] = ap.get("medium", float("nan"))
        row["ap_hard"]   = ap.get("hard", float("nan"))
        rows.append(row)

    df_sum = pd.DataFrame(rows).set_index("model")
    df_sum.to_csv(os.path.join(RESULTS_DIR, "benchmark_summary.csv"))
    print(f"[✓] benchmark_summary.csv saved")
    return df_sum


# ──────────────────────────────────────────────────────────────────────────────
# 4. Plots
# ──────────────────────────────────────────────────────────────────────────────

def _save(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[✓] {name}")


def plot_inference_distribution(logs: dict):
    """Overlapping KDE plot with vertical median lines and value labels."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for i, (label, df) in enumerate(logs.items()):
        t = df["total_ms"]
        pretty = LABEL_MAP.get(label, label).replace("\n", " ")
        color = PALETTE[i % len(PALETTE)]
        
        # Plot the density curve
        sns.kdeplot(t, ax=ax, label=pretty, color=color, fill=True, alpha=0.25)
        
        # --- NEW: Plot vertical line for median and add text label ---
        median_val = t.median()
        ax.axvline(median_val, color=color, linestyle="--", alpha=0.8)
        
        # Stagger the Y-position of the text slightly so multiple models don't overlap
        y_pos = ax.get_ylim()[1] * (0.05 + (i * 0.05))
        ax.text(median_val + 0.2, y_pos, f"Med: {median_val:.1f}ms", 
                color=color, fontsize=9, fontweight="bold")

    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Inference Time Distribution per Model")
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    _save("inference_time_distribution.png")



def plot_fps_bar(summary: pd.DataFrame):
    """Mean FPS per model with bold value labels and rotated x-axis to prevent overlap."""
    df = summary[["pretty", "fps"]].dropna().reset_index()
    df = df.sort_values("fps", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # FIX: Use numeric X positions so bars NEVER stack on top of each other
    x_positions = np.arange(len(df))
    bars = ax.bar(x_positions, df["fps"],
                  color=PALETTE[:len(df)], edgecolor="white", linewidth=0.8)
    
    # Value labels on top of the bars
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10, fontweight="bold", color="#333333")
    
    ax.set_ylabel("FPS")
    ax.set_title("Mean Frames Per Second per Model")
    ax.set_ylim(0, df["fps"].max() * 1.2)
    
    # Attach the text labels to the numeric positions
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df["pretty"], rotation=30, ha="right")
    
    _save("fps_bar.png")

def plot_latency_stats(summary: pd.DataFrame):
    """Grouped bar: mean / p95 / p99 latency with value labels."""
    df = summary[["pretty", "mean_ms", "p95_ms", "p99_ms"]].dropna().reset_index()
    x  = np.arange(len(df))
    w  = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w, df["mean_ms"], width=w, label="Mean",   color=PALETTE[0], alpha=0.85)
    b2 = ax.bar(x,     df["p95_ms"],  width=w, label="p95",    color=PALETTE[1], alpha=0.85)
    b3 = ax.bar(x + w, df["p99_ms"],  width=w, label="p99",    color=PALETTE[2], alpha=0.85)
    
    # --- NEW: Add value labels to the top of each bar ---
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8, color="#333333", fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels(df["pretty"], rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Statistics per Model (Mean / p95 / p99)")
    
    # Increase Y-limit slightly so labels don't clip the top of the chart
    ax.set_ylim(0, df[["mean_ms", "p95_ms", "p99_ms"]].max().max() * 1.15)
    ax.legend()
    _save("latency_stats.png")


def plot_ap_bar(summary: pd.DataFrame):
    """Grouped bar: Easy / Medium / Hard AP."""
    df = summary[["pretty", "ap_easy", "ap_medium", "ap_hard"]].dropna(
        subset=["ap_easy"]).reset_index()
    if df.empty:
        print("[!] No AP scores found – skipping ap_bar.png")
        return

    df = df.sort_values("ap_easy", ascending=False)
    x  = np.arange(len(df))
    w  = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w, df["ap_easy"],   width=w, label="Easy",   color=PALETTE[0], alpha=0.85)
    b2 = ax.bar(x,     df["ap_medium"], width=w, label="Medium", color=PALETTE[1], alpha=0.85)
    b3 = ax.bar(x + w, df["ap_hard"],   width=w, label="Hard",   color=PALETTE[2], alpha=0.85)
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(df["pretty"], rotation=20, ha="right")
    ax.set_ylabel("Average Precision (%)")
    ax.set_title("WiderFace Val AP per Model")
    ax.set_ylim(0, 105)
    ax.legend()
    _save("ap_bar.png")


def plot_total_detections_bar(metrics_payload: dict, difficulty: str = "hard", ref_threshold: float = 0.01):
    """Bar chart of total evaluated detections (TP+FP) vs evaluated GT (TP+FN)."""
    results = metrics_payload.get("results", [])
    if not results:
        print("[!] No model results found in metrics JSON – skipping total detections plot.")
        return

    labels = []
    totals = []
    gt_values = []

    for model_data in results:
        label = model_data.get("label", "unknown")
        curve = model_data.get("curves", {}).get(difficulty, {})
        thresholds = np.array(curve.get("thresholds", []), dtype=float)
        tps = np.array(curve.get("tp", []), dtype=float)
        fps = np.array(curve.get("fp", []), dtype=float)
        fns = np.array(curve.get("fn", []), dtype=float)

        if len(thresholds) == 0:
            continue

        idx = int(np.abs(thresholds - ref_threshold).argmin())
        total_pred = int(round(float(tps[idx] + fps[idx])))
        total_gt = int(round(float(tps[idx] + fns[idx])))

        labels.append(LABEL_MAP.get(label, label))
        totals.append(total_pred)
        gt_values.append(total_gt)

    if not totals:
        print("[!] No curve counts found – skipping total detections plot.")
        return

    gt_total = int(round(float(np.median(gt_values))))
    labels.insert(0, f"Ground Truth\n({difficulty.title()} Eval Faces)")
    totals.insert(0, gt_total)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#555555"] + list(PALETTE)[:len(totals) - 1]
    
    # FIX: Use numeric X positions so bars NEVER stack on top of each other
    x_positions = np.arange(len(labels))
    bars = ax.bar(x_positions, totals, color=colors, edgecolor="white", alpha=0.85)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(totals)*0.02), 
                f"{int(yval):,}", ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color="#333333")

    ax.set_ylabel("Total Count")
    ax.set_title(
        f"Total Evaluated Detections vs Ground Truth ({difficulty.title()}, threshold≈{ref_threshold})"
    )
    
    # Attach the text labels to the numeric positions
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    
    ax.set_ylim(0, max(totals) * 1.15)
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    _save("total_detections_bar.png")

def plot_stacked_latency_bar(logs: dict):
    """Stacked bar chart showing Preprocess vs Inference vs Postprocess times with labels."""
    records = []
    
    for label, df in logs.items():
        if "preprocess_ms" not in df.columns:
            print(f"[!] Granular timings missing for {label}, skipping in stacked bar.")
            continue
            
        records.append({
            "Model": LABEL_MAP.get(label, label).replace("\n", " "),
            "Preprocess": df["preprocess_ms"].mean(),
            "Inference": df["inference_ms"].mean(),
            "Postprocess": df["postprocess_ms"].mean(),
        })
        
    if not records:
        return

    plot_df = pd.DataFrame(records).set_index("Model")
    plot_df["Total"] = plot_df.sum(axis=1)
    plot_df = plot_df.sort_values("Total", ascending=False).drop(columns=["Total"])

    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_df.plot(kind="bar", stacked=True, ax=ax, 
                 color=[PALETTE[0], PALETTE[1], PALETTE[2]], alpha=0.85, edgecolor="white")
    
    # --- NEW: Add value labels to the center of each stack ---
    for c in ax.containers:
        # Only draw the label if the segment takes more than 0.5ms (prevents text overlapping)
        labels = [f"{v.get_height():.1f}" if v.get_height() > 0.5 else "" for v in c]
        ax.bar_label(c, labels=labels, label_type="center", fontsize=9, color="white", fontweight="bold")
    
    ax.set_ylabel("Average Time (ms)")
    ax.set_title("Latency Breakdown: Preprocess vs Inference vs Postprocess")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30, ha="right")
    
    # Move legend outside the plot so it doesn't cover the bars
    ax.legend(title="Pipeline Stage", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    _save("latency_breakdown_stacked.png")

def plot_precision_recall_curves(metrics_payload: dict):
    """Plot Precision-Recall curves side-by-side for Easy, Medium, and Hard difficulties, including Max F1."""
    results = metrics_payload.get("results", [])
    
    # Check if the first model has the 'curves' key we added earlier
    if not results or "curves" not in results[0]:
        print("[!] No PR curve data found in JSON – skipping PR plots.")
        return

    # Create a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    difficulties = ["easy", "medium", "hard"]
    titles = ["Easy Setting", "Medium Setting", "Hard Setting"]

    for i, diff in enumerate(difficulties):
        ax = axes[i]
        for j, model_data in enumerate(results):
            label = model_data.get("label", "unknown")
            pretty = LABEL_MAP.get(label, label).replace("\n", " ")
            
            curves = model_data.get("curves", {})
            diff_curve = curves.get(diff, {})
            
            # Extract lists and convert to numpy arrays for mathematical operations
            recall = np.array(diff_curve.get("recall", []))
            precision = np.array(diff_curve.get("precision", []))
            
            if len(recall) > 0 and len(precision) > 0:
                # 1. Calculate the F1 score for every point on the curve
                # We add a tiny epsilon (1e-8) to the denominator to prevent division by zero
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                # 2. Find the index of the highest F1 score
                best_idx = np.argmax(f1_scores)
                best_f1 = f1_scores[best_idx]
                best_r = recall[best_idx]
                best_p = precision[best_idx]
                
                color = PALETTE[j % len(PALETTE)]
                
                # Update the label to include the Max F1 score
                label_with_f1 = f"{pretty}\n(Max F1: {best_f1:.2f})"
                
                # Standard practice: Recall on X-axis, Precision on Y-axis
                ax.plot(recall, precision, label=label_with_f1, 
                        color=color, linewidth=2)
                
                # 3. Plot a dot at the optimal threshold (the "knee")
                ax.plot(best_r, best_p, marker='o', markersize=8, 
                        color=color, markeredgecolor='white', linestyle='None')
        
        ax.set_title(titles[i])
        ax.set_xlabel("Recall")
        if i == 0:
            ax.set_ylabel("Precision")
        
        # PR curves should always be scaled from 0 to 1
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)
        
        # Slightly smaller font to fit the F1 scores nicely
        ax.legend(loc="lower left", fontsize=8)

    plt.suptitle("Precision-Recall Curves by WiderFace Difficulty (Dots indicate Max F1 Score)", fontsize=14, y=1.05)
    
    # Use the existing _save helper
    path = os.path.join(PLOTS_DIR, "precision_recall_curves.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print("[✓] precision_recall_curves.png")

def plot_tp_fp_fn_stacked_bar(metrics_payload: dict):
    """Plot TP/FN/FP stacked bars as % of total correct faces (31,958)."""
    results = metrics_payload.get("results", [])
    if not results or "curves" not in results[0]:
        print("[!] No curve data found in JSON – skipping threshold garbage plots.")
        return

    difficulty = "hard"
    target_thresholds = [0.5, 0.6, 0.7, 0.85, 0.95]
    colors = ['#2ca02c', '#ff7f0e', '#d62728']

    for model_data in results:
        label = model_data.get("label", "unknown")
        pretty = LABEL_MAP.get(label, label).replace("\n", " ")

        curve = model_data.get("curves", {}).get(difficulty, {})
        thresh_array = np.array(curve.get("thresholds", []), dtype=float)
        tps = np.array(curve.get("tp", []), dtype=float)
        fps = np.array(curve.get("fp", []), dtype=float)
        fns = np.array(curve.get("fn", []), dtype=float)

        if len(thresh_array) == 0:
            continue

        # Existing sampled points used for stacked bars
        model_tp, model_fp, model_fn = [], [], []
        for t in target_thresholds:
            idx = int(np.abs(thresh_array - t).argmin())
            model_tp.append(tps[idx])
            model_fp.append(fps[idx])
            model_fn.append(fns[idx])

        # Convert raw counts to % of total correct faces.
        model_tp = (np.array(model_tp, dtype=float) / TOTAL_CORRECT_FACES) * 100.0
        model_fp = (np.array(model_fp, dtype=float) / TOTAL_CORRECT_FACES) * 100.0
        model_fn = (np.array(model_fn, dtype=float) / TOTAL_CORRECT_FACES) * 100.0

        # NEW: all thresholds where FP == 0 (from full evaluator thresholds)
        zero_fp_mask = fps == 0
        # Lowest threshold where FP == 0 (from full evaluator thresholds)
        zero_fp_thresholds = thresh_array[fps == 0]
        if zero_fp_thresholds.size > 0:
            lowest_zero_fp = float(np.min(zero_fp_thresholds))
            zero_fp_text = f"Lowest threshold with FP=0: {lowest_zero_fp:.3f}"
        else:
            zero_fp_text = "Lowest threshold with FP=0: None"

        # NEW: two-row figure (main stacked bars + additional threshold info panel)
        fig, (ax, ax_info) = plt.subplots(
            2, 1, figsize=(10, 8),
            gridspec_kw={"height_ratios": [4.5, 1.2], "hspace": 0.25}
        )

        x = np.arange(len(target_thresholds))
        width = 0.5

        b_fn = ax.bar(x, model_fn, width, label='False Negatives (Missed)', color=colors[1])
        b_tp = ax.bar(x, model_tp, width, bottom=model_fn, label='True Positives (Found)', color=colors[0])
        bottom_for_fp = model_fn + model_tp
        b_fp = ax.bar(x, model_fp, width, bottom=bottom_for_fp, label='False Positives (Garbage)', color=colors[2])

        bbox_style = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        for bars in (b_fn, b_tp, b_fp):
            labels = [f"{v.get_height():.2f}%" if v.get_height() > 0 else "" for v in bars]
            ax.bar_label(
                bars, labels=labels, label_type="center", fontsize=8,
                fontweight="bold", color="black", bbox=bbox_style
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Thresh: {t}" for t in target_thresholds], rotation=30, ha="right")
        ax.set_xlabel("Confidence Score Threshold")
        ax.set_ylabel(f"Detections (% of {TOTAL_CORRECT_FACES:,} Correct Faces, Log Scale)")
        ax.set_yscale('log')
        ax.set_title(f"{pretty} - True Faces vs. Garbage Detections (% Scale)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1))

        # NEW: additional plotted panel with FP=0 thresholds (all thresholds)
        ax_info.axis("off")
        ax_info.text(
            0.01, 0.95,
           f"{zero_fp_text}",
            ha="left", va="top", fontsize=9, wrap=True,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray")
        )

        safe_label = label.replace(" ", "_").replace("/", "_").replace("\n", "")
        filename = f"threshold_garbage_{safe_label}.png"
        path = os.path.join(PLOTS_DIR, filename)

        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[✓] {filename}")


def plot_models_tn_fn_fp_constrained(metrics_payload: dict):
    """Compare TP/FN/FP across all models at one threshold per model with FP<2%."""
    results = metrics_payload.get("results", [])
    if not results or "curves" not in results[0]:
        print("[!] No curve data found in JSON – skipping constrained multi-model plot.")
        return

    difficulty = "hard"
    max_fp_percent = 2.0

    rows = []
    for model_data in results:
        label = model_data.get("label", "unknown")
        pretty = LABEL_MAP.get(label, label).replace("\n", " ")

        curve = model_data.get("curves", {}).get(difficulty, {})
        thresholds = np.array(curve.get("thresholds", []), dtype=float)
        tps = np.array(curve.get("tp", []), dtype=float)
        fps = np.array(curve.get("fp", []), dtype=float)
        fns = np.array(curve.get("fn", []), dtype=float)

        if len(thresholds) == 0:
            continue

        tp_pct = (tps / TOTAL_CORRECT_FACES) * 100.0
        fp_pct = (fps / TOTAL_CORRECT_FACES) * 100.0
        fn_pct = (fns / TOTAL_CORRECT_FACES) * 100.0

        fp_ok_idx = np.where(fp_pct < max_fp_percent)[0]
        if fp_ok_idx.size == 0:
            print(f"[!] {label}: no threshold satisfies FP<{max_fp_percent}%.")
            continue

        # Prefer points with TP>FN; otherwise still pick the best TP under FP<2%.
        preferred_idx = fp_ok_idx[tp_pct[fp_ok_idx] > fn_pct[fp_ok_idx]]
        candidate_idx = preferred_idx if preferred_idx.size > 0 else fp_ok_idx

        # Pick operating point with highest TP%, then lowest threshold on ties.
        best_idx = candidate_idx[np.lexsort((thresholds[candidate_idx], -tp_pct[candidate_idx]))][0]
        tp_gt_fn = bool(tp_pct[best_idx] > fn_pct[best_idx])

        rows.append({
            "pretty": pretty,
            "threshold": float(thresholds[best_idx]),
            "tp_pct": float(tp_pct[best_idx]),
            "fn_pct": float(fn_pct[best_idx]),
            "fp_pct": float(fp_pct[best_idx]),
            "tp_gt_fn": tp_gt_fn,
        })

    if not rows:
        print("[!] No model has a valid threshold under constraints – skipping constrained multi-model plot.")
        return

    df = pd.DataFrame(rows).sort_values("tp_pct", ascending=False).reset_index(drop=True)

    x = np.arange(len(df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    b_tp = ax.bar(x - w, df["tp_pct"], width=w, label="True Positives", color="#2ca02c", alpha=0.9)
    b_fn = ax.bar(x, df["fn_pct"], width=w, label="False Negatives", color="#ff7f0e", alpha=0.9)
    b_fp = ax.bar(x + w, df["fp_pct"], width=w, label="False Positives", color="#d62728", alpha=0.9)

    for bars in (b_tp, b_fn, b_fp):
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)

    x_labels = [
        f"{r.pretty}\n(thr={r.threshold:.3f}, {'TP>FN' if r.tp_gt_fn else 'TP<=FN'})"
        for r in df.itertuples()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel(f"Percentage of {TOTAL_CORRECT_FACES:,} Correct Faces (%)")
    ax.set_xlabel("Model (Selected Threshold)")
    ax.set_title("TP / FN / FP Across Models at Thresholds with FP<2%")
    ax.set_ylim(0, max(df[["tp_pct", "fn_pct", "fp_pct"]].max()) * 1.2)
    ax.legend()

    _save("tp_fn_fp_all_models_constrained.png")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nLoading logs from: {RESULTS_DIR}")
    logs = load_inference_logs()
    if not logs:
        raise FileNotFoundError(f"No inference_log_*.csv files found in {RESULTS_DIR}")
    print(f"  Found {len(logs)} model log(s): {list(logs.keys())}")

    print(f"\nLoading benchmark metrics JSON: {METRICS_JSON}")
    metrics_payload = load_metrics_json(METRICS_JSON)
    print(f"  Models in JSON: {[m.get('label') for m in metrics_payload.get('results', [])]}")

    for m in metrics_payload.get("results", []):
        lbl = m.get("label")
        params = m.get("params", {})
        
        if not lbl: 
            continue
            
        # Extract the unique name before the underscore (e.g., 'scrfd10gkps')
        base_pretty = lbl.split('_')[0] 
        thr = params.get("score_threshold", "N/A")
        
        # Determine the full tensor shape string (NCHW format: 1x3xHxW)
        is_dynamic = params.get("dynamic_input", False)
        input_size = params.get("input_size", None)
        
        if is_dynamic:
            shape_str = "Dynamic Tensor"
        elif input_size and len(input_size) == 2:
            shape_str = f"1x3x{input_size[0]}x{input_size[1]}"
        else:
            shape_str = "Unknown Shape"
            
        # Create the final 3-line label (now guaranteed to be unique)
        LABEL_MAP[lbl] = f"{base_pretty}\n({shape_str})\n(test_thr: {thr})"

    print(f"\nBuilding summary table...")
    summary = build_summary(logs, metrics_payload)
    print(summary[["fps", "mean_ms", "p95_ms", "ap_easy", "ap_medium", "ap_hard"]].to_string())

    print(f"\nGenerating plots → {PLOTS_DIR}")
    plot_inference_distribution(logs)
    plot_stacked_latency_bar(logs)
    plot_fps_bar(summary)
    plot_latency_stats(summary)
    plot_ap_bar(summary)
    plot_total_detections_bar(metrics_payload)
    plot_precision_recall_curves(metrics_payload)
    plot_tp_fp_fn_stacked_bar(metrics_payload)
    plot_models_tn_fn_fp_constrained(metrics_payload)

    print(f"\nAll done. {len(os.listdir(PLOTS_DIR))} plot(s) saved to {PLOTS_DIR}")
