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
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
METRICS_JSON = os.path.join(RESULTS_DIR, "widerface_eval.json")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
PALETTE = sns.color_palette("tab10")

# ─── Pretty label map ─────────────────────────────────────────────────────────
LABEL_MAP = {
    "yunet_dynamic":        "YuNet\n(dynamic)",
    "yunet_640x640":        "YuNet\n(640×640)",
    "retinaface_dynamic":   "RetinaFace\n(dynamic)",
    "retinaface_640x640":   "RetinaFace\n(640×640)",
    "scrfd_640x640":        "SCRFD\n(640×640)",
}

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
        df = df[df["inference_time_ms"] > 0].reset_index(drop=True)
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

        t = df["inference_time_ms"]
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
    """Overlapping KDE + rug plot of per-image inference time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, df) in enumerate(logs.items()):
        t = df["inference_time_ms"]
        pretty = LABEL_MAP.get(label, label).replace("\n", " ")
        sns.kdeplot(t, ax=ax, label=pretty, color=PALETTE[i % len(PALETTE)], fill=True, alpha=0.25)
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Inference Time Distribution per Model")
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    _save("inference_time_distribution.png")


def plot_boxplot(logs: dict):
    """Box-and-whisker comparison of inference times."""
    data   = [df["inference_time_ms"].values for df in logs.values()]
    labels = [LABEL_MAP.get(l, l).replace("\n", " ") for l in logs]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Time Box Plot")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    _save("inference_time_boxplot.png")


def plot_fps_bar(summary: pd.DataFrame):
    """Mean FPS per model."""
    df = summary[["pretty", "fps"]].dropna().reset_index()
    df = df.sort_values("fps", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df["pretty"], df["fps"],
                  color=PALETTE[:len(df)], edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set_ylabel("FPS")
    ax.set_title("Mean Frames Per Second per Model")
    ax.set_ylim(0, df["fps"].max() * 1.2)
    _save("fps_bar.png")


def plot_latency_stats(summary: pd.DataFrame):
    """Grouped bar: mean / p95 / p99 latency."""
    df = summary[["pretty", "mean_ms", "p95_ms", "p99_ms"]].dropna().reset_index()
    x  = np.arange(len(df))
    w  = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, df["mean_ms"], width=w, label="Mean",   color=PALETTE[0], alpha=0.85)
    ax.bar(x,     df["p95_ms"],  width=w, label="p95",    color=PALETTE[1], alpha=0.85)
    ax.bar(x + w, df["p99_ms"],  width=w, label="p99",    color=PALETTE[2], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["pretty"], rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Statistics per Model (Mean / p95 / p99)")
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


def plot_latency_cdf(logs: dict):
    """Empirical CDF of inference time per model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, df) in enumerate(logs.items()):
        t      = np.sort(df["inference_time_ms"].values)
        cdf    = np.arange(1, len(t) + 1) / len(t)
        pretty = LABEL_MAP.get(label, label).replace("\n", " ")
        ax.plot(t, cdf, label=pretty, color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, label="p95")
    ax.axhline(0.99, color="gray", linestyle=":",  linewidth=0.8, label="p99")
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Inference Time CDF per Model")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    _save("latency_cdf.png")


def plot_detections_per_image(logs: dict):
    """Box plot of face detection count per image per model."""
    has_dets = {
        l: df for l, df in logs.items()
        if "num_detections" in df.columns and df["num_detections"].notna().any()
    }
    if not has_dets:
        print("[!] No Num_Detections column – skipping detections_per_image.png")
        return

    data   = [df["num_detections"].values for df in has_dets.values()]
    labels = [LABEL_MAP.get(l, l).replace("\n", " ") for l in has_dets]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Detections per Image")
    ax.set_title("Face Detections per Image Distribution")
    _save("detections_per_image.png")


def plot_performance_scatter(summary: pd.DataFrame):
    """Scatter: FPS vs AP-Easy (bubble = AP-Hard)."""
    df = summary[["pretty", "fps", "ap_easy", "ap_hard"]].dropna()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, row in df.reset_index().iterrows():
        size = max(row["ap_hard"] * 20, 50) if not np.isnan(row["ap_hard"]) else 80
        ax.scatter(row["fps"], row["ap_easy"],
                   s=size, color=PALETTE[i % len(PALETTE)],
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(row["pretty"].replace("\n", " "),
                    (row["fps"], row["ap_easy"]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("FPS (higher is better →)")
    ax.set_ylabel("Easy Val AP % (higher is better ↑)")
    ax.set_title("Speed vs Accuracy  (bubble size ∝ Hard AP)")
    ax.grid(True, linestyle="--", alpha=0.5)
    _save("speed_vs_accuracy.png")


def plot_precision_recall_summary(summary: pd.DataFrame):
    """
    Bar chart: Precision and Recall proxies from AP scores across difficulty.
    Uses (Easy+Medium+Hard)/3 as mean AP and Hard AP as a recall proxy.
    """
    df = summary[["pretty", "ap_easy", "ap_medium", "ap_hard"]].dropna(
        subset=["ap_easy"]).reset_index()
    if df.empty:
        return

    df["mean_ap"] = df[["ap_easy", "ap_medium", "ap_hard"]].mean(axis=1)
    df = df.sort_values("mean_ap", ascending=False)
    x  = np.arange(len(df))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w / 2, df["mean_ap"], width=w,
                label="Mean AP (avg difficulty)", color=PALETTE[0], alpha=0.85)
    b2 = ax.bar(x + w / 2, df["ap_hard"],  width=w,
                label="Hard AP (hardest cases)", color=PALETTE[3], alpha=0.85)
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["pretty"], rotation=20, ha="right")
    ax.set_ylabel("AP (%)")
    ax.set_title("Mean AP vs Hard AP per Model")
    ax.set_ylim(0, 105)
    ax.legend()
    _save("precision_recall_summary.png")


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

    print(f"\nBuilding summary table...")
    summary = build_summary(logs, metrics_payload)
    print(summary[["fps", "mean_ms", "p95_ms", "ap_easy", "ap_medium", "ap_hard"]].to_string())

    print(f"\nGenerating plots → {PLOTS_DIR}")
    plot_inference_distribution(logs)
    plot_boxplot(logs)
    plot_fps_bar(summary)
    plot_latency_stats(summary)
    plot_ap_bar(summary)
    plot_latency_cdf(logs)
    plot_detections_per_image(logs)
    plot_performance_scatter(summary)
    plot_precision_recall_summary(summary)

    print(f"\nAll done. {len(os.listdir(PLOTS_DIR))} plot(s) saved to {PLOTS_DIR}")
