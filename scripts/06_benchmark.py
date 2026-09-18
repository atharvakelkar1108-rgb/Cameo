"""
Builds the final comparison table, ordered by PIPELINE STAGE (Baseline ->
Quantized -> Pruned -> Optimized) rather than sorted by size — this makes
the methodology easy to follow in a report/viva. Also computes derived
metrics relative to baseline: compression ratio, size reduction %, latency
improvement %, and BLEU-4 difference from baseline.

This script does NOT re-run any model — it only reads results/*.json files
already produced by scripts 01/02/04/08.

Run: python scripts/06_benchmark.py
"""

import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Fixed pipeline order — this is the ordering that matches your project's
# methodology narrative, not an alphabetical/size sort.
PIPELINE_ORDER = [
    ("baseline.json", "Baseline (FP32)"),
    ("quantized_dynamic.json", "Quantized (INT8)"),
    ("pruned.json", "Pruned (30%)"),
    ("optimized.json", "Optimized (Pruned + INT8)"),
]


def load_pipeline_results():
    rows = []
    for filename, display_name in PIPELINE_ORDER:
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"  (skipping {filename} — not found yet)")
            continue
        with open(path) as f:
            data = json.load(f)
        data["stage"] = display_name
        rows.append(data)
    return rows


def add_derived_metrics(rows):
    """Adds compression ratio, size reduction %, latency improvement %, and
    BLEU-4 difference — all measured relative to the baseline row."""
    baseline = next((r for r in rows if "baseline" in r.get("model", "")), None)
    if baseline is None:
        print("WARNING: no baseline result found — derived metrics will be blank.")
        return rows

    base_size = baseline.get("size_mb")
    base_latency = baseline.get("mean_latency_ms")
    base_bleu = baseline.get("bleu4")

    for r in rows:
        size = r.get("size_mb")
        latency = r.get("mean_latency_ms")
        bleu = r.get("bleu4")

        r["compression_ratio_x"] = round(base_size / size, 2) if size else None
        r["size_reduction_pct"] = round(100 * (1 - size / base_size), 1) if size and base_size else None
        r["latency_improvement_pct"] = (
            round(100 * (base_latency - latency) / base_latency, 1) if latency and base_latency else None
        )
        r["bleu4_diff_vs_baseline"] = round(bleu - base_bleu, 4) if bleu is not None and base_bleu is not None else None

    return rows


def build_comparison_table():
    rows = load_pipeline_results()
    if not rows:
        print("No results found yet — run scripts 01/02/04/08 first.")
        return None

    rows = add_derived_metrics(rows)
    df = pd.DataFrame(rows)

    # Put the most report-relevant columns first; keep everything else after.
    priority_cols = [
        "stage", "model", "size_mb", "compression_ratio_x", "size_reduction_pct",
        "model_footprint_mb", "mean_latency_ms", "latency_improvement_pct",
        "bleu4", "bleu4_diff_vs_baseline", "sparsity_pct",
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    ordered_cols = [c for c in priority_cols if c in df.columns] + other_cols
    df = df[ordered_cols]

    out_csv = RESULTS_DIR / "comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved comparison table to {out_csv}\n")
    print(df.to_string(index=False))
    return df


def plot_comparisons(df):
    if df is None or df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = df["stage"]

    axes[0].bar(labels, df["size_mb"], color="#4C72B0")
    axes[0].set_title("Model Size (MB)")
    axes[0].tick_params(axis="x", rotation=25)

    if "mean_latency_ms" in df.columns:
        axes[1].bar(labels, df["mean_latency_ms"], color="#DD8452")
        axes[1].set_title("Mean Inference Latency (ms)")
        axes[1].tick_params(axis="x", rotation=25)

    if "bleu4" in df.columns:
        axes[2].bar(labels, df["bleu4"], color="#55A868")
        axes[2].set_title("BLEU-4 (accuracy)")
        axes[2].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    out_path = RESULTS_DIR / "comparison_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison plot to {out_path}")


if __name__ == "__main__":
    df = build_comparison_table()
    plot_comparisons(df)
