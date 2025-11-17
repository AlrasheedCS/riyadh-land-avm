#!/usr/bin/env python3
"""
Create tables and charts that summarize how well the available model runs performed.

The script looks for pred_test.parquet files under
notebooks/pipeline/data/sa-riyadh-capital/out/models and produces:
1. A CSV table of aggregate accuracy metrics per model variant.
2. Horizontal bar charts for MdAPE and pct. of predictions within ±20%.
3. Scatter + ratio distribution plots for the strongest variant in each model group.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = SCRIPT_DIR / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


import matplotlib

# Matplotlib must run headless inside the Codex runtime.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = SCRIPT_DIR / "data" / "sa-riyadh-capital" / "out"
MODELS_DIR = DATA_ROOT / "models"
CHART_DIR = DATA_ROOT / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = DATA_ROOT / "model_effectiveness_metrics.csv"


def _slugify(label: str) -> str:
    """Generate safe filenames from human friendly labels."""
    return (
        label.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("–", "_")
    )


def _pretty_label(model_id: str) -> str:
    """Convert a model_id like residential_land/main/ensemble into a readable label."""
    parts = model_id.split("/")
    group = parts[0].replace("_", " ").title()
    variant = " / ".join(p.replace("_", " ").title() for p in parts[1:])
    return f"{group} – {variant}"


def load_prediction_frames(models_dir: Path) -> List[pd.DataFrame]:
    """Collect every pred_test parquet file and return tidy frames with metadata."""
    frames: List[pd.DataFrame] = []
    for path in sorted(models_dir.rglob("pred_test.parquet")):
        rel = path.relative_to(models_dir)
        # Expect paths like residential_land/main/ensemble/pred_test.parquet
        model_id = "/".join(rel.parts[:-1])
        df = pd.read_parquet(path)
        if "sale_price_time_adj" not in df or "prediction" not in df:
            continue
        clean = df.copy()
        clean = clean[pd.to_numeric(clean["sale_price_time_adj"], errors="coerce") > 0]
        clean = clean[clean["prediction"].notna()]
        if clean.empty:
            continue
        clean = clean.assign(
            model_id=model_id,
            model_group=rel.parts[0],
            variant="/".join(rel.parts[1:-1]),
        )
        frames.append(clean)
    return frames


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error metrics per model_id."""
    records: List[Dict[str, float]] = []
    df = df.copy()
    df["sale_price_time_adj"] = pd.to_numeric(df["sale_price_time_adj"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["sale_price_time_adj", "prediction"])
    df = df[df["sale_price_time_adj"] > 0]
    df["ratio"] = df["prediction"] / df["sale_price_time_adj"]
    df["abs_pct_error"] = (df["ratio"] - 1.0).abs()
    df["abs_error"] = (df["prediction"] - df["sale_price_time_adj"]).abs()
    df["sq_error"] = (df["prediction"] - df["sale_price_time_adj"]) ** 2

    for model_id, slice_df in df.groupby("model_id"):
        ape = slice_df["abs_pct_error"].values
        ratio = slice_df["ratio"].values
        errors = slice_df["prediction"] - slice_df["sale_price_time_adj"]
        records.append(
            {
                "model_id": model_id,
                "model_group": slice_df["model_group"].iloc[0],
                "variant": slice_df["variant"].iloc[0],
                "count_sales": len(slice_df),
                "mdape_pct": float(np.median(ape) * 100.0),
                "mape_pct": float(np.mean(ape) * 100.0),
                "pct_within_10_pct": float(np.mean(ape <= 0.10) * 100.0),
                "pct_within_20_pct": float(np.mean(ape <= 0.20) * 100.0),
                "mean_ratio": float(np.mean(ratio)),
                "median_ratio": float(np.median(ratio)),
                "mae_sar": float(np.mean(np.abs(errors))),
                "rmse_sar": float(math.sqrt(np.mean(errors**2))),
                "rmse_pct": float(math.sqrt(np.mean(ape**2)) * 100.0),
            }
        )
    summary = pd.DataFrame.from_records(records)
    summary["pretty_label"] = summary["model_id"].map(_pretty_label)
    return summary.sort_values("mdape_pct")


def _format_pct_axis(ax, max_value: float) -> None:
    ax.set_xlim(0, max_value * 1.05)
    ax.set_xlabel("Percent")
    ax.grid(axis="x", linestyle=":", alpha=0.4)


def plot_mdape(summary: pd.DataFrame) -> Path:
    """Horizontal bar chart ordered by MdAPE."""
    path = CHART_DIR / "mdape_by_model.png"
    ordered = summary.sort_values("mdape_pct")
    fig_height = max(2.5, 0.35 * len(ordered))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(ordered["pretty_label"], ordered["mdape_pct"], color="#1f77b4")
    ax.set_xlabel("MdAPE (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.invert_yaxis()
    ax.set_title("Median absolute percent error by model variant")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_within_20(summary: pd.DataFrame) -> Path:
    """Horizontal bar chart for % of sales predicted within ±20%."""
    path = CHART_DIR / "within_20pct_by_model.png"
    ordered = summary.sort_values("pct_within_20_pct", ascending=False)
    fig_height = max(2.5, 0.35 * len(ordered))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(ordered["pretty_label"], ordered["pct_within_20_pct"], color="#2ca02c")
    ax.set_xlabel("% of sales within ±20%")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.invert_yaxis()
    ax.set_title("Share of back-test sales predicted within ±20%")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_scatter(df: pd.DataFrame, summary_row: pd.Series) -> Path:
    """Scatter plot for actual vs predicted SAR millions."""
    pretty_label = summary_row["pretty_label"]
    slug = _slugify(pretty_label)
    path = CHART_DIR / f"scatter_{slug}.png"
    subset = df[df["model_id"] == summary_row["model_id"]].copy()
    actual = subset["sale_price_time_adj"] / 1e6
    predicted = subset["prediction"] / 1e6
    lim = max(actual.max(), predicted.max()) * 1.05
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, predicted, s=16, alpha=0.6, color="#ff7f0e", edgecolor="none")
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.0, label="Perfect fit")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual price (SAR millions)")
    ax.set_ylabel("Predicted price (SAR millions)")
    ax.set_title(f"{pretty_label}\nActual vs predicted")
    annotation = (
        f"MdAPE: {summary_row['mdape_pct']:.1f}%\n"
        f"Within ±20%: {summary_row['pct_within_20_pct']:.0f}%\n"
        f"N = {int(summary_row['count_sales'])}"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_ratio_hist(df: pd.DataFrame, summary_row: pd.Series) -> Path:
    """Histogram of prediction ratios for a specific model."""
    pretty_label = summary_row["pretty_label"]
    slug = _slugify(pretty_label)
    path = CHART_DIR / f"ratio_hist_{slug}.png"
    subset = df[df["model_id"] == summary_row["model_id"]]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(
        subset["ratio"],
        bins=30,
        color="#9467bd",
        edgecolor="white",
        alpha=0.9,
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prediction / actual")
    ax.set_ylabel("Contracts")
    ax.set_title(f"{pretty_label}\nDistribution of prediction ratios")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    frames = load_prediction_frames(MODELS_DIR)
    if not frames:
        raise SystemExit("No pred_test.parquet files were found under the models directory.")
    predictions = pd.concat(frames, ignore_index=True)
    predictions["sale_price_time_adj"] = pd.to_numeric(
        predictions["sale_price_time_adj"], errors="coerce"
    )
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="coerce")
    predictions = predictions.dropna(subset=["sale_price_time_adj", "prediction"])
    predictions = predictions[predictions["sale_price_time_adj"] > 0]
    predictions["ratio"] = predictions["prediction"] / predictions["sale_price_time_adj"]

    summary = compute_summary(predictions)
    summary.to_csv(SUMMARY_CSV, index=False)

    chart_paths = [
        plot_mdape(summary),
        plot_within_20(summary),
    ]

    # Highlight the best (lowest MdAPE) variant in each model_group.
    highlighted = summary.sort_values("mdape_pct").groupby("model_group", as_index=False).first()
    for _, row in highlighted.iterrows():
        chart_paths.append(plot_scatter(predictions, row))
        chart_paths.append(plot_ratio_hist(predictions, row))

    print(f"Wrote summary metrics to {SUMMARY_CSV}")
    for path in chart_paths:
        print(f"Created chart: {path}")


if __name__ == "__main__":
    main()
