#!/usr/bin/env python3
"""
Prototype a segmented calibration for the commercial land model.

We rescale predictions within each price band using 1 / median(pred/actual),
then compare the pre/post accuracy metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import math
from typing import Dict

import numpy as np
import pandas as pd

# Ensure Matplotlib/headless tooling has a writable cache even if we don't plot.
SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = SCRIPT_DIR / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = SCRIPT_DIR / "data" / "sa-riyadh-capital" / "out"
MODEL_PATH = (
    DATA_ROOT / "models" / "commercial_land" / "main" / "ensemble" / "pred_test.parquet"
)
OUT_DIR = DATA_ROOT / "commercial_land_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR = DATA_ROOT / "charts"
CHART_PATH = CHART_DIR / "commercial_land_within20_before_after.png"
CHART_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR = SCRIPT_DIR / "data" / "sa-riyadh-capital" / "in"
UNIVERSE_PATH = INPUT_DIR / "universe.parquet"

VALUE_BANDS = [0, 5, 10, 20, 40, 80, 200]
VALUE_LABELS = ["<5M", "5–10M", "10–20M", "20–40M", "40–80M", "80M+"]
MIN_COUNT_BAND = 10
MIN_COUNT_DISTRICT = 6
SCALAR_MIN = 0.25
SCALAR_MAX = 4.0


def load_predictions() -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing prediction file: {MODEL_PATH}")
    df = pd.read_parquet(MODEL_PATH)
    df = df.copy()
    if UNIVERSE_PATH.exists():
        universe = pd.read_parquet(UNIVERSE_PATH)[["key", "district_slug"]]
        df = df.merge(universe, on="key", how="left")
    else:
        df["district_slug"] = df["key"].str.extract(r"sa_riyadh_(.+?)_", expand=False)
    df["sale_price_time_adj"] = pd.to_numeric(df["sale_price_time_adj"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["sale_price_time_adj", "prediction"])
    df = df[df["sale_price_time_adj"] > 0]
    df["sale_price_million"] = df["sale_price_time_adj"] / 1e6
    df["ratio"] = df["prediction"] / df["sale_price_time_adj"]
    df["price_band"] = pd.cut(
        df["sale_price_million"],
        bins=VALUE_BANDS,
        labels=VALUE_LABELS,
        include_lowest=True,
        right=False,
    )
    return df


def compute_scalar_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    district_table = (
        df.groupby(["district_slug", "price_band"], observed=True)
        .agg(
            count=("ratio", "size"),
            median_ratio=("ratio", "median"),
            mdape_pct=("ratio", lambda r: np.median(np.abs(r - 1.0)) * 100.0),
            pct_within_20_pct=("ratio", lambda r: np.mean(np.abs(r - 1.0) <= 0.20) * 100.0),
        )
        .reset_index()
    )

    def district_scalar(row):
        if row["count"] < MIN_COUNT_DISTRICT or row["median_ratio"] <= 0:
            return math.nan
        return float(np.clip(1.0 / row["median_ratio"], SCALAR_MIN, SCALAR_MAX))

    district_table["scalar"] = district_table.apply(district_scalar, axis=1)

    price_table = (
        df.groupby("price_band", observed=True)
        .agg(
            count=("ratio", "size"),
            median_ratio=("ratio", "median"),
            mdape_pct=("ratio", lambda r: np.median(np.abs(r - 1.0)) * 100.0),
            pct_within_20_pct=("ratio", lambda r: np.mean(np.abs(r - 1.0) <= 0.20) * 100.0),
        )
        .reset_index()
    )

    def fallback_scalar(row):
        if row["count"] < MIN_COUNT_BAND or row["median_ratio"] <= 0:
            return 1.0
        return float(np.clip(1.0 / row["median_ratio"], SCALAR_MIN, SCALAR_MAX))

    price_table["scalar"] = price_table.apply(fallback_scalar, axis=1)

    fallback_lookup = price_table.set_index("price_band")["scalar"]

    def filled_scalar(row):
        if not math.isnan(row["scalar"]):
            return row["scalar"]
        return fallback_lookup.get(row["price_band"], 1.0)

    district_table["scalar_filled"] = district_table.apply(filled_scalar, axis=1)
    district_table["calibrated_median_ratio"] = (
        district_table["median_ratio"] * district_table["scalar_filled"]
    )

    return district_table, price_table


def apply_calibration(
    df: pd.DataFrame, district_table: pd.DataFrame, price_table: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()
    combo_lookup = (
        district_table.set_index(["district_slug", "price_band"])["scalar_filled"]
    )
    price_lookup = price_table.set_index("price_band")["scalar"]

    combo_index = df.set_index(["district_slug", "price_band"]).index
    df["scalar"] = combo_lookup.reindex(combo_index).to_numpy()
    df["scalar"] = df["scalar"].astype("float64")
    df["scalar"] = df["scalar"].fillna(df["price_band"].map(price_lookup)).fillna(1.0)
    df["prediction_calibrated"] = df["prediction"] * df["scalar"]
    df["ratio_calibrated"] = df["prediction_calibrated"] / df["sale_price_time_adj"]
    return df


def summarize_metrics(df: pd.DataFrame, ratio_col: str) -> Dict[str, float]:
    ratios = df[ratio_col].values
    ape = np.abs(ratios - 1.0)
    return {
        "count": int(len(ratios)),
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(np.mean(ratios)),
        "mdape_pct": float(np.median(ape) * 100.0),
        "mape_pct": float(np.mean(ape) * 100.0),
        "pct_within_10_pct": float(np.mean(ape <= 0.10) * 100.0),
        "pct_within_20_pct": float(np.mean(ape <= 0.20) * 100.0),
    }


def plot_within20(before: pd.Series, after: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.arange(len(before))
    width = 0.35
    ax.bar(x - width / 2, before.values, width=width, label="Before")
    ax.bar(x + width / 2, after.values, width=width, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(before.index, rotation=30, ha="right")
    ax.set_ylabel("% within ±20%")
    ax.set_ylim(0, 100)
    ax.set_title("Commercial land accuracy by price band")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    df = load_predictions()
    district_table, price_table = compute_scalar_tables(df)
    calibrated = apply_calibration(df, district_table, price_table)

    before_metrics = summarize_metrics(df, "ratio")
    after_metrics = summarize_metrics(calibrated, "ratio_calibrated")

    band_metrics_before = (
        df.groupby("price_band", observed=True)["ratio"]
        .apply(lambda r: np.mean(np.abs(r - 1.0) <= 0.20) * 100)
        .rename("pct_within_20")
    )
    band_metrics_after = (
        calibrated.groupby("price_band", observed=True)["ratio_calibrated"]
        .apply(lambda r: np.mean(np.abs(r - 1.0) <= 0.20) * 100)
        .rename("pct_within_20")
    )
    combined_band_table = (
        price_table.merge(
            band_metrics_before.reset_index(),
            on="price_band",
            how="left",
        )
        .merge(
            band_metrics_after.reset_index(),
            on="price_band",
            suffixes=("_before", "_after"),
            how="left",
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    district_table_path = OUT_DIR / "commercial_land_district_price_band_scalars.csv"
    district_table.to_csv(district_table_path, index=False)
    price_table_path = OUT_DIR / "commercial_land_price_band_calibration.csv"
    price_table.to_csv(price_table_path, index=False)

    combined_band_path = OUT_DIR / "commercial_land_price_band_within20.csv"
    combined_band_table.to_csv(combined_band_path, index=False)

    metrics_path = OUT_DIR / "commercial_land_calibration_metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(
            {"before": before_metrics, "after": after_metrics},
            fh,
            indent=2,
        )

    calibrated_path = OUT_DIR / "commercial_land_predictions_calibrated.parquet"
    calibrated.to_parquet(calibrated_path, index=False)

    plot_within20(band_metrics_before, band_metrics_after)

    print(f"Saved district+price scalars -> {district_table_path}")
    print(f"Saved price-band calibration table -> {price_table_path}")
    print(f"Saved band accuracy table -> {combined_band_path}")
    print(f"Saved summary metrics -> {metrics_path}")
    print(f"Calibrated predictions -> {calibrated_path}")
    print(f"Chart -> {CHART_PATH}")


if __name__ == "__main__":
    main()
