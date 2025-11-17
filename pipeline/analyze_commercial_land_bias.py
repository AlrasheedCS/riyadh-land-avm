#!/usr/bin/env python3
"""Diagnostics focused on the commercial land models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = SCRIPT_DIR / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = SCRIPT_DIR / "data" / "sa-riyadh-capital" / "out"
MODEL_PATH = (
    DATA_ROOT / "models" / "commercial_land" / "main" / "ensemble" / "pred_test.parquet"
)
OUT_DIR = DATA_ROOT / "commercial_land_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR = DATA_ROOT / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing prediction file: {path}")
    df = pd.read_parquet(path)
    df = df.copy()
    df["sale_price_time_adj"] = pd.to_numeric(df["sale_price_time_adj"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["sale_price_time_adj", "prediction"])
    df = df[df["sale_price_time_adj"] > 0]
    df["ratio"] = df["prediction"] / df["sale_price_time_adj"]
    df["pct_error"] = (df["ratio"] - 1.0) * 100.0
    df["sale_price_million"] = df["sale_price_time_adj"] / 1e6
    df["predicted_million"] = df["prediction"] / 1e6
    df["district_slug"] = (
        df["key"]
        .str.replace("sa_riyadh_", "", regex=False)
        .str.replace("_commercial_land", "", regex=False)
    )
    return df


def summarize_overall(df: pd.DataFrame) -> Dict[str, float]:
    ape = np.abs(df["ratio"] - 1.0)
    return {
        "count": len(df),
        "median_ratio": float(df["ratio"].median()),
        "mean_ratio": float(df["ratio"].mean()),
        "mdape_pct": float(np.median(ape) * 100.0),
        "pct_within_20_pct": float(np.mean(ape <= 0.20) * 100.0),
        "recommended_scalar": float(1.0 / df["ratio"].median())
        if df["ratio"].median() > 0
        else np.nan,
    }


def summarize_by_district(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("district_slug")
        .agg(
            count=("ratio", "size"),
            median_ratio=("ratio", "median"),
            mdape_pct=("pct_error", lambda s: np.median(np.abs(s)) ),
            pct_within_20_pct=("ratio", lambda r: np.mean(np.abs(r - 1.0) <= 0.20) * 100),
            median_sale_million=("sale_price_million", "median"),
        )
        .sort_values("median_ratio")
    )
    agg["share_pct"] = agg["count"] / agg["count"].sum() * 100.0
    return agg.reset_index()


def summarize_by_price_band(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 5, 10, 20, 40, 80, 200]
    labels = ["<5M", "5–10M", "10–20M", "20–40M", "40–80M", "80M+"]
    df = df.copy()
    df["price_band"] = pd.cut(
        df["sale_price_million"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    agg = (
        df.groupby("price_band", observed=True)
        .agg(
            count=("ratio", "size"),
            median_ratio=("ratio", "median"),
            mdape_pct=("pct_error", lambda s: np.median(np.abs(s))),
            pct_within_20_pct=("ratio", lambda r: np.mean(np.abs(r - 1.0) <= 0.20) * 100),
            median_sale_million=("sale_price_million", "median"),
        )
        .reset_index()
    )
    agg["share_pct"] = agg["count"] / agg["count"].sum() * 100.0
    return agg


def plot_district_bars(df: pd.DataFrame) -> Path:
    top = df.sort_values("count", ascending=False).head(15)
    ordered = top.sort_values("median_ratio")
    path = CHART_DIR / "commercial_land_ratio_by_district.png"
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(ordered))))
    ax.barh(ordered["district_slug"], ordered["median_ratio"], color="#d62728")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Median prediction / actual")
    ax.set_ylabel("")
    ax.set_title("Commercial land bias by district (top 15 by volume)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_ratio_vs_price(df: pd.DataFrame) -> Path:
    path = CHART_DIR / "commercial_land_ratio_vs_price.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["sale_price_million"], df["ratio"], s=18, alpha=0.5, color="#1f77b4")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Actual sale price (SAR millions)")
    ax.set_ylabel("Prediction / actual")
    ax.set_title("Commercial land prediction ratio vs. sale price")
    ax.set_ylim(0, min(2.0, df["ratio"].max() * 1.05))
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    df = load_predictions(MODEL_PATH)
    overall = summarize_overall(df)
    district = summarize_by_district(df)
    price_bands = summarize_by_price_band(df)

    overall_path = OUT_DIR / "commercial_land_overall_summary.json"
    pd.Series(overall).to_json(overall_path, indent=2)
    district_path = OUT_DIR / "commercial_land_ratio_by_district.csv"
    district.to_csv(district_path, index=False)
    price_path = OUT_DIR / "commercial_land_ratio_by_price_band.csv"
    price_bands.to_csv(price_path, index=False)

    chart_paths = [
        plot_district_bars(district),
        plot_ratio_vs_price(df),
    ]

    print(f"Overall summary -> {overall_path}")
    print(f"District table -> {district_path}")
    print(f"Price band table -> {price_path}")
    for path in chart_paths:
        print(f"Created chart: {path}")


if __name__ == "__main__":
    main()
