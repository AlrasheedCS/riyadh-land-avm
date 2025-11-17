#!/usr/bin/env python3
"""
Augment the Riyadh universe & sales tables with spatial preference features:

- Distance to King Fahd Road (meters) + exponential proximity score.
- Binary flag for parcels north of Riyadh's CBD and a continuous north-distance metric.
- Centroid latitude/longitude to make downstream feature engineering easier.

Run this script any time the base `universe.parquet` / `sales.csv` are regenerated.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString


BASE_DIR = Path(__file__).resolve().parent / "data" / "sa-riyadh-capital" / "in"
UNIVERSE_PATH = BASE_DIR / "universe.parquet"
SALES_PATH = BASE_DIR / "sales.csv"
ALLOWED_MODEL_GROUPS = {"residential_land", "commercial_land"}
FEATURE_COLUMNS = [
    "key",
    "centroid_lat",
    "centroid_lon",
    "north_of_northern_ring_flag",
    "submarket",
    "dist_to_northern_ring_m",
]

DEPRECATED_FEATURES = [
    "dist_to_king_fahd_m",
    "king_fahd_proximity_score",
    "near_king_fahd_flag",
    "north_of_northern_ring_distance_km",
    "north_of_riyadh_flag",
    "north_distance_km",
]
DISTRICT_STAT_COLUMNS = [
    "district_avg_sale_price",
    "district_median_sale_price",
    "district_avg_price_per_sqft",
    "district_sales_count",
]
SUBMARKET_BANDS = [
    ("south", -90.0, 24.60),
    ("center", 24.60, 24.85),
    ("north", 24.85, 90.0),
]

NORTHERN_RING_LINE = LineString(
    [
        (46.470, 24.846),
        (46.520, 24.853),
        (46.585, 24.860),
        (46.650, 24.864),
        (46.720, 24.868),
        (46.790, 24.872),
        (46.860, 24.874),
        (46.930, 24.872),
        (47.000, 24.868),
    ]
)

NORTHERN_RING_LAT_THRESHOLD = float(
    np.mean([lat for _, lat in NORTHERN_RING_LINE.coords])
)
CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:32638"  # UTM Zone 38N covers Riyadh and gives meter-based distances.


def load_universe() -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(UNIVERSE_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    else:
        gdf = gdf.to_crs(CRS_WGS84)
    return gdf


def compute_spatial_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf_metric = gdf.to_crs(CRS_METRIC)
    centroids_metric = gdf_metric.geometry.centroid
    centroids_wgs = gpd.GeoSeries(centroids_metric, crs=CRS_METRIC).to_crs(CRS_WGS84)
    gdf["centroid_lat"] = centroids_wgs.y
    gdf["centroid_lon"] = centroids_wgs.x

    gdf["north_of_northern_ring_flag"] = (
        gdf["centroid_lat"] >= NORTHERN_RING_LAT_THRESHOLD
    ).astype(int)
    northern_ring_geom = (
        gpd.GeoSeries([NORTHERN_RING_LINE], crs=CRS_WGS84).to_crs(CRS_METRIC).iloc[0]
    )
    gdf["dist_to_northern_ring_m"] = gdf_metric.geometry.distance(northern_ring_geom)
    def classify_submarket(lat: float) -> str:
        for name, lo, hi in SUBMARKET_BANDS:
            if lo <= lat < hi:
                return name
        return "unknown"

    gdf["submarket"] = gdf["centroid_lat"].apply(classify_submarket)
    return gdf


def compute_district_stats(sales_df: pd.DataFrame):
    df = sales_df.copy()
    df = df[df["model_group"].isin(ALLOWED_MODEL_GROUPS)].copy()
    df["sale_price_time_adj"] = pd.to_numeric(df["sale_price_time_adj"], errors="coerce")
    df["land_area_sqft"] = pd.to_numeric(df["land_area_sqft"], errors="coerce")
    df = df[(df["sale_price_time_adj"] > 0) & (df["land_area_sqft"] > 0)]
    df["price_per_sqft"] = df["sale_price_time_adj"] / df["land_area_sqft"]

    by_combo = (
        df.groupby(["district_slug", "model_group"])
        .agg(
            district_avg_sale_price=("sale_price_time_adj", "mean"),
            district_median_sale_price=("sale_price_time_adj", "median"),
            district_avg_price_per_sqft=("price_per_sqft", "mean"),
            district_sales_count=("sale_price_time_adj", "size"),
        )
        .reset_index()
    )

    overall = (
        df.groupby("district_slug")
        .agg(
            district_avg_sale_price=("sale_price_time_adj", "mean"),
            district_median_sale_price=("sale_price_time_adj", "median"),
            district_avg_price_per_sqft=("price_per_sqft", "mean"),
            district_sales_count=("sale_price_time_adj", "size"),
        )
        .reset_index()
    )

    return by_combo, overall


def merge_district_stats(
    frame: pd.DataFrame,
    stats_by_combo: pd.DataFrame,
    stats_overall: pd.DataFrame,
) -> pd.DataFrame:
    drop_cols = [
        col
        for col in frame.columns
        if any(
            col == stat
            or col.startswith(f"{stat}_")
            or col.endswith(f"_{suffix}")
            for stat in DISTRICT_STAT_COLUMNS
            for suffix in ["x", "y"]
        )
    ]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)
    merged = frame.merge(stats_by_combo, on=["district_slug", "model_group"], how="left")
    overall_map = stats_overall.set_index("district_slug")
    for col in DISTRICT_STAT_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
        if col in overall_map.columns:
            merged[col] = merged[col].fillna(merged["district_slug"].map(overall_map[col]))
    return merged


def update_sales(
    features: pd.DataFrame,
    sales_df: pd.DataFrame,
    stats_by_combo: pd.DataFrame,
    stats_overall: pd.DataFrame,
) -> None:
    sales = sales_df.copy()
    sales = sales[sales["model_group"].isin(ALLOWED_MODEL_GROUPS)].copy()
    sales["parcel_id"] = (
        "parcel-"
        + sales["district_slug"].fillna("unknown")
        + "-"
        + sales["model_group"].fillna("unknown")
        + "-"
        + sales.groupby(["district_slug", "model_group"]).cumcount().astype(str)
    )
    feature_cols = [col for col in FEATURE_COLUMNS[1:] if col != "submarket"]
    existing_feature_cols = [col for col in feature_cols if col in sales.columns]
    if existing_feature_cols:
        sales = sales.drop(columns=existing_feature_cols)
    existing_stat_cols = [col for col in DISTRICT_STAT_COLUMNS if col in sales.columns]
    if existing_stat_cols:
        sales = sales.drop(columns=existing_stat_cols)

    enriched = sales.merge(features, on="key", how="left")
    enriched = merge_district_stats(enriched, stats_by_combo, stats_overall)
    if enriched[feature_cols].isna().any().any():
        district_medians = enriched.groupby("district_slug")[feature_cols].transform("median")
        enriched[feature_cols] = enriched[feature_cols].fillna(district_medians)
    enriched.to_csv(SALES_PATH, index=False)


def main() -> None:
    sales_df = pd.read_csv(SALES_PATH)
    stats_by_combo, stats_overall = compute_district_stats(sales_df)

    gdf = load_universe()
    gdf = gdf[gdf["model_group"].isin(ALLOWED_MODEL_GROUPS)].copy()
    enriched_universe = compute_spatial_features(gdf)
    enriched_universe = merge_district_stats(enriched_universe, stats_by_combo, stats_overall)
    enriched_universe = enriched_universe.drop(
        columns=[col for col in DEPRECATED_FEATURES if col in enriched_universe.columns],
        errors="ignore",
    )
    enriched_universe["parcel_id"] = (
        "parcel-"
        + enriched_universe["district_slug"].fillna("unknown")
        + "-"
        + enriched_universe["model_group"].fillna("unknown")
    )
    enriched_universe.to_parquet(UNIVERSE_PATH, index=False)

    feature_df = enriched_universe[FEATURE_COLUMNS].copy()
    update_sales(feature_df, sales_df, stats_by_combo, stats_overall)
    sales_df = pd.read_csv(SALES_PATH)
    sales_df = sales_df.drop(
        columns=[col for col in DEPRECATED_FEATURES if col in sales_df.columns],
        errors="ignore",
    )
    sales_df.to_csv(SALES_PATH, index=False)
    print(
        f"Updated {UNIVERSE_PATH} and {SALES_PATH} with spatial + district average features."
    )
...


if __name__ == "__main__":
    main()
SUBMARKET_BANDS = [
    ("south", -90.0, 24.60),
    ("center", 24.60, 24.85),
    ("north", 24.85, 90.0),
]
