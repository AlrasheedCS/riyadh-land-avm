#!/usr/bin/env python3
"""
Convert quarterly Riyadh sales indicator CSVs (Q1–Q3 2025) into pseudo-contract
rows inside sales.csv so the modeling pipeline can consume them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data" / "sa-riyadh-capital" / "in"
SALES_PATH = DATA_DIR / "sales.csv"
UNIVERSE_PATH = DATA_DIR / "universe.parquet"
INDICATOR_GLOB = "Sales transaction indicators in Riyadh*.csv"
ALLOWED_CLASSIFICATIONS = {"تجاري", "سكني"}
LAND_PROPERTY_TYPES = {"أرض"}

QUARTER_MIDPOINTS = {
    1: (2, 15),
    2: (5, 15),
    3: (8, 15),
    4: (11, 15),
}

M2_TO_SQFT = 10.76391041671

PROPERTY_TYPE_EN = {
    "أرض": "land",
    "فيلا": "villa",
    "شقة": "apartment",
    "دور": "single_floor",
    "عمارة": "multi_unit",
    "مبنى": "building",
    "دوبلكس": "duplex",
    "أخرى": "other",
}


def infer_model_group(property_type: str, classification: str) -> str:
    classification = (classification or "").strip()
    prop = (property_type or "").strip()
    if prop == "أرض":
        if classification == "تجاري":
            return "commercial_land"
        if classification == "زراعي":
            return "agricultural_land"
        return "residential_land"
    if prop in {"فيلا", "دور", "دوبلكس"}:
        return "villa"
    if prop in {"شقة"}:
        return "apartment"
    if prop in {"عمارة", "مبنى"}:
        return "multi_unit"
    if classification == "تجاري":
        return "commercial_land"
    if classification == "زراعي":
        return "agricultural_land"
    return "multi_unit"


def slugify(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "unknown_district"
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned.lower()


def build_district_lookup() -> Dict[str, Tuple[str, str]]:
    gdf = gpd.read_parquet(UNIVERSE_PATH)
    lookup = (
        gdf[["district", "district_normalized", "district_slug"]]
        .drop_duplicates(subset=["district"])
        .set_index("district")
        .to_dict("index")
    )
    return {
        name: (info["district_normalized"], info["district_slug"])
        for name, info in lookup.items()
    }


def _dedupe_columns(columns: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    deduped: List[str] = []
    for col in columns:
        base = col.strip()
        count = seen.get(base, -1) + 1
        seen[base] = count
        deduped.append(base if count == 0 else f"{base}_{count}")
    return deduped


def load_indicator_frames() -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(DATA_DIR.glob(INDICATOR_GLOB)):
        df = pd.read_csv(csv_path, encoding="utf-8-sig", na_values=["NULL"])
        df.columns = _dedupe_columns(df.columns.tolist())
        year_col = next(col for col in df.columns if col.startswith("السنة"))
        quarter_col = "الربع"
        district_col = next(col for col in df.columns if col.endswith("الحي"))
        property_col = next(col for col in df.columns if col.startswith("نوع"))
        class_col = next(col for col in df.columns if col.startswith("تصنيف"))
        contracts_col = next(col for col in df.columns if "عدد" in col)
        value_col = next(col for col in df.columns if col.startswith("قيمة"))
        area_col = next(col for col in df.columns if "المساحة" in col)
        avg_col = next(col for col in df.columns if "متوسط" in col)
        max_col = next(col for col in df.columns if "الأعلى" in col)
        min_col = next(col for col in df.columns if "الأدنى" in col)
        df = df.rename(
            columns={
                year_col: "year",
                quarter_col: "quarter",
                district_col: "district",
                property_col: "property_type_ar",
                class_col: "classification_ar",
                contracts_col: "num_contracts",
                value_col: "transaction_value",
                area_col: "area_m2",
                avg_col: "avg_price_per_m2",
                max_col: "max_price_per_m2",
                min_col: "min_price_per_m2",
            }
        )
        df["source_file"] = csv_path.name
        frames.append(df)
    if not frames:
        raise SystemExit(f"No indicator CSV files matching {INDICATOR_GLOB}")
    return frames


def build_indicator_sales(district_lookup: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    frames = load_indicator_frames()
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")
    df["num_contracts"] = pd.to_numeric(df["num_contracts"], errors="coerce").fillna(0).astype(int)
    df["transaction_value"] = pd.to_numeric(df["transaction_value"], errors="coerce")
    df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    df = df.dropna(subset=["year", "quarter", "district", "property_type_ar"]).reset_index(drop=True)
    df = df[df["classification_ar"].isin(ALLOWED_CLASSIFICATIONS)].reset_index(drop=True)
    df = df[df["property_type_ar"].isin(LAND_PROPERTY_TYPES)].reset_index(drop=True)

    records = []
    for idx, row in df.iterrows():
        year = int(row["year"])
        quarter = int(row["quarter"])
        district_name = str(row["district"]).strip()
        prop_ar = str(row["property_type_ar"]).strip()
        classification = str(row["classification_ar"]).strip()
        contracts = max(int(row["num_contracts"]), 1)
        total_value = float(row["transaction_value"]) if not math.isnan(row["transaction_value"]) else 0.0
        total_area_m2 = float(row["area_m2"]) if not math.isnan(row["area_m2"]) else np.nan
        avg_price_per_m2 = float(row["avg_price_per_m2"]) if not math.isnan(row["avg_price_per_m2"]) else np.nan

        normalized, slug = district_lookup.get(district_name, (district_name, slugify(district_name)))

        property_en = PROPERTY_TYPE_EN.get(prop_ar, "other")
        model_group = infer_model_group(prop_ar, classification)
        base_key = f"sa_riyadh_{slug}_{model_group}"

        per_contract_value = total_value / contracts if contracts and total_value else total_value or 0.0
        per_contract_area_m2 = total_area_m2 / contracts if total_area_m2 and contracts else np.nan
        land_area_sqft = per_contract_area_m2 * M2_TO_SQFT if per_contract_area_m2 and model_group.endswith("land") else 0.0
        bldg_area_sqft = per_contract_area_m2 * M2_TO_SQFT if per_contract_area_m2 and not model_group.endswith("land") else 0.0

        month, day = QUARTER_MIDPOINTS.get(quarter, (quarter * 3 - 1, 15))
        sale_date = f"{year:04d}-{month:02d}-{day:02d}"
        sale_quarter = f"{year}-Q{quarter}"

        for n in range(contracts):
            key_sale = f"{base_key}-{year}Q{quarter:01d}I{idx:04d}{n:03d}"
            record = {
                "key_sale": key_sale,
                "key": base_key,
                "district": district_name,
                "district_normalized": normalized,
                "district_slug": slug,
                "property_type_ar": prop_ar,
                "property_type_en": property_en,
                "model_group": model_group,
                "bldg_area_finished_sqft": bldg_area_sqft if not model_group.endswith("land") else 0.0,
                "sale_price_sar": per_contract_value,
                "sale_date": sale_date,
                "sale_year": year,
                "sale_month": month,
                "sale_quarter": sale_quarter,
                "sale_price_time_adj": per_contract_value,
                "land_area_sqft": land_area_sqft if model_group.endswith("land") else 0.0,
                "contracts_source": row["num_contracts"],
                "valid_sale": True,
                "vacant_sale": model_group.endswith("land"),
            }
            records.append(record)
    return pd.DataFrame.from_records(records)


def main() -> None:
    district_lookup = build_district_lookup()
    indicator_sales = build_indicator_sales(district_lookup)
    existing = pd.read_csv(SALES_PATH)
    indicator_sales = indicator_sales.reindex(columns=existing.columns, fill_value=np.nan)
    combined = pd.concat([existing, indicator_sales], ignore_index=True)
    combined = combined.drop_duplicates(subset=["key_sale"], keep="first")
    combined.to_csv(SALES_PATH, index=False)
    print(f"Added {len(indicator_sales)} indicator-based pseudo-sales into {SALES_PATH}")


if __name__ == "__main__":
    main()
