from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ZIP_GEOGRAPHY_PATH
from src.io_utils import require_columns


ZIP_GEOGRAPHY_REQUIRED_COLUMNS = [
    "zip",
    "city",
    "state_id",
    "state_name",
    "zcta",
    "population",
    "density",
    "county_name",
]


def load_zip_geography(path: Path | None = None) -> pd.DataFrame:
    actual_path = ZIP_GEOGRAPHY_PATH if path is None else path
    geo = pd.read_csv(actual_path, dtype=str)
    geo.columns = geo.columns.str.strip().str.lower()
    require_columns(geo, ZIP_GEOGRAPHY_REQUIRED_COLUMNS, "ZIP geography data")

    geo = geo.rename(
        columns={
            "zip": "ZCTA",
            "city": "CITY",
            "state_id": "STATE",
            "state_name": "STATE_NAME",
            "county_name": "COUNTY",
            "county_fips": "COUNTY_FIPS",
            "lat": "LAT",
            "lng": "LNG",
            "population": "ZIP_POPULATION",
            "density": "POP_DENSITY",
            "zcta": "ZCTA_FLAG",
            "parent_zcta": "PARENT_ZCTA",
            "zcta_state_code": "ZCTA_STATE_CODE",
            "zcta_county_code": "ZCTA_COUNTY_CODE",
            "zcta_geoid": "ZCTA_GEOID",
            "county_geoid": "COUNTY_GEOID",
        }
    )

    geo["ZCTA"] = geo["ZCTA"].astype(str).str.zfill(5)
    geo["ZCTA_FLAG"] = geo["ZCTA_FLAG"].astype(str).str.strip().str.upper()
    geo = geo[geo["ZCTA_FLAG"].isin({"TRUE", "1", "YES"})].copy()

    for column in [
        "LAT",
        "LNG",
        "ZIP_POPULATION",
        "POP_DENSITY",
        "COUNTY_FIPS",
        "LAND_AREA_SQKM",
    ]:
        if column in geo.columns:
            geo[column] = pd.to_numeric(geo[column], errors="coerce")

    if "COUNTY_FIPS" in geo.columns:
        county_fips = pd.to_numeric(geo["COUNTY_FIPS"], errors="coerce")
        geo["COUNTY_FIPS"] = county_fips.apply(
            lambda value: f"{int(value):05d}" if pd.notna(value) else pd.NA
        )

    if "ZCTA_STATE_CODE" in geo.columns:
        geo["ZCTA_STATE_CODE"] = geo["ZCTA_STATE_CODE"].astype(str).str.zfill(2)

    if "ZCTA_COUNTY_CODE" in geo.columns:
        geo["ZCTA_COUNTY_CODE"] = geo["ZCTA_COUNTY_CODE"].astype(str).str.zfill(3)

    return geo


def load_zip_base(path: Path | None = None) -> pd.DataFrame:
    geo = load_zip_geography(path)
    columns = [
        column
        for column in [
            "ZCTA",
            "CITY",
            "STATE",
            "STATE_NAME",
            "COUNTY",
            "COUNTY_FIPS",
            "LAT",
            "LNG",
            "ZIP_POPULATION",
            "POP_DENSITY",
            "LAND_AREA_SQKM",
            "ZCTA_STATE_CODE",
            "ZCTA_COUNTY_CODE",
            "ZCTA_GEOID",
            "COUNTY_GEOID",
        ]
        if column in geo.columns
    ]
    return geo[columns].copy()


def _area_type_from_density(pop_density):
    if pd.isna(pop_density):
        return "UNKNOWN"
    if pop_density >= 3000:
        return "URBAN"
    if pop_density >= 1000:
        return "SUBURBAN"
    return "RURAL"


def get_zip_to_fips(path=None):
    base = load_zip_base(path)
    require_columns(
        base,
        ["ZCTA", "COUNTY_FIPS"],
        "ZIP geography data",
    )

    mapping = base[["ZCTA", "COUNTY_FIPS"]].copy()
    mapping["ZCTA"] = mapping["ZCTA"].astype(str).str.zfill(5)
    county_fips = pd.to_numeric(mapping["COUNTY_FIPS"], errors="coerce")
    mapping["COUNTY_FIPS"] = county_fips.apply(
        lambda value: f"{int(value):05d}" if pd.notna(value) else pd.NA
    )
    mapping = mapping.rename(columns={"ZCTA": "zcta"})

    return (
        mapping[["zcta", "COUNTY_FIPS"]]
        .dropna(subset=["zcta", "COUNTY_FIPS"])
        .drop_duplicates(subset=["zcta"])
        .sort_values("zcta")
        .reset_index(drop=True)
    )


def build_geographic_intelligence(
    population_df: pd.DataFrame,
    restaurants_df: pd.DataFrame,
    path=None,
):
    base = load_zip_base(path)
    require_columns(base, ["ZCTA", "CITY", "STATE", "COUNTY"], "ZIP geography data")
    require_columns(population_df, ["ZCTA", "POP"], "Population data")

    geo = base.copy()
    geo["ZCTA"] = geo["ZCTA"].astype(str).str.zfill(5)

    pop = population_df.copy()
    pop["ZCTA"] = pop["ZCTA"].astype(str).str.zfill(5)
    pop["POP"] = pd.to_numeric(pop["POP"], errors="coerce")

    if "ZCTA" in restaurants_df.columns:
        restaurant_geo = "ZCTA"
    elif "zip_or_postal_code" in restaurants_df.columns:
        restaurant_geo = "zip_or_postal_code"
    else:
        raise ValueError("Restaurant data must include ZCTA or zip_or_postal_code.")

    restaurant_counts = (
        restaurants_df.assign(
            **{
                restaurant_geo: restaurants_df[restaurant_geo]
                .astype(str)
                .str.zfill(5)
            }
        )
        .groupby(restaurant_geo)
        .size()
        .reset_index(name="RESTAURANT_COUNT")
        .rename(columns={restaurant_geo: "ZCTA"})
    )

    df = geo.merge(pop[["ZCTA", "POP"]], on="ZCTA", how="left")
    df = df.merge(restaurant_counts, on="ZCTA", how="left")

    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LNG"] = pd.to_numeric(df["LNG"], errors="coerce")
    df["POP_DENSITY"] = pd.to_numeric(df["POP_DENSITY"], errors="coerce")
    df["RESTAURANT_COUNT"] = pd.to_numeric(df["RESTAURANT_COUNT"], errors="coerce").fillna(0)
    df["COMPETITOR_DENSITY"] = np.where(
        df["POP"] > 0,
        (df["RESTAURANT_COUNT"] / df["POP"]) * 1000,
        np.nan,
    )
    df["AREA_TYPE"] = df["POP_DENSITY"].apply(_area_type_from_density)

    return df[
        [
            "ZCTA",
            "CITY",
            "STATE",
            "COUNTY",
            "LAT",
            "LNG",
            "POP",
            "POP_DENSITY",
            "AREA_TYPE",
            "RESTAURANT_COUNT",
            "COMPETITOR_DENSITY",
        ]
    ].copy()


__all__ = [
    "load_zip_geography",
    "load_zip_base",
    "get_zip_to_fips",
    "build_geographic_intelligence",
]
