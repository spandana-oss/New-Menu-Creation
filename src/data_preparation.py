from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    DCA_RESTAURANT_SOURCE_PATH,
    MEMPHIS_RESTAURANT_SOURCE_PATH,
)
from src.io_utils import require_columns


RESTAURANT_REQUIRED_COLUMNS = [
    "restaurant_object_key",
    "zip_or_postal_code",
]

RESTAURANT_SOURCE_BY_LOCATION = {
    "memphis": MEMPHIS_RESTAURANT_SOURCE_PATH,
    "dca": DCA_RESTAURANT_SOURCE_PATH,
}


def _load_restaurant_source(path, location):
    df_restaurant = pd.read_csv(path)
    require_columns(
        df_restaurant,
        RESTAURANT_REQUIRED_COLUMNS,
        f"Restaurant source data ({Path(path).name})",
    )

    df_restaurant = df_restaurant.copy()
    df_restaurant["restaurant_source_file"] = Path(path).name
    df_restaurant["restaurant_location"] = location
    return df_restaurant


def restaurants(location="memphis"):
    location_key = str(location).strip().lower()
    if location_key not in RESTAURANT_SOURCE_BY_LOCATION:
        raise ValueError(
            f"Unknown restaurant location: {location}. "
            f"Expected one of: {', '.join(sorted(RESTAURANT_SOURCE_BY_LOCATION))}."
        )

    path = RESTAURANT_SOURCE_BY_LOCATION[location_key]
    df_restaurant = _load_restaurant_source(path, location_key)

    df = df_restaurant.drop_duplicates(subset=["restaurant_object_key"]).copy()
    df["zip_or_postal_code"] = df["zip_or_postal_code"].astype(str).str.zfill(5)
    df = df[df["zip_or_postal_code"].str.match(r"^\d{5}$")].copy()
    df["ZCTA"] = df["zip_or_postal_code"]
    return df
