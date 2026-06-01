from __future__ import annotations

import pandas as pd

from src.config import RAW_DATA_DIR
from src.io_utils import require_columns


RESTAURANT_SOURCE_PATH = RAW_DATA_DIR / "MEM_compstore_restaurants.csv"

RESTAURANT_REQUIRED_COLUMNS = [
    "restaurant_object_key",
    "zip_or_postal_code",
]


def restaurants():
    df_restaurant = pd.read_csv(RESTAURANT_SOURCE_PATH)
    require_columns(
        df_restaurant,
        RESTAURANT_REQUIRED_COLUMNS,
        "Restaurant source data",
    )

    df = df_restaurant.drop_duplicates(subset=["restaurant_object_key"]).copy()
    df["zip_or_postal_code"] = df["zip_or_postal_code"].astype(str).str.zfill(5)
    df = df[df["zip_or_postal_code"].str.match(r"^\d{5}$")].copy()
    df["ZCTA"] = df["zip_or_postal_code"]
    return df
