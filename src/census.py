from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

from src.config import CACHE_DIR


load_dotenv()

API_KEY = os.getenv("API_KEY")
os.makedirs(CACHE_DIR, exist_ok=True)

AGE_FIELDS = {
    "S0101_C01_019E": 21.0,
    "S0101_C01_020E": 29.5,
    "S0101_C01_021E": 39.5,
    "S0101_C01_023E": 59.5,
    "S0101_C01_024E": 72.0,
}


def age_group_from_value(age):
    if pd.isna(age):
        return "unknown"
    if age <= 0 or age > 100:
        return "unknown"
    if age < 30:
        return "young"
    if age < 40:
        return "millennial"
    if age < 55:
        return "adult"
    return "senior"


def age_group_from_mean(age):
    return age_group_from_value(age)


def _normalize_age_group(value):
    text = str(value).strip().lower()
    return text if text and text not in {"nan", "none"} else "unknown"


def _zcta_column(df):
    if "zip code tabulation area" not in df.columns:
        raise KeyError("Census response is missing 'zip code tabulation area'.")
    return "zip code tabulation area"


def safe_request(url, params, retries=3, timeout=30):
    if not API_KEY:
        raise ValueError("API_KEY is missing. Add it to your environment as API_KEY=your_key.")

    for attempt in range(retries):
        try:
            request_url = f"{url}?{urlencode(params)}"
            request = Request(request_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            print(f"API error: {exc.code} | {exc.read().decode('utf-8', errors='ignore')}")
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"Attempt {attempt + 1} failed: {exc}")

        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    raise RuntimeError("All Census API retries failed.")


def load_cache(filename):
    path = CACHE_DIR / filename
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return None


def save_cache(df, filename):
    
    path = CACHE_DIR / filename
    df.to_csv(path, index=False)


def _coerce_cached_cbp_frame(cached):
    frame = cached.copy()
    frame["FIPS"] = frame["FIPS"].astype(str).str.zfill(5)

    for column in ("ESTAB", "EMP"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return (
        frame[["FIPS", "ESTAB", "EMP"]]
        .drop_duplicates(subset=["FIPS"])
        .sort_values("FIPS")
        .reset_index(drop=True)
    )


def _coerce_cached_ethnicity_frame(cached):
    frame = cached.copy()
    frame["ZCTA"] = frame["ZCTA"].astype(str).str.zfill(5)

    cols = ["WHITE_POP", "BLACK_POP", "ASIAN_POP", "HISPANIC_POP"]
    for column in cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return (
        frame[["ZCTA", *cols]]
        .drop_duplicates(subset=["ZCTA"])
        .sort_values("ZCTA")
        .reset_index(drop=True)
    )


def add_age_group(df):
    df = df.copy()
    available_age_cols = [column for column in AGE_FIELDS if column in df.columns]

    if "median_age" in df.columns:
        df["median_age"] = pd.to_numeric(df["median_age"], errors="coerce")
        df.loc[(df["median_age"] <= 0) | (df["median_age"] > 100), "median_age"] = np.nan
        df["age_group"] = df["median_age"].apply(age_group_from_value)
        return df

    if "age_group" in df.columns:
        df["age_group"] = df["age_group"].apply(_normalize_age_group)
        return df

    if "avg_age" in df.columns:
        df["avg_age"] = pd.to_numeric(df["avg_age"], errors="coerce")
        df.loc[(df["avg_age"] <= 0) | (df["avg_age"] > 100), "avg_age"] = np.nan
        df["age_group"] = df["avg_age"].apply(age_group_from_value)
        return df

    if not available_age_cols:
        df["age_group"] = "unknown"
        return df

    age_counts = df[available_age_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    total = age_counts.sum(axis=1)
    weighted_total = sum(
        age_counts[column] * midpoint
        for column, midpoint in AGE_FIELDS.items()
        if column in age_counts.columns
    )
    avg_age = np.where(total > 0, weighted_total / total, np.nan)
    df["age_group"] = pd.Series(avg_age, index=df.index).apply(age_group_from_value)
    return df


def add_avg_age(df):
    return add_age_group(df)


def fetch_population():
    cache_file = "population.csv"
    cached = load_cache(cache_file)
    if cached is not None:
        print("Loaded population from cache")
        return cached

    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "B01003_001E",
        "for": "zip code tabulation area:*",
        "key": API_KEY,
    }

    data = safe_request(url, params)
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={"B01003_001E": "POP"})
    df["POP"] = pd.to_numeric(df["POP"], errors="coerce")
    df["ZCTA"] = df[_zcta_column(df)].astype(str).str.zfill(5)

    result = df[["ZCTA", "POP"]]
    save_cache(result, cache_file)
    return result


def fetch_income():
    cache_file = "income.csv"
    cached = load_cache(cache_file)
    if cached is not None:
        print("Loaded income from cache")
        return cached

    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "B19013_001E",
        "for": "zip code tabulation area:*",
        "key": API_KEY,
    }

    data = safe_request(url, params)
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={"B19013_001E": "MEDIAN_INCOME"})
    df["MEDIAN_INCOME"] = pd.to_numeric(df["MEDIAN_INCOME"], errors="coerce")
    df["ZCTA"] = df[_zcta_column(df)].astype(str).str.zfill(5)

    result = df[["ZCTA", "MEDIAN_INCOME"]]
    save_cache(result, cache_file)
    return result


def fetch_household_size():
    cache_file = "household_size.csv"
    cached = load_cache(cache_file)
    if cached is not None:
        print("Loaded household_size from cache")
        return cached

    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "B25010_001E",
        "for": "zip code tabulation area:*",
        "key": API_KEY,
    }

    data = safe_request(url, params)
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={"B25010_001E": "AVG_HOUSEHOLD_SIZE"})
    df["AVG_HOUSEHOLD_SIZE"] = pd.to_numeric(df["AVG_HOUSEHOLD_SIZE"], errors="coerce")
    df["ZCTA"] = df[_zcta_column(df)].astype(str).str.zfill(5)

    result = df[["ZCTA", "AVG_HOUSEHOLD_SIZE"]]
    save_cache(result, cache_file)
    return result


def fetch_cbp():
    cache_file = "cbp.csv"
    cached = load_cache(cache_file)
    if cached is not None:
        print("Loaded CBP from cache")
        return _coerce_cached_cbp_frame(cached)

    url = "https://api.census.gov/data/2022/cbp"
    params = {
        "get": "ESTAB,EMP,NAICS2017",
        "for": "county:*",
        "in": "state:*",
        "NAICS2017": "7225",
        "key": API_KEY,
    }

    data = safe_request(url, params)
    df = pd.DataFrame(data[1:], columns=data[0])

    df["ESTAB"] = pd.to_numeric(df["ESTAB"], errors="coerce")
    df["EMP"] = pd.to_numeric(df["EMP"], errors="coerce")
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["FIPS"] = df["state"] + df["county"]

    result = (
        df[["FIPS", "ESTAB", "EMP"]]
        .drop_duplicates(subset=["FIPS"])
        .sort_values("FIPS")
        .reset_index(drop=True)
    )
    save_cache(result, cache_file)
    return result


def fetch_ethnicity():
    cache_file = "ethnicity.csv"
    cached = load_cache(cache_file)
    if cached is not None:
        print("Loaded ethnicity from cache")
        return _coerce_cached_ethnicity_frame(cached)

    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": (
            "B02001_002E,"
            "B02001_003E,"
            "B02001_005E,"
            "B03003_003E"
        ),
        "for": "zip code tabulation area:*",
        "key": API_KEY,
    }

    data = safe_request(url, params)
    df = pd.DataFrame(data[1:], columns=data[0])

    df = df.rename(
        columns={
            "B02001_002E": "WHITE_POP",
            "B02001_003E": "BLACK_POP",
            "B02001_005E": "ASIAN_POP",
            "B03003_003E": "HISPANIC_POP",
        }
    )

    cols = ["WHITE_POP", "BLACK_POP", "ASIAN_POP", "HISPANIC_POP"]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ZCTA"] = df[_zcta_column(df)].astype(str).str.zfill(5)

    result = (
        df[["ZCTA", *cols]]
        .drop_duplicates(subset=["ZCTA"])
        .sort_values("ZCTA")
        .reset_index(drop=True)
    )
    save_cache(result, cache_file)
    return result


def fetch_age_distribution():
    cache_file = "age_distribution.csv"
    cached = load_cache(cache_file)
    if cached is not None and {"median_age", "age_group"}.issubset(cached.columns):
        print("Loaded age_distribution from cache")
        cached = add_age_group(cached)
        result = cached[["ZCTA", "median_age", "age_group"]].copy()
        save_cache(result, cache_file)
        return result[["ZCTA", "age_group"]]
    if cached is not None:
        print("Stale age_distribution cache detected; rebuilding")

    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "B01002_001E",
        "for": "zip code tabulation area:*",
        "key": API_KEY,
    }

    data = safe_request(url, params)
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={"B01002_001E": "median_age"})
    df["median_age"] = pd.to_numeric(df["median_age"], errors="coerce")

    df["ZCTA"] = df[_zcta_column(df)].astype(str).str.zfill(5)
    result = add_age_group(df)[["ZCTA", "median_age", "age_group"]]
    save_cache(result, cache_file)
    return result[["ZCTA", "age_group"]]


def fetch_population_growth():
    cache_file = "population_growth.csv"
    cached = load_cache(cache_file)
    if cached is not None:
        print("Loaded population growth from cache")
        return cached

    try:
        url_2022 = "https://api.census.gov/data/2022/acs/acs5"
        params_2022 = {
            "get": "B01003_001E",
            "for": "zip code tabulation area:*",
            "key": API_KEY,
        }
        data_2022 = safe_request(url_2022, params_2022)
        df_2022 = pd.DataFrame(data_2022[1:], columns=data_2022[0])
        df_2022 = df_2022.rename(columns={"B01003_001E": "POP_2022"})

        url_2017 = "https://api.census.gov/data/2017/acs/acs5"
        params_2017 = {
            "get": "B01003_001E",
            "for": "zip code tabulation area:*",
            "key": API_KEY,
        }
        data_2017 = safe_request(url_2017, params_2017)
        df_2017 = pd.DataFrame(data_2017[1:], columns=data_2017[0])
        df_2017 = df_2017.rename(columns={"B01003_001E": "POP_2017"})

        zcta_column = "zip code tabulation area"
        for frame in (df_2022, df_2017):
            frame["ZCTA"] = frame[zcta_column].astype(str).str.zfill(5)

        df = df_2022.merge(df_2017[["ZCTA", "POP_2017"]], on="ZCTA", how="left")
        df["POP_2022"] = pd.to_numeric(df["POP_2022"], errors="coerce")
        df["POP_2017"] = pd.to_numeric(df["POP_2017"], errors="coerce")
        df["population_growth_rate"] = np.where(
            df["POP_2017"] > 0,
            ((df["POP_2022"] - df["POP_2017"]) / df["POP_2017"]) * 100,
            np.nan,
        )

        result = df[["ZCTA", "population_growth_rate"]]
        save_cache(result, cache_file)
        return result
    except Exception as exc:
        print("API failed for population growth, trying cache...")
        cached = load_cache(cache_file)
        if cached is not None:
            return cached
        raise RuntimeError("No cache available and API failed.") from exc


def fetch_demographic_bundle():
    return {
        "cbp": fetch_cbp(),
        "population": fetch_population(),
        "income": fetch_income(),
        "household_size": fetch_household_size(),
        "ethnicity": fetch_ethnicity(),
        "age": fetch_age_distribution(),
        "growth": fetch_population_growth(),
    }
