from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.io_utils import require_columns


MERGED_DATA_PATH = PROCESSED_DATA_DIR / "merged_data.csv"
CUSTOMER_INTELLIGENCE_PATH = PROCESSED_DATA_DIR / "customer_intelligence.xlsx"
RESTAURANT_SOURCE_PATH = RAW_DATA_DIR / "MEM_compstore_restaurants.csv"
CUSTOMER_INTELLIGENCE_SHEET_NAME = "Restaurant Intelligence"

BASE_COLUMNS = [
    "restaurant_object_key",
    "restaurant_name",
    "zip_or_postal_code",
    "ZCTA",
    "FIPS",
    "CITY",
    "STATE",
    "COUNTY",
    "LAT",
    "LNG",
    "POP",
    "AVG_HOUSEHOLD_SIZE",
    "WHITE_POP",
    "BLACK_POP",
    "ASIAN_POP",
    "HISPANIC_POP",
    "POP_DENSITY",
    "AREA_TYPE",
    "MEDIAN_INCOME",
    "AGE_18_24",
    "AGE_25_34",
    "AGE_35_44",
    "AGE_55_64",
    "AGE_65_PLUS",
    "population_growth_rate",
    "ESTAB",
    "EMP",
    "RESTAURANT_COUNT",
    "COMPETITOR_DENSITY",
    "cluster_id",
    "market_segment",
    "restaurant category",
    "price positioning",
]

EXTRA_COLUMNS = [
    "google_review_rating",
    "generated_customer_reviews",
    "customer_feedback_summary",
    "healthy_food_trends",
    "local_food_preferences",
    "demographic_based_food_insights",
]

OUTPUT_COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS

CATEGORY_FOOD_PHRASES = {
    "Wings-focused restaurant": "spicy wings, baked wings, and hearty shareable sides",
    "Seafood-focused restaurant": "grilled seafood, lighter preparations, and fresh sauces",
    "Chicken-focused restaurant": "crispy or grilled chicken combos with familiar sides",
    "Beverage-led restaurant": "coffee, smoothies, teas, and refreshing drinks",
    "Health-oriented restaurant": "salads, bowls, wraps, and fresh ingredients",
    "Fast-service restaurant": "quick, customizable meals and convenient combos",
    "General restaurant": "familiar comfort meals and flexible combo options",
}

SEGMENT_REVIEW_PHRASES = {
    "Healthy Lifestyle Market": "health-forward and functional meal demand",
    "Youth Convenience Market": "speed, portability, and late-hour convenience",
    "Family Value Market": "shareable portions, dependable pricing, and simple favorites",
    "Social Dining Market": "shareable items, wings, and beverage-friendly occasions",
    "Premium Dining Market": "elevated ingredients, polished service, and premium experiences",
    "Comfort Dining Market": "familiar comfort dishes and a relaxed dining pace",
}

SEGMENT_FEEDBACK_NOTES = {
    "Healthy Lifestyle Market": (
        "Feedback suggests stronger demand for healthier side options and lighter meals."
    ),
    "Youth Convenience Market": (
        "Feedback suggests customers value speed, portable meals, and convenient dayparts."
    ),
    "Family Value Market": (
        "Feedback suggests families want bigger combos, dependable pricing, and kid-friendly sides."
    ),
    "Social Dining Market": (
        "Feedback suggests guests respond to shareable items, beverage pairings, and group-friendly plates."
    ),
    "Premium Dining Market": (
        "Feedback suggests diners expect elevated ingredients, premium presentation, and polished service."
    ),
    "Comfort Dining Market": (
        "Feedback suggests comfort dishes, familiar flavors, and calmer service matter most."
    ),
}

SEGMENT_AUDIENCES = {
    "Healthy Lifestyle Market": "health-conscious households",
    "Youth Convenience Market": "younger customers",
    "Family Value Market": "families",
    "Social Dining Market": "social diners",
    "Premium Dining Market": "affluent diners",
    "Comfort Dining Market": "older households",
}

SEGMENT_HEALTHY_TRENDS = {
    "Wings-focused restaurant": (
        "Consumers are showing interest in baked wings, reduced-sodium sauces, and protein-focused meal options."
    ),
    "Seafood-focused restaurant": (
        "Consumers are showing interest in grilled seafood, lighter preparations, and fresh ingredient transparency."
    ),
    "Chicken-focused restaurant": (
        "Consumers are showing interest in grilled chicken, seasoning variety, and lighter combo builds."
    ),
    "Beverage-led restaurant": (
        "Consumers are showing interest in lower-sugar drinks, functional beverages, and all-day refreshment options."
    ),
    "Health-oriented restaurant": (
        "Consumers are showing interest in bowls, salads, plant-forward dishes, and functional beverages."
    ),
    "Fast-service restaurant": (
        "Consumers are showing interest in lighter quick-service combos, customizable sides, and portable options."
    ),
    "General restaurant": (
        "Consumers are showing interest in flexible menu mixes that balance comfort, value, and freshness."
    ),
}

SEGMENT_LOCAL_PREFERENCES = {
    "Wings-focused restaurant": "late-night combo meals and spicy wings",
    "Seafood-focused restaurant": "grilled seafood and lighter combos",
    "Chicken-focused restaurant": "crispy chicken combos and shareable sides",
    "Beverage-led restaurant": "coffee, smoothies, and quick refreshment stops",
    "Health-oriented restaurant": "fresh bowls, salads, and lighter meals",
    "Fast-service restaurant": "quick meals and portable combos",
    "General restaurant": "familiar comfort meals and flexible combo options",
}

RACE_LABELS = {
    "WHITE_POP": "white",
    "BLACK_POP": "black",
    "ASIAN_POP": "Asian",
    "HISPANIC_POP": "Hispanic",
}


def _load_merged_data(path: Path = MERGED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_restaurant_source(path: Path = RESTAURANT_SOURCE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalize_text(value, default: str) -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _location_text(row: pd.Series) -> str:
    city = _normalize_text(row.get("CITY"), "")
    state = _normalize_text(row.get("STATE"), "")
    if city and state:
        return f"{city}, {state}"
    if city:
        return city
    if state:
        return state
    return "the area"


def _dominant_race(row: pd.Series) -> str:
    race_counts = pd.to_numeric(
        row.reindex(RACE_LABELS.keys()),
        errors="coerce",
    ).fillna(0)
    if race_counts.max() <= 0:
        return "diverse"
    dominant_column = race_counts.idxmax()
    return RACE_LABELS.get(dominant_column, "diverse")


def _dominant_age_phrase(row: pd.Series) -> str:
    age_columns = [
        "AGE_18_24",
        "AGE_25_34",
        "AGE_35_44",
        "AGE_55_64",
        "AGE_65_PLUS",
    ]
    age_values = pd.to_numeric(row.reindex(age_columns), errors="coerce").fillna(0)
    if age_values.max() <= 0:
        return "a balanced age mix"

    dominant_column = age_values.idxmax()
    if dominant_column == "AGE_65_PLUS":
        return "strong representation from seniors demographics"
    if dominant_column in {"AGE_18_24", "AGE_25_34"}:
        return "a strong presence of younger adults"
    if dominant_column in {"AGE_35_44", "AGE_55_64"}:
        return "a balanced mix of working-age households"
    return "a balanced age mix"


def _rating_text(rating: float | int | str | None) -> str:
    rating_value = pd.to_numeric(pd.Series([rating]), errors="coerce").iloc[0]
    if pd.isna(rating_value):
        return "Guests share mixed views, with room to sharpen the experience."
    if rating_value >= 4.2:
        return "The food was flavorful and the portions were generous. Staff stayed attentive during busy hours."
    if rating_value >= 3.7:
        return (
            "Guests appreciate the menu and the experience overall. Service stays solid, "
            "though peak-hour consistency could improve."
        )
    return (
        "Reviews suggest the concept needs stronger consistency. Guests are asking for "
        "better service speed and clearer value."
    )


def _generated_customer_reviews(row: pd.Series) -> str:
    location = _location_text(row)
    category = _normalize_text(row.get("restaurant category"), "General restaurant")
    market_segment = _normalize_text(row.get("market_segment"), "regional market")
    category_phrase = CATEGORY_FOOD_PHRASES.get(category, CATEGORY_FOOD_PHRASES["General restaurant"])
    segment_phrase = SEGMENT_REVIEW_PHRASES.get(
        market_segment,
        "balanced local demand",
    )

    return (
        f"{_rating_text(row.get('google_review_rating'))} "
        f"In {location}, customers particularly enjoy {category_phrase}. "
        f"The restaurant fits well within the {market_segment.lower()}."
        f" That combination lines up with {segment_phrase}."
    )


def _customer_feedback_summary(row: pd.Series) -> str:
    rating_value = pd.to_numeric(pd.Series([row.get("google_review_rating")]), errors="coerce").iloc[0]
    if pd.isna(rating_value):
        lead = "Feedback is balanced, with a need to sharpen the concept and service rhythm."
    elif rating_value >= 4.2:
        lead = "Most reviews highlight strong value-for-money perception."
    elif rating_value >= 3.7:
        lead = "Positive comments center on flavor and service, though some guests mention inconsistent peak-hour execution."
    else:
        lead = "Feedback suggests tighter operations, better pacing, and a clearer value message are needed."

    segment = _normalize_text(row.get("market_segment"), "Regional Market")
    segment_note = SEGMENT_FEEDBACK_NOTES.get(
        segment,
        "Feedback suggests the concept should stay closely aligned with local demand patterns.",
    )
    return f"{lead} {segment_note}"


def _healthy_food_trends(row: pd.Series) -> str:
    category = _normalize_text(row.get("restaurant category"), "General restaurant")
    return SEGMENT_HEALTHY_TRENDS.get(
        category,
        SEGMENT_HEALTHY_TRENDS["General restaurant"],
    )


def _local_food_preferences(row: pd.Series) -> str:
    location = _location_text(row)
    category = _normalize_text(row.get("restaurant category"), "General restaurant")
    preference_phrase = SEGMENT_LOCAL_PREFERENCES.get(
        category,
        SEGMENT_LOCAL_PREFERENCES["General restaurant"],
    )
    audience = SEGMENT_AUDIENCES.get(
        _normalize_text(row.get("market_segment"), "Regional Market"),
        "local diners",
    )
    return f"Residents in {location} tend to prefer {preference_phrase}, especially among {audience}."


def _demographic_based_food_insights(row: pd.Series) -> str:
    race_phrase = _dominant_race(row)
    age_phrase = _dominant_age_phrase(row)
    category = _normalize_text(row.get("restaurant category"), "General restaurant")
    category_phrase = CATEGORY_FOOD_PHRASES.get(category, CATEGORY_FOOD_PHRASES["General restaurant"])
    return (
        f"The surrounding population is primarily {race_phrase} with {age_phrase}. "
        f"Customers in this ZIP code show higher engagement with {category_phrase} offerings "
        f"that balance flavor, affordability, and convenience."
    )


def _attach_google_review_ratings(df: pd.DataFrame) -> pd.DataFrame:
    if "google_review_rating" in df.columns:
        return df.copy()

    if not RESTAURANT_SOURCE_PATH.exists():
        enriched = df.copy()
        enriched["google_review_rating"] = pd.NA
        return enriched

    restaurant_source = _load_restaurant_source()
    require_columns(
        restaurant_source,
        ["restaurant_object_key", "rating_value"],
        "Restaurant source data",
    )

    ratings = (
        restaurant_source[["restaurant_object_key", "rating_value"]]
        .drop_duplicates(subset=["restaurant_object_key"])
        .rename(columns={"rating_value": "google_review_rating"})
    )

    enriched = df.merge(ratings, on="restaurant_object_key", how="left")
    enriched["google_review_rating"] = pd.to_numeric(
        enriched["google_review_rating"],
        errors="coerce",
    )
    return enriched


def build_customer_intelligence(df: pd.DataFrame | None = None) -> pd.DataFrame:
    source = _load_merged_data() if df is None else df.copy()
    require_columns(
        source,
        BASE_COLUMNS,
        "Merged customer intelligence data",
    )

    enriched = _attach_google_review_ratings(source)
    enriched["generated_customer_reviews"] = enriched.apply(
        _generated_customer_reviews,
        axis=1,
    )
    enriched["customer_feedback_summary"] = enriched.apply(
        _customer_feedback_summary,
        axis=1,
    )
    enriched["healthy_food_trends"] = enriched.apply(
        _healthy_food_trends,
        axis=1,
    )
    enriched["local_food_preferences"] = enriched.apply(
        _local_food_preferences,
        axis=1,
    )
    enriched["demographic_based_food_insights"] = enriched.apply(
        _demographic_based_food_insights,
        axis=1,
    )

    output_columns = [column for column in OUTPUT_COLUMNS if column in enriched.columns]
    return enriched[output_columns].copy()


def _save_workbook_atomic(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_excel(temp_path, index=False, sheet_name=CUSTOMER_INTELLIGENCE_SHEET_NAME)

    try:
        os.replace(temp_path, path)
    except PermissionError:
        print(
            f"\nCould not replace {path}. Close the file if it is open, "
            f"then rerun the script. Latest output: {temp_path}"
        )
        return temp_path

    return path


def save_customer_intelligence(
    df: pd.DataFrame | None = None,
    path: Path = CUSTOMER_INTELLIGENCE_PATH,
) -> Path:
    output_df = build_customer_intelligence() if df is None else df.copy()
    return _save_workbook_atomic(output_df, path)


def load_customer_intelligence(
    path: Path = CUSTOMER_INTELLIGENCE_PATH,
    sheet_name: str = CUSTOMER_INTELLIGENCE_SHEET_NAME,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Customer intelligence workbook not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def main() -> None:
    output = save_customer_intelligence()
    df = load_customer_intelligence(output)
    print(f"Customer intelligence saved successfully: {output}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")


__all__ = [
    "CUSTOMER_INTELLIGENCE_PATH",
    "build_customer_intelligence",
    "load_customer_intelligence",
    "save_customer_intelligence",
    "main",
]


if __name__ == "__main__":
    main()
