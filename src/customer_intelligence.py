from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.census import add_age_group
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.io_utils import require_columns


ZCTA_LEVEL_CUSTUMER_REVIEWS_PATH = RAW_DATA_DIR / "zcta_level_custumer_reviews.xlsx"
CUSTOMER_INTELLIGENCE_PATH = PROCESSED_DATA_DIR / "customer_intelligence.xlsx"
CUSTOMER_INTELLIGENCE_SHEET_NAME = "Restaurant Intelligence"

BASE_COLUMNS = [
    "ZCTA",
    "CITY",
    "STATE",
    "COUNTY",
    "POP",
    "AVG_HOUSEHOLD_SIZE",
    "age_group",
    "POP_DENSITY",
    "AREA_TYPE",
    "MEDIAN_INCOME",
    "population_growth_rate",
    "ESTAB",
    "EMP",
    "RESTAURANT_COUNT",
    "COMPETITOR_DENSITY",
    "cluster_id",
    "market_segment",
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

SEGMENT_MENU_PHRASES = {
    "Healthy Lifestyle Market": "fresh bowls, salads, and lighter meals",
    "Youth Convenience Market": "quick grab-and-go meals and portable combos",
    "Family Value Market": "value-driven combo meals and familiar favorites",
    "Social Dining Market": "shareable plates and beverage-friendly bites",
    "Premium Dining Market": "elevated dishes and polished service",
    "Comfort Dining Market": "classic comfort meals and a relaxed pace",
    "General restaurant": "balanced comfort, value, and freshness",
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
    "Healthy Lifestyle Market": (
        "Consumers are showing interest in bowls, salads, plant-forward dishes, and functional beverages."
    ),
    "Youth Convenience Market": (
        "Consumers are showing interest in lighter grab-and-go meals, protein snacks, and quick customizable options."
    ),
    "Family Value Market": (
        "Consumers are showing interest in affordable lighter sides, balanced combo meals, and family-friendly portions."
    ),
    "Social Dining Market": (
        "Consumers are showing interest in shareable lighter plates, balanced combos, and lower-sugar beverages."
    ),
    "Premium Dining Market": (
        "Consumers are showing interest in elevated light plates, fresh ingredients, and wellness-oriented specials."
    ),
    "Comfort Dining Market": (
        "Consumers are showing interest in lighter comfort dishes, grilled proteins, and fresh sides."
    ),
    "General restaurant": (
        "Consumers are showing interest in flexible menu mixes that balance comfort, value, and freshness."
    ),
}

SEGMENT_LOCAL_PREFERENCES = {
    "Healthy Lifestyle Market": "fresh bowls, salads, and lighter meals",
    "Youth Convenience Market": "quick meals and portable combos",
    "Family Value Market": "family-friendly comfort meals and combo deals",
    "Social Dining Market": "shareable plates and beverage-friendly bites",
    "Premium Dining Market": "elevated dining experiences and polished service",
    "Comfort Dining Market": "classic comfort meals and relaxed dining",
    "General restaurant": "familiar comfort meals and flexible combo options",
}


def _load_zcta_level_custumer_reviews(
    path: Path = ZCTA_LEVEL_CUSTUMER_REVIEWS_PATH,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Customer intelligence source file not found: {path}")

    df = pd.read_excel(path, sheet_name="Sheet1")

    if "zip_or_postal_code" in df.columns:
        df["zip_or_postal_code"] = df["zip_or_postal_code"].astype(str).str.zfill(5)
    if "ZCTA" in df.columns:
        df["ZCTA"] = df["ZCTA"].astype(str).str.zfill(5)
    return df


def _normalize_text(value, default: str) -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_rating_value(value):
    rating_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(rating_value):
        return pd.NA
    if rating_value > 10:
        rating_value = rating_value / 20
    return rating_value


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


def _dominant_age_phrase(row: pd.Series) -> str:
    age_group = _normalize_text(row.get("age_group"), "unknown").lower()
    if age_group in {"young", "millennial"}:
        return "a strong presence of younger adults"
    if age_group == "adult":
        return "a balanced mix of working-age households"
    if age_group == "senior":
        return "strong representation from seniors demographics"

    age_value = row.get("median_age", row.get("avg_age"))
    avg_age = pd.to_numeric(pd.Series([age_value]), errors="coerce").iloc[0]
    if pd.isna(avg_age):
        return "a balanced age mix"

    if avg_age < 30:
        return "a strong presence of younger adults"
    if avg_age < 45:
        return "a balanced mix of working-age households"
    if avg_age < 60:
        return "a mature adult customer base"
    return "strong representation from seniors demographics"


def _rating_text(rating: float | int | str | None) -> str:
    rating_value = _normalize_rating_value(rating)
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
    market_segment = _normalize_text(row.get("market_segment"), "regional market")
    menu_phrase = SEGMENT_MENU_PHRASES.get(
        market_segment,
        SEGMENT_MENU_PHRASES["General restaurant"],
    )
    segment_phrase = SEGMENT_REVIEW_PHRASES.get(
        market_segment,
        "balanced local demand",
    )

    return (
        f"{_rating_text(row.get('google_review_rating'))} "
        f"In {location}, customers particularly enjoy {menu_phrase}. "
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
    market_segment = _normalize_text(row.get("market_segment"), "General restaurant")
    return SEGMENT_HEALTHY_TRENDS.get(
        market_segment,
        SEGMENT_HEALTHY_TRENDS["General restaurant"],
    )


def _local_food_preferences(row: pd.Series) -> str:
    location = _location_text(row)
    market_segment = _normalize_text(row.get("market_segment"), "General restaurant")
    preference_phrase = SEGMENT_LOCAL_PREFERENCES.get(
        market_segment,
        SEGMENT_LOCAL_PREFERENCES["General restaurant"],
    )
    audience = SEGMENT_AUDIENCES.get(
        _normalize_text(row.get("market_segment"), "Regional Market"),
        "local diners",
    )
    return f"Residents in {location} tend to prefer {preference_phrase}, especially among {audience}."


def _demographic_based_food_insights(row: pd.Series) -> str:
    age_phrase = _dominant_age_phrase(row)
    market_segment = _normalize_text(row.get("market_segment"), "General restaurant")
    menu_phrase = SEGMENT_MENU_PHRASES.get(
        market_segment,
        SEGMENT_MENU_PHRASES["General restaurant"],
    )
    return (
        f"The surrounding population shows {age_phrase}. "
        f"Customers in this ZIP code show higher engagement with menu options centered on {menu_phrase} "
        f"that balance flavor, affordability, and convenience."
    )


def build_customer_intelligence(
    zcta_level_custumer_reviews: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source = (
        _load_zcta_level_custumer_reviews()
        if zcta_level_custumer_reviews is None
        else zcta_level_custumer_reviews.copy()
    )
    source = add_age_group(source)
    require_columns(
        source,
        BASE_COLUMNS,
        "ZCTA-level customer reviews data",
    )

    require_columns(
        source,
        ["google_review_rating"],
        "ZCTA-level customer reviews data",
    )

    source["google_review_rating"] = pd.to_numeric(
        source["google_review_rating"],
        errors="coerce",
    )

    enriched = source.copy()
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
    zcta_level_custumer_reviews: pd.DataFrame | None = None,
    path: Path = CUSTOMER_INTELLIGENCE_PATH,
) -> Path:
    output_df = build_customer_intelligence(zcta_level_custumer_reviews)
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
