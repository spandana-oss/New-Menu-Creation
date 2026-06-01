from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    KMeans = None
    StandardScaler = None


def _split_tags(value):
    if value != value:
        return []
    return [
        tag.strip().lower()
        for tag in str(value).split(",")
        if tag.strip()
    ]


def _row_tags(row):
    tags = []
    for column in ["style", "cuisines", "restaurant_type"]:
        tags.extend(_split_tags(row.get(column)))
    return tags


def _tag_matches_keyword(tag, keyword):
    if keyword == "bar":
        return tag in ["bar", "bars", "cocktail bar", "cocktail bars"]
    if keyword == "cafe":
        return tag in ["cafe", "cafes", "coffee shop", "coffee shops"]
    if keyword == "qsr":
        return tag == "qsr"
    return keyword in tag


def _has_any(tags, keywords):
    return any(
        _tag_matches_keyword(tag, keyword)
        for tag in tags
        for keyword in keywords
    )


def _series_or_default(group, column, default=0):
    if column in group.columns:
        return pd.to_numeric(group[column], errors="coerce")
    return pd.Series(default, index=group.index, dtype="float64")


def _build_food_behavior(df, geo_column):
    behavior_rows = []

    category_keywords = {
        "fast_food_share": [
            "fast food",
            "qsr",
            "burger",
            "fried chicken",
            "chicken wings",
            "wings",
            "sandwich",
        ],
        "healthy_share": [
            "healthy",
            "salad",
            "vegetarian",
            "vegan",
            "juice",
            "smoothie",
            "poke",
            "mediterranean",
        ],
        "beverage_share": [
            "coffee",
            "cafe",
            "juice",
            "smoothie",
            "bar",
            "cocktail",
        ],
        "wings_share": [
            "wings",
            "chicken wings",
            "wings joint",
        ],
    }

    for geo_value, group in df.groupby(geo_column):
        all_tags = []
        category_counts = {category: 0 for category in category_keywords}

        for _, row in group.iterrows():
            tags = _row_tags(row)
            all_tags.extend(tags)

            for category, keywords in category_keywords.items():
                if _has_any(tags, keywords):
                    category_counts[category] += 1

        restaurant_count = len(group)
        unique_tags = set(all_tags)
        dominant_tag_count = max(
            (all_tags.count(tag) for tag in unique_tags),
            default=0,
        )

        rating = _series_or_default(group, "rating_value", 0)
        reviews = _series_or_default(group, "review_count", 0)
        customer_engagement = (
            rating.fillna(0).clip(lower=0).mean()
            * math.log1p(reviews.fillna(0).clip(lower=0).sum())
        )

        behavior = {
            geo_column: geo_value,
            "restaurant_count": restaurant_count,
            "cuisine_variety": len(unique_tags) / max(restaurant_count, 1),
            "cuisine_focus": dominant_tag_count / max(len(all_tags), 1),
            "customer_engagement": customer_engagement,
        }

        for category, count in category_counts.items():
            behavior[category] = count / max(restaurant_count, 1)

        behavior_rows.append(behavior)

    return behavior_rows


def _rank_series(series):
    ranked = pd.to_numeric(series, errors="coerce").rank(method="average", pct=True)
    return ranked


def _assign_clusters(df_cluster, features, n_clusters):
    x = df_cluster[features].fillna(0)

    if len(df_cluster) == 0:
        return pd.Series(dtype="int64")

    cluster_count = max(1, min(int(n_clusters), len(df_cluster)))

    if KMeans is None or StandardScaler is None or len(df_cluster) < 2:
        composite = x.mean(axis=1)
        try:
            return pd.qcut(
                composite.rank(method="first"),
                q=cluster_count,
                labels=False,
                duplicates="drop",
            ).fillna(0).astype(int)
        except ValueError:
            return pd.Series(0, index=df_cluster.index, dtype="int64")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    return pd.Series(kmeans.fit_predict(x_scaled), index=df_cluster.index, dtype="int64")


def create_clusters(df, n_clusters=3):
    if df is None or df.empty:
        empty = pd.DataFrame(columns=["cluster_id"])
        return df, empty, empty

    df = df.copy()
    geo_column = "ZCTA" if "ZCTA" in df.columns else "FIPS"
    if geo_column not in df.columns:
        raise ValueError("Input data must include a ZCTA or FIPS column.")

    df[geo_column] = df[geo_column].astype(str).str.zfill(5)

    df_cluster = df.groupby(geo_column).first().reset_index()

    if "POP" in df_cluster.columns and "RESTAURANT_COUNT" in df_cluster.columns:
        pop = pd.to_numeric(df_cluster["POP"], errors="coerce")
        restaurants = pd.to_numeric(df_cluster["RESTAURANT_COUNT"], errors="coerce")
        df_cluster["competition_score"] = np.where(pop > 0, restaurants / pop, np.nan)
    elif "COMPETITOR_DENSITY" in df_cluster.columns:
        df_cluster["competition_score"] = pd.to_numeric(
            df_cluster["COMPETITOR_DENSITY"],
            errors="coerce",
        )
    else:
        df_cluster["competition_score"] = np.nan

    age_columns = [
        "AGE_18_24",
        "AGE_25_34",
        "AGE_35_44",
        "AGE_55_64",
        "AGE_65_PLUS",
    ]
    available_age_cols = [column for column in age_columns if column in df_cluster.columns]
    if available_age_cols:
        age_counts = df_cluster[available_age_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df_cluster["AGE_TOTAL"] = age_counts.sum(axis=1).replace(0, np.nan)
        if "AGE_18_24" in df_cluster.columns:
            df_cluster["PCT_18_24"] = pd.to_numeric(df_cluster["AGE_18_24"], errors="coerce") / df_cluster["AGE_TOTAL"]
        if "AGE_25_34" in df_cluster.columns:
            df_cluster["PCT_25_34"] = pd.to_numeric(df_cluster["AGE_25_34"], errors="coerce") / df_cluster["AGE_TOTAL"]
        if "AGE_65_PLUS" in df_cluster.columns:
            df_cluster["PCT_65_PLUS"] = pd.to_numeric(df_cluster["AGE_65_PLUS"], errors="coerce") / df_cluster["AGE_TOTAL"]
    else:
        df_cluster["PCT_18_24"] = np.nan
        df_cluster["PCT_25_34"] = np.nan
        df_cluster["PCT_65_PLUS"] = np.nan

    df_behavior = pd.DataFrame(_build_food_behavior(df, geo_column))
    if not df_behavior.empty:
        df_cluster = df_cluster.merge(df_behavior, on=geo_column, how="left")
    else:
        for column in [
            "restaurant_count",
            "cuisine_variety",
            "cuisine_focus",
            "customer_engagement",
            "fast_food_share",
            "healthy_share",
            "beverage_share",
            "wings_share",
        ]:
            df_cluster[column] = np.nan

    df_cluster["young_demand"] = (
        df_cluster.get("PCT_18_24", pd.Series(np.nan, index=df_cluster.index))
        + df_cluster.get("PCT_25_34", pd.Series(np.nan, index=df_cluster.index))
    )
    df_cluster["senior_demand"] = df_cluster.get("PCT_65_PLUS", pd.Series(np.nan, index=df_cluster.index))

    ranked_features = {
        "income_level": "MEDIAN_INCOME",
        "young_demand_level": "young_demand",
        "senior_demand_level": "senior_demand",
        "competition_level": "competition_score",
        "growth_level": "population_growth_rate",
        "cuisine_variety_level": "cuisine_variety",
        "cuisine_focus_level": "cuisine_focus",
        "customer_engagement_level": "customer_engagement",
        "fast_food_level": "fast_food_share",
        "healthy_level": "healthy_share",
        "beverage_level": "beverage_share",
        "wings_level": "wings_share",
    }

    for ranked_name, source_column in ranked_features.items():
        if source_column in df_cluster.columns:
            df_cluster[ranked_name] = _rank_series(df_cluster[source_column])
        else:
            df_cluster[ranked_name] = np.nan

    features = list(ranked_features.keys())
    df_cluster["cluster_id"] = _assign_clusters(df_cluster, features, n_clusters)

    df = df.merge(
        df_cluster[[geo_column, "cluster_id"]],
        on=geo_column,
        how="left",
    )

    cluster_profile = (
        df_cluster.groupby("cluster_id")[features]
        .mean()
        .sort_index()
    )

    return df, df_cluster, cluster_profile


__all__ = ["create_clusters"]
