import os
import pandas as pd

from src.zip_to_fips import get_zip_to_fips, build_geographic_intelligence
from src.data_preparation import restaurants
from src.clustering import create_clusters
from src.segmentation import assign_market_segments, assign_menu_analysis

from src.census import (
    fetch_cbp,
    fetch_population,
    fetch_income,
    fetch_household_size,
    fetch_ethnicity,
    fetch_age_distribution,
    fetch_population_growth
)


MASTER_FEATURE_COLUMNS = [
    'restaurant_object_key',
    'restaurant_name',
    'zip_or_postal_code',
    'ZCTA',
    'FIPS',
    'CITY',
    'STATE',
    'COUNTY',
    'LAT',
    'LNG',
    'POP',
    'AVG_HOUSEHOLD_SIZE',
    'WHITE_POP',
    'BLACK_POP',
    'ASIAN_POP',
    'HISPANIC_POP',
    'POP_DENSITY',
    'AREA_TYPE',
    'MEDIAN_INCOME',
    'AGE_18_24',
    'AGE_25_34',
    'AGE_35_44',
    'AGE_55_64',
    'AGE_65_PLUS',
    'population_growth_rate',
    'ESTAB',
    'EMP',
    'RESTAURANT_COUNT',
    'COMPETITOR_DENSITY',
    'cluster_id',
    'market_segment',
    'restaurant category',
    'price positioning'
]

SEGMENTATION_ANALYSIS_COLUMNS = [
    'restaurant_object_key',
    'restaurant_name',
    'zip_or_postal_code',
    'ZCTA',
    'FIPS',
    'CITY',
    'STATE',
    'COUNTY',
    'AREA_TYPE',
    'cluster_id',
    'market_segment',
    'restaurant category',
    'price positioning',
    'market trend',
    'menu_signal_strength',
    'Age Demand Level',
    'Income Level',
    'Growth Level',
    'Competition Level',
    'Cuisine Variety Level',
    'Cuisine Focus Level',
    'Customer Engagement Level',
    'Fast Food Level',
    'Healthy Level',
    'Beverage Level',
    'Wings Level'
]

CENSUS_COLUMNS = [
    'zip_or_postal_code',
    'ZCTA',
    'FIPS',
    'CITY',
    'STATE',
    'COUNTY',
    'LAT',
    'LNG',
    'POP',
    'AVG_HOUSEHOLD_SIZE',
    'WHITE_POP',
    'BLACK_POP',
    'ASIAN_POP',
    'HISPANIC_POP',
    'POP_DENSITY',
    'AREA_TYPE'
]


def _format_keys(df_rest, df_cbp, acs_dfs):
    df_cbp['FIPS'] = df_cbp['FIPS'].astype(str).str.zfill(5)

    for acs_df in acs_dfs:
        acs_df['ZCTA'] = (
            acs_df['ZCTA']
            .astype(str)
            .str.zfill(5)
        )

    df_rest['ZCTA'] = (
        df_rest['ZCTA']
        .astype(str)
        .str.zfill(5)
    )


def _clean_numeric_fields(df):
    df = df.copy()

    df['ESTAB'] = pd.to_numeric(
        df['ESTAB'],
        errors='coerce'
    ).fillna(0)

    df['EMP'] = pd.to_numeric(
        df['EMP'],
        errors='coerce'
    )

    df['POP'] = pd.to_numeric(
        df['POP'],
        errors='coerce'
    )

    df['AVG_HOUSEHOLD_SIZE'] = pd.to_numeric(
        df['AVG_HOUSEHOLD_SIZE'],
        errors='coerce'
    )

    ethnicity_cols = [
        'WHITE_POP',
        'BLACK_POP',
        'ASIAN_POP',
        'HISPANIC_POP'
    ]

    for col in ethnicity_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    return df


def _load_datasets():
    datasets = {
        'restaurants': restaurants(),
        'zip_map': get_zip_to_fips(),
        'cbp': fetch_cbp(),
        'population': fetch_population(),
        'income': fetch_income(),
        'household_size': fetch_household_size(),
        'ethnicity': fetch_ethnicity(),
        'age': fetch_age_distribution(),
        'growth': fetch_population_growth()
    }

    _format_keys(
        datasets['restaurants'],
        datasets['cbp'],
        [
            datasets['population'],
            datasets['income'],
            datasets['household_size'],
            datasets['ethnicity'],
            datasets['age'],
            datasets['growth']
        ]
    )

    return datasets


def _merge_datasets(datasets):
    geo_df = build_geographic_intelligence(
        population_df=datasets["population"],
        restaurants_df=datasets["restaurants"]
    )

    df = datasets['restaurants'].merge(
        datasets['zip_map'][['zcta', 'FIPS']],
        left_on="ZCTA",
        right_on="zcta",
        how="left"
    )

    df = df.merge(datasets['cbp'], on="FIPS", how="left")
    df = df.merge(
        geo_df,
        on="ZCTA",
        how="left"
    )
    df = df.merge(datasets['income'], on="ZCTA", how="left")
    df = df.merge(datasets['household_size'], on="ZCTA", how="left")
    df = df.merge(datasets['ethnicity'], on="ZCTA", how="left")
    df = df.merge(datasets['age'], on="ZCTA", how="left")
    df = df.merge(datasets['growth'], on="ZCTA", how="left")

    return _clean_numeric_fields(df)


def _final_output(df):
    return df[
        [column for column in MASTER_FEATURE_COLUMNS if column in df.columns]
    ]


def _segmentation_analysis_output(df):
    return df[
        [
            column
            for column in SEGMENTATION_ANALYSIS_COLUMNS
            if column in df.columns
        ]
    ]


def _census_output(df):
    return df[
        [column for column in CENSUS_COLUMNS if column in df.columns]
    ]


def _save_final_output(df, filename):
    output_dir = os.path.dirname(filename)
    os.makedirs(output_dir, exist_ok=True)

    temp_filename = f"{filename}.tmp"

    df.to_csv(
        temp_filename,
        index=False
    )

    try:
        os.replace(
            temp_filename,
            filename
        )
        return filename
    except PermissionError:
        print(
            f"\nCould not replace {filename}. "
            f"Close the file if it is open, then rerun the script. "
            f"The latest output is available at {temp_filename}."
        )
        return temp_filename


def merge_datasets():
    datasets = _load_datasets()
    df = _merge_datasets(datasets)

    census_df = _census_output(df)
    census_filename = "data/processed/census_data.csv"
    saved_census_filename = _save_final_output(
        census_df,
        census_filename
    )

    geo_column = (
        'ZCTA'
        if 'ZCTA' in df.columns
        else 'FIPS'
    )

    df, df_cluster, cluster_profile = create_clusters(
        df,
        n_clusters=15
    )

    df = assign_market_segments(
        df,
        cluster_profile
    )

    df = assign_menu_analysis(
        df,
        df_cluster,
        geo_column
    )

    segmentation_analysis_df = _segmentation_analysis_output(df)
    segmentation_filename = "data/processed/segmentation_analysis.csv"
    saved_segmentation_filename = _save_final_output(
        segmentation_analysis_df,
        segmentation_filename
    )

    final_df = _final_output(df)
    filename = "data/processed/merged_data.csv"

    saved_filename = _save_final_output(final_df, filename)

    print("\nFinal merged dataset saved successfully.")
    print(f"Output file: {saved_filename}")
    print(f"Segmentation analysis file: {saved_segmentation_filename}")
    print(f"Census data file: {saved_census_filename}")

    return final_df

