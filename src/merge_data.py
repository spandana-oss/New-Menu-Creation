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
    fetch_age_distribution,
    fetch_population_growth
)
from src.config import (
    DCA_CENSUS_DATA_PATH,
    MEMPHIS_CENSUS_DATA_PATH,
    MEMPHIS_MERGED_DATA_PATH,
    MEMPHIS_SEGMENTATION_ANALYSIS_PATH,
)


MASTER_FEATURE_COLUMNS = [
   
    'zip_or_postal_code',
    'ZCTA',
    'CITY',
    'STATE',
    'COUNTY',
    'LAT',
    'LNG',
    'POP',
    'AVG_HOUSEHOLD_SIZE',
    'POP_DENSITY',
    'AREA_TYPE',
    'MEDIAN_INCOME',
    'age_group',
    'population_growth_rate',
    'ESTAB',
    'EMP',
    'RESTAURANT_COUNT',
    'market_segment',
    'price positioning',
]

SEGMENTATION_ANALYSIS_COLUMNS = [

    'zip_or_postal_code',
    'ZCTA',
    'CITY',
    'STATE',
    'COUNTY',
    'AREA_TYPE',
    'market_segment',
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
   
]

CENSUS_COLUMNS = [
    'zip_or_postal_code',
    'ZCTA',
    'CITY',
    'STATE',
    'COUNTY',
    'LAT',
    'LNG',
    'POP',
    'AVG_HOUSEHOLD_SIZE',
    'age_group',
    'POP_DENSITY',
    'AREA_TYPE'
]

LOCATION_CENSUS_DATA_PATHS = {
    "memphis": MEMPHIS_CENSUS_DATA_PATH,
    "dca": DCA_CENSUS_DATA_PATH,
}

LOCATION_SEGMENTATION_ANALYSIS_PATHS = {
    "memphis": MEMPHIS_SEGMENTATION_ANALYSIS_PATH,
}

LOCATION_MERGED_DATA_PATHS = {
    "memphis": MEMPHIS_MERGED_DATA_PATH,
}


def _normalize_location(location):
    location_key = str(location).strip().lower()
    if location_key not in LOCATION_CENSUS_DATA_PATHS:
        raise ValueError(
            f"Unknown location: {location}. "
            f"Expected one of: {', '.join(sorted(LOCATION_CENSUS_DATA_PATHS))}."
        )
    return location_key


def _format_keys(df_rest, df_cbp, acs_dfs):
    df_cbp['COUNTY_FIPS'] = df_cbp['COUNTY_FIPS'].astype(str).str.zfill(5)

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

    return df


def _apply_segmentation(df):
    geo_column = (
        'ZCTA'
        if 'ZCTA' in df.columns
        else 'COUNTY_FIPS'
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

    return df, df_cluster, cluster_profile


def _load_datasets(location="memphis"):
    location_key = _normalize_location(location)
    datasets = {
        'restaurants': restaurants(location_key),
        'zip_map': get_zip_to_fips(),
        'cbp': fetch_cbp(),
        'population': fetch_population(),
        'income': fetch_income(),
        'household_size': fetch_household_size(),
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
            datasets['age'],
            datasets['growth']
        ]
    )

    datasets['zip_map'] = datasets['zip_map'].copy()
    if 'zcta' in datasets['zip_map'].columns:
        datasets['zip_map']['zcta'] = (
            datasets['zip_map']['zcta']
            .astype(str)
            .str.zfill(5)
        )
        datasets['zip_map'] = datasets['zip_map'].rename(columns={'zcta': 'ZCTA'})
    if 'ZCTA' in datasets['zip_map'].columns:
        datasets['zip_map']['ZCTA'] = (
            datasets['zip_map']['ZCTA']
            .astype(str)
            .str.zfill(5)
        )

    return datasets


def _merge_datasets(datasets):
    geo_df = build_geographic_intelligence(
        population_df=datasets["population"],
        restaurants_df=datasets["restaurants"]
    )

    df = datasets['restaurants'].merge(
        datasets['zip_map'][['ZCTA', 'COUNTY_FIPS']],
        on="ZCTA",
        how="left"
    )

    df = df.merge(datasets['cbp'], on="COUNTY_FIPS", how="left")
    df = df.merge(
        geo_df,
        on="ZCTA",
        how="left"
    )
    df = df.merge(datasets['income'], on="ZCTA", how="left")
    df = df.merge(datasets['household_size'], on="ZCTA", how="left")
    df = df.merge(datasets['age'], on="ZCTA", how="left")
    df = df.merge(datasets['growth'], on="ZCTA", how="left")

    return _clean_numeric_fields(df)


def _build_census_dataset(location="memphis"):
    location_key = _normalize_location(location)
    datasets = _load_datasets(location_key)
    df = _merge_datasets(datasets)

    census_df = _census_output(df)
    saved_census_filename = _save_final_output(
        census_df,
        LOCATION_CENSUS_DATA_PATHS[location_key],
    )

    return df, saved_census_filename


def _build_segmented_dataset(location="memphis"):
    location_key = _normalize_location(location)
    df, saved_census_filename = _build_census_dataset(location_key)

    df, df_cluster, cluster_profile = _apply_segmentation(df)

    segmentation_analysis_df = _segmentation_analysis_output(df)
    saved_segmentation_filename = _save_final_output(
        segmentation_analysis_df,
        LOCATION_SEGMENTATION_ANALYSIS_PATHS[location_key],
    )

    final_df = _final_output(df)
    saved_filename = _save_final_output(
        final_df,
        LOCATION_MERGED_DATA_PATHS[location_key],
    )

    return final_df, saved_census_filename, saved_segmentation_filename, saved_filename


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


def _dca_census_output(df):
    ordered_columns = []
    seen = set()

    for column in [*MASTER_FEATURE_COLUMNS, *SEGMENTATION_ANALYSIS_COLUMNS]:
        if column in seen:
            continue
        seen.add(column)
        ordered_columns.append(column)

    return df[
        [
            column
            for column in ordered_columns
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
    final_df, saved_census_filename, saved_segmentation_filename, saved_filename = _build_segmented_dataset("memphis")

    print("\nFinal merged dataset saved successfully.")
    print(f"Output file: {saved_filename}")
    print(f"Segmentation analysis file: {saved_segmentation_filename}")
    print(f"Census data file: {saved_census_filename}")

    return final_df


def build_dca_demographic_dataset():
    return build_dca_segmentation_dataset()


def build_dca_segmentation_dataset():
    datasets = _load_datasets("dca")
    df = _merge_datasets(datasets)
    df, _, _ = _apply_segmentation(df)

    dca_df = _dca_census_output(df)
    saved_filename = _save_final_output(dca_df, DCA_CENSUS_DATA_PATH)

    print("\nDCA segmented census saved successfully.")
    print(f"Output file: {saved_filename}")

    return dca_df
