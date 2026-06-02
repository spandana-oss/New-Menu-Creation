from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"

# Legacy compatibility paths used by restored history scripts.
MEMPHIS_RESTAURANT_SOURCE_PATH = RAW_DATA_DIR / "MEM_compstore_restaurants.csv"
DCA_RESTAURANT_SOURCE_PATH = RAW_DATA_DIR / "Comp_Restaurants_Curated_DCA.csv"
RESTAURANT_SOURCE_PATH = MEMPHIS_RESTAURANT_SOURCE_PATH
RESTAURANT_SOURCE_PATHS = (
    MEMPHIS_RESTAURANT_SOURCE_PATH,
    DCA_RESTAURANT_SOURCE_PATH,
)

MEMPHIS_CENSUS_DATA_PATH = PROCESSED_DATA_DIR / "census_data.csv"
MEMPHIS_SEGMENTATION_ANALYSIS_PATH = PROCESSED_DATA_DIR / "segmentation_analysis.csv"
MEMPHIS_MERGED_DATA_PATH = PROCESSED_DATA_DIR / "merged_data.csv"
DCA_CENSUS_DATA_PATH = PROCESSED_DATA_DIR / "dca_census_data.csv"

CUSTOMER_INTELLIGENCE_PATH = PROCESSED_DATA_DIR / "customer_intelligence.xlsx"
DATA_FETCHING_PATH = PROCESSED_DATA_DIR / "data_fetching.csv"
SEGMENTATION_ANALYSIS_PATH = PROCESSED_DATA_DIR / "segmentation_analysis.csv"
FINAL_SEGMENTED_DATASET_PATH = (
    PROCESSED_DATA_DIR / "final_segmented_intelligence_dataset.csv"
)

ZIP_GEOGRAPHY_PATH = RAW_DATA_DIR / "zip_geography.csv"

REGIONAL_MARKET_INTELLIGENCE_PATH = (
    PROCESSED_DATA_DIR / "regional_market_intelligence.csv"
)
