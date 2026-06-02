from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.customer_intelligence import CUSTOMER_INTELLIGENCE_PATH
from src.io_utils import require_columns


MERGED_DATA_PATH = PROCESSED_DATA_DIR / "merged_data.csv"
MERGED_CUSTOMER_INTELLIGENCE_PATH = (
    PROCESSED_DATA_DIR / "merged_customer_intelligence.xlsx"
)
MERGED_CUSTOMER_INTELLIGENCE_SHEET_NAME = "Merged Intelligence"
DEMOGRAPHIC_ONLY_RESTAURANT_SOURCE_FILES = {
    "Comp_Restaurants_Curated_DCA.csv",
}


def _normalize_zcta(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "ZCTA" in normalized.columns:
        normalized["ZCTA"] = normalized["ZCTA"].astype(str).str.zfill(5)
    return normalized


def _load_merged_data(path: Path = MERGED_DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Merged data file not found: {path}")
    return _normalize_zcta(pd.read_csv(path))


def _load_customer_intelligence(
    path: Path = CUSTOMER_INTELLIGENCE_PATH,
) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Customer intelligence file not found: {path}")
    return _normalize_zcta(pd.read_excel(path, sheet_name=0))


def _merge_customer_intelligence_rows(
    restaurant_rows: pd.DataFrame,
    customer_intelligence: pd.DataFrame,
) -> pd.DataFrame:
    extra_columns = [
        column
        for column in customer_intelligence.columns
        if column not in restaurant_rows.columns
    ]

    restaurant_rows = restaurant_rows.copy()
    customer_intelligence = customer_intelligence.copy()

    restaurant_rows["_zcta_row"] = restaurant_rows.groupby(
        "ZCTA",
        sort=False,
    ).cumcount()
    customer_intelligence["_zcta_row"] = customer_intelligence.groupby(
        "ZCTA",
        sort=False,
    ).cumcount()

    right = customer_intelligence[["ZCTA", "_zcta_row", *extra_columns]]
    combined = restaurant_rows.merge(
        right,
        on=["ZCTA", "_zcta_row"],
        how="left",
    )

    ordered_columns = [
        column
        for column in restaurant_rows.columns
        if column != "_zcta_row"
    ] + extra_columns
    return combined[ordered_columns].copy()


def _save_workbook_atomic(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_excel(
        temp_path,
        index=False,
        sheet_name=MERGED_CUSTOMER_INTELLIGENCE_SHEET_NAME,
    )

    try:
        os.replace(temp_path, path)
    except PermissionError:
        print(
            f"\nCould not replace {path}. Close the file if it is open, "
            f"then rerun the script. Latest output: {temp_path}"
        )
        return temp_path

    return path


def merge_processed_outputs(
    merged_data_path: Path = MERGED_DATA_PATH,
    customer_intelligence_path: Path = CUSTOMER_INTELLIGENCE_PATH,
) -> pd.DataFrame:
    merged_data = _load_merged_data(merged_data_path)
    customer_intelligence = _load_customer_intelligence(customer_intelligence_path)

    require_columns(merged_data, ["ZCTA"], "Merged data")
    require_columns(customer_intelligence, ["ZCTA"], "Customer intelligence data")
    require_columns(
        merged_data,
        ["restaurant_source_file"],
        "Merged data",
    )

    merged_data = merged_data.copy()
    merged_data["_output_order"] = range(len(merged_data))

    dca_mask = merged_data["restaurant_source_file"].isin(
        DEMOGRAPHIC_ONLY_RESTAURANT_SOURCE_FILES
    )
    dca_rows = merged_data.loc[dca_mask].copy()
    customer_intelligence_rows = merged_data.loc[~dca_mask].copy()

    if len(customer_intelligence_rows) != len(customer_intelligence):
        raise ValueError(
            "Customer intelligence should only be merged into the non-DCA restaurant "
            f"rows, but found {len(customer_intelligence_rows)} eligible rows and "
            f"{len(customer_intelligence)} customer intelligence rows."
        )

    merged_customer_intelligence = _merge_customer_intelligence_rows(
        customer_intelligence_rows,
        customer_intelligence,
    )

    extra_columns = [
        column
        for column in merged_customer_intelligence.columns
        if column not in merged_data.columns
    ]

    for column in extra_columns:
        dca_rows[column] = pd.NA

    dca_rows = dca_rows.reindex(columns=merged_customer_intelligence.columns)

    combined = pd.concat(
        [merged_customer_intelligence, dca_rows],
        ignore_index=True,
    ).sort_values("_output_order", kind="stable")

    ordered_columns = [
        column
        for column in combined.columns
        if column != "_output_order"
    ]
    return combined[ordered_columns].copy()


def save_merged_processed_outputs(
    merged_data_path: Path = MERGED_DATA_PATH,
    customer_intelligence_path: Path = CUSTOMER_INTELLIGENCE_PATH,
    path: Path = MERGED_CUSTOMER_INTELLIGENCE_PATH,
) -> Path:
    output_df = merge_processed_outputs(merged_data_path, customer_intelligence_path)
    return _save_workbook_atomic(output_df, path)


def main() -> None:
    output = save_merged_processed_outputs()
    df = pd.read_excel(output)
    print(f"Merged customer intelligence saved successfully: {output}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")


__all__ = [
    "MERGED_CUSTOMER_INTELLIGENCE_PATH",
    "merge_processed_outputs",
    "save_merged_processed_outputs",
    "main",
]


if __name__ == "__main__":
    main()
