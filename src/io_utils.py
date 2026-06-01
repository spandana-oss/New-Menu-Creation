import os
from pathlib import Path

import pandas as pd


def require_columns(df, required_columns, context):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context} is missing required columns: {', '.join(missing)}"
        )


def unique_columns_case_insensitive(columns):
    unique_columns = []
    seen = set()
    for column in columns:
        key = str(column).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_columns.append(column)
    return unique_columns


def drop_repeated_columns(df):
    unique_columns = unique_columns_case_insensitive(list(df.columns))
    return df.loc[:, unique_columns]


def save_csv_atomic(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    drop_repeated_columns(df).to_csv(temp_path, index=False)

    try:
        os.replace(temp_path, path)
    except PermissionError:
        print(
            f"\nCould not replace {path}. Close the file if it is open, "
            f"then rerun the script. Latest output: {temp_path}"
        )
        return temp_path

    return path
