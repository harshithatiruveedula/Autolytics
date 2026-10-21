"""Exploratory analysis helpers for the cleaned working dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _series_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "column": column,
                "type": _series_kind(series),
                "non_missing": n - missing,
                "missing": missing,
                "missing_pct": round((missing / n) * 100, 2) if n else 0.0,
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_unique: int = 30) -> list[str]:
    names: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if series.nunique(dropna=True) <= max_unique:
            names.append(str(column))
    return names


def value_counts_frame(df: pd.DataFrame, column: str, limit: int = 20) -> pd.DataFrame:
    counts = df[column].astype("string").fillna("(missing)").value_counts().head(limit)
    return counts.rename_axis(column).reset_index(name="count")


def histogram_frame(series: pd.Series, bins: int = 20) -> pd.DataFrame:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return pd.DataFrame(columns=["bin", "count"])
    grouped = pd.cut(numeric, bins=min(bins, max(1, numeric.nunique())))
    counts = grouped.value_counts().sort_index()
    return pd.DataFrame(
        {"bin": [str(interval) for interval in counts.index], "count": counts.to_list()}
    )


def correlation_frame(df: pd.DataFrame) -> pd.DataFrame:
    nums = numeric_columns(df)
    if len(nums) < 2:
        return pd.DataFrame()
    return df[nums].corr(numeric_only=True)
