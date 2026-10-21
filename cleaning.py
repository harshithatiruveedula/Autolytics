"""Safe cleaning and missing-value imputation for AUTOLYTICS."""
from __future__ import annotations
from typing import Any
import pandas as pd


def _unique_column_names(columns: list[str]) -> tuple[list[str], dict[str, str]]:
    seen: set[str] = set()
    unique: list[str] = []
    renamed: dict[str, str] = {}
    for index, raw in enumerate(columns):
        original = str(raw)
        name = original.strip()
        if not name or name.lower() in {"nan", "none", "unnamed"}:
            name = f"column_{index + 1}"
        candidate, suffix = name, 2
        while candidate in seen:
            candidate = f"{name}_{suffix}"
            suffix += 1
        seen.add(candidate)
        unique.append(candidate)
        if candidate != original:
            renamed[f"{original} [{index + 1}]"] = candidate
    return unique, renamed


def _infer_series_type(series: pd.Series) -> tuple[pd.Series, str | None]:
    if series.dropna().empty or pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return series, None
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if pd.api.types.is_float_dtype(series) and not non_null.empty and (non_null == non_null.round()).all() and non_null.abs().max() < 1e15:
            try:
                return series.astype("Int64"), "integer"
            except (TypeError, ValueError):
                pass
        return series, None
    as_string = series.astype("string")
    values = as_string.dropna()
    if values.empty or values.str.match(r"^0\d+$").any():
        return as_string, None
    numeric = pd.to_numeric(values.str.replace(",", "", regex=False), errors="coerce")
    if numeric.notna().all():
        full = pd.to_numeric(as_string.str.replace(",", "", regex=False), errors="coerce")
        if (numeric == numeric.round()).all() and numeric.abs().max() < 1e15:
            try:
                return full.astype("Int64"), "integer"
            except (TypeError, ValueError, OverflowError):
                pass
        return full.astype("float64"), "number"
    sample = values.head(20)
    if sample.str.contains(r"[-/]", regex=True, na=False).mean() >= .8:
        try:
            parsed = pd.to_datetime(as_string, errors="coerce", format="mixed")
            if parsed.dropna().size == values.size:
                return parsed, "datetime"
        except (ValueError, TypeError):
            pass
    return as_string, None


def _impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Impute missing cells column-wise; never drop rows just for nulls."""
    result = df.copy()
    details: dict[str, dict[str, Any]] = {}
    for col in result.columns:
        missing = int(result[col].isna().sum())
        if missing == 0:
            continue
        series = result[col]
        if pd.api.types.is_numeric_dtype(series):
            if series.notna().any():
                fill_value = series.median()
                result[col] = series.fillna(fill_value)
                method = f"median ({fill_value})"
            else:
                result[col] = series.astype("object").fillna("Unknown")
                method = "Unknown (column was entirely empty)"
        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null = series.dropna()
            if not non_null.empty:
                fill_value = non_null.mode().iloc[0] if not non_null.mode().empty else non_null.iloc[0]
                result[col] = series.fillna(fill_value)
                method = f"most frequent date ({fill_value})"
            else:
                result[col] = series.astype("object").fillna("Unknown")
                method = "Unknown (column was entirely empty)"
        else:
            non_null = series.dropna()
            if not non_null.empty and not non_null.mode().empty:
                fill_value = non_null.mode().iloc[0]
                result[col] = series.fillna(fill_value)
                method = f"mode ({fill_value})"
            else:
                result[col] = series.astype("object").fillna("Unknown")
                method = "Unknown (column was entirely empty)"
        details[str(col)] = {"cells_imputed": missing, "method": method}
    return result, details


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Trim text, standardize columns/types, deduplicate exact rows, and impute nulls.

    Rows with partial missing values are never removed. Fully empty rows are retained
    until imputation and will be populated with column-wise imputation values.
    """
    working = df.copy()
    summary: dict[str, Any] = {
        "duplicate_rows_removed": 0,
        "empty_rows_removed": 0,
        "empty_columns_removed": 0,
        "blank_cells_set_to_missing": 0,
        "columns_renamed": {},
        "types_converted": {},
        "imputed_cells": 0,
        "imputation_details": {},
        "missing_cells_remaining": 0,
        "whitespace_trimmed": True,
    }
    names, renamed = _unique_column_names([str(c) for c in working.columns])
    working.columns = names
    summary["columns_renamed"] = renamed
    blank_count = 0
    for col in working.columns:
        if working[col].dtype == object or pd.api.types.is_string_dtype(working[col]):
            before = working[col]
            trimmed = before.map(lambda v: v.strip() if isinstance(v, str) else v)
            blank_count += int(trimmed.map(lambda v: isinstance(v, str) and v == "").sum())
            working[col] = trimmed.map(lambda v: pd.NA if isinstance(v, str) and v == "" else v)
    summary["blank_cells_set_to_missing"] = blank_count

    # Remove exact duplicate records only. Do not delete rows because they contain nulls.
    # Deduplicate only complete rows. Rows containing missing values are retained,
    # even when two partially-empty rows look identical.
    complete_mask = working.notna().all(axis=1)
    complete = working.loc[complete_mask].drop_duplicates(keep="first")
    partial = working.loc[~complete_mask]
    summary["duplicate_rows_removed"] = int(complete_mask.sum() - len(complete))
    working = pd.concat([complete, partial]).sort_index()

    type_changes = {}
    for col in working.columns:
        converted, kind = _infer_series_type(working[col])
        if kind:
            working[col] = converted
            type_changes[str(col)] = kind
    summary["types_converted"] = type_changes

    before_missing = int(working.isna().sum().sum())
    working, imputation_details = _impute_missing_values(working)
    after_missing = int(working.isna().sum().sum())
    summary["imputation_details"] = imputation_details
    summary["imputed_cells"] = before_missing - after_missing
    summary["missing_cells_remaining"] = after_missing
    working = working.reset_index(drop=True)
    return working, summary


def format_cleaning_summary(summary: dict[str, Any]) -> str:
    parts = [
        f"{summary.get('duplicate_rows_removed', 0)} duplicate rows removed",
        f"{summary.get('blank_cells_set_to_missing', 0)} blank cells identified",
        f"{summary.get('imputed_cells', 0)} missing cells imputed",
    ]
    renamed = summary.get("columns_renamed") or {}
    if renamed:
        parts.append(f"{len(renamed)} column names adjusted")
    converted = summary.get("types_converted") or {}
    if converted:
        parts.append("types inferred for " + str(len(converted)) + " columns")
    parts.append(f"{summary.get('missing_cells_remaining', 0)} missing cells remaining")
    return " · ".join(parts)
