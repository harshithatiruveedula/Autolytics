"""Lightweight baseline modeling without mutating the working dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eda import numeric_columns


def candidate_targets(df: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for column in df.columns:
        series = df[column]
        nunique = series.nunique(dropna=True)
        if nunique < 2:
            continue
        if pd.api.types.is_numeric_dtype(series) or nunique <= 12:
            names.append(str(column))
    return names


def _encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            encoded[column] = pd.to_numeric(series, errors="coerce")
        else:
            encoded[column] = series.astype("string").fillna("(missing)").astype("category").cat.codes
    return encoded.astype(float)


def train_baseline(df: pd.DataFrame, target: str, feature_cols: list[str]) -> dict[str, Any]:
    if not feature_cols:
        raise ValueError("Select at least one feature column.")
    if target in feature_cols:
        raise ValueError("The target column cannot also be a feature.")

    working = df[[target, *feature_cols]].copy()
    working = working.dropna(subset=[target])
    if len(working) < 8:
        raise ValueError("Need at least 8 rows with a non-missing target to train a model.")

    features = _encode_features(working[feature_cols]).fillna(0.0)
    y_raw = working[target]
    numeric_target = pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > 12
    task = "regression" if numeric_target else "classification"

    rng = np.random.default_rng(42)
    order = rng.permutation(len(working))
    split = max(1, int(len(working) * 0.8))
    train_idx, test_idx = order[:split], order[split:]
    if len(test_idx) == 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    X = features.to_numpy(dtype=float)
    X_train, X_test = X[train_idx], X[test_idx]
    design_train = np.column_stack([np.ones(len(X_train)), X_train])
    design_test = np.column_stack([np.ones(len(X_test)), X_test])

    if task == "regression":
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        y_train, y_test = y[train_idx], y[test_idx]
        coef, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
        pred = design_test @ coef
        resid = y_test - pred
        rmse = float(np.sqrt(np.mean(resid**2))) if len(resid) else float("nan")
        mae = float(np.mean(np.abs(resid))) if len(resid) else float("nan")
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2)) if len(y_test) else 0.0
        r2 = float(1 - ss_res / ss_tot) if ss_tot else float("nan")
        metrics = {"task": task, "rmse": rmse, "mae": mae, "r2": r2}
        preview = pd.DataFrame({"actual": y_test, "predicted": pred})
    else:
        labels = y_raw.astype("string").fillna("(missing)")
        codes, uniques = pd.factorize(labels, sort=True)
        y = codes.astype(float)
        y_train, y_test = y[train_idx], y[test_idx]
        coef, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
        raw_pred = design_test @ coef
        pred_codes = np.clip(np.rint(raw_pred), 0, max(0, len(uniques) - 1)).astype(int)
        actual_codes = np.clip(np.rint(y_test), 0, max(0, len(uniques) - 1)).astype(int)
        accuracy = float(np.mean(pred_codes == actual_codes)) if len(actual_codes) else float("nan")
        metrics = {
            "task": task,
            "accuracy": accuracy,
            "classes": [str(v) for v in uniques.tolist()],
        }
        preview = pd.DataFrame(
            {
                "actual": [str(uniques[i]) for i in actual_codes],
                "predicted": [str(uniques[i]) for i in pred_codes],
            }
        )

    metrics.update(
        {
            "target": target,
            "features": feature_cols,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "numeric_features_available": numeric_columns(df),
        }
    )
    return {"metrics": metrics, "preview": preview.head(25)}
