"""Exploratory analysis helpers and Altair chart builders for AUTOLYTICS.

All functions here are pure computation (pandas + Altair).  No Streamlit calls
are made so these can be tested independently of the UI layer.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

try:
    import altair as alt
    _ALTAIR = True
except ImportError:
    alt = None  # type: ignore[assignment]
    _ALTAIR = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Altair color palette matching the AUTOLYTICS navy/blue theme
_PALETTE = [
    "#2864f0", "#7c3aed", "#10b981", "#f59e0b", "#ef4444",
    "#06b6d4", "#8b5cf6", "#f97316", "#14b8a6", "#6366f1",
]
_CORR_SCHEME = "blueorange"

# Max categories to show in a bar / donut chart before truncating
_MAX_CAT = 20
# Max rows used for scatter plots (perf guard for large datasets)
_SCATTER_SAMPLE = 2000
# Min rows needed to draw any chart
_MIN_ROWS = 3
# Max numeric columns in correlation heatmap
_MAX_CORR_COLS = 12


# ---------------------------------------------------------------------------
# Column classification helpers (public — used by app.py)
# ---------------------------------------------------------------------------

def _series_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _is_identifier(col: str, series: pd.Series) -> bool:
    """True when a column looks like a row ID (high cardinality text or integer)."""
    name_lower = col.lower()
    id_words = {"id", "identifier", "uuid", "key", "code", "ref", "number", "no", "num"}
    tokens = set(re.split(r"[_\s\-]+", name_lower))
    if tokens & id_words and series.nunique(dropna=True) / max(len(series), 1) > 0.9:
        return True
    # pure sequential integers with unique values ≈ row count are also IDs
    if pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) == len(series):
        return True
    return False


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return column lists keyed by role: numeric, categorical, datetime, boolean, identifier."""
    result: dict[str, list[str]] = {
        "numeric": [], "categorical": [], "datetime": [], "boolean": [], "identifier": [],
    }
    for col in df.columns:
        series = df[col]
        kind = _series_kind(series)
        if kind == "boolean":
            result["boolean"].append(col)
        elif kind == "datetime":
            result["datetime"].append(col)
        elif kind == "numeric":
            if _is_identifier(col, series):
                result["identifier"].append(col)
            else:
                result["numeric"].append(col)
        else:
            if _is_identifier(col, series):
                result["identifier"].append(col)
            else:
                result["categorical"].append(col)
    return result


# ---------------------------------------------------------------------------
# Profile / statistics helpers (public)
# ---------------------------------------------------------------------------

def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        rows.append({
            "column": column,
            "type": _series_kind(series),
            "non_missing": n - missing,
            "missing": missing,
            "missing_pct": round((missing / n) * 100, 2) if n else 0.0,
            "unique": int(series.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
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


def descriptive_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    sub = df[cols].copy()
    desc = sub.describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index()
    desc = desc.rename(columns={"index": "column"})
    # Round floats for readability
    for c in desc.columns:
        if desc[c].dtype == float:
            desc[c] = desc[c].round(3)
    return desc


def outlier_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < _MIN_ROWS:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        rows.append({"column": col, "outliers": n_out,
                     "outlier_pct": round(n_out / len(s) * 100, 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KPI helpers (public)
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, cols: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Return a list of KPI dicts: {label, value, sub, icon}."""
    kpis: list[dict[str, Any]] = []
    n_rows = len(df)
    n_cols = len(df.columns)
    missing_total = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())

    kpis.append({"label": "Total Records", "value": f"{n_rows:,}",
                 "sub": "Rows in dataset", "icon": "📋"})
    kpis.append({"label": "Total Columns", "value": str(n_cols),
                 "sub": "Features / variables", "icon": "📊"})
    kpis.append({"label": "Missing Values", "value": f"{missing_total:,}",
                 "sub": f"{round(missing_total / max(n_rows * n_cols, 1) * 100, 1)}% of all cells",
                 "icon": "⚠️"})
    kpis.append({"label": "Duplicate Rows", "value": f"{dup_rows:,}",
                 "sub": f"{round(dup_rows / max(n_rows, 1) * 100, 1)}% of rows",
                 "icon": "🔁"})
    kpis.append({"label": "Numeric Columns", "value": str(len(cols["numeric"])),
                 "sub": "Quantitative features", "icon": "🔢"})
    kpis.append({"label": "Categorical Columns", "value": str(len(cols["categorical"])),
                 "sub": "Qualitative features", "icon": "🏷️"})

    # --- Domain-aware bonus KPIs (only when columns clearly match) ---
    num_cols = cols["numeric"]

    # Sales / revenue
    sales_col = _pick_col(df, num_cols, {"amount", "sales", "revenue", "price",
                                          "total", "billing", "payment", "spend"})
    if sales_col:
        total_sales = pd.to_numeric(df[sales_col], errors="coerce").sum()
        kpis.append({"label": f"Total {sales_col}", "value": _fmt_number(total_sales),
                     "sub": "Sum of all values", "icon": "💰"})

    # Average order / transaction value
    if sales_col:
        avg_val = pd.to_numeric(df[sales_col], errors="coerce").mean()
        kpis.append({"label": f"Avg {sales_col}", "value": _fmt_number(avg_val),
                     "sub": "Per record average", "icon": "📈"})

    # Unique customers / patients / students
    id_col = _pick_col(df, cols["identifier"],
                       {"customer", "patient", "student", "employee", "user", "client"})
    if id_col:
        n_unique = df[id_col].nunique(dropna=True)
        role = id_col.split("_")[0].capitalize()
        kpis.append({"label": f"Unique {role}s", "value": f"{n_unique:,}",
                     "sub": f"Distinct {id_col} values", "icon": "👥"})

    return kpis


def _pick_col(df: pd.DataFrame, candidates: list[str], keywords: set[str]) -> str | None:
    for col in candidates:
        tokens = set(re.split(r"[_\s\-]+", col.lower()))
        if tokens & keywords:
            return col
    return None


def _fmt_number(n: float) -> str:
    if math.isnan(n) or math.isinf(n):
        return "N/A"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,.2f}"


# ---------------------------------------------------------------------------
# Automatic insight generation (public)
# ---------------------------------------------------------------------------

def generate_insights(df: pd.DataFrame, cols: dict[str, list[str]],
                      corr: pd.DataFrame | None = None) -> list[str]:
    insights: list[str] = []
    n = len(df)

    # Missing values
    missing_by_col = df.isna().sum()
    worst_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    if not worst_missing.empty:
        col = worst_missing.index[0]
        pct = round(worst_missing.iloc[0] / n * 100, 1)
        insights.append(
            f"**{col}** has the most missing data: {worst_missing.iloc[0]:,} cells ({pct}% of rows)."
        )
    else:
        insights.append("No missing values detected — the dataset is complete.")

    # Top category
    for cat_col in cols["categorical"][:3]:
        vc = df[cat_col].value_counts(dropna=True)
        if vc.empty:
            continue
        top_val = vc.index[0]
        top_pct = round(vc.iloc[0] / n * 100, 1)
        insights.append(
            f"In **{cat_col}**, the most frequent value is **{top_val}** "
            f"({vc.iloc[0]:,} records, {top_pct}%)."
        )
        break

    # Numeric column with highest mean
    for num_col in cols["numeric"]:
        vals = pd.to_numeric(df[num_col], errors="coerce").dropna()
        if vals.empty:
            continue
        insights.append(
            f"**{num_col}** ranges from {vals.min():,.2f} to {vals.max():,.2f} "
            f"with a mean of {vals.mean():,.2f}."
        )
        break

    # Strongest correlation
    if corr is not None and not corr.empty and corr.shape[0] >= 2:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        stacked = upper.stack()
        if not stacked.empty:
            strongest = stacked.abs().idxmax()
            val = round(stacked[strongest], 3)
            a, b = strongest
            direction = "positive" if val > 0 else "negative"
            insights.append(
                f"The strongest correlation is between **{a}** and **{b}** "
                f"({direction}, r = {val}). Note: correlation does not imply causation."
            )

    # Duplicate rows
    dup = int(df.duplicated().sum())
    if dup > 0:
        insights.append(
            f"{dup:,} duplicate row{'s' if dup != 1 else ''} found "
            f"({round(dup / n * 100, 1)}% of records)."
        )

    # Datetime trend hint
    if cols["datetime"]:
        dt_col = cols["datetime"][0]
        insights.append(
            f"A date column **{dt_col}** was detected. "
            "A time-series trend chart has been generated where applicable."
        )

    return insights[:8]  # cap at 8 to keep the panel concise


# ---------------------------------------------------------------------------
# Altair chart builders (public)
# ---------------------------------------------------------------------------

def _safe_df(df: pd.DataFrame, col: str, extra: list[str] | None = None) -> pd.DataFrame:
    """Return a clean subset with nulls dropped for charting."""
    keep = [col] + (extra or [])
    keep = [c for c in keep if c in df.columns]
    return df[keep].dropna(subset=[col]).copy()


def chart_histogram(df: pd.DataFrame, col: str, color: str = _PALETTE[0]) -> Any:
    if not _ALTAIR:
        return None
    s = _safe_df(df, col)
    if len(s) < _MIN_ROWS:
        return None
    chart = (
        alt.Chart(s)
        .mark_bar(color=color, opacity=0.85)
        .encode(
            alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col),
            alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip(f"{col}:Q", bin=True), alt.Tooltip("count()", title="Count")],
        )
        .properties(title=f"Distribution of {col}", height=240)
    )
    return chart


def chart_boxplot(df: pd.DataFrame, col: str, color: str = _PALETTE[0]) -> Any:
    if not _ALTAIR:
        return None
    s = _safe_df(df, col)
    if len(s) < _MIN_ROWS:
        return None
    chart = (
        alt.Chart(s)
        .mark_boxplot(color=color, size=40)
        .encode(
            alt.X(f"{col}:Q", title=col),
            tooltip=[alt.Tooltip(f"{col}:Q")],
        )
        .properties(title=f"Box Plot — {col}", height=180)
    )
    return chart


def chart_bar_categorical(df: pd.DataFrame, col: str) -> Any:
    if not _ALTAIR:
        return None
    vc = (
        df[col].dropna().astype(str)
        .value_counts().head(_MAX_CAT)
        .rename_axis("category").reset_index(name="count")
    )
    if vc.empty:
        return None
    chart = (
        alt.Chart(vc)
        .mark_bar()
        .encode(
            alt.X("count:Q", title="Count"),
            alt.Y("category:N", sort="-x", title=col),
            color=alt.Color("category:N",
                            scale=alt.Scale(range=_PALETTE),
                            legend=None),
            tooltip=["category:N", "count:Q"],
        )
        .properties(title=f"Top categories — {col}", height=max(200, len(vc) * 22))
    )
    return chart


def chart_donut(df: pd.DataFrame, col: str) -> Any:
    if not _ALTAIR:
        return None
    vc = (
        df[col].dropna().astype(str)
        .value_counts().head(10)
        .rename_axis("category").reset_index(name="count")
    )
    if vc.empty:
        return None
    chart = (
        alt.Chart(vc)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("category:N",
                            scale=alt.Scale(range=_PALETTE),
                            legend=alt.Legend(title=col)),
            tooltip=["category:N", "count:Q"],
        )
        .properties(title=f"Distribution — {col}", height=280)
    )
    return chart


def chart_timeseries(df: pd.DataFrame, date_col: str,
                     value_col: str, color: str = _PALETTE[0]) -> Any:
    if not _ALTAIR:
        return None
    s = _safe_df(df, date_col, [value_col])
    if len(s) < _MIN_ROWS:
        return None
    s = s.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col, value_col])
    if len(s) < _MIN_ROWS:
        return None
    # Aggregate by month for readability
    s["_period"] = s[date_col].dt.to_period("M").dt.to_timestamp()
    agg = s.groupby("_period")[value_col].sum().reset_index()
    agg.columns = ["period", value_col]
    chart = (
        alt.Chart(agg)
        .mark_line(point=True, color=color)
        .encode(
            alt.X("period:T", title="Date"),
            alt.Y(f"{value_col}:Q", title=value_col),
            tooltip=[alt.Tooltip("period:T", title="Month"),
                     alt.Tooltip(f"{value_col}:Q")],
        )
        .properties(title=f"{value_col} over Time", height=260)
    )
    return chart


def chart_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                  color_col: str | None = None) -> Any:
    if not _ALTAIR:
        return None
    extra = [y_col] + ([color_col] if color_col else [])
    s = _safe_df(df, x_col, extra)
    if len(s) < _MIN_ROWS:
        return None
    if len(s) > _SCATTER_SAMPLE:
        s = s.sample(_SCATTER_SAMPLE, random_state=42)
    enc: dict[str, Any] = {
        "x": alt.X(f"{x_col}:Q", title=x_col),
        "y": alt.Y(f"{y_col}:Q", title=y_col),
        "tooltip": [f"{x_col}:Q", f"{y_col}:Q"],
    }
    if color_col and color_col in s.columns:
        enc["color"] = alt.Color(f"{color_col}:N",
                                 scale=alt.Scale(range=_PALETTE))
    chart = (
        alt.Chart(s).mark_circle(size=40, opacity=0.6)
        .encode(**enc)
        .properties(title=f"{x_col} vs {y_col}", height=260)
    )
    return chart


def chart_grouped_bar(df: pd.DataFrame, cat_col: str, num_col: str,
                      agg: str = "mean") -> Any:
    if not _ALTAIR:
        return None
    s = _safe_df(df, cat_col, [num_col])
    if len(s) < _MIN_ROWS:
        return None
    fn = "mean" if agg == "mean" else "sum"
    grouped = (
        s.groupby(cat_col)[num_col]
        .agg(fn)
        .reset_index()
        .rename(columns={num_col: "value"})
        .sort_values("value", ascending=False)
        .head(_MAX_CAT)
    )
    if grouped.empty:
        return None
    label = "Average" if agg == "mean" else "Total"
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            alt.X("value:Q", title=f"{label} {num_col}"),
            alt.Y(f"{cat_col}:N", sort="-x", title=cat_col),
            color=alt.Color(f"{cat_col}:N",
                            scale=alt.Scale(range=_PALETTE),
                            legend=None),
            tooltip=[f"{cat_col}:N",
                     alt.Tooltip("value:Q", title=f"{label} {num_col}", format=".2f")],
        )
        .properties(title=f"{label} {num_col} by {cat_col}", height=max(200, len(grouped) * 22))
    )
    return chart


def chart_cat_cat_grouped(df: pd.DataFrame, col_a: str, col_b: str,
                          max_a: int = 12, max_b: int = 8) -> Any:
    """Grouped / stacked bar: count of col_b values within each col_a value.

    col_a is the X-axis grouping (primary category, e.g. Medical Condition).
    col_b is the color/legend dimension (secondary category, e.g. Medication).
    Only the top-N categories of each column are kept to stay readable.
    """
    if not _ALTAIR:
        return None
    s = df[[col_a, col_b]].dropna().copy()
    if len(s) < _MIN_ROWS:
        return None

    # Keep only top categories for each axis to avoid unreadable charts
    top_a = s[col_a].value_counts().head(max_a).index.tolist()
    top_b = s[col_b].value_counts().head(max_b).index.tolist()
    s = s[s[col_a].isin(top_a) & s[col_b].isin(top_b)]
    if len(s) < _MIN_ROWS:
        return None

    agg = s.groupby([col_a, col_b]).size().reset_index(name="count")
    n_b = agg[col_b].nunique()
    height = max(220, len(top_a) * max(20, int(200 / max(n_b, 1))))
    height = min(height, 480)

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            alt.X(f"{col_a}:N", sort="-y", title=col_a,
                  axis=alt.Axis(labelAngle=-35, labelLimit=120)),
            alt.Y("count:Q", title="Record Count"),
            alt.Color(f"{col_b}:N",
                      scale=alt.Scale(range=_PALETTE),
                      legend=alt.Legend(title=col_b, labelLimit=120)),
            alt.Order(f"{col_b}:N"),
            tooltip=[f"{col_a}:N", f"{col_b}:N",
                     alt.Tooltip("count:Q", title="Count")],
        )
        .properties(title=f"{col_b} by {col_a}", height=height)
    )
    return chart


def chart_cat_num_box(df: pd.DataFrame, cat_col: str, num_col: str,
                      max_cats: int = 15) -> Any:
    """Box plot of a numerical column split by a categorical column."""
    if not _ALTAIR:
        return None
    s = df[[cat_col, num_col]].dropna().copy()
    if len(s) < _MIN_ROWS:
        return None
    top_cats = s[cat_col].value_counts().head(max_cats).index.tolist()
    s = s[s[cat_col].isin(top_cats)]
    if len(s) < _MIN_ROWS:
        return None
    height = max(220, len(top_cats) * 22)
    chart = (
        alt.Chart(s)
        .mark_boxplot(size=14)
        .encode(
            alt.Y(f"{cat_col}:N", sort="-x", title=cat_col),
            alt.X(f"{num_col}:Q", title=num_col),
            alt.Color(f"{cat_col}:N",
                      scale=alt.Scale(range=_PALETTE),
                      legend=None),
            tooltip=[f"{cat_col}:N", f"{num_col}:Q"],
        )
        .properties(title=f"{num_col} Distribution by {cat_col}", height=height)
    )
    return chart


def chart_time_cat(df: pd.DataFrame, date_col: str, cat_col: str,
                   max_cats: int = 6) -> Any:
    """Line chart: count of records per category over time (monthly bins)."""
    if not _ALTAIR:
        return None
    s = df[[date_col, cat_col]].dropna().copy()
    if len(s) < _MIN_ROWS:
        return None
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    if len(s) < _MIN_ROWS:
        return None
    top_cats = s[cat_col].value_counts().head(max_cats).index.tolist()
    s = s[s[cat_col].isin(top_cats)]
    s["_period"] = s[date_col].dt.to_period("M").dt.to_timestamp()
    agg = s.groupby(["_period", cat_col]).size().reset_index(name="count")
    agg.columns = ["period", cat_col, "count"]
    chart = (
        alt.Chart(agg)
        .mark_line(point=True)
        .encode(
            alt.X("period:T", title="Date"),
            alt.Y("count:Q", title="Record Count"),
            alt.Color(f"{cat_col}:N",
                      scale=alt.Scale(range=_PALETTE),
                      legend=alt.Legend(title=cat_col)),
            tooltip=["period:T", f"{cat_col}:N",
                     alt.Tooltip("count:Q", title="Count")],
        )
        .properties(title=f"{cat_col} Counts over Time", height=260)
    )
    return chart


def chart_correlation_heatmap(df: pd.DataFrame, num_cols: list[str]) -> Any:
    if not _ALTAIR or len(num_cols) < 2:
        return None
    cols = num_cols[:_MAX_CORR_COLS]
    corr = df[cols].corr(numeric_only=True)
    # Long-form for Altair
    records = []
    for i, r in enumerate(cols):
        for j, c in enumerate(cols):
            val = corr.loc[r, c]
            records.append({"row": r, "col": c, "value": round(val, 3)})
    corr_long = pd.DataFrame(records)
    chart = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            alt.X("col:N", sort=cols, title="", axis=alt.Axis(labelAngle=-45)),
            alt.Y("row:N", sort=cols, title=""),
            alt.Color("value:Q",
                      scale=alt.Scale(scheme=_CORR_SCHEME, domain=[-1, 1]),
                      legend=alt.Legend(title="r")),
            tooltip=["row:N", "col:N", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(title="Correlation Heatmap", height=max(220, len(cols) * 30))
    )
    # Overlay text labels
    text = chart.mark_text(baseline="middle", fontSize=10).encode(
        text=alt.Text("value:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.value) > 0.6",
            alt.value("white"),
            alt.value("#1c2430"),
        ),
    )
    return (chart + text)


# ---------------------------------------------------------------------------
# Main EDA runner (public)
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Run full EDA on *df* and return a result dict consumed by app.py.

    Returns
    -------
    {
        "kpis": list[dict],
        "insights": list[str],
        "cols": dict[str, list[str]],
        "profile": pd.DataFrame,
        "stats": pd.DataFrame,
        "outliers": pd.DataFrame,
        "missing": pd.DataFrame,
        "charts": list[dict],   # {"title": str, "chart": alt.Chart}
        "corr_df": pd.DataFrame,
        "n_rows": int,
        "n_cols": int,
    }
    """
    n = len(df)
    cols = classify_columns(df)

    # --- Profile ---
    profile = profile_columns(df)
    missing_df = (
        profile[profile["missing"] > 0][["column", "missing", "missing_pct"]]
        .sort_values("missing", ascending=False)
        .reset_index(drop=True)
    )

    # --- Stats ---
    stats = descriptive_stats(df, cols["numeric"])
    outliers = outlier_summary(df, cols["numeric"])

    # --- Correlation ---
    corr_df = pd.DataFrame()
    if len(cols["numeric"]) >= 2:
        corr_df = df[cols["numeric"]].corr(numeric_only=True)

    # --- KPIs ---
    kpis = compute_kpis(df, cols)

    # --- Insights ---
    insights = generate_insights(df, cols, corr_df if not corr_df.empty else None)

    # --- Charts ---
    charts: list[dict[str, Any]] = []

    # 1. Time-series — first date col × first numeric measure
    if cols["datetime"] and cols["numeric"]:
        dt_col = cols["datetime"][0]
        val_col = _pick_col(df, cols["numeric"],
                            {"amount", "sales", "revenue", "price", "total",
                             "billing", "score", "value", "count", "qty", "quantity"}) \
                  or cols["numeric"][0]
        c = chart_timeseries(df, dt_col, val_col)
        if c:
            charts.append({"title": f"{val_col} over Time", "section": "time", "chart": c})

    # 2. Donut for first suitable categorical (<=12 unique)
    donut_done = False
    for cat in cols["categorical"]:
        nu = df[cat].nunique(dropna=True)
        if 2 <= nu <= 12:
            c = chart_donut(df, cat)
            if c:
                charts.append({"title": f"Distribution — {cat}", "section": "donut", "chart": c})
                donut_done = True
                break

    # 3. Histograms for up to 3 numeric cols
    for num in cols["numeric"][:3]:
        c = chart_histogram(df, num)
        if c:
            charts.append({"title": f"Histogram — {num}", "section": "hist", "chart": c})

    # 4. Grouped bar: top categorical × top numeric
    if cols["categorical"] and cols["numeric"]:
        cat_col = cols["categorical"][0]
        num_col = _pick_col(df, cols["numeric"],
                            {"amount", "sales", "revenue", "price", "total",
                             "billing", "score", "value", "count"}) \
                  or cols["numeric"][0]
        if df[cat_col].nunique(dropna=True) <= _MAX_CAT:
            c = chart_grouped_bar(df, cat_col, num_col, agg="sum"
                                  if any(kw in num_col.lower() for kw in
                                         ("amount", "sales", "revenue", "total", "billing"))
                                  else "mean")
            if c:
                charts.append({"title": f"{num_col} by {cat_col}", "section": "groupbar", "chart": c})

    # 5. Horizontal bar for second categorical × numeric (top-N products style)
    if len(cols["categorical"]) >= 2 and cols["numeric"]:
        cat2 = cols["categorical"][1]
        num_col2 = cols["numeric"][0]
        if 2 <= df[cat2].nunique(dropna=True) <= _MAX_CAT:
            c = chart_grouped_bar(df, cat2, num_col2, agg="mean")
            if c:
                charts.append({"title": f"Top {cat2} by {num_col2}", "section": "topbar", "chart": c})

    # 6. Box plot for first numeric col
    if cols["numeric"]:
        c = chart_boxplot(df, cols["numeric"][0])
        if c:
            charts.append({"title": f"Box Plot — {cols['numeric'][0]}", "section": "box", "chart": c})

    # 7. Scatter — first two non-correlated numeric cols
    if len(cols["numeric"]) >= 2:
        x_col, y_col = cols["numeric"][0], cols["numeric"][1]
        color_hint = cols["categorical"][0] if cols["categorical"] else None
        c = chart_scatter(df, x_col, y_col, color_hint)
        if c:
            charts.append({"title": f"Scatter — {x_col} vs {y_col}", "section": "scatter", "chart": c})

    # 8. Correlation heatmap
    if len(cols["numeric"]) >= 2:
        c = chart_correlation_heatmap(df, cols["numeric"])
        if c:
            charts.append({"title": "Correlation Heatmap", "section": "corr", "chart": c})

    # 9. Bar chart for remaining categoricals (up to 2 more)
    bar_count = 0
    for cat in cols["categorical"]:
        if bar_count >= 2:
            break
        nu = df[cat].nunique(dropna=True)
        if 2 <= nu <= _MAX_CAT:
            c = chart_bar_categorical(df, cat)
            if c:
                charts.append({"title": f"Bar — {cat}", "section": "catbar", "chart": c})
                bar_count += 1

    return {
        "kpis": kpis,
        "insights": insights,
        "cols": cols,
        "profile": profile,
        "stats": stats,
        "outliers": outliers,
        "missing": missing_df,
        "charts": charts,
        "corr_df": corr_df,
        "n_rows": n,
        "n_cols": len(df.columns),
    }
