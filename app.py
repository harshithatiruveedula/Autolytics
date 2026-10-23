from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import base64
import gzip
import uuid
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from cleaning import clean_dataset, format_cleaning_summary
from query import TABLE_NAME, run_user_query

st.set_page_config(
    page_title="AUTOLYTICS | Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stHeader"]           { height:0!important;visibility:hidden!important; }
[data-testid="stToolbar"]          { right:0.5rem; }
[data-testid="stMainBlockContainer"]{ max-width:100%!important;padding:0!important; }
[data-testid="stAppViewContainer"] > .main { padding-top:0!important; }
[data-testid="stVerticalBlock"]    { gap:0!important; }
iframe[title="autolytics_workspace"]{ width:100%!important;border:0!important; }

/* ── EDA page chrome ─────────────────────────────────────────────────── */
.kpi-grid  { display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin:4px 0 16px; }
.kpi-card  { background:#fff;border:1px solid #e4ebf6;border-radius:12px;padding:14px 16px;box-shadow:0 3px 12px rgba(25,51,95,.04); }
.kpi-icon  { font-size:22px;margin-bottom:4px; }
.kpi-label { font-size:10px;color:#7285a7;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px; }
.kpi-value { font-size:20px;font-weight:850;color:#0b1c3a; }
.kpi-sub   { font-size:10px;color:#94a3b8;margin-top:3px; }
.sec-title { font-size:14px;font-weight:800;color:#14264a;margin:18px 0 8px;display:flex;align-items:center;gap:7px; }
.insight-bar{background:linear-gradient(90deg,#eef4ff,#f5f7ff);border:1px solid #dde9ff;border-radius:9px;padding:9px 14px;font-size:12px;color:#3558a0;line-height:1.6;margin-bottom:6px;}
.chart-card{ background:#fff;border:1px solid #e4ebf6;border-radius:12px;padding:14px 16px 10px;margin-bottom:0;box-shadow:0 3px 12px rgba(25,51,95,.04); }
.chart-head{ display:flex;justify-content:space-between;align-items:center;margin-bottom:6px; }
.chart-title{ font-size:13px;font-weight:750;color:#14264a; }
.chart-desc { font-size:11px;color:#7285a7;margin-top:2px; }
</style>
""", unsafe_allow_html=True)

_COMPONENT_DIR = Path(__file__).parent / "frontend"
_workspace = components.declare_component("autolytics_workspace", path=str(_COMPONENT_DIR))
NAV_PAGES = {"home", "upload", "query", "eda", "modeling", "reports"}

# Chart-type display names and the section tags they map to
_CHART_TYPE_OPTIONS = [
    "Auto (recommended)",
    "Histogram",
    "Box Plot",
    "Bar Chart",
    "Donut / Pie Chart",
    "Scatter Plot",
    "Line / Time Series",
    "Correlation Heatmap",
    "Grouped Bar",
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    defaults: dict = {
        "nav": "home",
        "original_df": None,
        "cleaned_df": None,
        "filename": None,
        "file_size": 0,
        "cleaning_summary": None,
        "query_payload": None,
        "query_error": None,
        "upload_error": None,
        "preview_page": 1,
        "last_event_id": None,
        # EDA
        "eda_kpis": None,
        "eda_insights": None,
        "eda_profile": None,
        "eda_stats": None,
        "eda_outliers": None,
        "eda_missing": None,
        "eda_corr_df": None,
        "eda_cols": None,
        "eda_dataset_id": None,
        # per-chart list: [{"id": str, "title": str, "desc": str, "section": str, "chart": alt.Chart}]
        "eda_charts": None,
        "eda_error": None,
        # user selections
        "eda_selected_cols": None,
        "eda_chart_type": "Auto (recommended)",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def human_size(n: int) -> str:
    size = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_bytes(name: str, payload: bytes) -> pd.DataFrame:
    if not payload:
        raise ValueError("The uploaded file is empty.")
    try:
        if name.lower().endswith(".csv"):
            try:
                return pd.read_csv(BytesIO(payload), low_memory=False, engine="c")
            except UnicodeDecodeError:
                return pd.read_csv(BytesIO(payload), encoding="latin-1", low_memory=False, engine="c")
        if name.lower().endswith(".xlsx"):
            return pd.read_excel(BytesIO(payload), engine="openpyxl")
        raise ValueError("Please upload a CSV or Excel (.xlsx) file.")
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Please upload"):
            raise
        raise ValueError(f"Could not read this file: {exc}") from exc


# ---------------------------------------------------------------------------
# Event processor
# ---------------------------------------------------------------------------

def process_event(event) -> bool:
    if not isinstance(event, dict):
        return False
    event_id = event.get("event_id")
    if event_id and event_id == st.session_state.last_event_id:
        return False
    if event_id:
        st.session_state.last_event_id = event_id

    action = event.get("action")

    if action == "nav":
        nav = event.get("nav")
        if nav in NAV_PAGES:
            st.session_state.nav = nav

    elif action == "page":
        st.session_state.preview_page = max(1, int(event.get("page", 1)))

    elif action == "clear_file":
        for key in ("original_df", "cleaned_df", "filename", "file_size",
                    "cleaning_summary", "query_payload", "query_error",
                    "eda_kpis", "eda_insights", "eda_profile", "eda_stats",
                    "eda_outliers", "eda_missing", "eda_corr_df", "eda_cols",
                    "eda_charts", "eda_dataset_id", "eda_error",
                    "eda_selected_cols"):
            st.session_state[key] = None if key != "file_size" else 0
        st.session_state.preview_page = 1
        st.session_state.upload_error = None

    elif action == "upload_error":
        st.session_state.upload_error = str(event.get("error", "Upload failed."))

    elif action == "upload":
        try:
            name = str(event.get("name", "dataset.csv"))
            payload = base64.b64decode(event.get("data", ""), validate=True)
            if event.get("compressed"):
                payload = gzip.decompress(payload)
            original = load_bytes(name, payload)
            if original.shape[1] == 0:
                raise ValueError("This file does not contain any columns.")
            cleaned, summary = clean_dataset(original)
            if cleaned is None or cleaned.shape[1] == 0:
                raise ValueError("Cleaning did not produce a usable dataset.")
            st.session_state.original_df = original
            st.session_state.cleaned_df = cleaned
            st.session_state.filename = name
            st.session_state.file_size = len(payload)
            st.session_state.cleaning_summary = summary
            st.session_state.query_payload = None
            st.session_state.query_error = None
            st.session_state.upload_error = None
            st.session_state.preview_page = 1
            # Invalidate EDA cache
            for key in ("eda_kpis", "eda_insights", "eda_profile", "eda_stats",
                        "eda_outliers", "eda_missing", "eda_corr_df", "eda_cols",
                        "eda_charts", "eda_dataset_id", "eda_error", "eda_selected_cols"):
                st.session_state[key] = None
        except Exception as exc:
            st.session_state.upload_error = str(exc)

    elif action == "query":
        df = st.session_state.cleaned_df
        question = str(event.get("query", "")).strip()
        if df is None:
            st.session_state.query_error = "Upload a dataset before running a query."
            st.session_state.query_payload = None
        elif not question:
            st.session_state.query_error = "Enter a SQL query or a question about your data."
            st.session_state.query_payload = None
        else:
            try:
                result = run_user_query(df, question)
                result_df = result.get("result")
                if isinstance(result_df, pd.DataFrame):
                    display_df = result_df.head(100)
                    display_rows = display_df.astype(object).where(pd.notna(display_df), None).to_dict("records")
                    display_columns = [str(c) for c in display_df.columns]
                    result_count = len(result_df)
                else:
                    display_rows, display_columns, result_count = [], [], 0
                st.session_state.query_payload = {
                    "sql": result.get("sql", ""),
                    "generated_sql": bool(result.get("generated_sql")),
                    "columns": display_columns,
                    "rows": display_rows,
                    "count": result_count,
                }
                st.session_state.query_error = None
            except Exception as exc:
                st.session_state.query_error = str(exc)
                st.session_state.query_payload = None

    return True


# ---------------------------------------------------------------------------
# EDA computation helpers
# ---------------------------------------------------------------------------

def _ensure_base_eda(df: pd.DataFrame) -> None:
    """Compute KPIs, insights, profile, stats — cheap operations done once per dataset."""
    from eda import (classify_columns, compute_kpis, generate_insights,
                     profile_columns, descriptive_stats, outlier_summary)
    current_id = id(df)
    if st.session_state.eda_dataset_id == current_id and st.session_state.eda_kpis is not None:
        return  # already computed for this exact df object
    cols = classify_columns(df)
    profile = profile_columns(df)
    missing_df = (
        profile[profile["missing"] > 0][["column", "missing", "missing_pct"]]
        .sort_values("missing", ascending=False)
        .reset_index(drop=True)
    )
    stats = descriptive_stats(df, cols["numeric"])
    outliers = outlier_summary(df, cols["numeric"])
    corr_df = pd.DataFrame()
    if len(cols["numeric"]) >= 2:
        corr_df = df[cols["numeric"]].corr(numeric_only=True)
    kpis = compute_kpis(df, cols)
    insights = generate_insights(df, cols, corr_df if not corr_df.empty else None)
    st.session_state.eda_cols = cols
    st.session_state.eda_kpis = kpis
    st.session_state.eda_insights = insights
    st.session_state.eda_profile = profile
    st.session_state.eda_stats = stats
    st.session_state.eda_outliers = outliers
    st.session_state.eda_missing = missing_df
    st.session_state.eda_corr_df = corr_df
    st.session_state.eda_dataset_id = current_id


def _build_charts_for_selection(df: pd.DataFrame, selected_cols: list[str],
                                 chart_type: str) -> list[dict]:
    """Chart planner: evaluate all meaningful column combinations and build charts."""
    from eda import (
        classify_columns,
        chart_histogram, chart_boxplot, chart_bar_categorical, chart_donut,
        chart_timeseries, chart_scatter, chart_grouped_bar, chart_correlation_heatmap,
    )

    # ── inline helpers for combination charts ────────────────────────────────
    # These are also defined in eda.py; we redefine them here so this function
    # works even if an older cached eda.pyc is loaded on Streamlit Cloud.
    import altair as _alt
    _PALETTE = [
        "#2864f0","#7c3aed","#10b981","#f59e0b","#ef4444",
        "#06b6d4","#8b5cf6","#f97316","#14b8a6","#6366f1",
    ]
    _MIN_ROWS = 3

    def _ccc(df2, col_a, col_b, max_a=12, max_b=8):
        """Grouped bar: count of col_b within each col_a."""
        s = df2[[col_a, col_b]].dropna().copy()
        if len(s) < _MIN_ROWS:
            return None
        top_a = s[col_a].value_counts().head(max_a).index.tolist()
        top_b = s[col_b].value_counts().head(max_b).index.tolist()
        s = s[s[col_a].isin(top_a) & s[col_b].isin(top_b)]
        if len(s) < _MIN_ROWS:
            return None
        agg = s.groupby([col_a, col_b]).size().reset_index(name="count")
        h = min(480, max(220, len(top_a) * max(20, int(200 / max(agg[col_b].nunique(), 1)))))
        return (
            _alt.Chart(agg).mark_bar()
            .encode(
                _alt.X(f"{col_a}:N", sort="-y", title=col_a,
                       axis=_alt.Axis(labelAngle=-35, labelLimit=120)),
                _alt.Y("count:Q", title="Record Count"),
                _alt.Color(f"{col_b}:N",
                           scale=_alt.Scale(range=_PALETTE),
                           legend=_alt.Legend(title=col_b, labelLimit=120)),
                _alt.Order(f"{col_b}:N"),
                tooltip=[f"{col_a}:N", f"{col_b}:N",
                         _alt.Tooltip("count:Q", title="Count")],
            )
            .properties(title=f"{col_b} by {col_a}", height=h)
        )

    def _cnb(df2, cat_col, num_col, max_cats=15):
        """Box plot of num_col split by cat_col."""
        s = df2[[cat_col, num_col]].dropna().copy()
        if len(s) < _MIN_ROWS:
            return None
        top = s[cat_col].value_counts().head(max_cats).index.tolist()
        s = s[s[cat_col].isin(top)]
        if len(s) < _MIN_ROWS:
            return None
        return (
            _alt.Chart(s).mark_boxplot(size=14)
            .encode(
                _alt.Y(f"{cat_col}:N", sort="-x", title=cat_col),
                _alt.X(f"{num_col}:Q", title=num_col),
                _alt.Color(f"{cat_col}:N",
                           scale=_alt.Scale(range=_PALETTE), legend=None),
                tooltip=[f"{cat_col}:N", f"{num_col}:Q"],
            )
            .properties(title=f"{num_col} Distribution by {cat_col}",
                        height=max(220, len(top) * 22))
        )

    def _tc(df2, date_col, cat_col, max_cats=6):
        """Line chart: category record counts over monthly time bins."""
        s = df2[[date_col, cat_col]].dropna().copy()
        if len(s) < _MIN_ROWS:
            return None
        s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
        s = s.dropna(subset=[date_col])
        if len(s) < _MIN_ROWS:
            return None
        top = s[cat_col].value_counts().head(max_cats).index.tolist()
        s = s[s[cat_col].isin(top)].copy()
        s["_p"] = s[date_col].dt.to_period("M").dt.to_timestamp()
        agg = s.groupby(["_p", cat_col]).size().reset_index(name="count")
        agg.columns = ["period", cat_col, "count"]
        return (
            _alt.Chart(agg).mark_line(point=True)
            .encode(
                _alt.X("period:T", title="Date"),
                _alt.Y("count:Q", title="Record Count"),
                _alt.Color(f"{cat_col}:N",
                           scale=_alt.Scale(range=_PALETTE),
                           legend=_alt.Legend(title=cat_col)),
                tooltip=["period:T", f"{cat_col}:N",
                         _alt.Tooltip("count:Q", title="Count")],
            )
            .properties(title=f"{cat_col} Counts over Time", height=260)
        )

    # Try to use the eda.py versions first (preferred); fall back to locals.
    try:
        from eda import chart_cat_cat_grouped as _ccc_fn
    except ImportError:
        _ccc_fn = _ccc
    try:
        from eda import chart_cat_num_box as _cnb_fn
    except ImportError:
        _cnb_fn = _cnb
    try:
        from eda import chart_time_cat as _tc_fn
    except ImportError:
        _tc_fn = _tc

    if not selected_cols:
        return []

    sub = df[selected_cols].copy()
    info = classify_columns(sub)
    num_cols  = info["numeric"]
    cat_cols  = info["categorical"]
    dt_cols   = info["datetime"]
    bool_cols = info["boolean"]
    all_cat   = cat_cols + bool_cols

    charts: list[dict] = []
    seen_pairs: set[frozenset] = set()
    MAX_CHARTS = 20

    def add(title, section, desc, cols_used, chart_obj):
        if chart_obj is not None and len(charts) < MAX_CHARTS:
            charts.append({
                "id": str(uuid.uuid4()),
                "title": title,
                "section": section,
                "desc": desc,
                "cols_used": cols_used,
                "chart": chart_obj,
            })

    auto = (chart_type == "Auto (recommended)")

    # ── Explicit chart-type selection ────────────────────────────────────────
    if not auto:
        if chart_type == "Histogram":
            for col in num_cols[:5]:
                add(f"Histogram — {col}", "hist", f"Distribution of {col}.", [col],
                    chart_histogram(sub, col))
        elif chart_type == "Box Plot":
            if all_cat and num_cols:
                cat_col = all_cat[0]
                for num in num_cols[:3]:
                    add(f"{num} by {cat_col}", "catbox",
                        f"Distribution of {num} across {cat_col}.", [cat_col, num],
                        _cnb_fn(sub, cat_col, num))
            else:
                for col in num_cols[:4]:
                    add(f"Box Plot — {col}", "box", f"Spread of {col}.", [col],
                        chart_boxplot(sub, col))
        elif chart_type == "Bar Chart":
            for col in all_cat[:4]:
                add(f"Bar Chart — {col}", "catbar",
                    f"Frequency of each category in {col}.", [col],
                    chart_bar_categorical(sub, col))
        elif chart_type == "Donut / Pie Chart":
            for col in all_cat[:3]:
                if 2 <= sub[col].nunique(dropna=True) <= 12:
                    add(f"Donut — {col}", "donut",
                        f"Proportional breakdown of {col}.", [col],
                        chart_donut(sub, col))
        elif chart_type == "Scatter Plot":
            if len(num_cols) >= 2:
                c_col = all_cat[0] if all_cat else None
                add(f"Scatter — {num_cols[0]} vs {num_cols[1]}", "scatter",
                    f"Relationship between {num_cols[0]} and {num_cols[1]}.",
                    [num_cols[0], num_cols[1]],
                    chart_scatter(sub, num_cols[0], num_cols[1], c_col))
            else:
                st.warning("Scatter plot needs at least 2 numeric columns.")
        elif chart_type == "Line / Time Series":
            if dt_cols and num_cols:
                add(f"{num_cols[0]} over Time", "time",
                    f"Trend of {num_cols[0]} over {dt_cols[0]}.",
                    [dt_cols[0], num_cols[0]],
                    chart_timeseries(sub, dt_cols[0], num_cols[0]))
            elif dt_cols and all_cat:
                add(f"{all_cat[0]} Counts over Time", "time_cat",
                    f"How {all_cat[0]} changes over {dt_cols[0]}.",
                    [dt_cols[0], all_cat[0]],
                    _tc_fn(sub, dt_cols[0], all_cat[0]))
            else:
                st.warning("Time series needs a date column in your selection.")
        elif chart_type == "Correlation Heatmap":
            if len(num_cols) >= 2:
                add("Correlation Heatmap", "corr",
                    "Pairwise correlations between numeric columns.", num_cols,
                    chart_correlation_heatmap(sub, num_cols))
            else:
                st.warning("Correlation heatmap needs at least 2 numeric columns.")
        elif chart_type == "Grouped Bar":
            if len(all_cat) >= 2:
                a, b = all_cat[0], all_cat[1]
                add(f"{b} by {a}", "cat_cat",
                    f"Count of {b} within each {a} group.", [a, b],
                    _ccc_fn(sub, a, b))
            elif all_cat and num_cols:
                cat_col, num_col = all_cat[0], num_cols[0]
                agg = "sum" if any(kw in num_col.lower() for kw in
                                   ("amount","sales","revenue","total","billing")) else "mean"
                add(f"{num_col} by {cat_col}", "groupbar",
                    f"{'Total' if agg=='sum' else 'Average'} {num_col} for each {cat_col}.",
                    [cat_col, num_col],
                    chart_grouped_bar(sub, cat_col, num_col, agg))
            else:
                st.warning("Grouped bar needs categorical columns in your selection.")
        return charts

    # ── Auto planner — Phase 1: all pairwise combinations ────────────────────
    # Cat × Cat
    cat_pairs_done = 0
    for i, col_a in enumerate(all_cat):
        for col_b in all_cat[i + 1:]:
            if cat_pairs_done >= 6:
                break
            pair_key = frozenset([col_a, col_b])
            if pair_key in seen_pairs:
                continue
            nu_a = sub[col_a].nunique(dropna=True)
            nu_b = sub[col_b].nunique(dropna=True)
            axis_col, color_col = (col_a, col_b) if nu_a >= nu_b else (col_b, col_a)
            c = _ccc_fn(sub, axis_col, color_col)
            if c is not None:
                add(f"{color_col} by {axis_col}", "cat_cat",
                    f"Count of {color_col} values within each {axis_col} group.",
                    [axis_col, color_col], c)
                seen_pairs.add(pair_key)
                cat_pairs_done += 1

    # Cat × Num
    cat_num_done = 0
    for cat_col in all_cat:
        for num_col in num_cols:
            if cat_num_done >= 8:
                break
            pair_key = frozenset([cat_col, num_col])
            if pair_key in seen_pairs:
                continue
            if sub[cat_col].nunique(dropna=True) < 2:
                continue
            agg = "sum" if any(kw in num_col.lower() for kw in
                               ("amount","sales","revenue","total","billing","price")) else "mean"
            label = "Total" if agg == "sum" else "Avg"
            c = chart_grouped_bar(sub, cat_col, num_col, agg)
            if c is not None:
                add(f"{label} {num_col} by {cat_col}", "groupbar",
                    f"{label} {num_col} for each {cat_col}.",
                    [cat_col, num_col], c)
                seen_pairs.add(pair_key)
                cat_num_done += 1
            if sub[cat_col].nunique(dropna=True) <= 12:
                c2 = _cnb_fn(sub, cat_col, num_col)
                if c2 is not None:
                    add(f"{num_col} Distribution by {cat_col}", "catbox",
                        f"Spread of {num_col} across {cat_col}.",
                        [cat_col, num_col], c2)

    # Date × Num
    for dt_col in dt_cols:
        for num_col in num_cols:
            pair_key = frozenset([dt_col, num_col])
            if pair_key in seen_pairs:
                continue
            c = chart_timeseries(sub, dt_col, num_col)
            if c is not None:
                add(f"{num_col} over Time", "time",
                    f"Monthly trend of {num_col}.", [dt_col, num_col], c)
                seen_pairs.add(pair_key)

    # Date × Cat
    for dt_col in dt_cols:
        for cat_col in all_cat[:2]:
            pair_key = frozenset([dt_col, cat_col])
            if pair_key in seen_pairs:
                continue
            if 2 <= sub[cat_col].nunique(dropna=True) <= 12:
                c = _tc_fn(sub, dt_col, cat_col)
                if c is not None:
                    add(f"{cat_col} Counts over Time", "time_cat",
                        f"How {cat_col} counts change over {dt_col}.",
                        [dt_col, cat_col], c)
                    seen_pairs.add(pair_key)

    # Num × Num
    for i, x_col in enumerate(num_cols):
        for y_col in num_cols[i + 1:]:
            pair_key = frozenset([x_col, y_col])
            if pair_key in seen_pairs:
                continue
            c_col = all_cat[0] if all_cat else None
            c = chart_scatter(sub, x_col, y_col, c_col)
            if c is not None:
                add(f"Scatter — {x_col} vs {y_col}", "scatter",
                    f"Relationship between {x_col} and {y_col}.",
                    [x_col, y_col], c)
                seen_pairs.add(pair_key)

    # Phase 2: individual distributions for remaining columns
    paired = set(col for pair in seen_pairs for col in pair)
    for cat_col in all_cat:
        if cat_col in paired:
            continue
        nu = sub[cat_col].nunique(dropna=True)
        if 2 <= nu <= 12:
            c = chart_donut(sub, cat_col)
        elif nu > 12:
            c = chart_bar_categorical(sub, cat_col)
        else:
            continue
        if c is not None:
            add(f"Distribution — {cat_col}", "donut" if nu <= 12 else "catbar",
                f"Breakdown of {cat_col}.", [cat_col], c)
            paired.add(cat_col)

    for num_col in num_cols:
        if num_col not in paired:
            c = chart_histogram(sub, num_col)
            if c is not None:
                add(f"Histogram — {num_col}", "hist",
                    f"Distribution of {num_col}.", [num_col], c)

    # Phase 3: correlation heatmap
    if len(num_cols) >= 2:
        c = chart_correlation_heatmap(sub, num_cols)
        if c is not None:
            add("Correlation Heatmap", "corr",
                "Pairwise Pearson correlations.", num_cols, c)

    return charts


# ---------------------------------------------------------------------------
# Component data builder
# ---------------------------------------------------------------------------

def build_component_data() -> dict:
    df = st.session_state.cleaned_df
    rows, columns, total_rows = [], [], 0
    if isinstance(df, pd.DataFrame):
        total_rows = len(df)
        columns = [{"name": str(c), "type": str(df[c].dtype)} for c in df.columns]
        page_size = 5
        pages = max(1, (len(df) + page_size - 1) // page_size)
        page = min(max(1, int(st.session_state.preview_page)), pages)
        st.session_state.preview_page = page
        part = df.iloc[(page - 1) * page_size:page * page_size]
        safe = part.astype(object).where(pd.notna(part), None)
        rows = [
            {str(k): (None if pd.isna(v) else
                      (v.isoformat() if hasattr(v, "isoformat") else
                       v.item() if hasattr(v, "item") else v))
             for k, v in row.items()}
            for row in safe.to_dict("records")
        ]
    summary = ""
    if st.session_state.cleaning_summary:
        summary = format_cleaning_summary(st.session_state.cleaning_summary)
    payload = st.session_state.query_payload
    if payload:
        payload = {
            **payload,
            "rows": [
                {str(k): (None if pd.isna(v) else
                          (v.isoformat() if hasattr(v, "isoformat") else
                           v.item() if hasattr(v, "item") else v))
                 for k, v in row.items()}
                for row in payload.get("rows", [])
            ],
        }
    charts = st.session_state.eda_charts
    return {
        "nav": st.session_state.nav,
        "data": {"columns": columns, "rows": rows, "total_rows": total_rows},
        "file": {"name": st.session_state.filename,
                 "size_label": human_size(st.session_state.file_size)}
              if st.session_state.filename else None,
        "cleaning_summary": summary,
        "query_error": st.session_state.query_error,
        "query_result": payload,
        "upload_error": st.session_state.upload_error,
        "eda_has_data": st.session_state.cleaned_df is not None,
        "eda_has_result": bool(charts),
        "eda_error": st.session_state.eda_error,
        "filename": st.session_state.filename,
    }


# ---------------------------------------------------------------------------
# EDA page renderer
# ---------------------------------------------------------------------------

def _render_eda_page() -> None:
    df = st.session_state.cleaned_df

    # ── No dataset ──────────────────────────────────────────────────────────
    if df is None:
        st.info("Upload a dataset from the **Upload Dataset** page to get started.", icon="📂")
        return

    # Compute cheap base stats once per dataset (KPIs, profile, insights)
    try:
        _ensure_base_eda(df)
    except Exception as exc:
        st.error(f"Could not analyse the dataset: {exc}")
        return

    cols_info      = st.session_state.eda_cols or {}
    all_cols       = list(df.columns)
    id_cols        = set(cols_info.get("identifier", []))
    default_sel    = [c for c in all_cols if c not in id_cols][:12]

    # ── 1. Page header (no empty columns that add height) ───────────────────
    st.markdown(
        "<h2 style='margin:0 0 2px;color:#0b1c3a;font-size:24px;font-weight:850'>"
        "📊 EDA &amp; Charts</h2>"
        "<p style='margin:0;color:#52698e;font-size:12px'>"
        "Select columns, choose a chart type, then click "
        "<strong>Generate Charts</strong>.</p>",
        unsafe_allow_html=True,
    )

    # ── 2. Dataset info banner ───────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#f1f6ff;border:1px solid #dde8ff;border-radius:8px;"
        f"padding:7px 13px;font-size:12px;color:#3558a0;margin:8px 0 12px'>"
        f"🗂 <b>{st.session_state.filename or 'dataset'}</b> — "
        f"{len(df):,} rows × {len(df.columns)} columns</div>",
        unsafe_allow_html=True,
    )

    # ── 3. Column & chart selection (collapsed after first generation) ───────
    has_charts = bool(st.session_state.eda_charts)
    with st.expander("⚙️ Column & Chart Selection", expanded=not has_charts):
        # Column multiselect
        if st.session_state.eda_selected_cols is None:
            st.session_state.eda_selected_cols = default_sel

        ms_col, btn_col = st.columns([5, 1])
        with ms_col:
            selected_cols = st.multiselect(
                "Columns to visualize",
                options=all_cols,
                default=st.session_state.eda_selected_cols,
                key="ms_eda_cols",
                help="ID/key columns are excluded by default but may be added manually.",
            )
            st.session_state.eda_selected_cols = selected_cols
        with btn_col:
            st.write("")          # small spacer so buttons align with the multiselect
            if st.button("All", key="btn_sel_all", use_container_width=True,
                         help="Select all columns"):
                st.session_state.eda_selected_cols = all_cols
                st.rerun()
            if st.button("Clear", key="btn_sel_clear", use_container_width=True,
                         help="Clear selection"):
                st.session_state.eda_selected_cols = []
                st.rerun()

        # Identifier warning
        selected_ids = [c for c in selected_cols if c in id_cols]
        if selected_ids:
            st.caption(
                f"⚠️ ID-like column(s) selected: **{', '.join(selected_ids)}** — "
                "these rarely produce meaningful charts."
            )

        # Chart type + Generate button on one row
        ct_col, gen_col = st.columns([3, 1])
        with ct_col:
            chart_type = st.selectbox(
                "Chart type",
                options=_CHART_TYPE_OPTIONS,
                index=_CHART_TYPE_OPTIONS.index(
                    st.session_state.eda_chart_type
                    if st.session_state.eda_chart_type in _CHART_TYPE_OPTIONS
                    else "Auto (recommended)"
                ),
                key="sel_chart_type",
                label_visibility="collapsed",
            )
            st.session_state.eda_chart_type = chart_type
        with gen_col:
            generate_clicked = st.button(
                "✦ Generate Charts",
                type="primary",
                use_container_width=True,
                key="btn_generate_eda",
            )

    # ── 4. Run chart generation (state mutation + rerun, no rendering here) ─
    if generate_clicked:
        if not selected_cols:
            st.warning("Select at least one column before generating charts.", icon="⚠️")
        else:
            with st.spinner("Building charts…"):
                try:
                    new_charts = _build_charts_for_selection(
                        df, selected_cols, st.session_state.eda_chart_type
                    )
                    if new_charts:
                        st.session_state.eda_charts = new_charts
                        st.session_state.eda_error  = None
                    else:
                        st.session_state.eda_error  = (
                            "No suitable charts could be built for those columns and "
                            "chart type. Try a different selection or choose 'Auto'."
                        )
                        st.session_state.eda_charts = []
                except Exception as exc:
                    st.session_state.eda_error  = f"Chart generation failed: {exc}"
                    st.session_state.eda_charts = []
            st.rerun()   # single controlled rerun — returns here with updated state

    # ── 5. Error banner ─────────────────────────────────────────────────────
    if st.session_state.eda_error:
        st.error(st.session_state.eda_error)

    # ── 6. KPI cards ────────────────────────────────────────────────────────
    kpis = st.session_state.eda_kpis or []
    if kpis:
        st.markdown("<div class='sec-title'>📌 Key Metrics</div>", unsafe_allow_html=True)
        _render_kpi_cards(kpis)

    # ── 7. Generated charts (directly below KPIs — no empty placeholder) ────
    charts = st.session_state.eda_charts
    if charts:
        st.markdown("<div class='sec-title'>📉 Generated Charts</div>", unsafe_allow_html=True)
        _render_chart_cards(charts)
    else:
        # Compact message only — no tall empty container
        st.markdown(
            "<div style='background:#f7f9fd;border:1px dashed #c8d8f0;border-radius:10px;"
            "padding:18px 20px;color:#7285a7;font-size:13px;margin:8px 0 16px;text-align:center'>"
            "📈 Select columns above and click <b>Generate Charts</b> to create visualizations."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── 8. Insights ─────────────────────────────────────────────────────────
    insights = st.session_state.eda_insights or []
    if insights:
        st.markdown("<div class='sec-title'>💡 Auto-generated Insights</div>",
                    unsafe_allow_html=True)
        for ins in insights:
            st.markdown(f"<div class='insight-bar'>• {ins}</div>",
                        unsafe_allow_html=True)

    # ── 9. Data overview (profile, stats, missing, outliers, correlation) ────
    st.markdown("<div class='sec-title'>🔍 Data Overview</div>", unsafe_allow_html=True)
    tab_profile, tab_stats, tab_missing, tab_outliers, tab_corr = st.tabs([
        "Column Profile", "Descriptive Stats", "Missing Values", "Outliers", "Correlation",
    ])
    with tab_profile:
        if st.session_state.eda_profile is not None:
            st.dataframe(st.session_state.eda_profile, use_container_width=True, hide_index=True)
    with tab_stats:
        s = st.session_state.eda_stats
        if s is None or s.empty:
            st.info("No numeric columns for descriptive statistics.")
        else:
            st.dataframe(s, use_container_width=True, hide_index=True)
    with tab_missing:
        m = st.session_state.eda_missing
        if m is None or m.empty:
            st.success("No missing values in this dataset.", icon="✅")
        else:
            st.dataframe(m, use_container_width=True, hide_index=True)
    with tab_outliers:
        o = st.session_state.eda_outliers
        if o is None or o.empty:
            st.info("No numeric columns for outlier analysis.")
        else:
            od = o[o["outliers"] > 0]
            if od.empty:
                st.success("No outliers detected (IQR method).", icon="✅")
            else:
                st.dataframe(od, use_container_width=True, hide_index=True)
    with tab_corr:
        c_df = st.session_state.eda_corr_df
        if c_df is None or c_df.empty:
            st.info("At least 2 numeric columns needed for correlation analysis.")
        else:
            st.dataframe(c_df.round(3), use_container_width=True)

    # ── 10. Downloads ────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border:0;border-top:1px solid #e4ebf6;margin:18px 0 10px'>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sec-title'>⬇️ Downloads</div>", unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("📄 EDA Summary (CSV)", data=_build_eda_csv(),
                           file_name="eda_summary.csv", mime="text/csv",
                           use_container_width=True)
    with dl2:
        st.download_button("📋 EDA Report (TXT)",
                           data=_build_report_text(st.session_state.filename),
                           file_name="eda_report.txt", mime="text/plain",
                           use_container_width=True)
    with dl3:
        if df is not None:
            st.download_button(
                "📊 Export Dataset (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"cleaned_{st.session_state.filename or 'dataset.csv'}",
                mime="text/csv", use_container_width=True,
            )


def _render_kpi_cards(kpis: list[dict]) -> None:
    html = "<div class='kpi-grid'>"
    for kpi in kpis:
        html += (
            f"<div class='kpi-card'>"
            f"<div class='kpi-icon'>{kpi['icon']}</div>"
            f"<div class='kpi-label'>{kpi['label']}</div>"
            f"<div class='kpi-value'>{kpi['value']}</div>"
            f"<div class='kpi-sub'>{kpi['sub']}</div>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_chart_cards(charts: list[dict]) -> None:
    """Render charts in 2-column grid; each card has an individual Remove button."""
    current = list(charts)
    pairs = [current[i:i + 2] for i in range(0, len(current), 2)]
    deleted_ids: list[str] = []

    for pair in pairs:
        cols_layout = st.columns(len(pair))
        for col_widget, item in zip(cols_layout, pair):
            with col_widget:
                cid     = item["id"]
                title   = item.get("title", "Chart")
                desc    = item.get("desc", "")
                section = item.get("section", "")
                cols_used = item.get("cols_used", [])

                # Card border via container + CSS class
                with st.container():
                    # Header row: title + delete button
                    th, td = st.columns([5, 1])
                    with th:
                        cols_label = " · ".join(cols_used) if cols_used else ""
                        st.markdown(
                            f"<div class='chart-head'>"
                            f"<div class='chart-title'>{title}</div>"
                            f"<div class='chart-desc'>{desc}</div>"
                            + (f"<div style='font-size:10px;color:#aab8cc;margin-top:3px'>"
                               f"Columns: {cols_label}</div>" if cols_label else "")
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                    with td:
                        if st.button("✕", key=f"del_{cid}", help="Remove this chart",
                                     use_container_width=True):
                            deleted_ids.append(cid)

                    # Chart rendering
                    try:
                        c = item["chart"]
                        if section not in ("donut", "corr"):
                            c = c.configure_axis(
                                labelFontSize=10, titleFontSize=11,
                                labelColor="#52698e", titleColor="#14264a",
                            ).configure_title(
                                fontSize=13, fontWeight=700,
                                color="#14264a", anchor="start",
                            ).configure_legend(
                                labelFontSize=10, titleFontSize=10,
                            )
                        st.altair_chart(c, use_container_width=True)
                    except Exception as exc:
                        st.caption(f"Chart unavailable: {exc}")

    # Apply deletions and rerun if any chart was removed
    if deleted_ids:
        st.session_state.eda_charts = [
            ch for ch in (st.session_state.eda_charts or [])
            if ch["id"] not in deleted_ids
        ]
        st.rerun()


def _build_eda_csv() -> bytes:
    buf = StringIO()
    profile = st.session_state.eda_profile
    stats = st.session_state.eda_stats
    missing = st.session_state.eda_missing
    outliers = st.session_state.eda_outliers
    insights = st.session_state.eda_insights or []
    buf.write("=== Column Profile ===\n")
    if profile is not None:
        profile.to_csv(buf, index=False)
    buf.write("\n=== Descriptive Statistics ===\n")
    if stats is not None and not stats.empty:
        stats.to_csv(buf, index=False)
    buf.write("\n=== Missing Values ===\n")
    if missing is not None and not missing.empty:
        missing.to_csv(buf, index=False)
    else:
        buf.write("No missing values\n")
    buf.write("\n=== Outlier Summary ===\n")
    if outliers is not None and not outliers.empty:
        outliers.to_csv(buf, index=False)
    buf.write("\n=== Key Insights ===\n")
    for ins in insights:
        buf.write(ins.replace("**", "") + "\n")
    return buf.getvalue().encode("utf-8")


def _build_report_text(filename: str | None) -> bytes:
    df = st.session_state.cleaned_df
    n_rows = len(df) if df is not None else 0
    n_cols = len(df.columns) if df is not None else 0
    profile = st.session_state.eda_profile
    stats = st.session_state.eda_stats
    missing = st.session_state.eda_missing
    insights = st.session_state.eda_insights or []
    lines = [
        "AUTOLYTICS — EDA Report", "=" * 50,
        f"Dataset: {filename or 'uploaded dataset'}",
        f"Rows: {n_rows:,}  |  Columns: {n_cols}", "",
        "KEY INSIGHTS", "-" * 40,
    ]
    for ins in insights:
        lines.append("• " + ins.replace("**", ""))
    if profile is not None:
        lines += ["", "COLUMN PROFILE", "-" * 40, profile.to_string(index=False)]
    if stats is not None and not stats.empty:
        lines += ["", "DESCRIPTIVE STATISTICS", "-" * 40, stats.to_string(index=False)]
    if missing is not None and not missing.empty:
        lines += ["", "MISSING VALUES", "-" * 40, missing.to_string(index=False)]
    lines += ["", "Generated by AUTOLYTICS"]
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_state()
    nav = st.session_state.nav

    # ── EDA page ─────────────────────────────────────────────────────────────
    # The EDA page is 100 % Streamlit-native.  We still instantiate the
    # component so nav events (user clicking another page) are processed,
    # but we collapse it to zero visible height with CSS and only render
    # it with height=0 so no blank space is injected above the EDA content.
    if nav == "eda":
        # Hide the iframe completely on the EDA page so it contributes 0 height.
        st.markdown(
            "<style>iframe[title='autolytics_workspace']"
            "{height:0!important;min-height:0!important;max-height:0!important;"
            "overflow:hidden!important;display:block!important;}</style>",
            unsafe_allow_html=True,
        )
        event = _workspace(
            **build_component_data(),
            key="autolytics_custom_workspace",
            default=None,
        )
        if event and process_event(event):
            st.rerun()
        # Render EDA content directly — no container that adds padding/height
        _render_eda_page()
        return

    # All other pages
    event = _workspace(
        **build_component_data(),
        key="autolytics_custom_workspace",
        default=None,
    )
    if event and process_event(event):
        st.rerun()

    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
    if st.session_state.cleaned_df is not None and st.session_state.cleaning_summary:
        details = st.session_state.cleaning_summary.get("imputation_details", {})
        if details:
            with st.expander("View detailed imputation report"):
                st.dataframe(
                    pd.DataFrame([{"Column": col, **info} for col, info in details.items()]),
                    use_container_width=True, hide_index=True,
                )


if __name__ == "__main__":
    main()
