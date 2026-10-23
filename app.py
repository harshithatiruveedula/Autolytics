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
.eda-wrap  { padding:20px 28px 36px;background:#f7f9fd;font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }
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
    """Build Altair chart objects for the user-selected columns and chart type."""
    from eda import (
        classify_columns, _pick_col,
        chart_histogram, chart_boxplot, chart_bar_categorical, chart_donut,
        chart_timeseries, chart_scatter, chart_grouped_bar, chart_correlation_heatmap,
        _MAX_CAT, _MIN_ROWS,
    )

    if not selected_cols:
        return []

    # Work only with the selected subset
    sub = df[selected_cols].copy()
    all_cols_info = classify_columns(sub)
    num_cols = all_cols_info["numeric"]
    cat_cols = all_cols_info["categorical"]
    dt_cols = all_cols_info["datetime"]
    bool_cols = all_cols_info["boolean"]

    charts: list[dict] = []

    def add(title: str, section: str, desc: str, chart_obj) -> None:
        if chart_obj is not None:
            charts.append({
                "id": str(uuid.uuid4()),
                "title": title,
                "section": section,
                "desc": desc,
                "chart": chart_obj,
            })

    auto = (chart_type == "Auto (recommended)")

    # ── Explicit type selection ─────────────────────────────────────────
    if not auto:
        if chart_type == "Histogram":
            for col in num_cols[:4]:
                add(f"Histogram — {col}", "hist",
                    f"Distribution of {col} values.",
                    chart_histogram(sub, col))
        elif chart_type == "Box Plot":
            for col in num_cols[:4]:
                add(f"Box Plot — {col}", "box",
                    f"Spread and outliers for {col}.",
                    chart_boxplot(sub, col))
        elif chart_type == "Bar Chart":
            for col in cat_cols[:3]:
                add(f"Bar Chart — {col}", "catbar",
                    f"Frequency of each category in {col}.",
                    chart_bar_categorical(sub, col))
        elif chart_type == "Donut / Pie Chart":
            for col in cat_cols[:2]:
                if 2 <= sub[col].nunique(dropna=True) <= 12:
                    add(f"Donut — {col}", "donut",
                        f"Proportional breakdown of {col}.",
                        chart_donut(sub, col))
        elif chart_type == "Scatter Plot":
            if len(num_cols) >= 2:
                c_col = cat_cols[0] if cat_cols else None
                add(f"Scatter — {num_cols[0]} vs {num_cols[1]}", "scatter",
                    f"Relationship between {num_cols[0]} and {num_cols[1]}.",
                    chart_scatter(sub, num_cols[0], num_cols[1], c_col))
            else:
                st.warning("Scatter plot needs at least 2 numeric columns in your selection.")
        elif chart_type == "Line / Time Series":
            if dt_cols and num_cols:
                add(f"{num_cols[0]} over Time", "time",
                    f"Trend of {num_cols[0]} over {dt_cols[0]}.",
                    chart_timeseries(sub, dt_cols[0], num_cols[0]))
            else:
                st.warning("Time series needs a date column and a numeric column in your selection.")
        elif chart_type == "Correlation Heatmap":
            if len(num_cols) >= 2:
                add("Correlation Heatmap", "corr",
                    "Pairwise correlations between selected numeric columns.",
                    chart_correlation_heatmap(sub, num_cols))
            else:
                st.warning("Correlation heatmap needs at least 2 numeric columns.")
        elif chart_type == "Grouped Bar":
            if cat_cols and num_cols:
                cat_col = cat_cols[0]
                num_col = num_cols[0]
                agg = "sum" if any(kw in num_col.lower() for kw in
                                   ("amount","sales","revenue","total","billing")) else "mean"
                add(f"{num_col} by {cat_col}", "groupbar",
                    f"{'Total' if agg=='sum' else 'Average'} {num_col} for each {cat_col}.",
                    chart_grouped_bar(sub, cat_col, num_col, agg))
            else:
                st.warning("Grouped bar needs at least one categorical and one numeric column.")
        return charts

    # ── Auto mode ─────────────────────────────────────────────────────────
    used_num: set[str] = set()
    used_cat: set[str] = set()

    # 1. Time series
    if dt_cols and num_cols:
        dt_col = dt_cols[0]
        val_col = (_pick_col(sub, num_cols, {"amount","sales","revenue","price","total",
                                              "billing","score","value","qty","quantity"})
                   or num_cols[0])
        add(f"{val_col} over Time", "time",
            f"Monthly aggregated trend of {val_col} over {dt_col}.",
            chart_timeseries(sub, dt_col, val_col))
        used_num.add(val_col)

    # 2. Donut for first low-cardinality categorical
    for cat in cat_cols:
        nu = sub[cat].nunique(dropna=True)
        if 2 <= nu <= 12:
            add(f"Distribution — {cat}", "donut",
                f"Proportional breakdown of {cat} categories.",
                chart_donut(sub, cat))
            used_cat.add(cat)
            break

    # 3. Grouped bar: first categorical × first numeric
    if cat_cols and num_cols:
        cat_col = cat_cols[0]
        num_col = (_pick_col(sub, num_cols, {"amount","sales","revenue","price","total",
                                              "billing","score","value"}) or num_cols[0])
        if sub[cat_col].nunique(dropna=True) <= _MAX_CAT:
            agg = "sum" if any(kw in num_col.lower() for kw in
                               ("amount","sales","revenue","total","billing")) else "mean"
            add(f"{'Total' if agg=='sum' else 'Avg'} {num_col} by {cat_col}", "groupbar",
                f"{'Total' if agg=='sum' else 'Average'} {num_col} grouped by {cat_col}.",
                chart_grouped_bar(sub, cat_col, num_col, agg))
            used_cat.add(cat_col); used_num.add(num_col)

    # 4. Histogram for remaining numeric cols (up to 3)
    hist_count = 0
    for num in num_cols:
        if hist_count >= 3:
            break
        add(f"Histogram — {num}", "hist",
            f"Distribution of {num} values across all records.",
            chart_histogram(sub, num))
        hist_count += 1

    # 5. Box plot for first numeric col
    if num_cols:
        add(f"Box Plot — {num_cols[0]}", "box",
            f"Median, spread, and outliers for {num_cols[0]}.",
            chart_boxplot(sub, num_cols[0]))

    # 6. Scatter
    remaining_num = [c for c in num_cols if c not in used_num]
    if len(remaining_num) >= 2:
        x, y = remaining_num[0], remaining_num[1]
        c_col = next((c for c in cat_cols if c not in used_cat), None)
        add(f"Scatter — {x} vs {y}", "scatter",
            f"Relationship between {x} and {y}.",
            chart_scatter(sub, x, y, c_col))
    elif len(num_cols) >= 2:
        c_col = next((c for c in cat_cols if c not in used_cat), None)
        add(f"Scatter — {num_cols[0]} vs {num_cols[1]}", "scatter",
            f"Relationship between {num_cols[0]} and {num_cols[1]}.",
            chart_scatter(sub, num_cols[0], num_cols[1], c_col))

    # 7. Second categorical bar (if different from donut)
    for cat in cat_cols:
        if cat in used_cat:
            continue
        nu = sub[cat].nunique(dropna=True)
        if 2 <= nu <= _MAX_CAT:
            add(f"Bar Chart — {cat}", "catbar",
                f"Frequency of each value in {cat}.",
                chart_bar_categorical(sub, cat))
            used_cat.add(cat)
            break

    # 8. Correlation heatmap
    if len(num_cols) >= 2:
        add("Correlation Heatmap", "corr",
            "Pairwise Pearson correlations between selected numeric columns.",
            chart_correlation_heatmap(sub, num_cols))

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

    # Compute cheap base stats once per dataset
    try:
        _ensure_base_eda(df)
    except Exception as exc:
        st.error(f"Could not analyse the dataset: {exc}")
        return

    cols_info = st.session_state.eda_cols or {}
    all_cols = list(df.columns)
    id_cols = set(cols_info.get("identifier", []))
    # Default selection: all non-identifier columns, capped at 12
    default_selection = [c for c in all_cols if c not in id_cols][:12]

    # ── Header row ──────────────────────────────────────────────────────────
    h_left, h_right = st.columns([3, 1])
    with h_left:
        st.markdown(
            "<h2 style='margin:0 0 2px;color:#0b1c3a;font-size:24px;font-weight:850'>"
            "📊 EDA &amp; Charts</h2>"
            "<p style='margin:0 0 6px;color:#52698e;font-size:12px'>"
            "Select columns, choose a chart type, then click <b>Generate Charts</b>.</p>",
            unsafe_allow_html=True,
        )
    with h_right:
        st.write("")

    # ── Dataset info ────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#f1f6ff;border:1px solid #dde8ff;border-radius:8px;"
        f"padding:8px 14px;font-size:12px;color:#3558a0;margin-bottom:12px'>"
        f"🗂 <b>{st.session_state.filename or 'dataset'}</b> — "
        f"{len(df):,} rows × {len(df.columns)} columns</div>",
        unsafe_allow_html=True,
    )

    # ── Column selection ────────────────────────────────────────────────────
    with st.expander("⚙️ Column & Chart Selection", expanded=True):
        sel_col, ctrl_col = st.columns([4, 1])
        with sel_col:
            # Use session state to preserve selection across reruns
            if st.session_state.eda_selected_cols is None:
                st.session_state.eda_selected_cols = default_selection
            selected_cols = st.multiselect(
                "Select columns to visualize",
                options=all_cols,
                default=st.session_state.eda_selected_cols,
                key="ms_eda_cols",
                help="Identifier columns (IDs, keys) are excluded by default but can be added manually.",
            )
            st.session_state.eda_selected_cols = selected_cols
        with ctrl_col:
            st.write("")
            st.write("")
            if st.button("Select All", key="btn_sel_all", use_container_width=True):
                st.session_state.eda_selected_cols = all_cols
                st.rerun()
            if st.button("Clear", key="btn_sel_clear", use_container_width=True):
                st.session_state.eda_selected_cols = []
                st.rerun()

        # Chart type selector
        chart_type = st.selectbox(
            "Chart type",
            options=_CHART_TYPE_OPTIONS,
            index=_CHART_TYPE_OPTIONS.index(
                st.session_state.eda_chart_type
                if st.session_state.eda_chart_type in _CHART_TYPE_OPTIONS
                else "Auto (recommended)"
            ),
            key="sel_chart_type",
            help="'Auto' recommends suitable charts based on the selected columns.",
        )
        st.session_state.eda_chart_type = chart_type

        # Identifier warning
        selected_ids = [c for c in selected_cols if c in id_cols]
        if selected_ids:
            st.caption(
                f"⚠️ Identifier column(s) selected: **{', '.join(selected_ids)}**. "
                "These are high-cardinality ID fields and usually won't produce meaningful charts."
            )

        # Generate Charts button
        generate_clicked = st.button(
            "✦ Generate Charts",
            type="primary",
            use_container_width=True,
            key="btn_generate_eda",
        )

    # ── Validation + chart generation ───────────────────────────────────────
    if generate_clicked:
        if not selected_cols:
            st.warning("Select at least one column before generating charts.", icon="⚠️")
        else:
            with st.spinner("Building charts…"):
                try:
                    charts = _build_charts_for_selection(
                        df, selected_cols, st.session_state.eda_chart_type
                    )
                    if not charts:
                        st.session_state.eda_error = (
                            "No suitable charts could be built for the selected columns "
                            "and chart type. Try different columns or choose 'Auto'."
                        )
                        st.session_state.eda_charts = []
                    else:
                        st.session_state.eda_charts = charts
                        st.session_state.eda_error = None
                except Exception as exc:
                    st.session_state.eda_error = f"Chart generation failed: {exc}"
                    st.session_state.eda_charts = []
            st.rerun()

    # ── Errors ──────────────────────────────────────────────────────────────
    if st.session_state.eda_error:
        st.error(st.session_state.eda_error)

    # ── KPI cards ───────────────────────────────────────────────────────────
    kpis = st.session_state.eda_kpis or []
    if kpis:
        st.markdown("<div class='sec-title'>📌 Key Metrics</div>", unsafe_allow_html=True)
        _render_kpi_cards(kpis)

    # ── Charts ──────────────────────────────────────────────────────────────
    charts = st.session_state.eda_charts
    if not charts and not generate_clicked:
        st.markdown(
            "<div style='text-align:center;padding:32px 16px;color:#7285a7'>"
            "<div style='font-size:40px;margin-bottom:8px'>📈</div>"
            "<div style='font-size:14px;font-weight:700;margin-bottom:4px'>No charts yet</div>"
            "<div style='font-size:12px'>Select columns above and click <b>Generate Charts</b>.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif charts:
        st.markdown("<div class='sec-title'>📉 Generated Charts</div>", unsafe_allow_html=True)
        _render_chart_cards(charts)

    # ── Insights ────────────────────────────────────────────────────────────
    insights = st.session_state.eda_insights or []
    if insights:
        st.markdown("<div class='sec-title'>💡 Auto-generated Insights</div>", unsafe_allow_html=True)
        for ins in insights:
            st.markdown(f"<div class='insight-bar'>• {ins}</div>", unsafe_allow_html=True)

    # ── Data overview tabs ───────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>🔍 Data Overview</div>", unsafe_allow_html=True)
    tab_profile, tab_stats, tab_missing, tab_outliers, tab_corr = st.tabs([
        "Column Profile", "Descriptive Stats", "Missing Values", "Outliers", "Correlation",
    ])
    with tab_profile:
        profile = st.session_state.eda_profile
        if profile is not None:
            st.dataframe(profile, use_container_width=True, hide_index=True)
    with tab_stats:
        stats = st.session_state.eda_stats
        if stats is None or stats.empty:
            st.info("No numeric columns found for descriptive statistics.")
        else:
            st.dataframe(stats, use_container_width=True, hide_index=True)
    with tab_missing:
        missing = st.session_state.eda_missing
        if missing is None or missing.empty:
            st.success("No missing values detected in this dataset.", icon="✅")
        else:
            st.dataframe(missing, use_container_width=True, hide_index=True)
    with tab_outliers:
        outliers = st.session_state.eda_outliers
        if outliers is None or outliers.empty:
            st.info("No numeric columns available for outlier analysis.")
        else:
            out_disp = outliers[outliers["outliers"] > 0]
            if out_disp.empty:
                st.success("No outliers detected (IQR method).", icon="✅")
            else:
                st.dataframe(out_disp, use_container_width=True, hide_index=True)
    with tab_corr:
        corr_df = st.session_state.eda_corr_df
        if corr_df is None or corr_df.empty:
            st.info("At least 2 numeric columns are needed for correlation analysis.")
        else:
            st.dataframe(corr_df.round(3), use_container_width=True)

    # ── Downloads ────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border:0;border-top:1px solid #e4ebf6;margin:20px 0 12px'>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sec-title'>⬇️ Downloads</div>", unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        csv_buf = _build_eda_csv()
        st.download_button(
            "📄 EDA Summary (CSV)",
            data=csv_buf, file_name="eda_summary.csv", mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        report_buf = _build_report_text(st.session_state.filename)
        st.download_button(
            "📋 EDA Report (TXT)",
            data=report_buf, file_name="eda_report.txt", mime="text/plain",
            use_container_width=True,
        )
    with dl3:
        if df is not None:
            st.download_button(
                "📊 Export Dataset (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"cleaned_{st.session_state.filename or 'dataset.csv'}",
                mime="text/csv",
                use_container_width=True,
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
    """Render charts in 2-column grid; each card has a Delete button."""
    # Work on a copy so deletion during iteration is safe
    current = list(charts)
    pairs = [current[i:i + 2] for i in range(0, len(current), 2)]
    deleted_ids: list[str] = []

    for pair in pairs:
        cols_layout = st.columns(len(pair))
        for col_widget, chart_item in zip(cols_layout, pair):
            with col_widget:
                cid = chart_item["id"]
                title = chart_item.get("title", "Chart")
                desc = chart_item.get("desc", "")
                section = chart_item.get("section", "")

                # Card header with title + delete button
                c_head, c_del = st.columns([5, 1])
                with c_head:
                    st.markdown(
                        f"<div class='chart-head'>"
                        f"<div><div class='chart-title'>{title}</div>"
                        f"<div class='chart-desc'>{desc}</div></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c_del:
                    if st.button("✕", key=f"del_{cid}", help="Remove this chart",
                                 use_container_width=True):
                        deleted_ids.append(cid)

                # Render the Altair chart
                try:
                    c = chart_item["chart"]
                    if section != "donut":
                        c = c.configure_axis(
                            labelFontSize=10, titleFontSize=11,
                            labelColor="#52698e", titleColor="#14264a",
                        ).configure_title(
                            fontSize=13, fontWeight=700, color="#14264a", anchor="start",
                        ).configure_legend(labelFontSize=10, titleFontSize=10)
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

def main():
    init_state()
    nav = st.session_state.nav

    # EDA page: render sidebar component (compact) + Streamlit content below
    if nav == "eda":
        event = _workspace(
            **build_component_data(),
            key="autolytics_custom_workspace",
            default=None,
        )
        if event and process_event(event):
            st.rerun()
        # Render the full EDA content directly — no wrapper div that adds height
        st.markdown("<div class='eda-wrap'>", unsafe_allow_html=True)
        _render_eda_page()
        st.markdown("</div>", unsafe_allow_html=True)
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
