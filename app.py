from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import base64
import gzip
import html
import uuid
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from cleaning import clean_dataset, format_cleaning_summary
from query import TABLE_NAME, run_user_query
from modeling import train_best_model, prepare_prediction
from domain import detect_domain

st.set_page_config(
    page_title="AUTOLYTICS | Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stHeader"]            { height:0!important;visibility:hidden!important; }
[data-testid="stToolbar"]           { right:0.5rem; }
[data-testid="stMainBlockContainer"]{ max-width:100%!important;padding:0!important; }
[data-testid="stAppViewContainer"] > .main { padding-top:0!important; }
[data-testid="stVerticalBlock"]     { gap:0.65rem!important; }
iframe[title="autolytics_workspace"]{ border:0!important; }

/* ── EDA page ────────────────────────────────────────────────────────────── */
.kpi-grid  { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
             gap:13px;margin:4px 0 18px; }
.kpi-card  { background:#fff;border:1px solid #e1eafb;border-radius:7px;
             padding:14px 15px;box-shadow:0 2px 8px rgba(25,51,95,.035);min-width:0; }
.kpi-icon  { font-size:24px;margin-bottom:6px; }
.kpi-label { font-size:10px;color:#7285a7;font-weight:700;text-transform:uppercase;
             letter-spacing:.04em;margin-bottom:3px; }
.kpi-value { font-size:22px;font-weight:850;color:#0b1c3a; }
.kpi-sub   { font-size:11px;color:#94a3b8;margin-top:4px; }
.sec-head  { font-size:15px;font-weight:800;color:#14264a;margin:22px 0 10px;
             display:flex;align-items:center;gap:8px; }
.insight-bar { background:linear-gradient(90deg,#eef4ff,#f5f7ff);
               border:1px solid #dde9ff;border-radius:9px;
               padding:10px 15px;font-size:12px;color:#3558a0;
               line-height:1.6;margin-bottom:7px; }
.chart-card  { background:#fff;border:1px solid #e1eafb;border-radius:7px;
               padding:12px 13px 8px;box-shadow:0 2px 8px rgba(25,51,95,.035);
               margin-bottom:12px;min-width:0;overflow:hidden; }
.chart-title { font-size:13px;font-weight:800;color:#14264a;line-height:1.35; }
.chart-desc  { font-size:11px;color:#7285a7;margin-top:3px;line-height:1.4; }
.chart-cols  { font-size:10px;color:#b0bcd4;margin-top:3px; }
.empty-charts{ background:#f7f9fd;border:1.5px dashed #c8d8f0;border-radius:8px;
               padding:24px;text-align:center;color:#7285a7;font-size:13px;
               margin:0 0 16px; }
[data-testid="stVerticalBlockBorderWrapper"]{border-color:#e1eafb!important;
               border-radius:7px!important;background:#fff!important;
               box-shadow:0 2px 8px rgba(25,51,95,.035)!important;}
[data-testid="stVerticalBlockBorderWrapper"] > div{padding:10px 12px 7px!important;}
[data-testid="stElementContainer"]{min-width:0!important;}
[data-testid="stAltairChart"]{width:100%!important;min-width:0!important;overflow:visible!important;}
[data-testid="stAltairChart"] > div{width:100%!important;min-width:0!important;}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"]{gap:.35rem!important;}
[data-testid="stAltairChart"] canvas,[data-testid="stAltairChart"] svg{max-width:100%!important;}
@media(max-width:900px){.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:520px){.kpi-grid{grid-template-columns:1fr;}}
</style>
""", unsafe_allow_html=True)

_COMPONENT_DIR = Path(__file__).parent / "frontend"
_workspace = components.declare_component("autolytics_workspace", path=str(_COMPONENT_DIR))
NAV_PAGES = {"home", "upload", "query", "eda", "modeling", "reports"}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    defaults: dict = {
        "nav": "home",
        "sidebar_collapsed": False,
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
        # EDA base stats (computed once per dataset)
        "eda_kpis": None,
        "eda_insights": None,
        "eda_profile": None,
        "eda_stats": None,
        "eda_outliers": None,
        "eda_missing": None,
        "eda_corr_df": None,
        "eda_cols": None,
        "eda_dataset_id": None,
        # domain detection result
        "eda_domain": None,
        "eda_domain_info": None,
        "eda_suggestions": None,
        "eda_story": None,
        # generated charts: list[{id, title, desc, kind, cols_used, chart}]
        "eda_charts": None,
        "eda_error": None,
        "model_info": None,
        "model_dataset_id": None,
        "model_target": None,
        "model_prediction": None,
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
                return pd.read_csv(BytesIO(payload), encoding="latin-1",
                                   low_memory=False, engine="c")
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

    if action == "sidebar_toggle":
        st.session_state.sidebar_collapsed = not bool(st.session_state.get("sidebar_collapsed", False))

    elif action == "nav":
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
                    "eda_charts", "eda_dataset_id", "eda_error", "eda_domain",
                    "eda_domain_info", "eda_suggestions", "eda_story"):
            st.session_state[key] = None if key != "file_size" else 0
        st.session_state.preview_page = 1
        st.session_state.model_info = None
        st.session_state.model_dataset_id = None
        st.session_state.model_target = None
        st.session_state.model_prediction = None
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
            st.session_state.model_info = None
            st.session_state.model_dataset_id = None
            st.session_state.model_target = None
            st.session_state.model_prediction = None
            for key in ("eda_kpis", "eda_insights", "eda_profile", "eda_stats",
                        "eda_outliers", "eda_missing", "eda_corr_df", "eda_cols",
                        "eda_charts", "eda_dataset_id", "eda_error", "eda_domain",
                    "eda_domain_info", "eda_suggestions", "eda_story"):
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
                    display_rows = display_df.astype(object).where(
                        pd.notna(display_df), None).to_dict("records")
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
# EDA base stats (computed once per dataset, cached in session state)
# ---------------------------------------------------------------------------

def _ensure_base_eda(df: pd.DataFrame) -> None:
    from eda import (classify_columns, compute_kpis, generate_insights,
                     profile_columns, descriptive_stats, outlier_summary)
    current_id = id(df)
    if st.session_state.eda_dataset_id == current_id and st.session_state.eda_kpis is not None:
        return
    cols    = classify_columns(df)
    profile = profile_columns(df)
    missing_df = (
        profile[profile["missing"] > 0][["column", "missing", "missing_pct"]]
        .sort_values("missing", ascending=False).reset_index(drop=True)
    )
    stats    = descriptive_stats(df, cols["numeric"])
    outliers = outlier_summary(df, cols["numeric"])
    corr_df  = pd.DataFrame()
    if len(cols["numeric"]) >= 2:
        corr_df = df[cols["numeric"]].corr(numeric_only=True)
    kpis     = compute_kpis(df, cols)
    insights = generate_insights(df, cols, corr_df if not corr_df.empty else None)
    st.session_state.eda_cols       = cols
    st.session_state.eda_kpis       = kpis
    st.session_state.eda_insights   = insights
    st.session_state.eda_profile    = profile
    st.session_state.eda_stats      = stats
    st.session_state.eda_outliers   = outliers
    st.session_state.eda_missing    = missing_df
    st.session_state.eda_corr_df    = corr_df
    st.session_state.eda_dataset_id = current_id


# ---------------------------------------------------------------------------
# Chart builder — converts a domain plan spec into a renderable dict
# ---------------------------------------------------------------------------

def _make_chart(df: pd.DataFrame, spec: dict) -> dict | None:
    """Build one Altair chart from a plan spec returned by domain.plan_hospital_charts."""
    import altair as alt

    _PALETTE = ["#2864f0","#7c3aed","#10b981","#f59e0b","#ef4444",
                "#06b6d4","#8b5cf6","#f97316","#14b8a6","#6366f1"]
    _MIN = 3

    kind  = spec["kind"]
    cols  = spec["cols"]
    title = spec["title"]
    desc  = spec.get("desc", "")

    for c in cols:
        if c not in df.columns:
            return None

    try:
        chart_obj = None

        if kind == "cat_cat":
            ca, cb = cols[0], cols[1]
            s = df[[ca, cb]].dropna().copy()
            if len(s) < _MIN:
                return None
            # Limit high-cardinality data without excluding the common clinical
            # categories; then rename fields to stable aliases so columns with
            # spaces, punctuation, or unusual names render reliably in Altair.
            top_a = s[ca].astype(str).value_counts().head(12).index.tolist()
            top_b = s[cb].astype(str).value_counts().head(10).index.tolist()
            s[ca] = s[ca].astype(str)
            s[cb] = s[cb].astype(str)
            s = s[s[ca].isin(top_a) & s[cb].isin(top_b)]
            if len(s) < _MIN:
                return None
            agg = s.groupby([ca, cb], observed=True).size().reset_index(name="count")
            agg = agg.rename(columns={ca: "category_label", cb: "series_label"})
            category_order = (agg.groupby("category_label")["count"].sum()
                                .sort_values(ascending=False).index.tolist())
            # Horizontal stacked bars preserve every category label in narrow
            # cards and make the full category list readable without clipping.
            chart_obj = (
                alt.Chart(agg).mark_bar()
                .encode(
                    alt.Y("category_label:N", sort=category_order,
                          title=ca,
                          scale=alt.Scale(paddingInner=0.18, paddingOuter=0.08),
                          axis=alt.Axis(labelLimit=240, labelOverlap=False,
                                        labelFontSize=11, tickSize=0)),
                    alt.X("count:Q", title="Record Count"),
                    alt.Color("series_label:N", scale=alt.Scale(range=_PALETTE),
                              legend=alt.Legend(title=cb, labelLimit=180,
                                                orient="bottom", columns=3)),
                    alt.Order("series_label:N"),
                    tooltip=[alt.Tooltip("category_label:N", title=ca),
                             alt.Tooltip("series_label:N", title=cb),
                             alt.Tooltip("count:Q", title="Count")],
                )
                .properties(title=title, height=max(320, len(category_order) * 48))
            )

        elif kind in ("cat_num_box", "catbox"):
            cat_col, num_col = cols[0], cols[1]
            s = df[[cat_col, num_col]].dropna().copy()
            if len(s) < _MIN:
                return None
            top = s[cat_col].value_counts().head(15).index.tolist()
            s = s[s[cat_col].isin(top)]
            if len(s) < _MIN:
                return None
            chart_obj = (
                alt.Chart(s).mark_boxplot(size=14)
                .encode(
                    alt.Y(f"{cat_col}:N", sort="-x", title=cat_col),
                    alt.X(f"{num_col}:Q", title=num_col),
                    alt.Color(f"{cat_col}:N",
                              scale=alt.Scale(range=_PALETTE), legend=None),
                    tooltip=[f"{cat_col}:N", f"{num_col}:Q"],
                )
                .properties(title=title, height=max(240, len(top) * 24))
            )

        elif kind == "cat_num_bar":
            cat_col, num_col = cols[0], cols[1]
            s = df[[cat_col, num_col]].dropna().copy()
            if len(s) < _MIN:
                return None
            agg = (s.groupby(cat_col)[num_col].mean()
                     .reset_index().rename(columns={num_col: "value"})
                     .sort_values("value", ascending=False).head(15))
            if agg.empty:
                return None
            chart_obj = (
                alt.Chart(agg).mark_bar()
                .encode(
                    alt.X("value:Q", title=f"Avg {num_col}"),
                    alt.Y(f"{cat_col}:N", sort="-x", title=cat_col),
                    alt.Color(f"{cat_col}:N",
                              scale=alt.Scale(range=_PALETTE), legend=None),
                    tooltip=[f"{cat_col}:N",
                             alt.Tooltip("value:Q", title=f"Avg {num_col}",
                                         format=".1f")],
                )
                .properties(title=title, height=max(240, len(agg) * 24))
            )

        elif kind == "timeseries_count":
            date_col = cols[0]
            s = df[[date_col]].dropna().copy()
            s[date_col] = pd.to_datetime(s[date_col], errors="coerce", dayfirst=True)
            s = s.dropna()
            if len(s) < _MIN:
                return None
            span_days = int((s[date_col].max() - s[date_col].min()).days)
            if span_days >= 730:
                s["_p"] = s[date_col].dt.to_period("Y").dt.to_timestamp()
                axis_title, axis_format, axis_angle, tooltip_title = "Year", "%Y", 0, "Year"
            else:
                s["_p"] = s[date_col].dt.to_period("M").dt.to_timestamp()
                multi_year = s[date_col].dt.year.nunique() > 1
                axis_title = "Month / Year" if multi_year else "Month"
                axis_format = "%b %Y" if multi_year else "%b"
                axis_angle = -40 if multi_year else 0
                tooltip_title = axis_title
            agg = s.groupby("_p").size().reset_index(name="count")
            agg.columns = ["period", "count"]
            if span_days >= 730:
                # Annual periods are discrete categories, not temporal ticks.
                # This prevents Vega-Lite from emitting duplicate year labels
                # when the browser/container is resized.
                agg["period"] = pd.to_datetime(agg["period"]).dt.year.astype(str)
                years = sorted(agg["period"].unique(), key=int)
                x_encoding = alt.X(
                    "period:O", title="Year", sort=years,
                    axis=alt.Axis(labelAngle=0, labelOverlap="greedy",
                                  labelLimit=70),
                )
                period_tooltip = alt.Tooltip("period:N", title="Year")
            else:
                x_encoding = alt.X(
                    "period:T", title=axis_title,
                    axis=alt.Axis(format=axis_format, labelAngle=axis_angle,
                                  labelOverlap="greedy", labelLimit=100, tickCount=6),
                )
                period_tooltip = alt.Tooltip("period:T", title=tooltip_title,
                                             format=axis_format)
            chart_obj = (
                alt.Chart(agg).mark_line(point=True, color=_PALETTE[0])
                .encode(
                    x_encoding,
                    alt.Y("count:Q", title="Patient Count"),
                    tooltip=[period_tooltip,
                             alt.Tooltip("count:Q", title="Admissions")],
                )
                .properties(title=title, height=300)
            )

        elif kind == "timeseries_cat":
            date_col, cat_col = cols[0], cols[1]
            s = df[[date_col, cat_col]].dropna().copy()
            s[date_col] = pd.to_datetime(s[date_col], errors="coerce", dayfirst=True)
            s = s.dropna(subset=[date_col])
            if len(s) < _MIN:
                return None
            top = s[cat_col].value_counts().head(6).index.tolist()
            s = s[s[cat_col].isin(top)].copy()
            span_days = int((s[date_col].max() - s[date_col].min()).days)
            if span_days >= 730:
                s["_p"] = s[date_col].dt.to_period("Y").dt.to_timestamp()
                axis_title, axis_format, axis_angle = "Year", "%Y", 0
            else:
                s["_p"] = s[date_col].dt.to_period("M").dt.to_timestamp()
                multi_year = s[date_col].dt.year.nunique() > 1
                axis_title = "Month / Year" if multi_year else "Month"
                axis_format = "%b %Y" if multi_year else "%b"
                axis_angle = -40 if multi_year else 0
            agg = s.groupby(["_p", cat_col]).size().reset_index(name="count")
            agg.columns = ["period", cat_col, "count"]
            if span_days >= 730:
                agg["period"] = pd.to_datetime(agg["period"]).dt.year.astype(str)
                years = sorted(agg["period"].unique(), key=int)
                x_encoding = alt.X(
                    "period:O", title="Year", sort=years,
                    axis=alt.Axis(labelAngle=0, labelOverlap="greedy", labelLimit=70),
                )
                period_tooltip = alt.Tooltip("period:N", title="Year")
            else:
                x_encoding = alt.X(
                    "period:T", title=axis_title,
                    axis=alt.Axis(format=axis_format, labelAngle=axis_angle,
                                  labelOverlap="greedy", labelLimit=100, tickCount=6),
                )
                period_tooltip = alt.Tooltip("period:T", title=axis_title,
                                             format=axis_format)
            chart_obj = (
                alt.Chart(agg).mark_line(point=True)
                .encode(
                    x_encoding,
                    alt.Y("count:Q", title="Record Count"),
                    alt.Color(f"{cat_col}:N", scale=alt.Scale(range=_PALETTE),
                              legend=alt.Legend(title=cat_col, labelLimit=160,
                                                orient="bottom", columns=3)),
                    tooltip=[period_tooltip, f"{cat_col}:N",
                             alt.Tooltip("count:Q", title="Count")],
                )
                .properties(title=title, height=320)
            )

        elif kind == "scatter":
            xc, yc = cols[0], cols[1]
            s = df[[xc, yc]].copy()
            s[xc] = pd.to_numeric(s[xc], errors="coerce")
            s[yc] = pd.to_numeric(s[yc], errors="coerce")
            s = s.dropna()
            if len(s) < _MIN:
                return None
            if len(s) > 2000:
                s = s.sample(2000, random_state=42)
            chart_obj = (
                alt.Chart(s).mark_circle(size=40, opacity=0.55, color=_PALETTE[0])
                .encode(
                    alt.X(f"{xc}:Q", title=xc),
                    alt.Y(f"{yc}:Q", title=yc),
                    tooltip=[f"{xc}:Q", f"{yc}:Q"],
                )
                .properties(title=title, height=260)
            )

        elif kind == "histogram":
            nc = cols[0]
            s = pd.to_numeric(df[nc], errors="coerce").dropna()
            if len(s) < _MIN:
                return None
            chart_obj = (
                alt.Chart(pd.DataFrame({nc: s}))
                .mark_bar(color=_PALETTE[0], opacity=0.85)
                .encode(
                    alt.X(f"{nc}:Q", bin=alt.Bin(maxbins=30), title=nc),
                    alt.Y("count()", title="Count"),
                    tooltip=[alt.Tooltip(f"{nc}:Q", bin=True),
                             alt.Tooltip("count()", title="Count")],
                )
                .properties(title=title, height=240)
            )

        elif kind == "bar":
            cat_col = cols[0]
            vc = (df[cat_col].dropna().astype(str)
                    .value_counts().head(20)
                    .rename_axis("category").reset_index(name="count"))
            if vc.empty:
                return None
            chart_obj = (
                alt.Chart(vc).mark_bar()
                .encode(
                    alt.X("count:Q", title="Count"),
                    alt.Y("category:N", sort="-x", title=cat_col),
                    alt.Color("category:N",
                              scale=alt.Scale(range=_PALETTE), legend=None),
                    tooltip=["category:N", "count:Q"],
                )
                .properties(title=title, height=max(200, len(vc) * 24))
            )

        elif kind == "corr":
            from eda import chart_correlation_heatmap
            chart_obj = chart_correlation_heatmap(df, cols)

        if chart_obj is None:
            return None

        return {
            "id": str(uuid.uuid4()),
            "title": title,
            "desc": desc,
            "kind": kind,
            "cols_used": cols,
            "chart": chart_obj,
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Automatic chart generation — domain-aware
# ---------------------------------------------------------------------------

def _generate_auto_charts(df: pd.DataFrame) -> list[dict]:
    """Generate domain-specific charts and evidence-backed narrative outputs."""
    from domain import detect_domain as detect_healthcare, plan_hospital_charts
    from domain_analytics import detect_domain_and_roles, build_domain_charts, generate_narrative

    domain_info = detect_domain_and_roles(df)
    narrative = generate_narrative(df, domain_info)
    st.session_state.eda_domain_info = domain_info
    st.session_state.eda_insights = narrative.get("insights", [])
    st.session_state.eda_suggestions = narrative.get("suggestions", [])
    st.session_state.eda_story = narrative.get("story", "")
    # Existing app banner and report code expect the legacy healthcare flag.
    legacy_domain = detect_healthcare(df)
    legacy_domain["domain_name"] = domain_info.get("name", "generic")
    legacy_domain["domain_label"] = domain_info.get("label", "General")
    legacy_domain["is_healthcare"] = domain_info.get("name") == "healthcare" or legacy_domain.get("is_healthcare", False)
    legacy_domain["confidence"] = max(float(legacy_domain.get("confidence", 0)), float(domain_info.get("confidence", 0)))
    st.session_state.eda_domain = legacy_domain

    # Build charts from semantic roles first. These are single-metric charts;
    # category breakdowns are used only when they are meaningful for the domain.
    charts = build_domain_charts(df, domain_info, max_charts=20)

    # Retain the mature healthcare planner for additional operational plots,
    # without allowing it to replace the new charts or duplicate chart titles.
    if domain_info.get("name") == "healthcare":
        try:
            mappings = legacy_domain.get("mappings", {})
            specs = plan_hospital_charts(df, mappings, max_charts=20)
            known_titles = {item.get("title") for item in charts}
            for spec in specs:
                if len(charts) >= 20:
                    break
                if spec.get("title") in known_titles:
                    continue
                item = _make_chart(df, spec)
                if item:
                    charts.append(item)
                    known_titles.add(item.get("title"))
        except Exception:
            # A missing optional healthcare metric must not block other charts.
            pass

    # If chart libraries are unavailable or the schema is unusually sparse,
    # let the caller display the narrative and generic profile without crashing.
    return charts[:20]


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
        "sidebar_collapsed": bool(st.session_state.get("sidebar_collapsed", False)),
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
        "eda_analysis": {
            "domain": (st.session_state.get("eda_domain_info") or {}).get("label", "General"),
            "key_insights": st.session_state.get("eda_insights") or [],
            "suggestions": st.session_state.get("eda_suggestions") or [],
            "story": st.session_state.get("eda_story") or "",
            "chart_titles": [c.get("title", "Chart") for c in (charts or [])],
        },
        "filename": st.session_state.filename,
    }


# ---------------------------------------------------------------------------
# EDA page renderer
# ---------------------------------------------------------------------------

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
    """Render stable responsive chart cards; charts remain inside their own card."""
    deleted: list[str] = []
    for idx in range(0, len(charts), 2):
        items = charts[idx:idx + 2]
        cols = st.columns([1, 1][:len(items)], gap="medium")
        for col_w, item in zip(cols, items):
            with col_w:
                cid = item.get("id", str(uuid.uuid4()))
                title = str(item.get("title") or "Chart")
                desc = str(item.get("desc") or "")
                kind = str(item.get("kind") or "")
                cols_used = item.get("cols_used", []) or []
                cols_lbl = " · ".join(map(str, cols_used)) if cols_used else ""
                with st.container(border=True):
                    title_col, action_col = st.columns([12, 1], vertical_alignment="top")
                    with title_col:
                        st.markdown(
                            f"<div class='chart-title'>{title}</div>"
                            f"<div class='chart-desc'>{desc}</div>"
                            + (f"<div class='chart-cols'>Columns: {cols_lbl}</div>" if cols_lbl else ""),
                            unsafe_allow_html=True,
                        )
                    with action_col:
                        if st.button("×", key=f"del_{cid}", help="Remove chart"):
                            deleted.append(cid)
                    try:
                        chart = item.get("chart")
                        if chart is None:
                            st.info("This chart could not be built from the available data.")
                            continue
                        chart = chart.properties(title="")
                        # Explicit responsive sizing prevents Altair's default
                        # width from overflowing narrow Streamlit columns.
                        chart = chart.properties(width="container")
                        needs_cfg = kind not in ("donut", "corr")
                        if needs_cfg:
                            chart = (chart
                                .configure_axis(labelFontSize=10, titleFontSize=11,
                                                labelColor="#52698e", titleColor="#14264a",
                                                gridColor="#edf2fa", labelLimit=220)
                                .configure_legend(labelFontSize=10, titleFontSize=10,
                                                  symbolSize=95, padding=8)
                                .configure_view(strokeOpacity=0))
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as exc:
                        st.warning(f"Unable to render this chart: {exc}")
    if deleted:
        deleted_ids = set(deleted)
        st.session_state.eda_charts = [
            chart for chart in (st.session_state.eda_charts or [])
            if chart.get("id") not in deleted_ids
        ]
        st.rerun()


def _render_eda_page() -> None:
    """Render the full EDA & Charts page — no manual column selection."""
    df = st.session_state.cleaned_df

    # ── No dataset ──────────────────────────────────────────────────────────
    if df is None:
        st.markdown(
            "<div style='text-align:center;padding:60px 20px;color:#7285a7'>"
            "<div style='font-size:48px;margin-bottom:12px'>📂</div>"
            "<b>No Dataset Loaded</b><br>"
            "<span style='font-size:13px'>Upload a dataset from the "
            "<b>Upload Dataset</b> page first.</span></div>",
            unsafe_allow_html=True,
        )
        return

    # Compute cheap base stats once per dataset
    try:
        _ensure_base_eda(df)
    except Exception as exc:
        st.error(f"Could not analyse the dataset: {exc}")
        return

    # ── 1. Header + Generate Charts button ──────────────────────────────────
    hdr_l, hdr_r = st.columns([3.2, 2.0], vertical_alignment="center")
    with hdr_l:
        st.markdown(
            "<h2 style='margin:0 0 3px;color:#0b1c3a;font-size:24px;font-weight:850'>"
            "📊 EDA &amp; Charts</h2>"
            "<p style='margin:0;color:#52698e;font-size:12px;line-height:1.5'>"
            "Explore your data with automatic EDA, visualizations and insights. "
            "Discover patterns, trends and key findings — all in one place.</p>",
            unsafe_allow_html=True,
        )
    with hdr_r:
        ds_col, btn_col = st.columns([1.35, 1], vertical_alignment="bottom")
        with ds_col:
            st.markdown(
                f"<div style='font-size:10px;color:#7285a7;margin:0 0 5px'>Dataset</div>"
                f"<div style='background:#fff;border:1px solid #dde8f8;border-radius:7px;"
                f"padding:9px 10px;font-size:11px;color:#243658;overflow:hidden;"
                f"text-overflow:ellipsis;white-space:nowrap'>▤ {st.session_state.filename or 'dataset'}"
                f"</div>", unsafe_allow_html=True,
            )
        with btn_col:
            gen_clicked = st.button(
                "✦ Generate EDA", type="primary", use_container_width=True,
                key="btn_gen_charts",
            )

    # Small metadata line keeps the design compact without a separate full-width panel.
    st.markdown(
        f"<div style='font-size:10px;color:#8295b5;margin:3px 0 10px'>"
        f"{len(df):,} rows · {len(df.columns)} columns</div>",
        unsafe_allow_html=True,
    )

    # ── 3. Chart generation ──────────────────────────────────────────────────
    if gen_clicked:
        with st.spinner("Detecting domain and building charts…"):
            try:
                charts = _generate_auto_charts(df)
                st.session_state.eda_charts = charts
                st.session_state.eda_error  = None
            except Exception as exc:
                st.session_state.eda_error  = f"Chart generation failed: {exc}"
                st.session_state.eda_charts = []
        st.rerun()

    # ── 4. Error ─────────────────────────────────────────────────────────────
    if st.session_state.eda_error:
        st.error(st.session_state.eda_error)

    # ── 5. KPI cards ─────────────────────────────────────────────────────────
    kpis = st.session_state.eda_kpis or []
    if kpis:
        st.markdown(
            "<div class='sec-head'>📌 Key Metrics "
            "<span style='font-size:11px;font-weight:400;color:#94a3b8'>"
            "(Auto-generated)</span></div>",
            unsafe_allow_html=True,
        )
        _render_kpi_cards(kpis)

    # ── 6. Charts ─────────────────────────────────────────────────────────────
    charts = st.session_state.eda_charts
    if charts:
        domain = st.session_state.eda_domain
        domain_info = st.session_state.get("eda_domain_info") or {}
        domain_label = domain_info.get("label", "General")
        domain_icons = {"Healthcare": "🏥", "E-commerce": "🛍️", "Finance": "💳", "Education": "🎓", "General": "📊"}
        st.markdown(
            "<div style='background:#eef6ff;border:1px solid #cfe0ff;"
            "border-radius:8px;padding:7px 14px;font-size:12px;color:#244d91;"
            "margin-bottom:12px'>" + domain_icons.get(domain_label, "📊") + " " +
            domain_label + " dataset detected — showing relevant analysis charts.</div>",
            unsafe_allow_html=True,
        )
        _render_chart_cards(charts)
    else:
        st.markdown(
            "<div class='empty-charts'>"
            "📈 Click <b>Generate Charts</b> above to automatically analyse your dataset."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── 7. Evidence-backed narrative outputs (below generated charts) ────────
    insights = st.session_state.eda_insights or []
    suggestions = st.session_state.get("eda_suggestions") or []
    story = st.session_state.get("eda_story") or ""
    # Keep the requested analytical sequence: WHAT → WHY → WHAT TO DO.
    if charts and insights:
        st.markdown("<div class='sec-head' style='margin-top:24px'>💡 Key Insights</div>", unsafe_allow_html=True)
        for ins in insights:
            safe_insight = html.escape(str(ins).replace("**", ""))
            st.markdown(f"<div class='insight-bar'>• {safe_insight}</div>", unsafe_allow_html=True)
    if charts and story:
        st.markdown("<div class='sec-head' style='margin-top:24px'>📖 Data Storytelling</div>", unsafe_allow_html=True)
        safe_story = html.escape(str(story)).replace("\n\n", "<br><br>").replace("\n", "<br>")
        st.markdown("<div class='chart-card' style='font-size:12px;color:#355078;line-height:1.8'>" + safe_story + "</div>", unsafe_allow_html=True)
    if charts and suggestions:
        st.markdown("<div class='sec-head' style='margin-top:24px'>🧭 Suggestions</div>", unsafe_allow_html=True)
        for suggestion in suggestions:
            safe_suggestion = html.escape(str(suggestion))
            st.markdown(f"<div class='insight-bar'>→ {safe_suggestion}</div>", unsafe_allow_html=True)

    # ── 8. Data overview tabs ─────────────────────────────────────────────────
    st.markdown(
        "<div class='sec-head' style='margin-top:24px'>🔍 Data Overview</div>",
        unsafe_allow_html=True,
    )
    tab_profile, tab_stats, tab_missing, tab_outliers, tab_corr = st.tabs([
        "Column Profile", "Descriptive Stats", "Missing Values", "Outliers", "Correlation",
    ])
    with tab_profile:
        p = st.session_state.eda_profile
        if p is not None:
            st.dataframe(p, use_container_width=True, hide_index=True)
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
            od = o[o["outliers"] > 0] if "outliers" in o.columns else o
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



def _build_report_pdf(filename: str | None) -> bytes:
    """Build a privacy-conscious PDF using the same calculated EDA findings."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export requires reportlab. Install the project requirements and try again.") from exc

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=17*mm, leftMargin=17*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="AutoTitle", parent=styles["Title"], textColor=colors.HexColor("#14264a"), alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name="AutoSection", parent=styles["Heading2"], textColor=colors.HexColor("#2864f0"), spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="AutoBody", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=5))
    story_flow = [Paragraph("AUTOLYTICS - EDA Report", styles["AutoTitle"])]
    df = st.session_state.cleaned_df
    domain_info = st.session_state.get("eda_domain_info") or {}
    story_flow.append(Paragraph(f"Dataset: {html.escape(str(filename or 'uploaded dataset'))}", styles["AutoBody"]))
    if df is not None:
        story_flow.append(Paragraph(f"Records: {len(df):,} &nbsp;&nbsp; Columns: {len(df.columns):,} &nbsp;&nbsp; Domain: {html.escape(str(domain_info.get('label', 'General')))}", styles["AutoBody"]))
    story_flow.append(Spacer(1, 4*mm))
    sections = [
        ("Key Insights", st.session_state.get("eda_insights") or []),
        ("Suggestions", st.session_state.get("eda_suggestions") or []),
    ]
    for heading, items in sections:
        story_flow.append(Paragraph(heading, styles["AutoSection"]))
        if items:
            for item in items:
                story_flow.append(Paragraph("&#8226; " + html.escape(str(item).replace("**", "")), styles["AutoBody"]))
        else:
            story_flow.append(Paragraph("No supported findings are available yet. Generate EDA after uploading a dataset.", styles["AutoBody"]))
    story_text = str(st.session_state.get("eda_story") or "").strip()
    story_flow.append(Paragraph("Data Storytelling", styles["AutoSection"]))
    if story_text:
        for para in story_text.split("\n\n"):
            story_flow.append(Paragraph(html.escape(para), styles["AutoBody"]))
    charts = st.session_state.get("eda_charts") or []
    story_flow.append(Paragraph("Generated Chart References", styles["AutoSection"]))
    if charts:
        chart_rows = [["Chart", "Columns used"]]
        for item in charts:
            chart_rows.append([str(item.get("title", "Chart")), ", ".join(map(str, item.get("cols_used", []) or []))])
        table = Table(chart_rows, colWidths=[75*mm, 90*mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf1ff")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#14264a")),
            ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#d9e4f5")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story_flow.append(table)
    else:
        story_flow.append(Paragraph("No charts have been generated yet.", styles["AutoBody"]))
    profile = st.session_state.get("eda_profile")
    if profile is not None and not profile.empty:
        story_flow.append(Paragraph("Column Profile", styles["AutoSection"]))
        rows = [[str(c) for c in profile.columns]]
        rows.extend([[str(v) for v in row] for row in profile.head(35).itertuples(index=False, name=None)])
        widths = [165*mm / max(1, len(rows[0]))] * len(rows[0])
        table = Table(rows, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf1ff")), ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#d9e4f5")), ("FONTSIZE", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "TOP")]))
        story_flow.append(table)
    story_flow.append(Spacer(1, 5*mm))
    story_flow.append(Paragraph("This report contains aggregate analysis only; raw records and sensitive identifiers are not included.", styles["AutoBody"]))
    doc.build(story_flow)
    return buffer.getvalue()


def _build_eda_csv() -> bytes:
    buf      = StringIO()
    profile  = st.session_state.eda_profile
    stats    = st.session_state.eda_stats
    missing  = st.session_state.eda_missing
    outliers = st.session_state.eda_outliers
    insights = st.session_state.eda_insights or []
    suggestions = st.session_state.get("eda_suggestions") or []
    story = st.session_state.get("eda_story") or ""
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
    if outliers is not None and not outliers.empty:
        buf.write("\n=== Outlier Summary ===\n")
        outliers.to_csv(buf, index=False)
    buf.write("\n=== Key Insights ===\n")
    for ins in insights:
        buf.write(ins.replace("**", "") + "\n")
    buf.write("\n=== Suggestions ===\n")
    for suggestion in suggestions:
        buf.write(suggestion + "\n")
    buf.write("\n=== Data Storytelling ===\n")
    buf.write(story + "\n")
    return buf.getvalue().encode("utf-8")


def _build_report_text(filename: str | None) -> bytes:
    df     = st.session_state.cleaned_df
    n_rows = len(df) if df is not None else 0
    n_cols = len(df.columns) if df is not None else 0
    profile  = st.session_state.eda_profile
    stats    = st.session_state.eda_stats
    missing  = st.session_state.eda_missing
    insights = st.session_state.eda_insights or []
    domain   = st.session_state.eda_domain
    suggestions = st.session_state.get("eda_suggestions") or []
    story = st.session_state.get("eda_story") or ""
    domain_info = st.session_state.get("eda_domain_info") or {}
    lines = [
        "AUTOLYTICS — EDA Report", "=" * 50,
        f"Dataset: {filename or 'uploaded dataset'}",
        f"Rows: {n_rows:,}  |  Columns: {n_cols}",
    ]
    if domain:
        dom_label = domain_info.get("label") or ("Healthcare" if domain.get("is_healthcare") else "General")
        lines.append(f"Domain: {dom_label} (confidence {domain_info.get('confidence', domain.get('confidence', 0)):.0%})")
    lines += ["", "KEY INSIGHTS", "-" * 40]
    for ins in insights:
        lines.append("• " + ins.replace("**", ""))
    lines += ["", "SUGGESTIONS", "-" * 40]
    for suggestion in suggestions:
        lines.append("• " + suggestion)
    if story:
        lines += ["", "DATA STORYTELLING", "-" * 40, story]
    if profile is not None:
        lines += ["", "COLUMN PROFILE", "-" * 40, profile.to_string(index=False)]
    if stats is not None and not stats.empty:
        lines += ["", "DESCRIPTIVE STATISTICS", "-" * 40,
                  stats.to_string(index=False)]
    if missing is not None and not missing.empty:
        lines += ["", "MISSING VALUES", "-" * 40, missing.to_string(index=False)]
    lines += ["", "Generated by AUTOLYTICS"]
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Modeling page
# ---------------------------------------------------------------------------
def _render_modeling_page() -> None:
    df = st.session_state.get("cleaned_df")
    st.markdown("""
    <style>
    .model-hero{display:flex;align-items:center;gap:16px;margin:8px 0 16px}
    .model-hero-icon{width:68px;height:68px;border-radius:13px;background:#e2efff;color:#0878ff;display:flex;align-items:center;justify-content:center;font-size:34px;flex-shrink:0}
    .model-hero h1{font-size:27px!important;color:#0c2047!important;margin:0!important;font-weight:850!important}
    .model-hero p{color:#5b78b1;font-size:13px;margin:4px 0 0}
    .model-dataset{border:1px solid #d9e7ff;background:#fff;border-radius:10px;padding:14px 18px;margin:0 0 16px;color:#132951}
    .model-muted{font-size:12px;color:#6780ae}
    .model-config{background:#f0f7ff;border-radius:9px;padding:14px 17px;color:#2457a5;font-size:12px;line-height:2}
    .model-result{border:1px solid #d8e7fb;background:#f8fbff;border-radius:10px;padding:16px}
    .model-result-value{font-size:24px;font-weight:850;color:#078a65;overflow-wrap:anywhere}
    .model-step{color:#0c2047;font-size:16px;font-weight:800;margin-bottom:4px}
    .model-hero{max-width:100%;overflow-wrap:anywhere}
    [data-testid="stMainBlockContainer"]{padding-top:16px!important;min-height:100vh!important}
    [data-testid="stAppViewContainer"]{min-height:100vh!important}
    
    @media(max-width:700px){.model-hero h1{font-size:23px!important}.model-hero-icon{width:52px;height:52px}}
    </style>
    <div class="model-hero"><div class="model-hero-icon">♧</div><div><h1>Modeling</h1><p>Automatically detect the best model, train it and make predictions from your data.</p></div></div>
    """, unsafe_allow_html=True)
    if df is None or df.empty:
        st.info("Upload a dataset first. The Modeling page will use the cleaned dataset already stored in AUTOLYTICS.")
        return
    dataset_id = f"{st.session_state.get('filename')}:{len(df)}:{len(df.columns)}:{hash(tuple(map(str, df.columns)))}"
    if st.session_state.get("model_dataset_id") != dataset_id:
        st.session_state.model_info = None
        st.session_state.model_prediction = None
        st.session_state.model_dataset_id = dataset_id
    try:
        domain_info = detect_domain(df)
        domain_label = domain_info.get("label", "General") if isinstance(domain_info, dict) else str(domain_info)
    except Exception:
        domain_label = "General"
    st.markdown(f"<div class='model-dataset'><div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center'><div><b>▤ &nbsp;{html.escape(str(st.session_state.get('filename') or 'Uploaded dataset'))}</b> &nbsp; <span style='background:#dcf8ef;color:#07855f;padding:4px 9px;border-radius:20px;font-size:11px;font-weight:700'>✓ Cleaned</span><div class='model-muted' style='margin-top:7px'>{len(df):,} rows &nbsp; · &nbsp; {len(df.columns)} columns &nbsp; · &nbsp; {html.escape(str(domain_label))}</div></div><div style='background:#eef6ff;border-radius:8px;padding:10px 18px;color:#1256ad'><b>Detected Domain</b><br>{html.escape(str(domain_label))}</div></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='model-step'>1. Select Target Variable</div><p class='model-muted'>Choose the column you want to predict. The application will automatically detect the model type and configure the best model.</p>", unsafe_allow_html=True)
    left, right, action = st.columns([1.35, 1, .58], gap="medium")
    with left:
        target = st.selectbox("Target Variable", options=list(map(str, df.columns)), key="model_target_selector", help="Choose the outcome you want the model to learn.")
    with right:
        st.markdown("<div class='model-config'><b>✦ &nbsp;Auto Configuration</b><br>✓ &nbsp; Detects classification / regression<br>✓ &nbsp; Selects suitable algorithms<br>✓ &nbsp; Handles missing values and encodes categories<br>✓ &nbsp; Scales numerical features when needed<br>✓ &nbsp; Chooses task-appropriate metrics</div>", unsafe_allow_html=True)
    with action:
        st.write("")
        st.write("")
        train_clicked = st.button("▶  Train Model", type="primary", use_container_width=True, key="model_train_button")
    current_model = st.session_state.get("model_info")
    if current_model and current_model.get("target") != target:
        st.session_state.model_info = None
        st.session_state.model_prediction = None
        current_model = None
        st.caption("Target changed. Train the model again to use the new target.")
    st.divider()
    st.markdown("<div class='model-step'>2. Model Training</div><p class='model-muted'>Training starts when you click Train Model. Evaluation uses a held-out test split; the selected pipeline is then refit on the available valid rows.</p>", unsafe_allow_html=True)
    if train_clicked:
        try:
            with st.status("Preparing data and comparing models…", expanded=True) as status:
                st.write("Validating target and selecting suitable features")
                info = train_best_model(df, target)
                st.write("Preprocessing features and comparing candidate algorithms")
                st.write(f"Best model: {info['model_name']} · {info['metric']}: {info['best_score']:.3f}")
                status.update(label="Model training completed", state="complete", expanded=False)
            st.session_state.model_info = info
            st.session_state.model_prediction = None
            current_model = info
            st.success(f"Training completed. {info['model_name']} selected using {info['metric']} ({info['best_score']:.3f}).")
        except Exception as exc:
            st.session_state.model_info = None
            current_model = None
            st.error(str(exc))
    if current_model:
        st.markdown(f"<div class='model-result'><b>✓ Model ready</b><br><span class='model-muted'>Task: {html.escape(current_model['task'].title())} · Best model: {html.escape(current_model['model_name'])} · Train rows: {current_model['train_rows']:,} · Test rows: {current_model['test_rows']:,}</span></div>", unsafe_allow_html=True)
        with st.expander("Model comparison details"):
            results = pd.DataFrame(current_model['results'])
            if 'score' in results: results['score'] = results['score'].map(lambda x: round(x, 4) if pd.notna(x) else x)
            st.dataframe(results, use_container_width=True, hide_index=True)
    else:
        st.info("Training has not started. Select a target variable and click Train Model.")
    st.divider()
    st.markdown("<div class='model-step'>3. Make Predictions</div><p class='model-muted'>Enter values for the features used by the trained model. No additional file upload is needed.</p>", unsafe_allow_html=True)
    if not current_model:
        st.info("Select a target and train a model before making predictions.")
        return
    info = current_model
    raw_values = {}
    feature_cols = info['features']
    form_left, form_right = st.columns([1, 1], gap="medium")
    with form_left:
        with st.form("model_prediction_form", clear_on_submit=False):
            st.markdown("**Input Details**")
            grid = st.columns(3)
            for idx, col in enumerate(feature_cols):
                series = df[col]
                nonnull = series.dropna()
                with grid[idx % 3]:
                    if pd.api.types.is_numeric_dtype(series):
                        numeric = pd.to_numeric(series, errors='coerce').dropna()
                        default = float(numeric.median()) if not numeric.empty else 0.0
                        if pd.api.types.is_integer_dtype(series) and abs(default) < 1e12:
                            raw_values[col] = st.number_input(str(col), value=int(round(default)), step=1, key=f"model_input_{col}")
                        else:
                            raw_values[col] = st.number_input(str(col), value=default, key=f"model_input_{col}", format="%.4f")
                    else:
                        vals = nonnull.astype(str).drop_duplicates().tolist()
                        if 0 < len(vals) <= 50:
                            raw_values[col] = st.selectbox(str(col), vals, key=f"model_input_{col}")
                        else:
                            raw_values[col] = st.text_input(str(col), value=str(nonnull.mode().iloc[0]) if not nonnull.empty else "", key=f"model_input_{col}")
            predict_clicked = st.form_submit_button("✦  Predict", type="primary", use_container_width=True)
    with form_right:
        st.markdown("**Prediction Result**")
        result = st.session_state.get("model_prediction")
        if result is None:
            st.markdown("<div class='model-result'><span class='model-muted'>Your prediction will appear here after you enter the feature values and click Predict.</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='model-result'><div class='model-muted'>Predicted {html.escape(info['target'])}</div><div class='model-result-value'>{html.escape(str(result['prediction']))}</div></div>", unsafe_allow_html=True)
            if result.get('probabilities'):
                st.markdown("**Class probabilities**")
                for cls, prob in result['probabilities']:
                    st.write(f"{cls}: {prob:.1%}")
            st.caption("Classification probabilities are model estimates, not guarantees of real-world certainty.")
    if predict_clicked:
        try:
            X_new = prepare_prediction(raw_values, info)
            prediction = info['pipeline'].predict(X_new)[0]
            out = {'prediction': prediction.item() if hasattr(prediction, 'item') else prediction, 'probabilities': None}
            if info['task'] == 'classification' and hasattr(info['pipeline'].named_steps['model'], 'predict_proba'):
                probs = info['pipeline'].predict_proba(X_new)[0]
                classes = info['pipeline'].named_steps['model'].classes_
                out['probabilities'] = sorted([(str(c), float(p)) for c,p in zip(classes, probs)], key=lambda x:x[1], reverse=True)[:5]
            st.session_state.model_prediction = out
            st.rerun()
        except Exception as exc:
            st.error(f"Could not generate prediction: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_state()
    nav = st.session_state.nav

    # ── EDA page ──────────────────────────────────────────────────────────────
    # The EDA page is 100 % Streamlit-native.  We still instantiate the
    # component so nav-click events (switching to another page) are processed,
    # but we must prevent it from taking up vertical space.
    # Two-pronged fix:
    #   1. CSS: display:none removes the iframe from layout flow entirely.
    #   2. index.html reports height:1 (not scrollHeight) for the EDA page,
    #      so Streamlit never allocates space for the component wrapper.
    if nav == "modeling":
        st.markdown(
            "<style>iframe[title*='autolytics_workspace'],[data-testid='stCustomComponentV1'] iframe{position:fixed!important;left:0!important;top:0!important;width:248px!important;height:100vh!important;min-height:100vh!important;max-height:none!important;z-index:1000!important;display:block!important;border:0!important;}[data-testid='stCustomComponentV1']{position:fixed!important;left:0!important;top:0!important;width:0!important;height:0!important;min-width:0!important;min-height:0!important;padding:0!important;margin:0!important;overflow:visible!important;}[data-testid='stMainBlockContainer']{position:relative!important;margin-left:248px!important;width:calc(100% - 248px)!important;max-width:none!important;padding:14px 24px 24px!important;box-sizing:border-box!important}@media(max-width:700px){[data-testid='stMainBlockContainer']{padding:8px!important;margin-left:248px!important;width:calc(100% - 248px)!important}}</style>",
            unsafe_allow_html=True,
        )
        event = _workspace(**build_component_data(), key="autolytics_custom_workspace", default=None)
        if event and process_event(event):
            st.rerun()
        _render_modeling_page()
        return

    if nav == "eda":
        st.markdown(
            "<style>"
            "iframe[title*='autolytics_workspace'],[data-testid='stCustomComponentV1'] iframe{"
            "position:fixed!important;left:0!important;top:0!important;width:"
            + ("64px" if st.session_state.get("sidebar_collapsed", False) else "188px")
            + "!important;height:100vh!important;min-height:100vh!important;"
            "z-index:1000!important;display:block!important;border:0!important;}"
            "[data-testid='stCustomComponentV1']{position:fixed!important;left:0!important;top:0!important;"
            "width:0!important;height:0!important;min-width:0!important;min-height:0!important;"
            "padding:0!important;margin:0!important;overflow:visible!important;}"
            "[data-testid='stMainBlockContainer']{position:relative!important;margin-left:"
            + ("64px" if st.session_state.get("sidebar_collapsed", False) else "188px")
            + "!important;width:calc(100% - "
            + ("64px" if st.session_state.get("sidebar_collapsed", False) else "188px")
            + ")!important;max-width:none!important;padding:14px 18px 18px!important;"
            "box-sizing:border-box!important;}"
            ".chart-title{font-size:13px!important;}"
            "@media(max-width:700px){[data-testid='stMainBlockContainer']{padding:8px!important;}}"
            "@media(max-width:520px){[data-testid='stMainBlockContainer']{margin-left:64px!important;"
            "width:calc(100% - 64px)!important;}}"
            "</style>",
            unsafe_allow_html=True,
        )
        event = _workspace(
            **build_component_data(),
            key="autolytics_custom_workspace",
            default=None,
        )
        if event and process_event(event):
            st.rerun()
        _render_eda_page()
        return

    # ── All other pages ────────────────────────────────────────────────────────
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
                    pd.DataFrame([{"Column": col, **info}
                                  for col, info in details.items()]),
                    use_container_width=True, hide_index=True,
                )


if __name__ == "__main__":
    main()
