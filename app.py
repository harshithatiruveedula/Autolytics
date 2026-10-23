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
[data-testid="stHeader"]            { height:0!important;visibility:hidden!important; }
[data-testid="stToolbar"]           { right:0.5rem; }
[data-testid="stMainBlockContainer"]{ max-width:100%!important;padding:0!important; }
[data-testid="stAppViewContainer"] > .main { padding-top:0!important; }
[data-testid="stVerticalBlock"]     { gap:0!important; }
iframe[title="autolytics_workspace"]{ width:100%!important;border:0!important; }

/* ── EDA page ────────────────────────────────────────────────────────────── */
.kpi-grid  { display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
             gap:14px;margin:4px 0 20px; }
.kpi-card  { background:#fff;border:1px solid #e4ebf6;border-radius:13px;
             padding:16px 18px;box-shadow:0 4px 16px rgba(25,51,95,.05); }
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
.chart-card  { background:#fff;border:1px solid #e4ebf6;border-radius:13px;
               padding:16px 18px 12px;box-shadow:0 4px 16px rgba(25,51,95,.05);
               margin-bottom:16px; }
.chart-title { font-size:13px;font-weight:800;color:#14264a; }
.chart-desc  { font-size:11px;color:#7285a7;margin-top:2px; }
.chart-cols  { font-size:10px;color:#b0bcd4;margin-top:3px; }
.empty-charts{ background:#f7f9fd;border:1.5px dashed #c8d8f0;border-radius:12px;
               padding:24px;text-align:center;color:#7285a7;font-size:13px;
               margin:0 0 20px; }
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
        # generated charts: list[{id, title, desc, kind, cols_used, chart}]
        "eda_charts": None,
        "eda_error": None,
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
                    "eda_charts", "eda_dataset_id", "eda_error", "eda_domain"):
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
            for key in ("eda_kpis", "eda_insights", "eda_profile", "eda_stats",
                        "eda_outliers", "eda_missing", "eda_corr_df", "eda_cols",
                        "eda_charts", "eda_dataset_id", "eda_error", "eda_domain"):
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
            top_a = s[ca].value_counts().head(12).index.tolist()
            top_b = s[cb].value_counts().head(8).index.tolist()
            s = s[s[ca].isin(top_a) & s[cb].isin(top_b)]
            if len(s) < _MIN:
                return None
            agg = s.groupby([ca, cb]).size().reset_index(name="count")
            h = min(480, max(240, len(top_a) * max(22, int(220 / max(agg[cb].nunique(), 1)))))
            chart_obj = (
                alt.Chart(agg).mark_bar()
                .encode(
                    alt.X(f"{ca}:N", sort="-y", title=ca,
                          axis=alt.Axis(labelAngle=-35, labelLimit=130)),
                    alt.Y("count:Q", title="Record Count"),
                    alt.Color(f"{cb}:N", scale=alt.Scale(range=_PALETTE),
                              legend=alt.Legend(title=cb, labelLimit=130)),
                    alt.Order(f"{cb}:N"),
                    tooltip=[f"{ca}:N", f"{cb}:N",
                             alt.Tooltip("count:Q", title="Count")],
                )
                .properties(title=title, height=h)
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
            s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
            s = s.dropna()
            if len(s) < _MIN:
                return None
            s["_p"] = s[date_col].dt.to_period("M").dt.to_timestamp()
            agg = s.groupby("_p").size().reset_index(name="count")
            agg.columns = ["period", "count"]
            chart_obj = (
                alt.Chart(agg).mark_line(point=True, color=_PALETTE[0])
                .encode(
                    alt.X("period:T", title="Date"),
                    alt.Y("count:Q", title="Patient Count"),
                    tooltip=[alt.Tooltip("period:T", title="Month"),
                             alt.Tooltip("count:Q", title="Admissions")],
                )
                .properties(title=title, height=260)
            )

        elif kind == "timeseries_cat":
            date_col, cat_col = cols[0], cols[1]
            s = df[[date_col, cat_col]].dropna().copy()
            s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
            s = s.dropna(subset=[date_col])
            if len(s) < _MIN:
                return None
            top = s[cat_col].value_counts().head(6).index.tolist()
            s = s[s[cat_col].isin(top)].copy()
            s["_p"] = s[date_col].dt.to_period("M").dt.to_timestamp()
            agg = s.groupby(["_p", cat_col]).size().reset_index(name="count")
            agg.columns = ["period", cat_col, "count"]
            chart_obj = (
                alt.Chart(agg).mark_line(point=True)
                .encode(
                    alt.X("period:T", title="Date"),
                    alt.Y("count:Q", title="Record Count"),
                    alt.Color(f"{cat_col}:N", scale=alt.Scale(range=_PALETTE),
                              legend=alt.Legend(title=cat_col)),
                    tooltip=["period:T", f"{cat_col}:N",
                             alt.Tooltip("count:Q", title="Count")],
                )
                .properties(title=title, height=260)
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
    """Detect domain, run the appropriate chart planner, return chart dicts."""
    from domain import detect_domain, plan_hospital_charts
    from eda import classify_columns

    domain = detect_domain(df)
    st.session_state.eda_domain = domain
    mappings = domain["mappings"]

    charts: list[dict] = []

    if domain["is_healthcare"]:
        plan = plan_hospital_charts(df, mappings, max_charts=8)
        for spec in plan:
            result = _make_chart(df, spec)
            if result is not None:
                charts.append(result)
    else:
        # Generic auto planner for non-hospital datasets
        import altair as alt
        _PALETTE = ["#2864f0","#7c3aed","#10b981","#f59e0b","#ef4444",
                    "#06b6d4","#8b5cf6","#f97316","#14b8a6","#6366f1"]
        info = classify_columns(df)
        num_cols = info["numeric"]
        cat_cols = info["categorical"]
        dt_cols  = info["datetime"]
        MAX = 8
        seen: set[frozenset] = set()

        def _add(spec):
            if len(charts) < MAX:
                r = _make_chart(df, spec)
                if r:
                    charts.append(r)

        # Time series
        for dt_col in dt_cols[:1]:
            _add({"kind": "timeseries_count",
                  "title": "Records over Time",
                  "desc": f"Monthly record counts from {dt_col}.",
                  "cols": [dt_col]})

        # Cat × Num grouped bars
        for cat in cat_cols[:3]:
            nu = df[cat].nunique(dropna=True)
            if nu < 2 or nu > 20:
                continue
            for num in num_cols[:2]:
                pk = frozenset([cat, num])
                if pk in seen:
                    continue
                agg = "sum" if any(kw in num.lower() for kw in
                                   ("amount","sales","revenue","total","billing","price")) else "mean"
                label = "Total" if agg == "sum" else "Avg"
                _add({"kind": "cat_num_bar",
                      "title": f"{label} {num} by {cat}",
                      "desc": f"{label} {num} per {cat}.",
                      "cols": [cat, num]})
                seen.add(pk)

        # Cat × Cat
        for i, ca in enumerate(cat_cols[:3]):
            for cb in cat_cols[i + 1:i + 3]:
                pk = frozenset([ca, cb])
                if pk in seen:
                    continue
                nua = df[ca].nunique(dropna=True)
                nub = df[cb].nunique(dropna=True)
                if nua < 2 or nub < 2 or nua > 20 or nub > 20:
                    continue
                _add({"kind": "cat_cat",
                      "title": f"{cb} by {ca}",
                      "desc": f"Count of {cb} within each {ca}.",
                      "cols": [ca, cb]})
                seen.add(pk)

        # Scatter
        if len(num_cols) >= 2:
            _add({"kind": "scatter",
                  "title": f"{num_cols[0]} vs {num_cols[1]}",
                  "desc": f"Relationship between {num_cols[0]} and {num_cols[1]}.",
                  "cols": [num_cols[0], num_cols[1]]})

        # Histograms
        for num in num_cols[:2]:
            if len(charts) >= MAX:
                break
            _add({"kind": "histogram",
                  "title": f"Distribution of {num}",
                  "desc": f"Value distribution of {num}.",
                  "cols": [num]})

        # Category bars
        for cat in cat_cols[:2]:
            if len(charts) >= MAX:
                break
            if 2 <= df[cat].nunique(dropna=True) <= 20:
                _add({"kind": "bar",
                      "title": f"Top {cat} Categories",
                      "desc": f"Frequency of {cat} values.",
                      "cols": [cat]})

        # Correlation heatmap
        if len(num_cols) >= 3 and len(charts) < MAX:
            _add({"kind": "corr",
                  "title": "Correlation Heatmap",
                  "desc": "Pairwise correlations between numerical columns.",
                  "cols": num_cols[:8]})

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
    """Render charts 2-up in a responsive grid; each card has a Remove button."""
    current  = list(charts)
    pairs    = [current[i:i + 2] for i in range(0, len(current), 2)]
    deleted: list[str] = []

    for pair in pairs:
        col_widgets = st.columns(len(pair), gap="medium")
        for col_w, item in zip(col_widgets, pair):
            with col_w:
                cid       = item["id"]
                title     = item.get("title", "Chart")
                desc      = item.get("desc", "")
                kind      = item.get("kind", "")
                cols_used = item.get("cols_used", [])
                cols_lbl  = " · ".join(cols_used) if cols_used else ""

                # Card header HTML
                st.markdown(
                    f"<div class='chart-card'>"
                    f"<div class='chart-title'>{title}</div>"
                    f"<div class='chart-desc'>{desc}</div>"
                    + (f"<div class='chart-cols'>Columns: {cols_lbl}</div>"
                       if cols_lbl else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

                # Delete button (right-aligned)
                _, del_col = st.columns([9, 1])
                with del_col:
                    if st.button("✕", key=f"del_{cid}", help="Remove chart"):
                        deleted.append(cid)

                # Altair chart
                try:
                    c = item["chart"]
                    # configure_* makes it top-level; skip for already-configured kinds
                    needs_cfg = kind not in ("donut", "corr", "cat_cat")
                    if needs_cfg:
                        c = (c
                             .configure_axis(
                                 labelFontSize=10, titleFontSize=11,
                                 labelColor="#52698e", titleColor="#14264a",
                             )
                             .configure_title(
                                 fontSize=13, fontWeight=700,
                                 color="#14264a", anchor="start",
                             )
                             .configure_legend(
                                 labelFontSize=10, titleFontSize=10,
                             ))
                    st.altair_chart(c, use_container_width=True)
                except Exception as exc:
                    st.caption(f"Chart unavailable: {exc}")

    if deleted:
        st.session_state.eda_charts = [
            ch for ch in (st.session_state.eda_charts or [])
            if ch["id"] not in deleted
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
    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown(
            "<h2 style='margin:0 0 3px;color:#0b1c3a;font-size:26px;font-weight:850'>"
            "📊 EDA &amp; Charts</h2>"
            "<p style='margin:0;color:#52698e;font-size:13px'>"
            "Explore your data with automatic analysis, visualizations, and insights. "
            "Discover patterns, trends, and key findings — all in one place.</p>",
            unsafe_allow_html=True,
        )
    with hdr_r:
        st.write("")
        gen_clicked = st.button(
            "✦ Generate Charts",
            type="primary",
            use_container_width=True,
            key="btn_gen_charts",
        )

    # ── 2. Dataset info ──────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#fff;border:1px solid #e4ebf6;border-radius:9px;"
        f"padding:8px 14px;font-size:12px;color:#3558a0;margin:10px 0 14px;"
        f"display:inline-flex;align-items:center;gap:8px'>"
        f"🗂 <b>{st.session_state.filename or 'dataset'}</b> — "
        f"{len(df):,} rows × {len(df.columns)} columns</div>",
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
            "<div class='sec-head'>📌 Key Insights "
            "<span style='font-size:11px;font-weight:400;color:#94a3b8'>"
            "(Auto-generated)</span></div>",
            unsafe_allow_html=True,
        )
        _render_kpi_cards(kpis)

    # ── 6. Charts ─────────────────────────────────────────────────────────────
    charts = st.session_state.eda_charts
    if charts:
        domain = st.session_state.eda_domain
        if domain and domain.get("is_healthcare"):
            st.markdown(
                "<div style='background:#ecfdf4;border:1px solid #a3e6c7;"
                "border-radius:8px;padding:7px 14px;font-size:12px;color:#166442;"
                "margin-bottom:12px'>🏥 Healthcare dataset detected — "
                "showing hospital analysis charts.</div>",
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

    # ── 7. Auto-generated insights ────────────────────────────────────────────
    insights = st.session_state.eda_insights or []
    if insights:
        st.markdown(
            "<div class='sec-head' style='margin-top:24px'>💡 Key Insights</div>",
            unsafe_allow_html=True,
        )
        for ins in insights:
            st.markdown(f"<div class='insight-bar'>• {ins}</div>",
                        unsafe_allow_html=True)

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

    # ── 9. Downloads ─────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border:0;border-top:1px solid #e4ebf6;margin:24px 0 14px'>",
        unsafe_allow_html=True,
    )
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "📄 EDA Summary (CSV)", data=_build_eda_csv(),
            file_name="eda_summary.csv", mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📋 EDA Report (TXT)",
            data=_build_report_text(st.session_state.filename),
            file_name="eda_report.txt", mime="text/plain",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            "📊 Export Dataset (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"cleaned_{st.session_state.filename or 'dataset.csv'}",
            mime="text/csv", use_container_width=True,
        )


def _build_eda_csv() -> bytes:
    buf      = StringIO()
    profile  = st.session_state.eda_profile
    stats    = st.session_state.eda_stats
    missing  = st.session_state.eda_missing
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
    if outliers is not None and not outliers.empty:
        buf.write("\n=== Outlier Summary ===\n")
        outliers.to_csv(buf, index=False)
    buf.write("\n=== Key Insights ===\n")
    for ins in insights:
        buf.write(ins.replace("**", "") + "\n")
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
    lines = [
        "AUTOLYTICS — EDA Report", "=" * 50,
        f"Dataset: {filename or 'uploaded dataset'}",
        f"Rows: {n_rows:,}  |  Columns: {n_cols}",
    ]
    if domain:
        dom_label = "Healthcare" if domain.get("is_healthcare") else "General"
        lines.append(f"Domain: {dom_label} "
                     f"(confidence {domain.get('confidence', 0):.0%})")
    lines += ["", "KEY INSIGHTS", "-" * 40]
    for ins in insights:
        lines.append("• " + ins.replace("**", ""))
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
    if nav == "eda":
        st.markdown(
            "<style>"
            "iframe[title='autolytics_workspace']"
            "{display:none!important;height:0!important;}"
            "[data-testid='stCustomComponentV1']"
            "{height:0!important;min-height:0!important;"
            "padding:0!important;margin:0!important;overflow:hidden!important;}"
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
