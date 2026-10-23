from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import base64
import gzip
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
[data-testid="stHeader"] { height: 0 !important; visibility: hidden !important; }
[data-testid="stToolbar"] { right: 0.5rem; }
[data-testid="stMainBlockContainer"] {
    max-width: 100% !important;
    padding: 0 !important;
}
[data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
iframe[title="autolytics_workspace"] { width: 100% !important; border: 0 !important; }

/* ── EDA section styles ─────────────────────────────────────────────────── */
.eda-page {
    padding: 24px 28px 36px;
    background: #f7f9fd;
    min-height: 100vh;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
}
.eda-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 14px;
    margin-bottom: 24px;
}
.eda-title { font-size: 26px; font-weight: 850; color: #0b1c3a; margin: 0 0 4px; }
.eda-sub   { font-size: 13px; color: #52698e; margin: 0; }
.eda-controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}
.kpi-card {
    background: #fff;
    border: 1px solid #e4ebf6;
    border-radius: 13px;
    padding: 16px 18px;
    box-shadow: 0 4px 14px rgba(25,51,95,.04);
}
.kpi-icon  { font-size: 26px; margin-bottom: 6px; }
.kpi-label { font-size: 11px; color: #7285a7; margin-bottom: 3px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.kpi-value { font-size: 22px; font-weight: 850; color: #0b1c3a; }
.kpi-sub   { font-size: 11px; color: #94a3b8; margin-top: 4px; }
.section-title {
    font-size: 15px; font-weight: 800; color: #14264a;
    margin: 24px 0 10px;
    display: flex; align-items: center; gap: 8px;
}
.insight-bar {
    background: linear-gradient(90deg, #eef4ff, #f5f7ff);
    border: 1px solid #dde9ff;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 12px;
    color: #3558a0;
    line-height: 1.6;
    margin-bottom: 8px;
}
.dl-bar {
    display: flex; gap: 10px; flex-wrap: wrap;
    background: #fff;
    border-top: 1px solid #e4ebf6;
    padding: 14px 28px;
    position: sticky;
    bottom: 0;
}
</style>
""", unsafe_allow_html=True)

_COMPONENT_DIR = Path(__file__).parent / "frontend"
_workspace = components.declare_component("autolytics_workspace", path=str(_COMPONENT_DIR))
NAV_PAGES = {"home", "upload", "query", "eda", "modeling", "reports"}


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
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
        "eda_result": None,
        "eda_dataset_id": None,   # id() of the df used for last EDA run
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

def process_event(event):
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
                    "eda_result", "eda_dataset_id", "eda_error"):
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
            # Invalidate cached EDA when a new file is uploaded
            st.session_state.eda_result = None
            st.session_state.eda_dataset_id = None
            st.session_state.eda_error = None
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

    elif action == "generate_eda":
        _run_eda_action()

    return True


def _run_eda_action():
    """Run EDA on the current cleaned dataset and cache the result."""
    df = st.session_state.cleaned_df
    if df is None:
        st.session_state.eda_error = "Upload a dataset before generating charts."
        st.session_state.eda_result = None
        return
    current_id = id(df)
    # Skip recompute if the dataset hasn't changed
    if (st.session_state.eda_result is not None
            and st.session_state.eda_dataset_id == current_id):
        return
    try:
        from eda import run_eda
        with st.spinner("Generating charts and insights…"):
            result = run_eda(df)
        st.session_state.eda_result = result
        st.session_state.eda_dataset_id = current_id
        st.session_state.eda_error = None
    except Exception as exc:
        st.session_state.eda_error = f"EDA failed: {exc}"
        st.session_state.eda_result = None


# ---------------------------------------------------------------------------
# Component data builder
# ---------------------------------------------------------------------------

def build_component_data():
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
        # EDA flags
        "eda_has_data": st.session_state.cleaned_df is not None,
        "eda_has_result": st.session_state.eda_result is not None,
        "eda_error": st.session_state.eda_error,
        "filename": st.session_state.filename,
    }


# ---------------------------------------------------------------------------
# EDA page renderer (Streamlit-native)
# ---------------------------------------------------------------------------

def _render_eda_page():
    """Render the full EDA & Charts page using Streamlit native widgets."""
    df = st.session_state.cleaned_df
    eda = st.session_state.eda_result
    eda_error = st.session_state.eda_error

    # ── Header ──────────────────────────────────────────────────────────────
    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.markdown(
            "<h2 style='margin:0 0 2px;color:#0b1c3a;font-size:26px;font-weight:850'>"
            "📊 EDA &amp; Charts</h2>"
            "<p style='margin:0;color:#52698e;font-size:13px'>"
            "Explore your data with automatic analysis, visualizations, and insights.</p>",
            unsafe_allow_html=True,
        )
    with col_btn:
        st.write("")  # spacing
        generate_clicked = st.button(
            "✦ Generate Charts",
            type="primary",
            use_container_width=True,
            key="btn_generate_eda",
        )

    if generate_clicked:
        _run_eda_action()
        st.rerun()

    # ── No dataset ──────────────────────────────────────────────────────────
    if df is None:
        st.info("Upload a dataset from the **Upload Dataset** page to get started.", icon="📂")
        return

    # ── Errors ──────────────────────────────────────────────────────────────
    if eda_error:
        st.error(eda_error)

    # ── Prompt if not yet generated ─────────────────────────────────────────
    if eda is None:
        st.markdown(
            "<div style='text-align:center;padding:48px 24px;color:#7285a7'>"
            "<div style='font-size:48px;margin-bottom:12px'>📈</div>"
            "<div style='font-size:16px;font-weight:700;margin-bottom:6px'>Ready to analyse</div>"
            "<div style='font-size:13px'>"
            f"Dataset loaded: <b>{st.session_state.filename or 'dataset'}</b> "
            f"({len(df):,} rows × {len(df.columns)} columns)<br>"
            "Click <b>Generate Charts</b> to run automatic EDA."
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    # ── KPI cards ───────────────────────────────────────────────────────────
    kpis = eda["kpis"]
    _render_kpi_cards(kpis)

    # ── Insights ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>💡 Key Insights <span style='font-size:11px;font-weight:400;color:#94a3b8'>(auto-generated)</span></div>", unsafe_allow_html=True)
    for insight in eda["insights"]:
        st.markdown(f"<div class='insight-bar'>• {insight}</div>", unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────────────────────
    charts = eda["charts"]
    if charts:
        st.markdown("<div class='section-title'>📉 Visualizations</div>", unsafe_allow_html=True)

        # Group charts for 2-column layout
        chart_pairs = [charts[i:i+2] for i in range(0, len(charts), 2)]
        for pair in chart_pairs:
            cols_layout = st.columns(len(pair))
            for col_widget, chart_item in zip(cols_layout, pair):
                with col_widget:
                    try:
                        c = chart_item["chart"]
                        # configure_* returns a top-level chart; skip for arc/donut
                        # charts which don't support axis configuration
                        section = chart_item.get("section", "")
                        if section != "donut":
                            c = c.configure_axis(
                                labelFontSize=10, titleFontSize=11,
                                labelColor="#52698e", titleColor="#14264a",
                            ).configure_title(
                                fontSize=13, fontWeight=700, color="#14264a", anchor="start",
                            ).configure_legend(
                                labelFontSize=10, titleFontSize=10,
                            )
                        st.altair_chart(c, use_container_width=True)
                    except Exception as exc:
                        st.caption(f"Chart unavailable: {exc}")
    else:
        st.info("No suitable charts could be generated for this dataset.")

    # ── Data quality + stats ─────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🔍 Data Overview</div>", unsafe_allow_html=True)
    tab_profile, tab_stats, tab_missing, tab_outliers, tab_corr = st.tabs([
        "Column Profile", "Descriptive Stats", "Missing Values", "Outliers", "Correlation",
    ])

    with tab_profile:
        profile = eda["profile"]
        st.dataframe(profile, use_container_width=True, hide_index=True)

    with tab_stats:
        stats = eda["stats"]
        if stats.empty:
            st.info("No numeric columns found for descriptive statistics.")
        else:
            st.dataframe(stats, use_container_width=True, hide_index=True)

    with tab_missing:
        missing = eda["missing"]
        if missing.empty:
            st.success("No missing values detected in this dataset.", icon="✅")
        else:
            st.dataframe(missing, use_container_width=True, hide_index=True)

    with tab_outliers:
        outliers = eda["outliers"]
        if outliers.empty:
            st.info("No numeric columns available for outlier analysis.")
        else:
            outliers_display = outliers[outliers["outliers"] > 0]
            if outliers_display.empty:
                st.success("No outliers detected (IQR method).", icon="✅")
            else:
                st.dataframe(outliers_display, use_container_width=True, hide_index=True)

    with tab_corr:
        corr_df = eda["corr_df"]
        if corr_df.empty:
            st.info("At least 2 numeric columns are needed for correlation analysis.")
        else:
            st.dataframe(corr_df.round(3), use_container_width=True)

    # ── Downloads ────────────────────────────────────────────────────────────
    st.markdown("<hr style='border:0;border-top:1px solid #e4ebf6;margin:28px 0 14px'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⬇️ Downloads</div>", unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        csv_buf = _build_eda_csv(eda)
        st.download_button(
            "📄 Download EDA Summary (CSV)",
            data=csv_buf,
            file_name="eda_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col2:
        report_buf = _build_report_text(eda, st.session_state.filename)
        st.download_button(
            "📋 Download EDA Report (TXT)",
            data=report_buf,
            file_name="eda_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with dl_col3:
        if df is not None:
            csv_full = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Export Dataset (CSV)",
                data=csv_full,
                file_name=f"cleaned_{st.session_state.filename or 'dataset.csv'}",
                mime="text/csv",
                use_container_width=True,
            )


def _render_kpi_cards(kpis: list[dict]):
    """Render KPI cards as HTML inside a Streamlit container."""
    cards_html = "<div class='kpi-grid'>"
    for kpi in kpis:
        cards_html += (
            f"<div class='kpi-card'>"
            f"<div class='kpi-icon'>{kpi['icon']}</div>"
            f"<div class='kpi-label'>{kpi['label']}</div>"
            f"<div class='kpi-value'>{kpi['value']}</div>"
            f"<div class='kpi-sub'>{kpi['sub']}</div>"
            f"</div>"
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def _build_eda_csv(eda: dict) -> bytes:
    """Combine profile + stats into a single CSV for download."""
    buf = StringIO()
    buf.write("=== Column Profile ===\n")
    eda["profile"].to_csv(buf, index=False)
    buf.write("\n=== Descriptive Statistics ===\n")
    if not eda["stats"].empty:
        eda["stats"].to_csv(buf, index=False)
    buf.write("\n=== Missing Values ===\n")
    if not eda["missing"].empty:
        eda["missing"].to_csv(buf, index=False)
    else:
        buf.write("No missing values\n")
    buf.write("\n=== Outlier Summary ===\n")
    if not eda["outliers"].empty:
        eda["outliers"].to_csv(buf, index=False)
    buf.write("\n=== Key Insights ===\n")
    for ins in eda["insights"]:
        buf.write(ins.replace("**", "") + "\n")
    return buf.getvalue().encode("utf-8")


def _build_report_text(eda: dict, filename: str | None) -> bytes:
    lines = [
        "AUTOLYTICS — EDA Report",
        "=" * 50,
        f"Dataset: {filename or 'uploaded dataset'}",
        f"Rows: {eda['n_rows']:,}  |  Columns: {eda['n_cols']}",
        "",
        "KEY INSIGHTS",
        "-" * 40,
    ]
    for ins in eda["insights"]:
        lines.append("• " + ins.replace("**", ""))
    lines += ["", "COLUMN PROFILE", "-" * 40]
    lines.append(eda["profile"].to_string(index=False))
    if not eda["stats"].empty:
        lines += ["", "DESCRIPTIVE STATISTICS", "-" * 40]
        lines.append(eda["stats"].to_string(index=False))
    if not eda["missing"].empty:
        lines += ["", "MISSING VALUES", "-" * 40]
        lines.append(eda["missing"].to_string(index=False))
    lines += ["", "Generated by AUTOLYTICS"]
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_state()

    nav = st.session_state.nav

    # ── EDA page is rendered entirely in Streamlit (not inside the iframe) ──
    if nav == "eda":
        # Still render the component (for nav sidebar) but collapsed/minimal
        event = _workspace(
            **build_component_data(),
            key="autolytics_custom_workspace",
            default=None,
        )
        if event and process_event(event):
            st.rerun()
        # Render full EDA content below the component
        with st.container():
            st.markdown("<div class='eda-page'>", unsafe_allow_html=True)
            _render_eda_page()
            st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── All other pages go through the normal custom component flow ─────────
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
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
