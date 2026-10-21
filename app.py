from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import base64
import gzip
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from cleaning import clean_dataset, format_cleaning_summary
from query import TABLE_NAME, run_user_query

st.set_page_config(page_title="AUTOLYTICS | Data Analytics", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Let the custom interface use the full browser width instead of Streamlit's
# default centered/padded content area. The component itself owns the layout.
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
</style>
""", unsafe_allow_html=True)

_COMPONENT_DIR = Path(__file__).parent / "frontend"
_workspace = components.declare_component("autolytics_workspace", path=str(_COMPONENT_DIR))
NAV_PAGES = {"home", "upload", "query", "eda", "modeling", "reports"}


def init_state():
    defaults = {
        "nav": "home", "original_df": None, "cleaned_df": None,
        "filename": None, "file_size": 0, "cleaning_summary": None,
        "query_payload": None, "query_error": None, "upload_error": None,
        "preview_page": 1, "last_event_id": None,
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
        for key in ("original_df", "cleaned_df", "filename", "file_size", "cleaning_summary", "query_payload", "query_error"):
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
            # Keep the frames returned by the parser/cleaner instead of making
            # extra full-size copies; copying large datasets delays first upload.
            st.session_state.original_df = original
            st.session_state.cleaned_df = cleaned
            # Defer SQLite serialization until a SQL query needs it. This keeps
            # the upload flow responsive for large files; common natural-language
            # aggregations run directly on pandas without creating the SQLite copy.
            st.session_state.filename = name
            st.session_state.file_size = len(payload)
            st.session_state.cleaning_summary = summary
            st.session_state.query_payload = None
            st.session_state.query_error = None
            st.session_state.upload_error = None
            st.session_state.preview_page = 1
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
        rows = [{str(k): (None if pd.isna(v) else (v.isoformat() if hasattr(v, "isoformat") else v.item() if hasattr(v, "item") else v)) for k, v in row.items()} for row in safe.to_dict("records")]
    summary = ""
    if st.session_state.cleaning_summary:
        summary = format_cleaning_summary(st.session_state.cleaning_summary)
    payload = st.session_state.query_payload
    if payload:
        payload = {**payload, "rows": [{str(k): (None if pd.isna(v) else (v.isoformat() if hasattr(v, "isoformat") else v.item() if hasattr(v, "item") else v)) for k, v in row.items()} for row in payload.get("rows", [])]}
    return {
        "nav": st.session_state.nav,
        "data": {"columns": columns, "rows": rows, "total_rows": total_rows},
        "file": {"name": st.session_state.filename, "size_label": human_size(st.session_state.file_size)} if st.session_state.filename else None,
        "cleaning_summary": summary,
        "query_error": st.session_state.query_error,
        "query_result": payload,
        "upload_error": st.session_state.upload_error,
    }


def main():
    init_state()
    event = _workspace(**build_component_data(), key="autolytics_custom_workspace", default=None)
    if event and process_event(event):
        # Rerun only for a newly received event. Re-running the same component
        # value caused a rerun loop and made the page appear to jump.
        st.rerun()
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
    if st.session_state.cleaned_df is not None and st.session_state.cleaning_summary:
        details = st.session_state.cleaning_summary.get("imputation_details", {})
        if details:
            with st.expander("View detailed imputation report"):
                st.dataframe(pd.DataFrame([{"Column": col, **info} for col, info in details.items()]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
