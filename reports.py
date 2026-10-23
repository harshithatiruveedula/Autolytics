"""Build a downloadable analysis report from session artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cleaning import format_cleaning_summary
from eda import profile_columns


def build_report_markdown(
    filename: str | None,
    original_df: pd.DataFrame | None,
    cleaned_df: pd.DataFrame,
    cleaning_summary: dict[str, Any] | None,
    query_payload: dict[str, Any] | None = None,
    model_result: dict[str, Any] | None = None,
    analysis_payload: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# AUTOLYTICS Report",
        "",
        f"**Source file:** {filename or 'uploaded dataset'}",
    ]
    if original_df is not None:
        lines.append(f"**Original shape:** {original_df.shape[0]:,} rows × {original_df.shape[1]:,} columns")
    lines.append(f"**Working shape:** {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]:,} columns")
    lines.append("")
    lines.append("## Cleaning")
    if cleaning_summary:
        lines.append(format_cleaning_summary(cleaning_summary))
    else:
        lines.append("No cleaning summary is available.")
    lines.append("")
    lines.append("## Column profile")
    profile = profile_columns(cleaned_df)
    lines.append(profile.to_string(index=False))
    lines.append("")
    if query_payload and query_payload.get("sql"):
        lines.append("## Last query")
        lines.append("```sql")
        lines.append(str(query_payload["sql"]))
        lines.append("```")
        # query_payload stores row count under "count" and raw rows under "rows";
        # there is no "result" DataFrame key at this layer.
        result_count = query_payload.get("count")
        if result_count is not None:
            lines.append(f"Returned {int(result_count):,} row(s).")
        lines.append("")
    if model_result and model_result.get("metrics"):
        lines.append("## Model")
        metrics = model_result["metrics"]
        for key, value in metrics.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
    if analysis_payload:
        domain_label = analysis_payload.get("domain")
        if domain_label:
            lines.extend(["", "## Detected domain", str(domain_label)])
        for key, heading in (("key_insights", "Key Insights"), ("suggestions", "Suggestions")):
            items = analysis_payload.get(key) or []
            lines.extend(["", f"## {heading}"])
            lines.extend([f"- {str(item).replace('**', '')}" for item in items] or ["No supported findings available."])
        story = str(analysis_payload.get("story") or "").strip()
        if story:
            lines.extend(["", "## Data Storytelling", story])
        chart_titles = analysis_payload.get("chart_titles") or []
        if chart_titles:
            lines.extend(["", "## Generated chart references"])
            lines.extend([f"- {title}" for title in chart_titles])
    lines.append("The original uploaded file is unchanged. This report uses the cleaned working copy.")
    return "\n".join(lines)
