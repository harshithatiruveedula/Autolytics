"""Domain-aware chart planning, evidence-backed insights, suggestions and storytelling.

This module is deliberately deterministic: all statements are derived from the
uploaded dataframe. It supports healthcare, e-commerce, finance and education,
then falls back to safe generic EDA when domain confidence is low.
"""
from __future__ import annotations
import re
from typing import Any
import numpy as np
import pandas as pd

try:
    import altair as alt
except ImportError:  # lets non-visual tests use the analysis functions
    alt = None

PALETTE = ["#2864f0", "#7c3aed", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#8b5cf6", "#f97316"]

ALIASES: dict[str, dict[str, list[str]]] = {
    "ecommerce": {
        "revenue": ["revenue", "sales", "sales amount", "order value", "order amount", "total amount", "amount", "purchase value", "gross sales"],
        "unit_price": ["unit price", "item price", "product price", "price per unit", "price"],
        "product": ["product", "product name", "sku", "item", "style", "stock item", "product id"],
        "category": ["category", "product category", "item category", "department", "product type"],
        "date": ["order date", "purchase date", "sale date", "date", "created at", "order timestamp", "transaction date"],
        "status": ["order status", "status", "fulfilment status", "fulfillment status", "delivery status", "shipment status"],
        "customer": ["customer id", "client id", "buyer id", "customer", "buyer", "client"],
        "order_id": ["order id", "order_id", "order number", "order no", "order reference", "purchase id"],
        "quantity": ["quantity", "qty", "units sold", "unit count", "items sold", "units"],
        "region": ["region", "state", "city", "location", "ship state", "shipping state", "country", "market"],
        "fulfilment": ["fulfilment", "fulfillment", "fulfilment method", "fulfillment method", "shipping service", "ship service"],
        "channel": ["sales channel", "channel", "platform", "marketplace"],
        "returns": ["return status", "returned", "return", "refund status", "cancelled", "canceled", "cancellation status"],
        "customer_type": ["b2b", "business customer", "customer type", "buyer type"],
        "size": ["size", "product size", "item size", "clothing size"],
        "courier_status": ["courier status", "carrier status", "delivery tracking status"],
        "shipping_service": ["shipping service", "ship service", "service level", "delivery speed"],
    },
    "healthcare": {
        "condition": ["medical condition", "condition", "diagnosis", "primary dx", "disease", "ailment", "illness", "disorder"],
        "medication": ["medication", "medicine", "drug", "drug name", "prescription"],
        "billing": ["billing amount", "bill amount", "medical cost", "treatment cost", "hospital charge", "charges", "cost"],
        "date": ["date of admission", "admission date", "admit date", "encounter date", "visit date", "date of visit", "admission timestamp"],
        "blood_type": ["blood type", "blood group", "abo group"],
        "test_result": ["test results", "test result", "lab results", "lab result", "test outcome", "lab outcome"],
        "gender": ["gender", "sex", "patient sex", "patient gender"],
        "age": ["age", "patient age", "age years", "age at admission"],
        "los": ["length of stay", "los", "stay duration", "days stayed", "days in hospital", "hospital stay duration"],
        "discharge": ["discharge status", "discharge outcome", "patient outcome", "disposition", "outcome"],
        "admission_type": ["admission type", "encounter class", "encounter type", "visit type", "admission category", "patient class", "admit type"],
        "department": ["department", "unit", "ward", "specialty", "service line"],
        "waiting_time": ["waiting time", "wait time", "queue time", "wait duration"],
        "satisfaction": ["patient satisfaction", "satisfaction score", "experience score", "patient rating"],
        "readmission": ["readmission", "readmitted", "readmit"],
        "insurance": ["insurance provider", "insurance", "insurer", "payer"],
    },
    "finance": {
        "income": ["income", "revenue", "deposit", "deposits", "earnings", "salary", "interest income"],
        "expense": ["expense", "expenses", "expenditure", "debit", "withdrawal", "spend", "spending", "cost"],
        "profit": ["profit", "net profit", "net income", "operating profit", "margin"],
        "amount": ["transaction amount", "amount", "transaction value", "value", "payment amount", "purchase amount", "transfer amount"],
        "date": ["transaction date", "payment date", "date", "posted date", "timestamp", "account opening date", "opened date"],
        "category": ["category", "transaction type", "expense category", "merchant category", "spending category", "purpose"],
        "account": ["account id", "client id", "customer id", "account number", "customer", "client"],
        "credit_limit": ["credit limit", "maximum credit", "credit ceiling"],
        "available_credit": ["available credit", "remaining credit", "credit available"],
        "card_brand": ["card brand", "brand", "network", "issuer brand"],
        "card_type": ["card type", "type of card", "debit credit", "payment card type"],
        "chip": ["has chip", "chip enabled", "chip", "chip present"],
        "dark_web": ["card on dark web", "dark web", "compromised card", "fraud flag"],
        "cards_issued": ["num cards issued", "number of cards issued", "cards issued", "card count"],
        "card_expiry": ["expires", "expiry", "expiration date", "card expiry"],
        "pin_change": ["pin last changed", "pin change year", "pin changed", "pin last changed year"],
        "account_opened": ["account opened", "account opening date", "opened date", "open date", "date opened"],
    },
    "education": {
        "student": ["student id", "student_id", "learner id", "student number", "pupil id", "learner number"],
        "gender": ["gender", "sex"],
        "age": ["age", "student age", "learner age", "age years"],
        "grade_level": ["grade level", "class", "year group", "school year", "academic year", "level"],
        "math_score": ["math score", "maths score", "mathematics score", "math marks", "math grade"],
        "reading_score": ["reading score", "reading marks", "literacy score"],
        "writing_score": ["writing score", "writing marks", "composition score"],
        "science_score": ["science score", "science marks", "science grade"],
        "score": ["score", "marks", "exam score", "test score", "assessment score", "percentage", "final grade"],
        "attendance": ["attendance", "attendance rate", "attendance percentage", "attendance percent", "days attended"],
        "study_time": ["study hours", "study time", "hours studied", "study duration"],
        "extracurricular": ["extracurricular", "extracurriculars", "activities", "participation", "club", "activity"],
        "subject": ["subject", "course", "department", "module"],
        "outcome": ["pass fail", "pass/fail", "result", "outcome", "completion status", "graduated"],
        "date": ["graduation date", "enrollment date", "enrolment date", "exam date", "assessment date", "date"],
    },
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _role_match(column: str, aliases: list[str]) -> float:
    name = _norm(column)
    best = 0.0
    for alias in aliases:
        a = _norm(alias)
        if name == a:
            best = max(best, 1.0)
        elif len(a) >= 4 and (a in name or name in a):
            best = max(best, 0.84)
        else:
            overlap = len(set(name.split()) & set(a.split())) / max(1, len(set(a.split())))
            best = max(best, overlap * 0.7)
    return best


def _coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = s.astype("string").str.replace(r"[$£€,%,\s]", "", regex=True).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _eligible_sales_df(df: pd.DataFrame, status_col: str | None) -> pd.DataFrame:
    """Exclude explicitly cancelled orders from sales-value charts when status exists."""
    if not status_col or status_col not in df:
        return df
    status = df[status_col].astype("string").str.lower().str.strip()
    cancelled = status.str.contains(r"cancelled|canceled", regex=True, na=False)
    return df.loc[~cancelled]


def _looks_date(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    # Numeric values are not dates merely because pandas can interpret them as
    # nanoseconds since epoch. For text, require visible date separators or a
    # plausible four-digit year representation before parsing.
    if pd.api.types.is_numeric_dtype(s):
        return False
    sample = s.dropna().astype(str).head(100)
    if len(sample) < 3:
        return False
    date_like = sample.str.contains(r"[/.:T-]", regex=True).mean()
    year_like = sample.str.fullmatch(r"(?:19|20)\d{2}").mean()
    if date_like < 0.5 and year_like < 0.75:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    return parsed.notna().mean() >= 0.75


def detect_domain_and_roles(df: pd.DataFrame) -> dict[str, Any]:
    """Detect a domain and map semantic roles, allowing aliases and value cues.

    This function is intentionally pure (no Streamlit calls) so it can be
    wrapped with st.cache_data in app.py.
    """
    all_maps: dict[str, dict[str, str]] = {}
    scores: dict[str, float] = {}
    for domain, role_aliases in ALIASES.items():
        mappings: dict[str, str] = {}
        for role, aliases in role_aliases.items():
            ranked = sorted(((_role_match(str(c), aliases), str(c)) for c in df.columns), reverse=True)
            if ranked and ranked[0][0] >= 0.55 and ranked[0][1] not in mappings.values():
                col = ranked[0][1]
                if domain == "education" and role == "score" and _norm(col) == "grade":
                    # A numeric Grade field is more likely an academic result than a class/year label.
                    if not pd.api.types.is_numeric_dtype(df[col]) and _coerce_numeric(df[col]).notna().mean() < 0.6:
                        continue
                # Validate field shape for numeric/date concepts.
                if role in {"revenue", "billing", "income", "expense", "profit", "amount", "credit_limit", "available_credit", "cards_issued", "quantity", "age", "los", "waiting_time", "satisfaction", "attendance", "study_time", "math_score", "reading_score", "writing_score", "science_score", "score"}:
                    numeric = _coerce_numeric(df[col])
                    if numeric.notna().mean() < 0.45:
                        continue
                if role in {"date", "card_expiry"} and not _looks_date(df[col]):
                    continue
                mappings[role] = col
        # Value-based support for unfamiliar headers. Use narrow vocabularies and
        # require low cardinality so ordinary text columns are not misclassified.
        for col in df.columns:
            if str(col) in mappings.values():
                continue
            sample = df[col].dropna().astype(str).str.lower().str.strip().head(250)
            vals = set(sample.unique())
            if not vals or len(vals) > 30:
                continue
            inferred: list[tuple[str, float]] = []
            gender_values = {"male", "female", "m", "f", "man", "woman", "other", "unknown", "nonbinary", "non binary"}
            blood_values = {"a+", "a-", "b+", "b-", "ab+", "ab-", "o+", "o-"}
            test_values = {"normal", "abnormal", "inconclusive", "positive", "negative", "detected", "not detected", "high", "low", "borderline"}
            admission_values = {"emergency", "urgent", "elective", "routine", "planned", "inpatient", "outpatient", "observation", "ambulatory", "scheduled"}
            status_values = {"shipped", "cancelled", "canceled", "returned", "refunded", "pending", "delivered", "processing", "unshipped", "in transit", "complete", "completed"}
            fulfil_values = {"amazon", "merchant", "fba", "fbm", "standard", "expedited", "express", "prime", "self ship", "self-ship"}
            brand_values = {"visa", "mastercard", "amex", "american express", "discover", "diners club"}
            card_type_values = {"credit", "debit", "prepaid", "charge"}
            outcome_values = {"pass", "fail", "passed", "failed", "complete", "completed", "graduated", "not graduated", "withdrawn"}
            if vals <= gender_values and (domain in ("healthcare", "education")):
                inferred.append(("gender", .78))
            if domain == "healthcare":
                if vals <= blood_values: inferred.append(("blood_type", .8))
                if vals <= test_values: inferred.append(("test_result", .76))
                if vals <= admission_values: inferred.append(("admission_type", .72))
            if domain == "ecommerce":
                if vals <= status_values: inferred.append(("status", .76))
                if vals <= fulfil_values: inferred.append(("fulfilment", .72))
            if domain == "finance":
                if vals <= brand_values: inferred.append(("card_brand", .82))
                if vals <= card_type_values: inferred.append(("card_type", .72))
            if domain == "education" and vals <= outcome_values:
                inferred.append(("outcome", .72))
            # Only infer a role if that role is not already mapped; one column
            # can never satisfy two concepts in the same domain.
            for role, confidence in inferred:
                if role not in mappings and str(col) not in mappings.values():
                    mappings[role] = str(col)
                    break
            # If domain signals already identify the domain, discover a date
            # column by parseability even when its header is unfamiliar.
            if "date" not in mappings and _looks_date(df[col]):
                mappings["date"] = str(col)
        # Core concept weights are intentionally domain-specific.
        core = {
            "ecommerce": {"revenue": 1.0, "product": 1.3, "category": 1.0, "status": 0.8, "quantity": 0.8, "customer": 0.8, "fulfilment": 0.8},
            "healthcare": {"condition": 1.2, "medication": 1.0, "billing": 0.9, "blood_type": 0.9, "test_result": 0.9, "admission_type": 0.9, "los": 0.8, "department": 0.8},
            "finance": {"income": 1.0, "expense": 1.0, "profit": 1.2, "amount": 0.7, "credit_limit": 1.0, "card_brand": 1.0, "card_type": 0.8, "account": 0.8, "dark_web": 1.0},
            "education": {"student": 0.8, "math_score": 1.2, "reading_score": 1.2, "writing_score": 1.2, "score": 0.8, "attendance": 1.0, "study_time": 0.8, "grade_level": 0.9, "extracurricular": 0.8},
        }[domain]
        score = sum(core.get(role, 0.35) for role in mappings)
        # Strong disambiguation: finance card fields, education score fields, etc.
        scores[domain] = score
        all_maps[domain] = mappings
    winner = max(scores, key=scores.get) if scores else "generic"
    top_score = scores.get(winner, 0)
    runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    # Require multiple independent domain signals; generic fallback on ambiguity.
    minimum = {"ecommerce": 1.8, "healthcare": 1.8, "finance": 1.8, "education": 1.8}
    if top_score < minimum[winner] or top_score - runner_up < 0.25:
        winner = "generic"
    return {"name": winner, "label": {"ecommerce":"E-commerce", "healthcare":"Healthcare", "finance":"Finance", "education":"Education", "generic":"General"}[winner],
            "confidence": round(top_score / max(top_score + runner_up, 1), 3), "scores": scores,
            "mappings": all_maps.get(winner, {}), "all_mappings": all_maps}


def _chart(title: str, desc: str, kind: str, cols: list[str], chart: Any) -> dict[str, Any]:
    return {"id": f"domain-{_norm(title).replace(' ', '-')}", "title": title, "desc": desc,
            "kind": kind, "cols_used": cols, "chart": chart}


def _cat_counts(df: pd.DataFrame, col: str, top: int = 15) -> pd.DataFrame:
    return df[col].astype("string").fillna("(missing)").value_counts().head(top).rename_axis("category").reset_index(name="count")


def _bar_count(df: pd.DataFrame, col: str, title: str, desc: str) -> dict[str, Any] | None:
    if alt is None or col not in df or df[col].nunique(dropna=True) < 2:
        return None
    data = _cat_counts(df, col)
    if data.empty: return None
    c = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("count:Q", title="Record Count"),
        y=alt.Y("category:N", sort="-x", title=col, axis=alt.Axis(labelLimit=420, labelOverlap=False, labelFontSize=12, labelPadding=6)),
        color=alt.Color("category:N", scale=alt.Scale(range=PALETTE), legend=None),
        tooltip=[alt.Tooltip("category:N", title=col), alt.Tooltip("count:Q", title="Records")]
    ).properties(height=max(220, min(520, len(data)*34)), padding={"left": 8, "right": 18, "top": 8, "bottom": 8})
    return _chart(title, desc, "bar", [col], c)


def _hist(df: pd.DataFrame, col: str, title: str, desc: str) -> dict[str, Any] | None:
    if alt is None: return None
    values = _coerce_numeric(df[col]).dropna()
    if len(values) < 3 or values.nunique() < 2: return None
    data = pd.DataFrame({"value": values})
    c = alt.Chart(data).mark_bar(color=PALETTE[0], opacity=.88).encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=28), title=col), y=alt.Y("count():Q", title="Records"),
        tooltip=[alt.Tooltip("value:Q", bin=True), alt.Tooltip("count():Q", title="Records")]
    ).properties(height=250)
    return _chart(title, desc, "histogram", [col], c)


def _cat_num(df: pd.DataFrame, cat: str, num: str, title: str, desc: str, agg: str = "mean") -> dict[str, Any] | None:
    if alt is None: return None
    s = pd.DataFrame({"category": df[cat].astype("string"), "value": _coerce_numeric(df[num])}).dropna()
    if len(s) < 3 or s["category"].nunique() < 2: return None
    grouped = s.groupby("category")["value"].agg(agg).sort_values(ascending=False).head(15).reset_index()
    grouped.columns = ["category", "value"]
    metric_label = "Total" if agg == "sum" else "Median" if agg == "median" else "Average"
    c = alt.Chart(grouped).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("value:Q", title=f"{metric_label} {num}"),
        y=alt.Y("category:N", sort="-x", title=cat, axis=alt.Axis(labelLimit=420, labelOverlap=False, labelFontSize=12, labelPadding=6)),
        color=alt.Color("category:N", scale=alt.Scale(range=PALETTE), legend=None),
        tooltip=[alt.Tooltip("category:N", title=cat), alt.Tooltip("value:Q", title=num, format=",.2f")]
    ).properties(height=max(240, min(520, len(grouped)*36)), padding={"left": 8, "right": 18, "top": 8, "bottom": 8})
    return _chart(title, desc, "cat_num_bar", [cat, num], c)


def _cat_cat(df: pd.DataFrame, cat_a: str, cat_b: str, title: str, desc: str) -> dict[str, Any] | None:
    if alt is None: return None
    s = df[[cat_a, cat_b]].dropna().astype(str)
    if len(s) < 3 or s[cat_a].nunique() < 2 or s[cat_b].nunique() < 2: return None
    top_a = s[cat_a].value_counts().head(12).index
    top_b = s[cat_b].value_counts().head(8).index
    s = s[s[cat_a].isin(top_a) & s[cat_b].isin(top_b)]
    agg = s.groupby([cat_a, cat_b]).size().reset_index(name="count")
    c = alt.Chart(agg).mark_bar().encode(
        x=alt.X("count:Q", title="Record Count"),
        y=alt.Y(f"{cat_a}:N", sort="-x", title=cat_a, axis=alt.Axis(labelLimit=420, labelOverlap=False, labelFontSize=12, labelPadding=6)),
        color=alt.Color(f"{cat_b}:N", scale=alt.Scale(range=PALETTE), legend=alt.Legend(title=cat_b, orient="bottom", columns=3)),
        tooltip=[f"{cat_a}:N", f"{cat_b}:N", alt.Tooltip("count:Q", title="Records")]
    ).properties(height=max(240, min(500, s[cat_a].nunique()*38)))
    return _chart(title, desc, "cat_cat", [cat_a, cat_b], c)


def _time_count(df: pd.DataFrame, date_col: str, title: str, desc: str, value_col: str | None = None) -> dict[str, Any] | None:
    if alt is None: return None
    s = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)})
    if value_col and value_col in df:
        s["value"] = _coerce_numeric(df[value_col])
        s = s.dropna(subset=["date", "value"])
    else:
        s = s.dropna(subset=["date"])
        s["value"] = 1.0
    if len(s) < 3: return None
    span = (s["date"].max() - s["date"].min()).days
    if span > 730:
        s["period"] = s["date"].dt.year.astype(str)
        period_type, axis_title = "period:N", "Year"
        agg = s.groupby("period")["value"].sum().reset_index()
        sort = sorted(agg["period"].unique(), key=lambda x: int(x))
        x = alt.X(period_type, title=axis_title, sort=sort, axis=alt.Axis(labelOverlap="greedy", labelAngle=0))
    else:
        s["period"] = s["date"].dt.to_period("M").dt.to_timestamp()
        agg = s.groupby("period")["value"].sum().reset_index()
        multi_year = s["date"].dt.year.nunique() > 1
        # Always initialise the axis title before constructing tooltip encodings.
        # Previously axis_title was only assigned in the annual branch, causing
        # UnboundLocalError for common ecommerce datasets spanning <= 730 days.
        axis_title = "Month / Year" if multi_year else "Month"
        axis_format = "%b %Y" if multi_year else "%b"
        axis_angle = -35 if multi_year else 0
        x = alt.X("period:T", title=axis_title,
                  axis=alt.Axis(format=axis_format, labelAngle=axis_angle, tickCount=8, labelOverlap="greedy"))
    y_title = f"Total {value_col}" if value_col else "Record Count"
    period_tooltip = alt.Tooltip("period:N", title=axis_title) if span > 730 else alt.Tooltip("period:T", title=axis_title, format=axis_format)
    c = alt.Chart(agg).mark_line(point=True, color=PALETTE[0]).encode(
        x=x, y=alt.Y("value:Q", title=y_title), tooltip=[period_tooltip, alt.Tooltip("value:Q", title=y_title)]
    ).properties(height=270)
    return _chart(title, desc, "timeseries", [date_col] + ([value_col] if value_col else []), c)


def _multi_score_chart(df: pd.DataFrame, score_cols: list[str]) -> dict[str, Any] | None:
    if alt is None or not score_cols: return None
    records=[]
    for col in score_cols:
        vals = _coerce_numeric(df[col]).dropna()
        if len(vals): records.append({"subject": col.replace("_", " ").title(), "average": float(vals.mean()), "n": int(vals.size)})
    if len(records) < 2: return None
    data=pd.DataFrame(records)
    c=alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("average:Q", title="Average Score"), y=alt.Y("subject:N", sort="-x", title="Subject"),
        color=alt.Color("subject:N", scale=alt.Scale(range=PALETTE), legend=None),
        tooltip=["subject:N", alt.Tooltip("average:Q", title="Average", format=".2f"), "n:Q"]
    ).properties(height=max(190, len(records)*34))
    return _chart("Average Scores by Subject", "Mean score for each available subject column.", "cat_num_bar", score_cols, c)


def build_domain_charts(df: pd.DataFrame, domain_info: dict[str, Any], max_charts: int = 20) -> list[dict[str, Any]]:
    """Build non-combination charts relevant to the detected domain plus fallbacks."""
    d = domain_info["name"]
    m = domain_info.get("mappings", {})
    charts: list[dict[str, Any]] = []
    def add(item):
        if item and len(charts) < max_charts and item["title"] not in {x["title"] for x in charts}:
            charts.append(item)
    def get(role): return m.get(role)
    if d == "healthcare":
        cond, med, bill, date = get("condition"), get("medication"), get("billing"), get("date")
        gender, age, bt, test = get("gender"), get("age"), get("blood_type"), get("test_result")
        adm, dept, los, disc = get("admission_type"), get("department"), get("los"), get("discharge")
        if cond and med: add(_cat_cat(df, cond, med, "Medication Usage by Medical Condition", f"Record counts of {med} within each {cond}."))
        if cond and bill: add(_cat_num(df, cond, bill, "Billing Amount by Medical Condition", f"Average recorded {bill} for each {cond}.", "mean"))
        if date: add(_time_count(df, date, "Patient Admissions Over Time", f"Admission record counts aggregated by time from {date}."))
        if cond and bt: add(_cat_cat(df, cond, bt, "Blood Type Distribution by Medical Condition", f"Counts of blood-type categories across {cond}."))
        if cond and test: add(_cat_cat(df, cond, test, "Test Results by Medical Condition", f"Recorded test-result categories across {cond}."))
        if age: add(_hist(df, age, "Patient Age Distribution", f"Distribution of recorded values in {age}."))
        if gender: add(_bar_count(df, gender, "Gender Distribution", f"Patient record counts by {gender}."))
        if adm and cond: add(_cat_cat(df, cond, adm, "Admission Type by Medical Condition", f"Record counts for admission types within each {cond}."))
        if los and adm: add(_cat_num(df, adm, los, "Length of Stay by Admission Type", f"Average recorded {los} by {adm}.", "mean"))
        if disc: add(_bar_count(df, disc, "Discharge Outcome Distribution", f"Record counts by {disc}."))
        if dept and get("waiting_time"): add(_cat_num(df, dept, get("waiting_time"), "Waiting Time by Department", f"Average recorded waiting time by {dept}."))
        if dept and get("satisfaction"): add(_cat_num(df, dept, get("satisfaction"), "Patient Satisfaction by Department", f"Average recorded satisfaction by {dept}."))
    elif d == "ecommerce":
        rev, prod, cat, date = get("revenue"), get("product"), get("category"), get("date")
        status, qty, region, fulfil = get("status"), get("quantity"), get("region"), get("fulfilment")
        if date: add(_time_count(df, date, "Order Records Over Time", f"Order-record counts by month or year from {date}; this is not necessarily a distinct-order count."))
        sales_df = _eligible_sales_df(df, status)
        if date and rev: add(_time_count(sales_df, date, "Recorded Order Amount Over Time", f"Sum of recorded {rev} by month or year; explicitly cancelled records are excluded when order status is available.", rev))
        if cat: add(_bar_count(df, cat, "Orders by Product Category", f"Record counts by {cat}."))
        if cat and rev: add(_cat_num(sales_df, cat, rev, "Recorded Amount by Product Category", f"Total recorded {rev} by {cat}; explicitly cancelled records are excluded when order status is available.", "sum"))
        if prod: add(_bar_count(df, prod, "Top Products by Record Count", f"Most frequent products in {prod}."))
        if prod and rev: add(_cat_num(sales_df, prod, rev, "Top Products by Recorded Amount", f"Total recorded {rev} by product, showing the leading 15; explicitly cancelled records are excluded when status is available.", "sum"))
        if qty and cat: add(_cat_num(sales_df, cat, qty, "Recorded Quantity by Category", f"Total recorded {qty} by {cat}; explicitly cancelled records are excluded when order status is available.", "sum"))
        if status: add(_bar_count(df, status, "Order Status Distribution", f"Record counts by {status}."))
        if region: add(_bar_count(df, region, "Orders by Region", f"Record counts by {region}."))
        if region and rev: add(_cat_num(sales_df, region, rev, "Recorded Amount by Region", f"Total recorded {rev} by {region}; explicitly cancelled records are excluded when status is available.", "sum"))
        if fulfil: add(_bar_count(df, fulfil, "Fulfilment Method Distribution", f"Record counts by {fulfil}."))
        if get("courier_status"): add(_bar_count(df, get("courier_status"), "Courier Status Distribution", f"Record counts by {get('courier_status')}."))
        if get("shipping_service"): add(_bar_count(df, get("shipping_service"), "Shipping Service Distribution", f"Record counts by {get('shipping_service')}."))
        if get("size"): add(_bar_count(df, get("size"), "Sales Records by Product Size", f"Record counts by {get('size')}."))
        if get("size") and rev: add(_cat_num(df, get("size"), rev, "Sales by Product Size", f"Total recorded {rev} by size.", "sum"))
        if get("customer_type"): add(_bar_count(df, get("customer_type"), "Customer Type Distribution", f"Record counts by {get('customer_type')}."))
        if rev: add(_hist(df, rev, "Order Value Distribution", f"Distribution of recorded values in {rev}."))
        if get("unit_price"): add(_hist(df, get("unit_price"), "Unit Price Distribution", f"Distribution of recorded values in {get('unit_price')}."))
    elif d == "finance":
        date = get("date")
        income, expense, profit, amount = get("income"), get("expense"), get("profit"), get("amount")
        category = get("category")
        brand, card_type = get("card_brand"), get("card_type")
        credit, avail = get("credit_limit"), get("available_credit")
        if date: add(_time_count(df, date, "Financial Records Over Time", f"Record counts by month or year from {date}."))
        if date and income: add(_time_count(df, date, "Income Over Time", f"Sum of recorded {income} by month or year.", income))
        if date and expense: add(_time_count(df, date, "Expenses Over Time", f"Sum of recorded {expense} by month or year.", expense))
        if date and profit: add(_time_count(df, date, "Profit Over Time", f"Sum of recorded {profit} by month or year.", profit))
        if category and amount: add(_cat_num(df, category, amount, "Transaction Amount by Category", f"Total recorded {amount} by {category}.", "sum"))
        if category: add(_bar_count(df, category, "Financial Records by Category", f"Record counts by {category}."))
        if amount: add(_hist(df, amount, "Transaction Amount Distribution", f"Distribution of recorded values in {amount}."))
        if credit: add(_hist(df, credit, "Credit Limit Distribution", f"Distribution of recorded values in {credit}."))
        if card_type: add(_bar_count(df, card_type, "Card Type Distribution", f"Record counts by {card_type}."))
        if brand: add(_bar_count(df, brand, "Card Brand Distribution", f"Record counts by {brand}."))
        if brand and credit: add(_cat_num(df, brand, credit, "Credit Limit by Card Brand", f"Median credit limit by {brand}.", "median"))
        if get("chip"): add(_bar_count(df, get("chip"), "Chip Availability", f"Records by chip availability in {get('chip')}."))
        if get("dark_web"): add(_bar_count(df, get("dark_web"), "Card Risk Flag Distribution", f"Recorded flag distribution in {get('dark_web')}; this is not proof of fraud."))
        if get("cards_issued"): add(_hist(df, get("cards_issued"), "Cards Issued Distribution", f"Distribution of recorded card counts in {get('cards_issued')}."))
        if get("card_expiry") and _looks_date(df[get("card_expiry")]): add(_time_count(df, get("card_expiry"), "Card Expiry Timeline", f"Distribution of card expiry dates from {get('card_expiry')}."))
        elif get("card_expiry"): add(_bar_count(df, get("card_expiry"), "Card Expiry Distribution", f"Record counts by expiry value in {get('card_expiry')}."))
        if get("pin_change"): add(_bar_count(df, get("pin_change"), "PIN Change Distribution", f"Record counts by recorded PIN-change value in {get('pin_change')}."))
    elif d == "education":
        score_cols = [get(k) for k in ("math_score", "reading_score", "writing_score", "science_score") if get(k)]
        if len(score_cols) >= 2: add(_multi_score_chart(df, score_cols))
        for role, label in (("math_score", "Math"), ("reading_score", "Reading"), ("writing_score", "Writing"), ("science_score", "Science"), ("score", "Academic Score")):
            col = get(role)
            if col: add(_hist(df, col, f"{label} Score Distribution", f"Distribution of recorded values in {col}."))
        grade, gender, age = get("grade_level"), get("gender"), get("age")
        attend, study = get("attendance"), get("study_time")
        extra, outcome = get("extracurricular"), get("outcome")
        primary_score = next((x for x in score_cols if x), get("score"))
        if get("subject") and primary_score: add(_cat_num(df, get("subject"), primary_score, "Average Score by Subject", f"Average recorded {primary_score} by {get('subject')}.", "mean"))
        if grade: add(_bar_count(df, grade, "Students by Grade Level", f"Record counts by {grade}."))
        if gender: add(_bar_count(df, gender, "Student Gender Distribution", f"Record counts by {gender}."))
        if age: add(_hist(df, age, "Student Age Distribution", f"Distribution of recorded values in {age}."))
        if attend: add(_hist(df, attend, "Attendance Distribution", f"Distribution of recorded values in {attend}."))
        if grade and primary_score: add(_cat_num(df, grade, primary_score, f"Academic Performance by {grade}", f"Average {primary_score} by {grade}.", "mean"))
        if extra: add(_bar_count(df, extra, "Extracurricular Participation", f"Record counts by {extra}; missing values are shown separately where present."))
        if extra and primary_score: add(_cat_num(df, extra, primary_score, "Scores by Extracurricular Activity", f"Average {primary_score} by recorded activity group.", "mean"))
        if study: add(_hist(df, study, "Study Time Distribution", f"Distribution of recorded values in {study}."))
        if outcome: add(_bar_count(df, outcome, "Education Outcome Distribution", f"Record counts by {outcome}; pass rates are not inferred without a reliable outcome definition."))
    # Generic fallback for unknown domains or a sparse domain dataset.
    if len(charts) < min(5, max_charts):
        used = {c for item in charts for c in item.get("cols_used", [])}
        for col in df.columns:
            if len(charts) >= max_charts: break
            if str(col) in used: continue
            if _looks_date(df[col]):
                add(_time_count(df, str(col), f"Records Over Time — {col}", f"Record counts grouped by time using {col}."))
            elif pd.api.types.is_numeric_dtype(df[col]) or _coerce_numeric(df[col]).notna().mean() > .85:
                add(_hist(df, str(col), f"Distribution of {col}", f"Distribution of values in {col}."))
            elif 2 <= df[col].nunique(dropna=True) <= 20:
                add(_bar_count(df, str(col), f"Distribution of {col}", f"Record counts by {col}."))
    return charts[:max_charts]


def generate_narrative(df: pd.DataFrame, domain_info: dict[str, Any]) -> dict[str, list[str] | str]:
    """Create domain-specific business findings, evidence-based story, and actions.

    Findings are calculated from the uploaded data. Explanations are labelled as
    hypotheses when the dataset cannot establish causation; absent measures are
    never replaced with invented values.
    """
    label = domain_info.get("label", "General")
    domain = domain_info.get("name", "generic")
    m = domain_info.get("mappings", {}) or {}
    n = len(df)
    insights: list[str] = []
    story_parts: list[str] = []
    suggestions: list[str] = []
    evidence_links: list[tuple[str, str, str]] = []  # finding, interpretation, action

    def col(role: str) -> str | None:
        c = m.get(role)
        return c if c in df.columns else None

    def num(role: str) -> pd.Series:
        c = col(role)
        return _coerce_numeric(df[c]) if c else pd.Series(dtype=float)

    def category_finding(role: str, business_label: str) -> tuple[str, str, float] | None:
        c = col(role)
        if not c: return None
        vals = df[c].dropna().astype(str).str.strip()
        counts = vals.value_counts()
        if len(counts) < 2: return None
        top, low = str(counts.index[0]), str(counts.index[-1])
        pct = float(counts.iloc[0] / max(len(vals), 1) * 100)
        return top, low, pct

    def add(finding: str, interpretation: str, action: str) -> None:
        insights.append(finding)
        evidence_links.append((finding, interpretation, action))

    # Business/domain findings first; data-quality details are secondary.
    if domain == "ecommerce":
        status_col = col("status")
        sales_df = _eligible_sales_df(df, status_col)
        amount_col, qty_col = col("revenue"), col("quantity")
        category_col, product_col = col("category"), col("product")
        region_col, date_col = col("region"), col("date")
        if amount_col:
            amounts = _coerce_numeric(sales_df[amount_col]).dropna()
            if len(amounts):
                total = float(amounts.sum())
                add(f"Recorded sales value is {total:,.2f} across {len(amounts):,} usable order-line records" + (" after excluding explicitly cancelled records." if status_col else "."),
                    "This is the recorded amount in the source data, not necessarily net revenue after discounts, refunds, tax, or shipping. A single-period total alone cannot prove growth or decline.",
                    f"Use {total:,.2f} as the baseline only after validating the meaning of '{amount_col}', currency, and whether rows are order lines or complete orders. Track net sales and order count by week/month.")
                if category_col:
                    tmp = pd.DataFrame({"group": sales_df[category_col].astype("string"), "amount": _coerce_numeric(sales_df[amount_col])}).dropna()
                    grouped = tmp.groupby("group")["amount"].sum().sort_values(ascending=False)
                    if len(grouped) >= 2 and grouped.sum() != 0:
                        top, low = str(grouped.index[0]), str(grouped.index[-1])
                        share = float(grouped.iloc[0] / grouped.sum() * 100)
                        add(f"'{top}' is the top sales category at {grouped.iloc[0]:,.2f} ({share:.1f}% of category-level recorded sales); '{low}' is lowest at {grouped.iloc[-1]:,.2f}.",
                            f"Sales are concentrated in '{top}'. The dataset shows the outcome, but not by itself whether the difference is caused by demand, pricing, availability, promotion, or product mix.",
                            f"Protect availability and monitor margin for '{top}', then test targeted merchandising or promotion for '{low}'. Compare category conversion, units, returns, and profit before shifting budget.")
                if product_col:
                    tmp = pd.DataFrame({"group": sales_df[product_col].astype("string"), "amount": _coerce_numeric(sales_df[amount_col])}).dropna()
                    grouped = tmp.groupby("group")["amount"].sum().sort_values(ascending=False)
                    if len(grouped) >= 2:
                        add(f"Highest recorded sales product/item is '{grouped.index[0]}' ({grouped.iloc[0]:,.2f}); lowest among grouped items is '{grouped.index[-1]}' ({grouped.iloc[-1]:,.2f}).",
                            "Product-level totals can reveal a best seller and weak seller, but they do not show profitability or inventory constraints unless those fields are present.",
                            "Check stock availability, margin, returns, and product-page conversion for the strongest and weakest items; avoid discontinuing a low-sales item until exposure and stock availability are checked.")
                if region_col:
                    tmp = pd.DataFrame({"group": sales_df[region_col].astype("string"), "amount": _coerce_numeric(sales_df[amount_col])}).dropna()
                    grouped = tmp.groupby("group")["amount"].sum().sort_values(ascending=False)
                    if len(grouped) >= 2:
                        add(f"'{grouped.index[0]}' leads recorded sales by region at {grouped.iloc[0]:,.2f}, while '{grouped.index[-1]}' records {grouped.iloc[-1]:,.2f}.",
                            "Regional variation may reflect customer demand, delivery coverage, marketing, or availability; those causes need additional supporting columns.",
                            "Compare regional order counts, average order value, delivery performance, and marketing spend before reallocating regional campaigns.")
        if qty_col:
            q = num("quantity").dropna()
            if len(q):
                add(f"The dataset records {q.sum():,.0f} units across {len(q):,} usable quantity values (average {q.mean():.2f} units per record).",
                    "Units and sales value together help distinguish high-volume products from high-value products; neither metric alone establishes profit.",
                    "Compare units sold with sales value and margin by category/product to identify volume leaders and premium-value items.")
        if status_col:
            statuses = df[status_col].dropna().astype(str).str.lower().str.strip()
            cancel_mask = statuses.str.contains(r"cancelled|canceled", regex=True, na=False)
            return_mask = statuses.str.contains(r"return|refund", regex=True, na=False)
            if len(statuses):
                if cancel_mask.any():
                    pct = float(cancel_mask.mean()*100)
                    add(f"{int(cancel_mask.sum()):,} records ({pct:.1f}% of non-missing status records) are marked cancelled.",
                        "Cancellations may be reducing fulfilled sales, but this is a record-level rate unless unique order IDs are available and status granularity is confirmed.",
                        "Break cancellations down by product, region, and fulfilment status; track distinct-order cancellation rate and cancelled sales value after validating order-level definitions.")
                if return_mask.any():
                    add(f"{int(return_mask.sum()):,} records ({return_mask.mean()*100:.1f}% of non-missing status records) contain a return/refund status.",
                        "Returns can reduce retained revenue and signal product or fulfilment issues, but the status field alone does not explain the reason.",
                        "Review return reasons, product quality, listing accuracy, and delivery outcomes; track return rate and retained revenue by product/category.")
        if date_col and amount_col:
            dates = pd.to_datetime(sales_df[date_col], errors="coerce", dayfirst=True)
            trend = pd.DataFrame({"month": dates.dt.to_period("M"), "sales": _coerce_numeric(sales_df[amount_col])}).dropna()
            monthly = trend.groupby("month")["sales"].sum().sort_index()
            if len(monthly) >= 2 and monthly.iloc[0] != 0:
                change = (monthly.iloc[-1] / monthly.iloc[0] - 1) * 100
                direction = "increased" if change > 0 else "decreased" if change < 0 else "was unchanged"
                add(f"Recorded sales {direction} by {abs(change):.1f}% from {monthly.index[0]} ({monthly.iloc[0]:,.2f}) to {monthly.index[-1]} ({monthly.iloc[-1]:,.2f}).",
                    "This compares the first and last represented months only. Seasonality, partial months, missing dates, and one-off campaigns may affect the comparison; it is not proof of sustained growth or decline.",
                    "Inspect all intermediate months and compare equivalent full periods. Break the change into order volume, average order value, category mix, cancellations, and returns.")

    elif domain == "education":
        score_roles = [(r, col(r)) for r in ("math_score", "reading_score", "writing_score", "science_score", "score") if col(r)]
        score_summary = []
        for role, c in score_roles:
            v = _coerce_numeric(df[c]).dropna()
            if len(v): score_summary.append((c, v, float(v.mean())))
        if score_summary:
            best = max(score_summary, key=lambda x: x[2]); weak = min(score_summary, key=lambda x: x[2])
            summary = "; ".join(f"{c}: {mean:.1f}" for c, _, mean in score_summary)
            add(f"Average student performance by available score field: {summary}. Strongest average: {best[0]} ({best[2]:.1f}); lowest average: {weak[0]} ({weak[2]:.1f}).",
                f"The gap suggests {weak[0]} may be a learning-support priority, assuming the score fields use comparable scales. A score gap does not identify its cause on its own.",
                f"Review the skills and assessment topics represented by '{weak[0]}'; provide targeted practice and compare student-level progress and score distributions after support. Track the mean, median, and share meeting the school's proficiency benchmark.")
            for c, v, mean in score_summary:
                if v.min() >= 0 and v.max() <= 100 and len(v) >= 5:
                    low = int((v < 60).sum()); pct = low / len(v) * 100
                    if pct >= 10:
                        add(f"{low:,} records ({pct:.1f}%) score below 60 in '{c}', indicating a potential group needing academic support if 60 is an appropriate benchmark.",
                            "A lower-score group can be hidden by the overall average. The 60-point threshold is only a screening assumption and should be replaced with the institution's official benchmark.",
                            f"Identify students below the official proficiency threshold in '{c}', diagnose specific skill gaps, and provide structured support. Track the proportion meeting the benchmark over the next assessment cycle.")
                        break
        attendance_col = col("attendance")
        if attendance_col:
            attendance = num("attendance").dropna()
            if len(attendance):
                add(f"Average recorded attendance is {attendance.mean():.1f} in the source field's units; the middle 50% of values range from {attendance.quantile(.25):.1f} to {attendance.quantile(.75):.1f}.",
                    "The spread helps identify whether low attendance is concentrated among a subset of students. Units and thresholds should be confirmed before classifying attendance as poor.",
                    "Review students with consistently low attendance using the institution's attendance policy, then check whether attendance and assessment results move together. Track attendance and academic progress jointly.")
                score_role = next((r for r in ("math_score", "score", "reading_score", "writing_score", "science_score") if col(r)), None)
                if score_role:
                    pair = pd.DataFrame({"attendance": num("attendance"), "score": num(score_role)}).dropna()
                    if len(pair) >= 5 and pair.nunique().min() > 1:
                        corr = float(pair.corr().iloc[0, 1])
                        strength = "positive" if corr > .15 else "negative" if corr < -.15 else "weak"
                        add(f"Attendance and {col(score_role)} show a {strength} relationship (Pearson r={corr:.2f}; {len(pair):,} students/records with both values).",
                            "This association can help identify students for closer review, but it does not prove that attendance causes score changes; prior achievement and other factors may matter.",
                            "Compare score bands across attendance bands and review other available student context before intervention. Track whether targeted attendance support is followed by improved attendance and scores.")
        group_role = "grade_level" if col("grade_level") else "gender" if col("gender") else None
        score_col = col("score") or col("math_score") or col("reading_score") or col("writing_score")
        if group_role and score_col:
            tmp = pd.DataFrame({"group": df[col(group_role)].astype("string"), "score": _coerce_numeric(df[score_col])}).dropna()
            grouped = tmp.groupby("group")["score"].mean().sort_values(ascending=False)
            if len(grouped) >= 2:
                add(f"Average '{score_col}' differs across {group_role.replace('_',' ')} groups: highest '{grouped.index[0]}' ({grouped.iloc[0]:.1f}) and lowest '{grouped.index[-1]}' ({grouped.iloc[-1]:.1f}).",
                    "Group averages can flag inequities or different support needs; group membership is not itself an explanation for the difference.",
                    f"Compare assessment coverage, sample sizes, attendance, and available learning context for these {group_role.replace('_',' ')} groups. Use the comparison to investigate support needs, not to label individual students.")

    elif domain == "healthcare":
        condition_col, billing_col = col("condition"), col("billing")
        if condition_col:
            counts = df[condition_col].dropna().astype(str).value_counts()
            if len(counts) >= 2:
                top, low = str(counts.index[0]), str(counts.index[-1])
                pct = float(counts.iloc[0] / max(counts.sum(), 1) * 100)
                add(f"'{top}' is the most frequently recorded condition ({counts.iloc[0]:,} records; {pct:.1f}% of records with a condition); '{low}' is least frequent among the listed conditions.",
                    "This describes the uploaded case mix, not disease prevalence in the wider population. Differences may reflect the facility, sampling window, or coding practices.",
                    "Use the case mix for capacity planning and validate it against the full patient population and period. Review service demand and outcomes by condition before changing resources.")
        if billing_col:
            bills = num("billing").dropna()
            if len(bills):
                add(f"Recorded billing averages {bills.mean():,.2f} (median {bills.median():,.2f}; range {bills.min():,.2f}–{bills.max():,.2f}) across {len(bills):,} usable records.",
                    "The difference between mean and median can indicate a skewed cost distribution. Billing amounts alone do not establish clinical quality, treatment necessity, or inefficiency.",
                    "Review unusually high bills and compare billing by condition, admission type, insurer, and length of stay where available. Validate coding and case mix before operational changes.")
            if condition_col:
                tmp = pd.DataFrame({"condition": df[condition_col].astype("string"), "bill": num("billing")}).dropna()
                grouped = tmp.groupby("condition")["bill"].agg(["mean", "count"]).query("count >= 2").sort_values("mean", ascending=False)
                if len(grouped) >= 2:
                    add(f"Average billing is highest for '{grouped.index[0]}' ({grouped.iloc[0]['mean']:,.2f}; n={int(grouped.iloc[0]['count'])}) and lowest for '{grouped.index[-1]}' ({grouped.iloc[-1]['mean']:,.2f}; n={int(grouped.iloc[-1]['count'])}).",
                        "Cost variation may reflect case complexity, treatment mix, or length of stay; these explanations require clinical and operational context and are not proven by billing data alone.",
                        "Compare like-for-like cases and review service utilization and billing components before drawing efficiency conclusions. Track adjusted cost per encounter alongside quality and outcome measures.")
        result_col = col("test_result")
        if result_col:
            results = df[result_col].dropna().astype(str).str.lower().str.strip()
            abnormal = results.str.contains(r"abnormal|positive|detected|high|low", regex=True, na=False)
            if len(results) and abnormal.any():
                add(f"{int(abnormal.sum()):,} records ({abnormal.mean()*100:.1f}% of non-missing '{result_col}' values) are labelled abnormal/positive/detected/high/low by text matching.",
                    "This is a broad label-based screening count, not a diagnosis or severity measure; the meaning of results varies by test.",
                    "Validate the allowed result categories with clinical definitions and review follow-up completion and result-specific patterns with qualified staff. Do not treat this aggregate as a diagnosis.")
        admit_col, discharge_col = col("date"), col("discharge")
        # If the mapping has date + a real discharge date header, calculate length of stay safely.
        discharge_date = next((str(c) for c in df.columns if _norm(c) in {"discharge date", "date of discharge", "discharged at"}), None)
        if admit_col and discharge_date:
            admit = pd.to_datetime(df[admit_col], errors="coerce", dayfirst=True)
            discharge = pd.to_datetime(df[discharge_date], errors="coerce", dayfirst=True)
            stay = (discharge - admit).dt.total_seconds() / 86400
            stay = stay[(stay >= 0) & (stay <= 3650)]
            if len(stay):
                add(f"Calculated length of stay averages {stay.mean():.1f} days (median {stay.median():.1f}) across {len(stay):,} records with valid admission and discharge dates.",
                    "A long stay may reflect case complexity or discharge processes; this dataset alone may not separate clinical need from operational delay.",
                    "Review long-stay cases by condition and admission type with clinical teams. Track median stay alongside readmissions, outcomes, and patient safety so shorter stays are not pursued at the expense of care.")

    elif domain == "finance":
        income_col, expense_col, profit_col = col("income"), col("expense"), col("profit")
        if income_col or expense_col or profit_col:
            totals = {}
            for role, title in (("income", "income"), ("expense", "expenses"), ("profit", "profit")):
                if col(role):
                    values = num(role).dropna()
                    if len(values):
                        totals[role] = float(values.sum())
                        add(f"Total recorded {title} is {values.sum():,.2f} across {len(values):,} usable values in '{col(role)}'.",
                            "A total is meaningful only if the records cover a consistent period, currency, and accounting basis. It should not be interpreted as profitability unless the fields support that definition.",
                            f"Validate the period and accounting meaning of '{col(role)}', then monitor the total by month and category. Compare like-for-like periods to detect material increases or declines.")
            if income_col and expense_col and not profit_col:
                inc, exp = num("income").dropna(), num("expense").dropna()
                if len(inc) and len(exp):
                    balance = float(inc.sum() - exp.sum())
                    add(f"Recorded income minus recorded expenses is {balance:,.2f} as a provisional balance, provided both fields share compatible periods, currency, and accounting definitions.",
                        "This is not formal net profit unless the source includes all relevant revenue and costs and uses matching records/periods.",
                        "Reconcile income and expense coverage and classify major cost drivers before treating the balance as performance. Track operating margin only when the underlying definitions are validated.")
        credit_col = col("credit_limit")
        if credit_col:
            credit = num("credit_limit").dropna()
            if len(credit):
                add(f"Median credit limit is {credit.median():,.2f}; the middle 50% spans {credit.quantile(.25):,.2f}–{credit.quantile(.75):,.2f} across {len(credit):,} usable card records.",
                    "This describes the card portfolio's assigned limits, not customer spending, income, or profitability. Limits alone do not indicate credit utilization without balances or available-credit data.",
                    "Segment limits by card type and customer risk context where permitted, and add balance/available-credit fields to calculate utilization. Track delinquency and exposure metrics only if those data are available.")
        dark_col = col("dark_web")
        if dark_col:
            flags = df[dark_col].dropna().astype(str).str.lower().str.strip()
            positive = flags.isin(["yes", "true", "1", "flagged", "positive", "detected"])
            if len(flags):
                add(f"{int(positive.sum()):,} records ({positive.mean()*100:.1f}% of non-missing flags) are marked as a positive dark-web/compromise flag in '{dark_col}'.",
                    "A positive flag warrants validation and appropriate security handling; it is not, by itself, proof of a fraudulent transaction or financial loss.",
                    "Follow the organization's approved security process to validate positive flags and assess exposure. Track confirmed cases and remediation time; do not publish sensitive card details.")
        for role, title in (("card_brand", "card brand"), ("card_type", "card type")):
            c = col(role)
            if c:
                counts = df[c].dropna().astype(str).value_counts()
                if len(counts) >= 2:
                    add(f"'{counts.index[0]}' is the most common {title} ({counts.iloc[0]:,} records; {counts.iloc[0]/counts.sum()*100:.1f}% of non-missing records).",
                        "Portfolio composition may affect exposure and servicing needs, but category frequency does not indicate revenue or risk without transaction and outcome data.",
                        f"Review portfolio mix alongside utilization, losses, and service outcomes if available; do not infer spend or profitability from '{c}' alone.")
                    break

    # Keep data quality visible, but do not let it dominate business analysis.
    missing = df.isna().sum().sort_values(ascending=False)
    if len(missing) and int(missing.iloc[0]) > 0:
        c = str(missing.index[0]); count = int(missing.iloc[0]); pct = count / max(n, 1) * 100
        insights.append(f"Data quality: '{c}' is missing in {count:,} records ({pct:.1f}%), which may limit analysis that relies on this field.")
        suggestions.append(f"Check why '{c}' is missing and whether missingness is concentrated in a particular group or period before using related metrics for decisions.")
    dup = int(df.duplicated().sum())
    if dup:
        insights.append(f"Data quality: {dup:,} fully duplicated rows ({dup/max(n,1)*100:.1f}% of records) were found; verify whether they are true duplicates before counting outcomes.")
        suggestions.append("Validate duplicate rows against the appropriate business key (order, patient encounter, account, or student) before removing them.")

    if not insights:
        insights.append(f"The uploaded {label.lower()} dataset contains {n:,} records across {len(df.columns):,} fields, but no domain-specific performance metric could be calculated from the recognized columns.")
        suggestions.append("Add or map the domain's core measures and outcomes (for example, sales and orders, student scores, patient outcomes, or transaction measures) to enable business-level analysis.")

    # Prioritize business findings and avoid filling the UI with repetitive generic checks.
    insights = list(dict.fromkeys(insights))[:7]
    suggestions = list(dict.fromkeys(suggestions))[:7]

    # Storytelling links a few of the findings to evidence and possible meaning.
    selected = evidence_links[:3]
    if selected:
        for i, (finding, interpretation, action) in enumerate(selected, 1):
            story_parts.append(f"{i}. {finding}\nWhat the data suggests: {interpretation}")
        story_parts.append("Interpretation boundary: these are patterns in the uploaded records, not proof of cause and effect. Confirm definitions, coverage, and relevant business context before making high-impact decisions.")
        # Align the first recommendations with the story's actual findings.
        suggestions = list(dict.fromkeys([action for _, _, action in selected] + suggestions))[:7]
    else:
        story_parts.append(f"This {label.lower()} dataset has {n:,} records and {len(df.columns):,} fields. The recognized columns do not support a reliable domain-specific explanation yet; add relevant outcome measures or clarify field definitions before drawing business conclusions.")

    if domain == "generic":
        # Generic fallback should be candid, not pretend to know the business domain.
        story_parts = [f"The dataset has {n:,} records and {len(df.columns):,} columns. Its business domain could not be confidently identified, so the analysis avoids labelling generic statistics as sales, performance, costs, or outcomes."]
    return {"insights": insights, "suggestions": suggestions, "story": "\n\n".join(story_parts)}

