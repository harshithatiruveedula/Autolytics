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
    """Detect a domain and map semantic roles, allowing aliases and value cues."""
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
        y=alt.Y("category:N", sort="-x", title=col, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("category:N", scale=alt.Scale(range=PALETTE), legend=None),
        tooltip=[alt.Tooltip("category:N", title=col), alt.Tooltip("count:Q", title="Records")]
    ).properties(height=max(180, min(420, len(data)*25)))
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
        y=alt.Y("category:N", sort="-x", title=cat, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("category:N", scale=alt.Scale(range=PALETTE), legend=None),
        tooltip=[alt.Tooltip("category:N", title=cat), alt.Tooltip("value:Q", title=num, format=",.2f")]
    ).properties(height=max(200, min(420, len(grouped)*26)))
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
        y=alt.Y(f"{cat_a}:N", sort="-x", title=cat_a, axis=alt.Axis(labelLimit=180)),
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
    """Return evidence-based key insights, suggestions, and a concise data story."""
    label = domain_info.get("label", "General")
    m = domain_info.get("mappings", {})
    n = len(df)
    insights: list[str] = [f"The uploaded {label.lower()} dataset contains {n:,} records across {len(df.columns):,} columns."]
    suggestions: list[str] = []
    def col(role): return m.get(role)
    # Data quality insight is always valid.
    missing = df.isna().sum().sort_values(ascending=False)
    if len(missing) and int(missing.iloc[0]) > 0:
        c = str(missing.index[0]); count = int(missing.iloc[0]); pct = count / max(n, 1) * 100
        insights.append(f"{c} has the most missing values: {count:,} ({pct:.1f}% of records).")
        suggestions.append(f"Review missingness in '{c}' ({pct:.1f}% of records) before using it for decisions; confirm whether the gaps are expected or systematic.")
    dup = int(df.duplicated().sum())
    if dup:
        insights.append(f"The dataset contains {dup:,} fully duplicated rows ({dup/max(n,1)*100:.1f}% of records).")
        suggestions.append("Review duplicate rows and confirm whether they represent repeated events or accidental duplicates before counting transactions, patients, or students.")
    # Most frequent categorical values from semantically meaningful roles.
    preferred = {
        "ecommerce": ["category", "status", "region", "product"],
        "healthcare": ["condition", "medication", "department", "discharge", "gender"],
        "finance": ["category", "card_brand", "card_type", "dark_web"],
        "education": ["grade_level", "gender", "extracurricular", "outcome"],
        "generic": [],
    }.get(domain_info.get("name"), [])
    for role in preferred:
        c = col(role)
        if not c or c not in df: continue
        vc = df[c].dropna().astype(str).value_counts()
        if len(vc) and len(vc) >= 2:
            top, cnt = str(vc.index[0]), int(vc.iloc[0]); pct = cnt / max(int(df[c].notna().sum()), 1) * 100
            insights.append(f"In {c}, '{top}' is the most frequent category ({cnt:,} records; {pct:.1f}% of non-missing values).")
            suggestions.append(f"Review the drivers behind '{top}' in {c} ({pct:.1f}% of non-missing values) and compare its outcomes with other groups before taking action.")
            break
    # Domain-specific headline metrics. These are calculated from the current
    # dataset and use explicit labels so row counts are not confused with orders.
    domain_name = domain_info.get("name")
    if domain_name == "ecommerce":
        status_col = col("status")
        sales_df = _eligible_sales_df(df, status_col)
        amount_col = col("revenue")
        if amount_col and amount_col in sales_df:
            amount_values = _coerce_numeric(sales_df[amount_col]).dropna()
            if len(amount_values):
                insights.append(f"Recorded order-line amount totals {amount_values.sum():,.2f} across {len(amount_values):,} usable records after excluding explicitly cancelled rows where status is available.")
                order_id = col("order_id")
                if order_id and order_id in sales_df:
                    per_order = pd.DataFrame({"order": sales_df[order_id], "amount": _coerce_numeric(sales_df[amount_col])}).dropna().groupby("order")["amount"].sum()
                    if len(per_order):
                        insights.append(f"{len(per_order):,} distinct orders are represented; mean recorded order total is {per_order.mean():,.2f} based on summing available line amounts per order.")
        if status_col and status_col in df:
            status_values = df[status_col].astype("string").str.lower().str.strip()
            cancelled = status_values.str.contains(r"cancelled|canceled", regex=True, na=False)
            if status_values.notna().sum():
                count_cancelled = int(cancelled.sum())
                pct_cancelled = count_cancelled / int(status_values.notna().sum()) * 100
                insights.append(f"{count_cancelled:,} records ({pct_cancelled:.1f}% of non-missing status records) are marked cancelled; confirm whether statuses are line-level or order-level before interpreting this as an order cancellation rate.")
                if count_cancelled:
                    suggestions.append(f"Review the {count_cancelled:,} records marked cancelled and compare their product, region, and fulfilment patterns; this status-based record percentage is not necessarily a distinct-order cancellation rate.")
    elif domain_name == "finance":
        income_col, expense_col, profit_col = col("income"), col("expense"), col("profit")
        for role_label, column in (("income", income_col), ("expenses", expense_col), ("profit", profit_col)):
            if column and column in df:
                vals = _coerce_numeric(df[column]).dropna()
                if len(vals):
                    insights.append(f"Total recorded {role_label} is {vals.sum():,.2f} across {len(vals):,} usable values in '{column}'.")
        if not profit_col and income_col and expense_col:
            inc = _coerce_numeric(df[income_col]).dropna()
            exp = _coerce_numeric(df[expense_col]).dropna()
            if len(inc) and len(exp):
                insights.append(f"The difference between total recorded income and expenses is {inc.sum()-exp.sum():,.2f}; this is a derived balance only if both fields use compatible periods, currencies, and accounting definitions.")
        if col("credit_limit") and col("credit_limit") in df:
            credit_values = _coerce_numeric(df[col("credit_limit")]).dropna()
            if len(credit_values):
                insights.append(f"The median recorded credit limit is {credit_values.median():,.2f} across {len(credit_values):,} usable card records; credit limit is not income or spending.")
    elif domain_name == "education":
        subject_cols = [(role, col(role)) for role in ("math_score", "reading_score", "writing_score", "science_score", "score") if col(role) and col(role) in df]
        subject_means = []
        for role, column in subject_cols:
            vals = _coerce_numeric(df[column]).dropna()
            if len(vals): subject_means.append((column, float(vals.mean()), len(vals)))
        if len(subject_means) >= 2:
            highest = max(subject_means, key=lambda x: x[1]); lowest = min(subject_means, key=lambda x: x[1])
            summary = "; ".join(f"{name}: {mean:.2f}" for name, mean, _ in subject_means)
            insights.append(f"Average scores by available score field are {summary}. Highest mean: {highest[0]} ({highest[1]:.2f}); lowest mean: {lowest[0]} ({lowest[1]:.2f}). These comparisons are descriptive and depend on comparable score scales.")
        if col("attendance") and col("attendance") in df:
            vals = _coerce_numeric(df[col("attendance")]).dropna()
            if len(vals): insights.append(f"Average recorded attendance is {vals.mean():.2f} in the source field's units across {len(vals):,} usable records.")

    # Numeric measure insight: avoid treating any finance amount as income/profit.
    numeric_role = {
        "ecommerce": col("revenue"), "healthcare": col("billing"),
        "finance": col("profit") or col("income") or col("expense") or col("amount") or col("credit_limit"),
        "education": col("math_score") or col("score") or col("attendance"),
    }.get(domain_info.get("name"))
    if numeric_role and numeric_role in df:
        measure_df = df
        measure_note = ""
        if domain_info.get("name") == "ecommerce" and col("status"):
            measure_df = _eligible_sales_df(df, col("status"))
            measure_note = " after excluding explicitly cancelled records"
        values = _coerce_numeric(measure_df[numeric_role]).dropna()
        if len(values):
            insights.append(f"{numeric_role}{measure_note} has a mean of {values.mean():,.2f}, a median of {values.median():,.2f}, and a range from {values.min():,.2f} to {values.max():,.2f} across {len(values):,} usable values.")
            q1, q3 = values.quantile(.25), values.quantile(.75); iqr = q3-q1
            outliers = int(((values < q1-1.5*iqr) | (values > q3+1.5*iqr)).sum()) if iqr else 0
            if outliers:
                suggestions.append(f"Review {outliers:,} unusually low/high values in '{numeric_role}' using business context and data validation; statistical outliers are not automatically errors or fraud.")
    # Time trend insight, with correct semantic caveats.
    date_col = col("date")
    if date_col and date_col in df:
        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dropna()
        if len(dates) >= 3 and dates.dt.to_period("M").nunique() >= 2:
            monthly = dates.dt.to_period("M").value_counts().sort_index()
            first, last = int(monthly.iloc[0]), int(monthly.iloc[-1])
            change = ((last-first)/first*100) if first else None
            if change is not None:
                insights.append(f"Record volume changed from {first:,} in {monthly.index[0]} to {last:,} in {monthly.index[-1]} ({change:+.1f}% between the first and last represented months). This compares endpoint counts, not a causal trend.")
                suggestions.append("Inspect intermediate periods and check for incomplete date coverage or seasonality before interpreting the first-to-last change.")
    # Education association: descriptive correlation only.
    if domain_info.get("name") == "education" and col("attendance"):
        score = col("math_score") or col("score")
        if score and score in df:
            pair = pd.DataFrame({"attendance": _coerce_numeric(df[col("attendance")]), "score": _coerce_numeric(df[score])}).dropna()
            if len(pair) >= 5 and pair.nunique().min() > 1:
                corr = pair.corr().iloc[0,1]
                insights.append(f"Attendance and {score} have a Pearson correlation of {corr:.2f} across {len(pair):,} complete records; this is an association, not evidence that one causes the other.")
                suggestions.append("Explore attendance and score patterns by grade and other available student context; do not infer that attendance alone explains performance.")
    # Healthcare safety and finance caveats
    if domain_info.get("name") == "healthcare":
        suggestions.append("Use these patterns for operational review only; recorded associations do not establish clinical causes or treatment recommendations.")
    if domain_info.get("name") == "finance":
        if col("dark_web"):
            vals = df[col("dark_web")].dropna().astype(str).str.lower()
            positive = vals.isin(["yes", "true", "1", "flagged", "positive"]).sum()
            insights.append(f"{int(positive):,} records are marked positive in '{col('dark_web')}' out of {len(vals):,} non-missing flags; validate the flag definition before interpreting risk.")
        suggestions.append("Treat unusual values or risk flags as items for validation, not confirmed fraud or investment advice.")
    if not suggestions:
        suggestions.append("No domain-specific action could be supported confidently from the available columns. Add well-defined measures, dates, or outcomes to enable more targeted recommendations.")
    insights = list(dict.fromkeys(insights))[:6]
    suggestions = list(dict.fromkeys(suggestions))[:6]
    # Connected, data-grounded story rather than a fixed domain template.
    story_parts = [f"This analysis covers {n:,} records and {len(df.columns):,} columns in a dataset classified as {label.lower()} (rule-based confidence {domain_info.get('confidence', 0):.0%})."]
    if len(insights) > 1: story_parts.append(insights[1])
    if len(insights) > 2: story_parts.append(insights[2])
    story_parts.append("These are descriptive patterns in the uploaded data. Missing fields, duplicate records, sampling, and data definitions can affect interpretation.")
    if suggestions: story_parts.append("A useful next step is: " + suggestions[0])
    return {"insights": insights, "suggestions": suggestions, "story": "\n\n".join(story_parts)}
