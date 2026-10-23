"""
AUTOLYTICS domain detection and semantic column mapping.

Supports hospital/healthcare datasets automatically, with extensible support
for education, ecommerce, finance, and other domains.

No paid API or LLM required.  All detection is rule-based using column names,
synonyms, data types, cardinality, and representative values.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Synonym dictionaries — concept → list of normalized name fragments
# ---------------------------------------------------------------------------

_HC_SYNONYMS: dict[str, list[str]] = {
    "condition": [
        "medical condition", "condition", "diagnosis", "disease", "ailment",
        "illness", "disorder", "medical diagnosis", "chief complaint",
        "primary diagnosis", "presenting complaint",
    ],
    "medication": [
        "medication", "medicine", "drug", "prescription", "treatment",
        "drug name", "prescribed drug", "drug prescribed", "medicine name",
        "therapy", "medication name",
    ],
    "department": [
        "department", "dept", "specialty", "speciality", "clinical unit",
        "medical department", "ward", "unit", "clinical department",
        "service", "care unit",
    ],
    "billing": [
        "billing amount", "billing", "amount", "cost", "charge", "charges",
        "total charge", "total cost", "treatment cost", "medical cost", "bill amount", "medical bill",
        "healthcare cost", "medical expense", "expense", "total charges",
        "healthcare expenses", "fee", "total fee", "payment amount",
    ],
    "admission_date": [
        "date of admission", "admission date", "admit date", "admitted",
        "date admitted", "arrival date", "admission", "admitdate",
        "date_admitted", "date_of_admission",
    ],
    "discharge_date": [
        "date of discharge", "discharge date", "discharged", "date discharged",
        "release date", "discharge", "dischargedate", "date_discharged",
    ],
    "blood_type": [
        "blood type", "blood group", "blood", "bloodtype", "blood_type",
        "blood_group",
    ],
    "test_result": [
        "test results", "test result", "lab results", "lab result",
        "test outcome", "examination outcome", "lab outcome", "test",
        "results", "test_result", "lab_result",
    ],
    "age": [
        "age", "patient age", "age years", "age_years",
    ],
    "age_group": [
        "age group", "age_group", "age bracket", "age category",
        "patient age group",
    ],
    "gender": [
        "gender", "sex", "patient gender", "patient sex",
    ],
    "los": [
        "length of stay", "los", "stay duration", "hospital stay",
        "stay", "days stayed", "days in hospital", "length_of_stay",
        "lengthofstay", "hospital stay duration", "hospital stay duration days",
    ],
    "waiting_time": [
        "waiting time", "wait time", "queue time", "wait duration",
        "wait", "waiting", "time waited", "waiting_time",
    ],
    "satisfaction": [
        "patient satisfaction", "satisfaction", "satisfaction score",
        "experience score", "rating", "patient rating", "experience",
        "satisfaction_score",
    ],
    "risk_level": [
        "risk level", "risk", "acuity", "severity", "risk category",
        "risk_level", "acuity level",
    ],
    "discharge_status": [
        "discharge status", "discharge outcome", "patient outcome",
        "outcome", "disposition", "status at discharge",
        "discharge_status",
    ],
    "readmission": [
        "readmission", "readmitted", "return admission", "re-admission",
        "readmit", "re admit",
    ],
    "insurance": [
        "insurance provider", "insurance", "insurer", "payer",
        "health insurer", "coverage", "insurance_provider",
    ],
    "admission_type": [
        "admission type", "visit type", "encounter type", "type of admission",
        "admission_type",
    ],
    "patient_id": [
        "patient id", "patient identifier", "patient number", "patient_id",
        "patientid", "pid",
    ],
    "doctor": [
        "doctor", "physician", "attending", "provider", "clinician",
        "consultant", "doctor name",
    ],
    "hospital": [
        "hospital", "facility", "clinic", "institution", "site",
        "hospital name",
    ],
    "room": [
        "room number", "room", "bed", "room_number", "roomnumber",
    ],
}

# Concepts that indicate the dataset is healthcare-related
_HC_INDICATOR_CONCEPTS = {
    "condition", "medication", "department", "billing", "admission_date",
    "discharge_date", "blood_type", "test_result", "los", "waiting_time",
    "satisfaction", "risk_level", "discharge_status", "readmission",
    "insurance", "admission_type", "patient_id",
}

# Minimum number of indicator concepts needed to flag a dataset as healthcare
_HC_MIN_CONCEPTS = 3

# Cardinality guards — concepts that should never be treated as IDs even if
# their name sounds like one
_NON_ID_CONCEPTS = {
    "condition", "medication", "department", "blood_type", "test_result",
    "age_group", "gender", "risk_level", "discharge_status", "readmission",
    "insurance", "admission_type",
}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase, strip punctuation/underscores to a plain space-separated string."""
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def _tok(text: str) -> set[str]:
    return set(_norm(text).split())


# ---------------------------------------------------------------------------
# Column → concept matching
# ---------------------------------------------------------------------------

def _concept_score(col_norm: str, synonyms: list[str]) -> float:
    """Return a confidence score 0–1 for how well col_norm matches a synonym list."""
    col_toks = set(col_norm.split())
    best = 0.0
    for syn in synonyms:
        syn_norm = _norm(syn)
        # Exact full match
        if col_norm == syn_norm:
            return 1.0
        # Synonym fully contained in column name
        if syn_norm in col_norm:
            best = max(best, 0.92)
            continue
        # Column name contained in synonym
        if col_norm in syn_norm and len(col_norm) >= 4:
            best = max(best, 0.80)
            continue
        # Token overlap
        syn_toks = set(syn_norm.split())
        if not syn_toks:
            continue
        overlap = len(col_toks & syn_toks) / len(syn_toks)
        best = max(best, overlap * 0.75)
    return best


def map_columns(df: pd.DataFrame,
                min_confidence: float = 0.55) -> dict[str, dict[str, Any]]:
    """Return a mapping from concept → {col, confidence, col_name}.

    Each concept is assigned at most one column (highest-confidence match).
    Identifier columns are protected: if a column looks like a row ID
    and the matched concept is not in _NON_ID_CONCEPTS, it is skipped.
    """
    # Pre-classify columns
    from eda import classify_columns, _is_identifier
    classification = classify_columns(df)
    id_cols = set(classification.get("identifier", []))

    best_match: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        col_norm = _norm(col)
        series = df[col]

        for concept, synonyms in _HC_SYNONYMS.items():
            score = _concept_score(col_norm, synonyms)
            if score < min_confidence:
                continue

            # Guard: skip if column is an identifier and concept is not a
            # semantic concept (don't map 'patient_id' to 'billing')
            if col in id_cols and concept not in _NON_ID_CONCEPTS:
                continue

            # Extra guard: billing/age/los/waiting_time must be numeric
            if concept in ("billing", "age", "los", "waiting_time", "satisfaction"):
                if not pd.api.types.is_numeric_dtype(series):
                    # Try coercing — some billing columns are stored as strings
                    coerced = pd.to_numeric(series, errors="coerce")
                    if coerced.notna().mean() < 0.5:
                        continue
                    score *= 0.9  # slight penalty for needing coercion

            # Admission/discharge dates must be parseable as dates
            if concept in ("admission_date", "discharge_date"):
                if not pd.api.types.is_datetime64_any_dtype(series):
                    sample = series.dropna().head(20)
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
                        if parsed.notna().mean() < 0.7:
                            continue
                    except Exception:
                        continue

            # Categorical-only concepts: skip pure-numeric columns
            if concept in ("condition", "medication", "department", "blood_type",
                           "test_result", "gender", "risk_level", "discharge_status",
                           "readmission", "insurance", "admission_type"):
                if pd.api.types.is_numeric_dtype(series):
                    # allow low-cardinality numerics that could be encoded categories
                    nu = series.nunique(dropna=True)
                    if nu > 20:
                        continue
                    score *= 0.7

            # Keep best-scoring column per concept
            existing = best_match.get(concept)
            if existing is None or score > existing["confidence"]:
                best_match[concept] = {
                    "col": col,
                    "col_name": col,      # original display name
                    "confidence": round(score, 3),
                }

    return best_match


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

def detect_domain(df: pd.DataFrame) -> dict[str, Any]:
    """Return domain metadata including whether this looks like a healthcare dataset.

    Returns
    -------
    {
        "is_healthcare": bool,
        "confidence": float (0–1),
        "mappings": dict[concept, {col, col_name, confidence}],
        "detected_concepts": list[str],
    }
    """
    mappings = map_columns(df)
    detected = [c for c in mappings if c in _HC_INDICATOR_CONCEPTS]
    n_detected = len(detected)
    confidence = min(1.0, n_detected / _HC_MIN_CONCEPTS) * 0.7 + (
        max(m["confidence"] for m in mappings.values()) * 0.3 if mappings else 0
    )
    return {
        "is_healthcare": n_detected >= _HC_MIN_CONCEPTS,
        "confidence": round(confidence, 3),
        "mappings": mappings,
        "detected_concepts": detected,
    }


# ---------------------------------------------------------------------------
# Hospital chart planner
# ---------------------------------------------------------------------------

def plan_hospital_charts(df: pd.DataFrame,
                          mappings: dict[str, dict[str, Any]],
                          max_charts: int = 8) -> list[dict[str, Any]]:
    """Return a prioritised list of chart spec dicts for a healthcare dataset.

    Each dict: {title, desc, kind, cols, priority}
    kind: "cat_cat" | "cat_num_box" | "cat_num_bar" | "timeseries" |
          "scatter" | "histogram" | "corr" | "bar"
    """

    def col(concept: str) -> str | None:
        m = mappings.get(concept)
        return m["col"] if m else None

    def safe(concept: str, min_unique: int = 2, max_unique: int = 999,
             numeric: bool = False) -> str | None:
        c = col(concept)
        if c is None or c not in df.columns:
            return None
        s = df[c]
        if numeric:
            if not pd.api.types.is_numeric_dtype(s):
                coerced = pd.to_numeric(s, errors="coerce")
                if coerced.notna().mean() < 0.5:
                    return None
        else:
            nu = s.nunique(dropna=True)
            if nu < min_unique or nu > max_unique:
                return None
        return c

    # ── Priority 1: important categorical × categorical combinations ─────────
    candidates: list[dict[str, Any]] = []

    # 1. Medical Condition × Medication
    c_cond = safe("condition", 2, 30)
    c_med  = safe("medication", 2, 30)
    if c_cond and c_med:
        candidates.append({
            "priority": 1,
            "title": "Medication Usage by Medical Condition",
            "desc": f"Record count for each {c_med} within each {c_cond}.",
            "kind": "cat_cat",
            "cols": [c_cond, c_med],
        })

    # 2. Department × Discharge Status
    c_dept = safe("department", 2, 25)
    c_disc = safe("discharge_status", 2, 15)
    if c_dept and c_disc:
        candidates.append({
            "priority": 2,
            "title": f"Discharge Outcomes by {c_dept}",
            "desc": f"Distribution of {c_disc} across {c_dept}.",
            "kind": "cat_cat",
            "cols": [c_dept, c_disc],
        })

    # 3. Medical Condition × Billing Amount
    c_bill = safe("billing", numeric=True)
    if c_cond and c_bill:
        candidates.append({
            "priority": 3,
            "title": f"Billing Distribution by Medical Condition",
            "desc": f"Distribution of {c_bill} for each {c_cond}.",
            "kind": "cat_num_box",
            "cols": [c_cond, c_bill],
        })

    # 4. Department × Waiting Time
    c_wait = safe("waiting_time", numeric=True)
    if c_dept and c_wait:
        candidates.append({
            "priority": 4,
            "title": f"Avg Waiting Time by {c_dept}",
            "desc": f"Average {c_wait} for each {c_dept}.",
            "kind": "cat_num_bar",
            "cols": [c_dept, c_wait],
        })

    # 5. Department × Patient Satisfaction
    c_sat = safe("satisfaction", numeric=True)
    if c_dept and c_sat:
        candidates.append({
            "priority": 5,
            "title": f"Patient Satisfaction by {c_dept}",
            "desc": f"Average {c_sat} for each {c_dept}.",
            "kind": "cat_num_bar",
            "cols": [c_dept, c_sat],
        })

    # 6. Admission Date × Patient Count (time series)
    c_adm = col("admission_date")
    if c_adm and c_adm in df.columns:
        candidates.append({
            "priority": 6,
            "title": "Patient Admissions Over Time",
            "desc": f"Monthly admission counts from {c_adm}.",
            "kind": "timeseries_count",
            "cols": [c_adm],
        })

    # 7. Medical Condition × Blood Type
    c_bt = safe("blood_type", 2, 12)
    if c_cond and c_bt:
        candidates.append({
            "priority": 7,
            "title": f"Blood Type Distribution by Medical Condition",
            "desc": f"Count of {c_bt} within each {c_cond}.",
            "kind": "cat_cat",
            "cols": [c_cond, c_bt],
        })

    # 8. Medical Condition × Test Results
    c_test = safe("test_result", 2, 10)
    if c_cond and c_test:
        candidates.append({
            "priority": 8,
            "title": f"Test Results by Medical Condition",
            "desc": f"Distribution of {c_test} for each {c_cond}.",
            "kind": "cat_cat",
            "cols": [c_cond, c_test],
        })

    # 9. Risk Level × Readmission
    c_risk  = safe("risk_level", 2, 10)
    c_readm = safe("readmission", 2, 5)
    if c_risk and c_readm:
        candidates.append({
            "priority": 9,
            "title": "Readmission by Risk Level",
            "desc": f"Readmission outcomes across {c_risk} categories.",
            "kind": "cat_cat",
            "cols": [c_risk, c_readm],
        })

    # 10. Age Group × Medical Condition
    c_age_grp = safe("age_group", 2, 15)
    if not c_age_grp:
        # Try to build from raw age if available
        c_age = col("age")
        if c_age and c_age in df.columns and pd.api.types.is_numeric_dtype(df[c_age]):
            c_age_grp = None  # we'll handle this in secondary charts
    if c_age_grp and c_cond:
        candidates.append({
            "priority": 10,
            "title": f"Medical Conditions by Age Group",
            "desc": f"Record counts for {c_cond} across {c_age_grp}.",
            "kind": "cat_cat",
            "cols": [c_age_grp, c_cond],
        })

    # 11. Admission Type × Length of Stay
    c_adm_type = safe("admission_type", 2, 10)
    c_los      = safe("los", numeric=True)
    if c_adm_type and c_los:
        candidates.append({
            "priority": 11,
            "title": f"Length of Stay by Admission Type",
            "desc": f"Distribution of {c_los} for each {c_adm_type}.",
            "kind": "cat_num_box",
            "cols": [c_adm_type, c_los],
        })

    # 12. Insurance Provider × Billing Amount
    c_ins = safe("insurance", 2, 20)
    if c_ins and c_bill:
        candidates.append({
            "priority": 12,
            "title": f"Billing by Insurance Provider",
            "desc": f"Distribution of {c_bill} for each {c_ins}.",
            "kind": "cat_num_box",
            "cols": [c_ins, c_bill],
        })

    # 13. Waiting Time × Patient Satisfaction (scatter)
    if c_wait and c_sat:
        candidates.append({
            "priority": 13,
            "title": "Waiting Time vs Patient Satisfaction",
            "desc": f"Relationship between {c_wait} and {c_sat}.",
            "kind": "scatter",
            "cols": [c_wait, c_sat],
        })

    # 14. Length of Stay × Billing Amount (scatter)
    if c_los and c_bill:
        candidates.append({
            "priority": 14,
            "title": "Length of Stay vs Billing Amount",
            "desc": f"Relationship between {c_los} and {c_bill}.",
            "kind": "scatter",
            "cols": [c_los, c_bill],
        })

    # 15. Admission Date × Medical Condition (time series by condition)
    if c_adm and c_cond:
        nu_cond = df[c_cond].nunique(dropna=True)
        if 2 <= nu_cond <= 8:
            candidates.append({
                "priority": 15,
                "title": f"Admissions by Medical Condition Over Time",
                "desc": f"Monthly counts per {c_cond} from {c_adm}.",
                "kind": "timeseries_cat",
                "cols": [c_adm, c_cond],
            })

    # ── Secondary: individual-column charts ─────────────────────────────────
    # Age distribution
    c_age = col("age")
    if c_age and c_age in df.columns and pd.api.types.is_numeric_dtype(df[c_age]):
        candidates.append({
            "priority": 20,
            "title": f"Age Distribution",
            "desc": f"Distribution of patient {c_age}.",
            "kind": "histogram",
            "cols": [c_age],
        })

    # Billing distribution
    if c_bill:
        candidates.append({
            "priority": 21,
            "title": f"Billing Amount Distribution",
            "desc": f"Distribution of {c_bill} values.",
            "kind": "histogram",
            "cols": [c_bill],
        })

    # LOS distribution
    if c_los and not any(c_los in sp["cols"] for sp in candidates if sp["kind"] in ("histogram",)):
        candidates.append({
            "priority": 22,
            "title": f"Length of Stay Distribution",
            "desc": f"Distribution of {c_los}.",
            "kind": "histogram",
            "cols": [c_los],
        })

    # Gender distribution is a core hospital chart, not a low-priority extra.
    # Keep it ahead of optional billing/LOS charts so it survives chart limits.
    c_gender = safe("gender", 2, 10)
    if c_gender:
        candidates.append({
            "priority": 9,
            "title": f"Patient Distribution by Gender",
            "desc": f"Record count by {c_gender}.",
            "kind": "bar",
            "cols": [c_gender],
        })

    # ── Pick top N by priority ───────────────────────────────────────────────
    candidates.sort(key=lambda x: x["priority"])

    # De-duplicate by col sets (same pair shouldn't appear twice)
    seen_col_sets: list[frozenset] = []
    result: list[dict] = []
    for spec in candidates:
        col_set = frozenset(spec["cols"])
        if col_set in seen_col_sets:
            continue
        seen_col_sets.append(col_set)
        result.append(spec)
        if len(result) >= max_charts:
            break

    return result
