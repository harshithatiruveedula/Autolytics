"""Safe SQL execution and flexible offline natural-language query helpers."""
from __future__ import annotations
import json
import os
import re
import sqlite3
import threading
import weakref
from typing import Any
import pandas as pd

try:
    import duckdb
except ImportError:  # SQLite remains as a fallback for local environments.
    duckdb = None

TABLE_NAME = "table"
_SQL_TABLE = '"table"'  # Quote the SQL identifier because TABLE is a reserved keyword.
RESULT_ROW_LIMIT = 1000

# Keep the SQLite copy of each active dataframe so every query does not reload
# the entire dataset. Cache is intentionally small to limit memory use.
_SQLITE_CACHE: dict[int, tuple[Any, sqlite3.Connection]] = {}
_SQLITE_CACHE_LOCK = threading.RLock()
_SQLITE_CACHE_LIMIT = 3
_DUCKDB_CACHE: dict[int, tuple[Any, Any]] = {}
_DUCKDB_CACHE_LOCK = threading.RLock()

# Cache distinct text values used for entity matching so repeated questions do not
# repeatedly convert and scan every unique value in large columns.
_VALUE_CACHE: dict[tuple[int, str], tuple[Any, list[tuple[str, str]]]] = {}
_VALUE_CACHE_LOCK = threading.RLock()
_SQL_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|grant|revoke|truncate|exec|execute|load_extension)\b", re.IGNORECASE)
_COMMENTS = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)


def looks_like_sql(text: str) -> bool:
    return bool(_SQL_START.match(text or ""))


def strip_sql_comments(sql: str) -> str:
    return _COMMENTS.sub(" ", sql)


def validate_sql(sql: str) -> str:
    cleaned = strip_sql_comments(sql or "").strip()
    if not cleaned:
        raise ValueError("Enter a SQL query or a question about your data.")
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed.")
    if not _SQL_START.match(cleaned):
        raise ValueError("Only SELECT queries are allowed against the dataset.")
    if _FORBIDDEN.search(cleaned):
        raise ValueError("This SQL statement is not allowed.")
    return cleaned


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _find_columns(question: str, columns: list[str]) -> list[str]:
    """Return mentioned columns without mistaking a token inside a longer name for another column.

    For example, when a dataset has both `id` and `client_id`, mentioning
    `client_id` must not also select `id`.
    """
    q = _norm(question)
    candidates = []
    for col in columns:
        name = _norm(col)
        if not name:
            continue
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", q):
            candidates.append((match.start(), match.end(), len(name), col))
    # Longest spans win; this prevents overlaps such as `id` within `client id`.
    candidates.sort(key=lambda item: (-(item[2]), item[0], -(item[1] - item[0])))
    accepted_spans = []
    accepted_columns = []
    for start, end, _, col in candidates:
        if any(start < used_end and end > used_start for used_start, used_end in accepted_spans):
            continue
        if col not in accepted_columns:
            accepted_columns.append(col)
            accepted_spans.append((start, end))
    # Keep natural order of the first occurrence in the question for stable,
    # intuitive SELECT/GROUP BY column ordering.
    first_positions = {}
    for col in accepted_columns:
        name = _norm(col)
        found = re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", q)
        first_positions[col] = found.start() if found else len(q)
    return sorted(accepted_columns, key=lambda col: first_positions[col])


def _match_column(text: str, columns: list[str]) -> str | None:
    found = _find_columns(text, columns)
    return found[0] if found else None


def _numeric(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def _sql_col(col: str) -> str:
    return _quote_identifier(col)


def _first_numeric_column(df: pd.DataFrame) -> str | None:
    return next((str(c) for c in df.columns if _numeric(df, c)), None)


def _first_categorical_column(df: pd.DataFrame) -> str | None:
    return next((str(c) for c in df.columns if not _numeric(df, c)), None)



def _entity_lookup_sql(question: str, df: pd.DataFrame) -> str | None:
    """Handle person lookups and multi-column category filters locally.

    Examples: age of Bobby Jackson; who has cancer; whose doctor is Mathew;
    people with cancer who are Mathew's patients.
    """
    q = _norm(question)
    if not q:
        return None

    columns = [str(c) for c in df.columns]
    col_tokens = {c: set(_norm(c).split()) for c in columns}

    def pick_col(tokens: set[str]) -> str | None:
        candidates = [c for c in columns if tokens.intersection(col_tokens[c])]
        return max(candidates, key=lambda c: len(tokens.intersection(col_tokens[c]))) if candidates else None

    name_col = pick_col({"name", "patient", "customer", "person", "identifier"})
    doctor_col = pick_col({"doctor", "physician"})
    condition_col = pick_col({"condition", "diagnosis", "disease", "illness"})
    hospital_col = pick_col({"hospital", "clinic"})
    medication_col = pick_col({"medication", "medicine", "drug"})

    def matching_values(col: str | None) -> list[str]:
        if not col:
            return []
        key = (id(df), col)
        with _VALUE_CACHE_LOCK:
            cached = _VALUE_CACHE.get(key)
            if cached is not None and cached[0]() is df:
                indexed_values = cached[1]
            else:
                indexed_values = [
                    (value, _norm(value))
                    for value in df[col].dropna().astype(str).str.strip().drop_duplicates().tolist()
                ]
                _VALUE_CACHE[key] = (weakref.ref(df), indexed_values)
                for old_key, (df_ref, _) in list(_VALUE_CACHE.items()):
                    if df_ref() is None:
                        _VALUE_CACHE.pop(old_key, None)
        found = [
            value for value, normalized in indexed_values
            if len(normalized) >= 2
            and re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", q)
        ]
        return sorted(found, key=lambda v: (len(_norm(v).split()), len(_norm(v))), reverse=True)

    # Map requested properties to the actual column name.
    target = None
    alias_specs = [
        (r"\b(date of admission|admission date|admitted on|when .* admitted)\b", {"admission", "admitted"}),
        (r"\b(date of discharge|discharge date|discharged on)\b", {"discharge", "discharged"}),
        (r"\b(age|how old)\b", {"age"}),
        (r"\b(gender|sex)\b", {"gender", "sex"}),
        (r"\b(medical condition|condition|diagnosis|disease|illness)\b", {"condition", "diagnosis", "disease", "illness"}),
        (r"\b(doctor|physician)\b", {"doctor", "physician"}),
        (r"\b(hospital|clinic)\b", {"hospital", "clinic"}),
        (r"\b(billing amount|bill amount|medical bill|billing|bill)\b", {"billing", "bill", "amount"}),
        (r"\b(medication|medicine|drug)\b", {"medication", "medicine", "drug"}),
        (r"\b(test results|test result|lab results|lab result)\b", {"test", "result", "results"}),
        (r"\b(room number|room no|room)\b", {"room"}),
        (r"\b(name|full name|patient name|customer name)\b", {"name", "patient", "customer"}),
    ]
    for pattern, tokens in alias_specs:
        if re.search(pattern, q):
            target = pick_col(tokens)
            if target:
                break

    # Person-specific detail: only scan the name column when the wording
    # actually asks for an attribute belonging to a person.
    asks_person_detail = bool(re.search(r"\b(of|for|about|belonging to|details for|information for)\b", q)) and bool(target)
    if name_col and asks_person_detail:
        # Extract the likely person name directly from the phrase after the last
        # relationship word (e.g. 'age of Bobby Jackson'). This avoids scanning
        # every distinct patient name before running a simple lookup.
        pieces = re.split(r"\b(?:of|for|about|belonging to|details for|information for)\b", question, flags=re.I)
        person_text = pieces[-1].strip(" \t\r\n?!.,;:'\"`") if pieces else ""
        person_text = re.sub(r"\b(patient|person|customer)\s*$", "", person_text, flags=re.I).strip()
        # If the extracted phrase contains only a name-like value, query SQLite
        # directly; if not, retain the cached value matcher as a fallback.
        if person_text and not re.search(r"\b(show|list|give|tell|what|which|who|the|is|are)\b", person_text, flags=re.I):
            person = person_text.replace("'", "''")
            where = f"LOWER(TRIM(CAST({_sql_col(name_col)} AS TEXT))) = LOWER('{person}')"
            return f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} WHERE {where} LIMIT {RESULT_ROW_LIMIT}"
        person_values = matching_values(name_col)
        if person_values:
            person = person_values[0].replace("'", "''")
            where = f"LOWER(TRIM(CAST({_sql_col(name_col)} AS TEXT))) = LOWER('{person}')"
            return f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} WHERE {where} LIMIT {RESULT_ROW_LIMIT}"

    # Apply every applicable filter, so combined questions do not degrade into
    # returning a whole column or a single unrelated filter.
    filters: list[str] = []
    asks_doctor = bool(re.search(r"\b(doctor|physician|patient|patients|treated by|under|whose)\b", q))
    doctor_values = matching_values(doctor_col) if asks_doctor else []
    if doctor_col and doctor_values and asks_doctor:
        value = doctor_values[0].replace("'", "''")
        filters.append(f"LOWER(TRIM(CAST({_sql_col(doctor_col)} AS TEXT))) = LOWER('{value}')")

    asks_condition = bool(re.search(r"\b(with|have|has|having|diagnosed|suffering|condition|cancer|diabetes|asthma|hypertension)\b", q))
    condition_values = matching_values(condition_col) if asks_condition else []
    if condition_col and condition_values and asks_condition:
        value = condition_values[0].replace("'", "''")
        filters.append(f"LOWER(TRIM(CAST({_sql_col(condition_col)} AS TEXT))) = LOWER('{value}')")

    asks_hospital = bool(re.search(r"\b(hospital|clinic|at|from)\b", q))
    hospital_values = matching_values(hospital_col) if asks_hospital else []
    if hospital_col and hospital_values and asks_hospital:
        value = hospital_values[0].replace("'", "''")
        filters.append(f"LOWER(TRIM(CAST({_sql_col(hospital_col)} AS TEXT))) = LOWER('{value}')")

    asks_medication = bool(re.search(r"\b(medication|medicine|drug|taking|prescribed)\b", q))
    medication_values = matching_values(medication_col) if asks_medication else []
    if medication_col and medication_values and asks_medication:
        value = medication_values[0].replace("'", "''")
        filters.append(f"LOWER(TRIM(CAST({_sql_col(medication_col)} AS TEXT))) = LOWER('{value}')")

    filters = list(dict.fromkeys(filters))
    if filters:
        return f"SELECT * FROM {_SQL_TABLE} WHERE {' AND '.join(filters)} LIMIT 100"

    # Only attempt a value/entity scan when the wording signals a filter or
    # lookup. Scanning every distinct value for a simple "show age" question
    # can dominate query time on large datasets.
    if re.search(r"\b(where|whose|named|called|equals?|equal to|matching|with value|for patient|for customer|of [a-z])\b", q):
        for col in _find_columns(question, columns):
            vals = matching_values(col)
            if vals:
                value = vals[0].replace("'", "''")
                return f"SELECT * FROM {_SQL_TABLE} WHERE LOWER(TRIM(CAST({_sql_col(col)} AS TEXT))) = LOWER('{value}') LIMIT 100"
    return None


def _individual_detail_sql(question: str, df: pd.DataFrame) -> str | None:
    """Resolve a single field for one record, including row/index and identifier lookups."""
    original = (question or "").strip()
    q_norm = _norm(original)
    columns = [str(c) for c in df.columns]
    if not q_norm or not re.search(r"\b(of|for|about)\b", q_norm):
        return None

    mentioned = _find_columns(original, columns)
    if not mentioned:
        return None

    def is_identifier(col: str) -> bool:
        tokens = _norm(col).split()
        return (
            bool(tokens and tokens[-1] in {"id", "identifier", "number", "no"})
            or _norm(col).endswith(" id")
            or _norm(col) in {"id", "student id", "patient id", "employee id",
                              "customer id", "record id", "order id", "client id"}
        )

    id_columns = [c for c in columns if is_identifier(c)]
    # A specifically mentioned ID column must win over generic columns like `id`.
    explicit_ids = _find_columns(original, id_columns)
    id_columns = explicit_ids + [c for c in id_columns if c not in explicit_ids]

    relation = re.search(r"\b(?:of|for|about)\b", q_norm)
    before_relation = original[:relation.start()] if relation else original
    target_candidates = _find_columns(before_relation, columns)
    # Usually the target is a field before "of"; allow an identifier field itself
    # as the target for queries such as "order id of index 1".
    target = next((c for c in target_candidates if c not in id_columns), None)
    target_is_id = False
    if target is None:
        target = next((c for c in target_candidates if c in id_columns), None)
        target_is_id = target is not None
    if target is None:
        target = next((c for c in mentioned if c not in id_columns), None)
    if target is None:
        return None

    # Explicit row/index syntax: `order id of index 1`, `amount for row 10`.
    selector = re.search(r"\b(?:of|for|about)\s+(?:the\s+)?(?:row\s+)?index\s+(\d+)\b", q_norm)
    if selector:
        idx_value = int(selector.group(1))
        index_col = next((c for c in columns if _norm(c) in {"index", "row index", "row number"}), None)
        if index_col:
            return (
                f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} "
                f"WHERE CAST({_sql_col(index_col)} AS TEXT) = '{idx_value}' LIMIT 1"
            )
        # Without an explicit index column, use zero-based row position.
        return f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} LIMIT 1 OFFSET {idx_value}"

    # If the target itself is an identifier, a trailing number without another
    # named identifier means the user is asking for that field at row/index N.
    # Prefer an explicit `index`/`row` column when available.
    trailing_number = re.search(r"\b(?:of|for|about)\s+(\d+)\s*$", q_norm)
    if target_is_id and trailing_number:
        idx_value = int(trailing_number.group(1))
        index_col = next((c for c in columns if _norm(c) in {"index", "row index", "row number"}), None)
        if index_col:
            return (
                f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} "
                f"WHERE CAST({_sql_col(index_col)} AS TEXT) = '{idx_value}' LIMIT 1"
            )
        return f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} LIMIT 1 OFFSET {idx_value}"

    # Resolve an explicitly named identifier by extracting its value directly,
    # avoiding a sort/scan of every distinct ID (important for large ecommerce files).
    explicit_named_id = next((c for c in id_columns if c in _find_columns(original, id_columns)), None)
    if explicit_named_id:
        # Match the raw question so punctuation in real-world identifiers
        # (e.g. Amazon order IDs containing hyphens) is preserved.
        col_pattern = r"(?<![A-Za-z0-9])" + r"\s+".join(re.escape(part) for part in explicit_named_id.split()) + r"(?![A-Za-z0-9])"
        raw_col_match = re.search(col_pattern, original, flags=re.I)
        tail = original[raw_col_match.end():].strip() if raw_col_match else ""
        candidate_match = re.match(
            r"(?:is|equals|equal\s+to|of|number)?\s*([A-Za-z0-9][A-Za-z0-9\-_/\.]+|[A-Za-z0-9])",
            tail, flags=re.I
        )
        if candidate_match:
            candidate = candidate_match.group(1).strip()
            if candidate.lower() not in {"of", "for", "about", "index", "row"}:
                series = df[explicit_named_id]
                if pd.api.types.is_numeric_dtype(series):
                    try:
                        numeric_candidate = float(candidate) if "." in candidate else int(candidate)
                        mask = pd.to_numeric(series, errors="coerce").eq(numeric_candidate)
                    except (ValueError, OverflowError):
                        mask = pd.Series(False, index=df.index)
                else:
                    mask = series.astype(str).str.strip().str.lower().eq(candidate.lower())
                if bool(mask.any()):
                    matched_value = str(df.loc[mask, explicit_named_id].iloc[0]).replace("'", "''")
                    return (
                        f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} "
                        f"WHERE CAST({_sql_col(explicit_named_id)} AS TEXT) = '{matched_value}' LIMIT 1"
                    )

    # Name/person lookups (e.g. "Age of Bobby JacksOn"). Use cached normalized
    # values so repeated questions do not repeatedly sort all distinct names.
    name_columns = [
        c for c in columns
        if any(token in set(_norm(c).split()) for token in ("name", "patient", "student", "employee", "customer"))
        and not any(token in set(_norm(c).split()) for token in ("id", "number"))
    ]
    for col in name_columns:
        key = (id(df), col)
        with _VALUE_CACHE_LOCK:
            cached = _VALUE_CACHE.get(key)
            if cached is None or cached[0]() is not df:
                values = [
                    (str(v).strip(), _norm(v))
                    for v in df[col].dropna().astype(str).str.strip().drop_duplicates().tolist()
                ]
                _VALUE_CACHE[key] = (weakref.ref(df), values)
                indexed_values = values
            else:
                indexed_values = cached[1]
        for value, normalized in sorted(indexed_values, key=lambda item: len(item[1]), reverse=True):
            if len(normalized) >= 2 and re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", q_norm):
                escaped = value.replace("'", "''")
                return (
                    f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} "
                    f"WHERE LOWER(TRIM(CAST({_sql_col(col)} AS TEXT))) = LOWER('{escaped}') LIMIT 1"
                )

    # Semantic entity aliases: `math score of student 1`, `amount for order 123`,
    # or `card type of client 825`, even when the exact ID column name is not stated.
    entity_selector = re.search(
        r"\b(?:of|for|about)\s+(?:the\s+)?(student|patient|employee|customer|client|order|card|account|record)\s+(?:id\s*)?([a-z0-9][a-z0-9\-_/\.]*)\s*$",
        q_norm
    )
    if entity_selector:
        entity = entity_selector.group(1)
        value = entity_selector.group(2)
        related_id_col = next(
            (c for c in id_columns if entity in set(_norm(c).split()) or _norm(c).startswith(entity + " ")),
            None
        )
        if related_id_col:
            series = df[related_id_col]
            if pd.api.types.is_numeric_dtype(series):
                try:
                    number = float(value) if "." in value else int(value)
                    mask = pd.to_numeric(series, errors="coerce").eq(number)
                except (ValueError, OverflowError):
                    mask = pd.Series(False, index=df.index)
            else:
                mask = series.astype(str).str.strip().str.lower().eq(value.lower())
            if bool(mask.any()):
                matched_value = str(df.loc[mask, related_id_col].iloc[0]).replace("'", "''")
                return (
                    f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} "
                    f"WHERE CAST({_sql_col(related_id_col)} AS TEXT) = '{matched_value}' LIMIT 1"
                )

    # Generic `field of 1`: use a matching numeric identifier, if present.
    if trailing_number:
        number = trailing_number.group(1)
        for col in id_columns:
            if col in explicit_ids:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                return (
                    f"SELECT {_sql_col(target)} FROM {_SQL_TABLE} "
                    f"WHERE {_sql_col(col)} = {number} LIMIT 1"
                )
    return None



def _grouping_query_sql(question: str, df: pd.DataFrame) -> str | None:
    """Build multi-column grouped queries before simpler column/count rules.

    Supports examples such as:
      - group extracurriculars by student_id
      - group extracurriculars and gender by student_id
      - group by gender and grade_level
      - average math_score by gender and grade_level
      - count extracurriculars by student_id
      - total sales by ship-state and Category
    """
    original = (question or "").strip()
    q = _norm(original)
    columns = [str(c) for c in df.columns]
    if not q:
        return None

    # A "GROUP BY ..." request explicitly names every grouping dimension after
    # the phrase. Use all matched columns, not just the first one.
    group_by_match = re.search(r"\bgroup(?:ed)?\s+by\s+(.+)$", original, flags=re.I)
    if group_by_match:
        dims = _find_columns(group_by_match.group(1), columns)
        dims = list(dict.fromkeys(dims))
        if dims:
            selected = ", ".join(f"{_sql_col(c)} AS {_sql_col(c)}" for c in dims)
            grouped = ", ".join(_sql_col(c) for c in dims)
            return (
                f"SELECT {selected}, COUNT(*) AS record_count FROM {_SQL_TABLE} "
                f"GROUP BY {grouped} ORDER BY {grouped} LIMIT {RESULT_ROW_LIMIT}"
            )

    by_match = re.search(r"\bby\s+(.+)$", original, flags=re.I)
    if not by_match:
        return None

    left_text = original[:by_match.start()]
    right_text = by_match.group(1)
    left_cols = _find_columns(left_text, columns)
    right_cols = _find_columns(right_text, columns)
    left_cols = list(dict.fromkeys(left_cols))
    right_cols = list(dict.fromkeys(right_cols))
    if not right_cols:
        return None

    # Aggregate requests use the left-side numeric field as a measure and all
    # right-side fields as grouping dimensions.
    wants_avg = bool(re.search(r"\b(average|avg|mean)\b", q))
    wants_sum = bool(re.search(r"\b(sum|total|revenue|sales|amount|spend|cost|quantity)\b", q))
    wants_count = bool(re.search(r"\b(count|counts|number of|how many|frequency)\b", q))
    explicit_group = bool(re.search(r"\b(group|grouped|grouping|breakdown|distribution)\b", q))

    numeric_measure = next((c for c in left_cols if _numeric(df, c)), None)
    if numeric_measure and (wants_avg or wants_sum):
        measures = [_sql_col(numeric_measure)]
        dims = [c for c in right_cols if c != numeric_measure]
        if not dims:
            return None
        aggregate = "AVG" if wants_avg else "SUM"
        alias = "average_value" if wants_avg else "total_value"
        dim_select = ", ".join(f"{_sql_col(c)} AS {_sql_col(c)}" for c in dims)
        dim_group = ", ".join(_sql_col(c) for c in dims)
        return (
            f"SELECT {dim_select}, {aggregate}({_sql_col(numeric_measure)}) AS {alias} "
            f"FROM {_SQL_TABLE} WHERE {_sql_col(numeric_measure)} IS NOT NULL "
            f"GROUP BY {dim_group} ORDER BY {alias} DESC LIMIT {RESULT_ROW_LIMIT}"
        )

    # "Count X by Y" means count the non-null X values for each Y dimension.
    # For explicit "group X by Y", X is also a grouping dimension.
    if wants_count and not explicit_group:
        dims = [c for c in right_cols]
        if dims:
            dim_select = ", ".join(f"{_sql_col(c)} AS {_sql_col(c)}" for c in dims)
            dim_group = ", ".join(_sql_col(c) for c in dims)
            measure = left_cols[0] if left_cols else None
            count_expr = f"COUNT({_sql_col(measure)})" if measure else "COUNT(*)"
            return (
                f"SELECT {dim_select}, {count_expr} AS record_count FROM {_SQL_TABLE} "
                f"GROUP BY {dim_group} ORDER BY record_count DESC LIMIT {RESULT_ROW_LIMIT}"
            )

    # Group/distribution questions group by every mentioned field on both
    # sides of "by", supporting two or more dimensions.
    if explicit_group or (left_cols and right_cols):
        dims = list(dict.fromkeys(left_cols + right_cols))
        if dims:
            dim_select = ", ".join(f"{_sql_col(c)} AS {_sql_col(c)}" for c in dims)
            dim_group = ", ".join(_sql_col(c) for c in dims)
            return (
                f"SELECT {dim_select}, COUNT(*) AS record_count FROM {_SQL_TABLE} "
                f"GROUP BY {dim_group} ORDER BY {dim_group} LIMIT {RESULT_ROW_LIMIT}"
            )
    return None



def _filtered_projection_sql(question: str, df: pd.DataFrame) -> str | None:
    """Handle projection + row filters, e.g. 'order id who have status as shipped'.

    Column names and categorical values are discovered from the active dataframe,
    so the same logic works across domains rather than relying on dataset-specific
    hard-coded fields.
    """
    original = (question or "").strip()
    columns = [str(c) for c in df.columns]
    if not original or not columns:
        return None

    marker = re.search(
        r"\b(?:who\s+(?:have|has)|that\s+(?:have|has)|which\s+(?:have|has)|where|with|having|whose)\b",
        original, flags=re.I,
    )
    if not marker:
        return None

    before = original[:marker.start()]
    after = original[marker.end():]
    target_cols = _find_columns(before, columns)
    filter_cols = _find_columns(after, columns)
    if not filter_cols:
        return None

    # Filter out instructional words and retain only actual target fields.
    target_cols = [c for c in target_cols if c not in filter_cols]
    if not target_cols:
        # For a query like 'show rows where status is shipped', all columns are useful.
        if not re.search(r"\b(row|rows|record|records|all|everything|data)\b", _norm(before)):
            return None
        target_cols = columns

    after_norm = _norm(after)
    filter_norm = after_norm
    where_parts: list[str] = []
    used_cols: set[str] = set()
    for col in filter_cols:
        if col in used_cols:
            continue
        used_cols.add(col)
        col_norm = _norm(col)
        match = re.search(rf"(?<![a-z0-9]){re.escape(col_norm)}(?![a-z0-9])", filter_norm)
        if not match:
            continue
        tail = filter_norm[match.end():].strip()
        # A conjunction usually begins the next filter. Column-name boundaries
        # also prevent the first filter from consuming a second condition.
        stop_positions = [m.start() for m in re.finditer(r"\b(?:and|or)\b", tail)]
        for other in filter_cols:
            if other == col:
                continue
            other_norm = _norm(other)
            other_match = re.search(rf"(?<![a-z0-9]){re.escape(other_norm)}(?![a-z0-9])", tail)
            if other_match:
                stop_positions.append(other_match.start())
        if stop_positions:
            tail = tail[:min(stop_positions)]
        tail = re.sub(r"^(?:is|are|was|were|as|equals?|equal to|set to|of|the|a|an|value|contains?|includes?|like)+\s*", "", tail).strip()
        if not tail:
            continue
        # Trim generic phrasing around the value.
        tail = re.sub(r"\b(?:status|type|value)\s+is\s+", "", tail).strip()
        value = tail
        # Exact dataset values win when the complete value is mentioned. For
        # abbreviated values such as 'shipped', use contains matching so values
        # like 'Shipped - Delivered to Buyer' are included as well.
        series = df[col]
        normalized_value = _norm(value)
        if not normalized_value:
            continue
        observed = series.dropna().astype(str).str.strip().drop_duplicates()
        exact = [v for v in observed.tolist() if _norm(v) == normalized_value]
        if exact:
            # If the requested value is a short prefix shared by multiple actual
            # categories, treat it as a category family (e.g. shipped statuses).
            variants = [v for v in observed.tolist() if normalized_value and _norm(v).startswith(normalized_value)]
            if len(variants) > 1 and len(normalized_value.split()) == 1:
                where_parts.append(
                    f"LOWER(TRIM(CAST({_sql_col(col)} AS TEXT))) LIKE '%{normalized_value.replace(chr(39), chr(39)*2)}%'"
                )
            else:
                val = exact[0].replace("'", "''")
                where_parts.append(f"LOWER(TRIM(CAST({_sql_col(col)} AS TEXT))) = LOWER('{val}')")
        else:
            # Use the requested phrase as a safe substring filter; this also
            # tolerates user abbreviations and punctuation differences.
            escaped = normalized_value.replace("'", "''")
            where_parts.append(f"LOWER(TRIM(CAST({_sql_col(col)} AS TEXT))) LIKE '%{escaped}%'")

    if not where_parts:
        return None
    selected = ", ".join(_sql_col(c) for c in target_cols)
    where_sql = " AND ".join(where_parts)
    return f"SELECT {selected} FROM {_SQL_TABLE} WHERE {where_sql} LIMIT {RESULT_ROW_LIMIT}"

def natural_language_to_sql(question: str, df: pd.DataFrame) -> str | None:
    """Map common natural-language requests to safe, read-only SQL without an API."""
    original = (question or "").strip()
    text = _norm(original)
    if not text:
        return None
    columns = [str(c) for c in df.columns]
    mentioned = _find_columns(original, columns)

    # Resolve individual-record questions before aggregate/column-only rules.
    detail_sql = _individual_detail_sql(original, df)
    if detail_sql:
        return detail_sql

    # Projection + filter questions must be handled before broad column selection.
    # Example: "order id who have status as shipped" should return only Order ID
    # values matching the Status condition, not the first 1,000 rows.
    filtered_sql = _filtered_projection_sql(original, df)
    if filtered_sql:
        return filtered_sql

    # Grouping must be resolved before simple column counts; otherwise
    # "count X by Y" or multi-column group requests can lose their dimensions.
    grouping_sql = _grouping_query_sql(original, df)
    if grouping_sql:
        return grouping_sql

    # Handle aggregation requests on a specifically named column before entity
    # matching. This prevents "count of doctor", "total blood type", etc. from
    # being mistaken for a request to list raw rows.
    if mentioned:
        primary_col = mentioned[0]
        quoted_col = _sql_col(primary_col)
        asks_count_specific = bool(re.search(
            r"\b(count|counts|number of|how many|frequency|distribution|breakdown|total of|count of)\b", text
        ))
        asks_total = bool(re.search(r"\b(total|sum)\b", text))
        asks_rows = bool(re.search(r"\b(rows|records|entries|all patients|all customers)\b", text))
        if asks_count_specific and not asks_rows:
            # A numeric total/sum should remain arithmetic; categorical totals
            # mean the frequency of each category.
            if asks_total and _numeric(df, primary_col) and not re.search(r"\b(count|counts|number of|how many|frequency|distribution|breakdown)\b", text):
                return f"SELECT SUM({quoted_col}) AS total_value FROM {_SQL_TABLE} WHERE {quoted_col} IS NOT NULL"
            return (f"SELECT {quoted_col} AS category, COUNT(*) AS record_count "
                    f"FROM {_SQL_TABLE} WHERE {quoted_col} IS NOT NULL "
                    f"GROUP BY {quoted_col} ORDER BY record_count DESC LIMIT {RESULT_ROW_LIMIT}")

    # Specific entity/value questions must be handled before broad column-only
    # rules, otherwise prompts like "date of admission of Bobby Jackson" can
    # accidentally return every date in the dataset.
    entity_sql = _entity_lookup_sql(original, df)
    if entity_sql:
        return entity_sql

    # A named-column request followed by 'by' is usually a grouped summary.
    # Examples: 'medical condition by age', 'group medical condition by age',
    # 'show sales by city', 'number of patients by hospital'.
    by_match = re.search(r"\bby\s+(.+)$", original, flags=re.I)
    if by_match:
        group_col = _match_column(by_match.group(1), columns)
        before_by = original[:by_match.start()]
        measure_cols = _find_columns(before_by, columns)
        # Remove common instruction words; if a column is explicitly named before 'by',
        # make a grouped summary. Numeric measures use SUM/AVG when requested;
        # categorical measures use COUNT by both columns.
        if group_col and measure_cols:
            measure_col = next((c for c in measure_cols if c != group_col), None)
            if measure_col:
                mc, gc = _sql_col(measure_col), _sql_col(group_col)
                asks_avg = bool(re.search(r"\b(average|avg|mean)\b", text))
                asks_sum = bool(re.search(r"\b(total|sum|sales|revenue|amount|spend|cost|price|quantity)\b", text))
                asks_count = bool(re.search(r"\b(count|number of|how many|frequency|distribution|group|breakdown)\b", text))
                if _numeric(df, measure_col) and asks_avg:
                    return f"SELECT {gc} AS group_value, AVG({mc}) AS average_value FROM {_SQL_TABLE} WHERE {mc} IS NOT NULL GROUP BY {gc} ORDER BY average_value DESC LIMIT {RESULT_ROW_LIMIT}"
                if _numeric(df, measure_col) and asks_sum:
                    return f"SELECT {gc} AS group_value, SUM({mc}) AS total_value FROM {_SQL_TABLE} WHERE {mc} IS NOT NULL GROUP BY {gc} ORDER BY total_value DESC LIMIT {RESULT_ROW_LIMIT}"
                # If a categorical field is requested 'by' another field, show its
                # distribution across both columns rather than returning raw rows.
                if not _numeric(df, measure_col) or asks_count:
                    return (f"SELECT {gc} AS {_sql_col(group_col)}, {mc} AS {_sql_col(measure_col)}, "
                            f"COUNT(*) AS record_count FROM {_SQL_TABLE} "
                            f"GROUP BY {gc}, {mc} ORDER BY {gc}, record_count DESC LIMIT {RESULT_ROW_LIMIT}")
            elif group_col and re.search(r"\b(group|grouped|distribution|breakdown|count|number|how many)\b", text):
                gc = _sql_col(group_col)
                return f"SELECT {gc} AS category, COUNT(*) AS record_count FROM {_SQL_TABLE} GROUP BY {gc} ORDER BY record_count DESC LIMIT {RESULT_ROW_LIMIT}"

    # First N rows / preview.
    first_rows = re.search(r"\b(?:show|list|display|get|give|fetch)?\s*(?:me\s+)?(?:the\s+)?first\s+(\d+)\s+rows?\b", text)
    if first_rows:
        n = max(1, min(int(first_rows.group(1)), RESULT_ROW_LIMIT))
        return f"SELECT * FROM {_SQL_TABLE} LIMIT {n}"
    if text in {"show data", "show dataset", "show table", "preview data", "show the data", "show all rows", "display dataset", "list all data"}:
        return f"SELECT * FROM {_SQL_TABLE} LIMIT 100"

    # Count a named category locally (e.g. "total males", "how many females").
    gender_col = next((str(c) for c in df.columns if _norm(c) in
                       {"gender", "sex", "patient gender", "customer gender", "patient sex"}), None)
    if gender_col and re.search(r"\b(total|count|number|how many|quantity|amount of|no of)\b", text):
        gender_aliases = {
            "male": {"male", "males", "m", "man", "men", "boy", "boys"},
            "female": {"female", "females", "f", "woman", "women", "girl", "girls"},
        }
        requested = next((label for label, words in gender_aliases.items()
                          if re.search(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", text)), None)
        if requested:
            values = df[gender_col].dropna().astype(str).str.strip().str.lower()
            observed = set(values.unique())
            aliases = gender_aliases[requested]
            found = sorted(observed.intersection(aliases))
            if found:
                vals = ", ".join("'" + v.replace("'", "''") + "'" for v in found)
                return (f"SELECT COUNT(*) AS {requested}_count FROM {_SQL_TABLE} "
                        f"WHERE LOWER(TRIM(CAST({_sql_col(gender_col)} AS TEXT))) IN ({vals})")

    # Counts.
    if re.search(r"\b(how many|count|number of|total)\b.*\b(rows|records|entries|patients|customers)\b|\b(row count|record count|total records)\b", text):
        return f"SELECT COUNT(*) AS record_count FROM {_SQL_TABLE}"

    # Missing/null values.
    if re.search(r"\b(missing|null|empty|blank)\b", text) and re.search(r"\b(value|cell|data|count|how many|column|columns)\b", text):
        if mentioned:
            expressions = [f"SUM(CASE WHEN {_sql_col(c)} IS NULL THEN 1 ELSE 0 END) AS {_sql_col(c + '_missing')}" for c in mentioned]
        else:
            expressions = [f"SUM(CASE WHEN {_sql_col(c)} IS NULL THEN 1 ELSE 0 END) AS {_sql_col(c + '_missing')}" for c in columns]
        return f"SELECT {', '.join(expressions)} FROM {_SQL_TABLE}" if expressions else None

    # Average / mean of a numeric field by a grouping field.
    if re.search(r"\b(average|avg|mean)\b", text):
        numeric_col = next((c for c in mentioned if _numeric(df, c)), None)
        group_col = next((c for c in mentioned if c != numeric_col), None)
        if numeric_col and group_col:
            v, g = _sql_col(numeric_col), _sql_col(group_col)
            return f"SELECT {g} AS category, AVG({v}) AS average_value FROM {_SQL_TABLE} WHERE {v} IS NOT NULL GROUP BY {g} ORDER BY average_value DESC LIMIT {RESULT_ROW_LIMIT}"
        if numeric_col:
            v = _sql_col(numeric_col)
            return f"SELECT AVG({v}) AS average_value FROM {_SQL_TABLE} WHERE {v} IS NOT NULL"

    # Sum / total numeric field by category.
    if re.search(r"\b(sum|total)\b", text):
        numeric_col = next((c for c in mentioned if _numeric(df, c)), None)
        group_col = next((c for c in mentioned if c != numeric_col), None)
        if numeric_col and group_col:
            v, g = _sql_col(numeric_col), _sql_col(group_col)
            return f"SELECT {g} AS category, SUM({v}) AS total_value FROM {_SQL_TABLE} WHERE {v} IS NOT NULL GROUP BY {g} ORDER BY total_value DESC LIMIT {RESULT_ROW_LIMIT}"
        if numeric_col:
            v = _sql_col(numeric_col)
            return f"SELECT SUM({v}) AS total_value FROM {_SQL_TABLE} WHERE {v} IS NOT NULL"

    # Count/distribution by a column, including 'how many patients per city'.
    if re.search(r"\b(count|number|how many|frequency|distribution|breakdown|group|grouped)\b", text):
        group_col = mentioned[-1] if mentioned else None
        if group_col:
            g = _sql_col(group_col)
            return f"SELECT {g} AS category, COUNT(*) AS record_count FROM {_SQL_TABLE} GROUP BY {g} ORDER BY record_count DESC LIMIT {RESULT_ROW_LIMIT}"

    # Show/list/filter a specifically named column, rather than all columns.
    if mentioned and re.search(r"\b(show|list|display|give|find|view|what are|which)\b", text):
        cols_sql = ", ".join(_sql_col(c) for c in mentioned)
        distinct = bool(re.search(r"\b(unique|distinct|different)\b", text))
        prefix = "SELECT DISTINCT" if distinct else "SELECT"
        return f"{prefix} {cols_sql} FROM {_SQL_TABLE} LIMIT {RESULT_ROW_LIMIT}"

    # Highest / lowest rows.
    if re.search(r"\b(highest|maximum|largest|top|most|greatest)\b", text):
        numeric_col = next((c for c in mentioned if _numeric(df, c)), None) or _first_numeric_column(df)
        if numeric_col:
            c = _sql_col(numeric_col)
            n_match = re.search(r"\btop\s+(\d+)\b", text)
            n = min(int(n_match.group(1)), RESULT_ROW_LIMIT) if n_match else 10
            return f"SELECT * FROM {_SQL_TABLE} WHERE {c} IS NOT NULL ORDER BY {c} DESC LIMIT {n}"
    if re.search(r"\b(lowest|minimum|smallest|least)\b", text):
        numeric_col = next((c for c in mentioned if _numeric(df, c)), None) or _first_numeric_column(df)
        if numeric_col:
            c = _sql_col(numeric_col)
            return f"SELECT * FROM {_SQL_TABLE} WHERE {c} IS NOT NULL ORDER BY {c} ASC LIMIT 10"

    # Unique / distinct values.
    if re.search(r"\b(unique|distinct|different)\b", text) and mentioned:
        c = _sql_col(mentioned[0])
        return f"SELECT DISTINCT {c} FROM {_SQL_TABLE} WHERE {c} IS NOT NULL LIMIT {RESULT_ROW_LIMIT}"

    # Simple column-only requests should return just the requested field(s).
    if mentioned and len(mentioned) <= 4:
        cols_sql = ", ".join(_sql_col(c) for c in mentioned)
        return f"SELECT {cols_sql} FROM {_SQL_TABLE} LIMIT {RESULT_ROW_LIMIT}"

    return None


NL_UNAVAILABLE_MESSAGE = (
    "I couldn't confidently interpret that question. Try naming the column(s) and "
    "the result you want, such as 'count medical condition by age', 'show medical "
    "condition by age', 'average age by city', or write a read-only SQL SELECT query. "
    "For unrestricted natural-language understanding, connect an AI language model."
)



def _llm_natural_language_to_sql(question: str, df: pd.DataFrame) -> str | None:
    """Use an optional OpenAI-compatible LLM to translate a question into read-only SQLite SQL.

    Requires OPENAI_API_KEY. The API key is read from environment variables / .env;
    it is never sent to the browser or stored in the dataset.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ValueError(
            "LLM query support is not installed. Run: python -m pip install openai"
        ) from exc

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)

    schema = []
    for col in df.columns:
        series = df[col]
        schema.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "examples": [None if pd.isna(v) else str(v)[:100] for v in series.dropna().head(3).tolist()],
        })

    # Send only schema metadata, never dataset rows or cell values. This is
    # especially important when users upload health, customer, or other private data.
    prompt = {
        "question": question,
        "table_name": TABLE_NAME,
        "columns": [{"name": item["name"], "dtype": item["dtype"]} for item in schema],
    }
    system = (
        "You translate natural-language questions into ONE SQLite SELECT query for a dataframe table. "
        "Return JSON only in the shape {\"sql\": \"SELECT ...\"}. The only table is \"table\" (always double-quote this table name because TABLE is a SQL keyword). "
        "Use exact column names and double-quote identifiers. Never create, update, delete, or modify data. "
        "No PRAGMA, ATTACH, extension loading, multiple statements, markdown, or explanations. "
        "For 'X by Y' when X is categorical and Y is numeric/age, return a useful grouped summary: "
        "group by the requested dimensions and include COUNT(*) as record_count. If the wording asks "
        "for an association/distribution, group by both named columns and count records. "
        "If a question is ambiguous, choose the safest useful interpretation based on column names. "
        "Limit raw row outputs to 100 rows. SQL dialect is SQLite."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)},
            ],
            timeout=8,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        sql = parsed.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("The language model did not return a SQL query.")
        # Validate before the query ever reaches SQLite.
        return validate_sql(sql)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "The AI query service could not complete the request. Check your API key, "
            "internet connection, model name, and API billing/limits. Details: " + str(exc)
        ) from exc

def _connection_for_dataframe(df: pd.DataFrame) -> sqlite3.Connection:
    """Return a cached read-only SQLite connection for this dataframe object."""
    key = id(df)
    with _SQLITE_CACHE_LOCK:
        cached = _SQLITE_CACHE.get(key)
        if cached is not None and cached[0]() is df:
            return cached[1]

        # Remove stale/dead entries and close their connections.
        for old_key, (df_ref, old_connection) in list(_SQLITE_CACHE.items()):
            if df_ref() is None or old_key == key:
                try:
                    old_connection.close()
                except Exception:
                    pass
                _SQLITE_CACHE.pop(old_key, None)

        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            # This database is temporary and in-memory, so durability is not
            # needed during the one-time bulk load. These settings reduce the
            # cost of the first upload without changing query results.
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA temp_store = MEMORY")
            # Chunked insertion avoids a single oversized insert for large files.
            # The connection is cached, so this load is not repeated per query.
            df.to_sql(
                TABLE_NAME,
                connection,
                index=False,
                if_exists="replace",
                chunksize=10000,
            )
            connection.execute("PRAGMA query_only = ON")
        except Exception:
            connection.close()
            raise

        # Keep a weak reference so discarded dataframes can be evicted.
        import weakref
        _SQLITE_CACHE[key] = (weakref.ref(df), connection)
        while len(_SQLITE_CACHE) > _SQLITE_CACHE_LIMIT:
            oldest_key = next(iter(_SQLITE_CACHE))
            _, old_connection = _SQLITE_CACHE.pop(oldest_key)
            try:
                old_connection.close()
            except Exception:
                pass
        return connection


def prepare_dataframe(df: pd.DataFrame) -> None:
    """Build the in-memory SQLite table during upload, not when the user first queries."""
    _connection_for_dataframe(df)


# ---------------------------------------------------------------------------
# Sensitive-column masking
# ---------------------------------------------------------------------------

# Columns whose raw values must never appear in public-facing query results.
# Matching is case-insensitive against the full column name. Extend as needed.
_SENSITIVE_TOKENS: tuple[str, ...] = (
    "card_number", "card number", "cardnumber",
    "cvv", "cvc", "security code",
    "ssn", "social security",
    "password", "passwd",
    "pin",
)


def _mask_sensitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace values in sensitive columns with '***' before returning to the UI.

    Only columns whose lowercased name contains a known sensitive token are
    masked. The original DataFrame is never mutated; a copy is returned only
    when at least one column needs masking.
    """
    cols_to_mask = [
        col for col in df.columns
        if any(token in str(col).lower() for token in _SENSITIVE_TOKENS)
    ]
    if not cols_to_mask:
        return df
    masked = df.copy()
    for col in cols_to_mask:
        masked[col] = "***"
    return masked


def execute_sql(df: pd.DataFrame, sql: str) -> tuple[pd.DataFrame, bool]:
    """Run validated SELECT SQL against the dataframe without bulk-copying it.

    DuckDB can query a registered pandas dataframe directly. This avoids the
    slow first-query step of serializing every cell into an in-memory SQLite
    table. SQLite is retained as a fallback when DuckDB is not installed.
    """
    safe_sql = validate_sql(sql)
    try:
        if duckdb is not None:
            key = id(df)
            with _DUCKDB_CACHE_LOCK:
                cached = _DUCKDB_CACHE.get(key)
                if cached is None or cached[0]() is not df:
                    # Close stale entries and avoid retaining discarded datasets.
                    for old_key, (old_ref, old_con) in list(_DUCKDB_CACHE.items()):
                        if old_ref() is None or old_key == key:
                            try:
                                old_con.close()
                            except Exception:
                                pass
                            _DUCKDB_CACHE.pop(old_key, None)
                    connection = duckdb.connect(database=":memory:")
                    connection.register(TABLE_NAME, df)
                    _DUCKDB_CACHE[key] = (weakref.ref(df), connection)
                    while len(_DUCKDB_CACHE) > _SQLITE_CACHE_LIMIT:
                        oldest_key = next(iter(_DUCKDB_CACHE))
                        _, old_con = _DUCKDB_CACHE.pop(oldest_key)
                        try:
                            old_con.close()
                        except Exception:
                            pass
                else:
                    connection = cached[1]
                result = connection.execute(safe_sql).fetchdf()
        else:
            with _SQLITE_CACHE_LOCK:
                connection = _connection_for_dataframe(df)
                result = pd.read_sql_query(safe_sql, connection)
    except Exception as exc:
        raise ValueError(_friendly_sql_error(exc)) from exc
    truncated = len(result) > RESULT_ROW_LIMIT
    if truncated:
        result = result.head(RESULT_ROW_LIMIT)
    return _mask_sensitive_columns(result), truncated


def _friendly_sql_error(exc: Exception) -> str:
    message = str(exc)
    if "no such table" in message.lower():
        return f"Use the table name `{TABLE_NAME}` in your SQL."
    if "no such column" in message.lower():
        return "A column in the query was not found. Check the dataset preview for exact names."
    return f"The query could not be run: {message}"


def run_user_query(df: pd.DataFrame, text: str) -> dict[str, Any]:
    query_text = (text or "").strip()
    if not query_text:
        raise ValueError("Enter a SQL query or a question about your data.")
    generated_sql = None
    if looks_like_sql(query_text):
        sql = query_text
    else:
        normalized = _norm(query_text)
        # Avoid natural-language entity matching and SQL initialization for the
        # common preview request. This is especially important on the first query.
        first_rows = re.search(r"\b(?:show|list|display|get|give|fetch)?\s*(?:me\s+)?(?:the\s+)?first\s+(\d+)\s+rows?\b", normalized)
        if first_rows:
            n = max(1, min(int(first_rows.group(1)), RESULT_ROW_LIMIT))
            result = _mask_sensitive_columns(df.head(n).copy())
            return {"sql": f"SELECT * FROM {_SQL_TABLE} LIMIT {n}", "generated_sql": None, "result": result, "truncated": len(df) > n}
        # Handle common full-dataset count requests directly with pandas. This
        # avoids constructing a complete in-memory SQLite copy on the first query.
        if re.search(r"\b(how many|count|number of|total)\b.*\b(rows|records|entries|patients|customers)\b|\b(row count|record count|total records)\b", normalized):
            return {
                "sql": f"SELECT COUNT(*) AS record_count FROM {_SQL_TABLE}",
                "generated_sql": None,
                "result": pd.DataFrame({"record_count": [len(df)]}),
                "truncated": False,
            }
        # Execute straightforward column aggregations directly with pandas
        # before natural-language matching. This avoids scanning unique values
        # or initializing a SQL engine for routine count/sum questions.
        mentioned_fast = _find_columns(query_text, [str(c) for c in df.columns])
        normalized_fast = _norm(query_text)
        by_match_fast = bool(re.search(r"\bby\b", normalized_fast))
        wants_count_fast = bool(re.search(r"\b(count|counts|number of|how many|frequency|distribution|breakdown|count of)\b", normalized_fast))
        wants_sum_fast = bool(re.search(r"\b(total|sum)\b", normalized_fast))
        asks_rows_fast = bool(re.search(r"\b(rows|records|entries|all patients|all customers)\b", normalized_fast))
        if mentioned_fast and not by_match_fast and not asks_rows_fast:
            col_fast = mentioned_fast[0]
            if wants_count_fast and not wants_sum_fast:
                counts_fast = df[col_fast].value_counts(dropna=True).rename_axis("category").reset_index(name="record_count")
                counts_fast = counts_fast.head(RESULT_ROW_LIMIT)
                return {"sql": f"SELECT {_sql_col(col_fast)} AS category, COUNT(*) AS record_count FROM {_SQL_TABLE} GROUP BY {_sql_col(col_fast)} ORDER BY record_count DESC LIMIT {RESULT_ROW_LIMIT}", "generated_sql": None, "result": _mask_sensitive_columns(counts_fast), "truncated": df[col_fast].nunique(dropna=True) > RESULT_ROW_LIMIT}
            if wants_sum_fast and _numeric(df, col_fast):
                total_fast = pd.to_numeric(df[col_fast], errors="coerce").sum()
                return {"sql": f"SELECT SUM({_sql_col(col_fast)}) AS total_value FROM {_SQL_TABLE}", "generated_sql": None, "result": pd.DataFrame({"total_value": [total_fast]}), "truncated": False}
        generated_sql = natural_language_to_sql(query_text, df)
        if generated_sql is None:
            generated_sql = _llm_natural_language_to_sql(query_text, df)
        if generated_sql is None:
            if not os.getenv("OPENAI_API_KEY", "").strip():
                raise ValueError(
                    "This question needs AI interpretation, but OPENAI_API_KEY is not configured. "
                    "Create a .env file from .env.example, add your API key, and restart AUTOLYTICS. "
                    "Simple built-in questions and SQL queries can still run without the key."
                )
            raise ValueError(NL_UNAVAILABLE_MESSAGE)
        sql = generated_sql
        # Fast path for common natural-language category-count questions. Avoid
        # building a second full SQLite copy of a large dataframe just to count
        # values in one column.
        if (mentioned_columns := _find_columns(query_text, [str(c) for c in df.columns])) and not re.search(r"\bby\b", _norm(query_text)):
            primary_col = mentioned_columns[0]
            normalized_question = _norm(query_text)
            wants_count = bool(re.search(r"\b(count|counts|number of|how many|frequency|distribution|breakdown|count of|total of)\b", normalized_question))
            wants_rows = bool(re.search(r"\b(rows|records|entries|all patients|all customers)\b", normalized_question))
            wants_sum = bool(re.search(r"\b(total|sum)\b", normalized_question))
            wants_count_word = bool(re.search(r"\b(count|counts|number of|how many|frequency|distribution|breakdown)\b", normalized_question))
            if wants_count and not wants_rows and (not wants_sum or not _numeric(df, primary_col) or wants_count_word):
                series = df[primary_col].dropna()
                counts = series.value_counts(dropna=True).rename_axis("category").reset_index(name="record_count")
                counts = counts.head(RESULT_ROW_LIMIT)
                return {"sql": validate_sql(sql), "generated_sql": generated_sql, "result": _mask_sensitive_columns(counts), "truncated": len(series.unique()) > RESULT_ROW_LIMIT}
            if wants_sum and _numeric(df, primary_col) and not wants_count_word:
                total = pd.to_numeric(df[primary_col], errors="coerce").sum()
                return {"sql": validate_sql(sql), "generated_sql": generated_sql, "result": pd.DataFrame({"total_value": [total]}), "truncated": False}
    result, truncated = execute_sql(df, sql)
    return {"sql": validate_sql(sql), "generated_sql": generated_sql, "result": result, "truncated": truncated}