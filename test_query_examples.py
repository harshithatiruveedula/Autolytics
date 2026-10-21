"""Self-contained smoke tests for AUTOLYTICS natural-language query patterns.
Run from this folder with: python test_query_examples.py
"""
import pandas as pd
from query import run_user_query

amazon = pd.DataFrame({
    "index": [0, 1, 2],
    "Order ID": ["405-8078784-5731545", "171-9198151-1101146", "404-0687676-7273146"],
    "Status": ["Cancelled", "Shipped - Delivered to Buyer", "Shipped"],
    "Category": ["Set", "kurta", "kurta"],
    "Amount": [647.62, 406.00, 329.00],
})
finance = pd.DataFrame({
    "id": [4524, 2731, 3701],
    "client_id": [825, 825, 1746],
    "card_brand": ["Visa", "Visa", "Mastercard"],
    "card_type": ["Debit", "Credit", "Credit"],
})
education = pd.DataFrame({
    "Student_ID": [1, 2, 3],
    "Gender": ["Female", "Female", "Male"],
    "Math_Score": [7, 94, 81],
    "Extracurriculars": ["Art", "Music", "Art"],
})

checks = [
    (amazon, "order id who have status as shipped", "Order ID", 2),
    (amazon, "order id where status is cancelled", "Order ID", 1),
    (amazon, "order id of index 1", "Order ID", 1),
    (finance, "client id of 1", "client_id", 1),
    (finance, "client id of index 1", "client_id", 1),
    (finance, "card_type of client 825", "card_type", 1),
    (education, "gender of student_id 1", "Gender", 1),
    (education, "student_id who have gender as female", "Student_ID", 2),
]
for df, question, expected_column, min_rows in checks:
    result = run_user_query(df, question)["result"]
    assert expected_column in result.columns, (question, list(result.columns))
    assert len(result) >= min_rows, (question, len(result), result)
    print(f"PASS: {question} -> {len(result)} row(s)")

# Confirm multi-column grouping still generates both requested dimensions.
from query import natural_language_to_sql
sql = natural_language_to_sql("group extracurriculars by student_id", education)
assert sql and '"Extracurriculars"' in sql and '"Student_ID"' in sql, sql
print("PASS: multi-column grouping retains requested columns")
