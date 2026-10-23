"""Focused regression tests for four-domain semantic detection and narrative outputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from domain_analytics import detect_domain_and_roles, generate_narrative, build_domain_charts

rng = np.random.default_rng(7)
N = 80
DATASETS = {
    "ecommerce": pd.DataFrame({
        "Purchase Date": pd.date_range("2024-01-01", periods=N, freq="D"),
        "Product Name": rng.choice(["A", "B", "C"], N),
        "Product Category": rng.choice(["Clothing", "Home", "Electronics"], N),
        "Order Value": rng.uniform(10, 250, N), "Units Sold": rng.integers(1, 5, N),
        "Fulfilment Status": rng.choice(["Shipped", "Cancelled", "Returned"], N),
        "Ship State": rng.choice(["CA", "NY", "TX"], N),
    }),
    "finance": pd.DataFrame({
        "Card Brand": rng.choice(["Visa", "Mastercard", "Amex"], N),
        "Card Type": rng.choice(["Credit", "Debit"], N),
        "Credit Limit": rng.integers(1000, 20000, N),
        "Num Cards Issued": rng.integers(1, 5, N),
        "Has Chip": rng.choice(["Yes", "No"], N),
        "Card on Dark Web": rng.choice(["No", "Yes"], N),
    }),
    "education": pd.DataFrame({
        "Learner ID": range(N), "Sex": rng.choice(["Male", "Female"], N),
        "Grade_Level": rng.choice(["6", "7", "8"], N),
        "Maths Score": rng.integers(40, 100, N), "Reading Score": rng.integers(40, 100, N),
        "Writing Score": rng.integers(40, 100, N), "Attendance Percentage": rng.uniform(60, 100, N),
        "Activities": rng.choice(["Art", "Music", "None"], N),
    }),
    "healthcare": pd.DataFrame({
        "Primary Dx": rng.choice(["Asthma", "Cancer", "Diabetes"], N),
        "Drug Name": rng.choice(["Aspirin", "Metformin"], N),
        "Treatment Cost": rng.uniform(100, 5000, N),
        "Admit Date": pd.date_range("2024-01-01", periods=N, freq="D"),
        "Blood Group": rng.choice(["A+", "O-"], N), "Sex": rng.choice(["Male", "Female"], N),
        "Patient Age": rng.integers(20, 90, N), "Encounter Class": rng.choice(["Emergency", "Routine"], N),
        "Lab Results": rng.choice(["Normal", "Abnormal"], N),
    }),
}

passed = failed = 0
for expected, df in DATASETS.items():
    info = detect_domain_and_roles(df)
    narrative = generate_narrative(df, info)
    checks = [
        (f"{expected}: domain detected", info["name"] == expected),
        (f"{expected}: insights generated", len(narrative["insights"]) >= 2),
        (f"{expected}: suggestions generated", len(narrative["suggestions"]) >= 1),
        (f"{expected}: story generated", len(narrative["story"]) > 30),
    ]
    for label, ok in checks:
        print(("PASS " if ok else "FAIL ") + label)
        passed += int(ok); failed += int(not ok)

# Finance-specific regression: credit limit must never be mislabelled as income.
finance_info = detect_domain_and_roles(DATASETS["finance"])
check = finance_info["mappings"].get("credit_limit") == "Credit Limit" and "income" not in finance_info["mappings"]
print(("PASS " if check else "FAIL ") + "finance: credit limit is not income")
passed += int(check); failed += int(not check)

# E-commerce headline amounts exclude explicitly cancelled records when status is known.
ecom_small = pd.DataFrame({"Order Value": [100.0, 200.0, 300.0], "Order Status": ["Shipped", "Cancelled", "Delivered"], "Product Category": ["A", "B", "A"]})
ecom_info = detect_domain_and_roles(ecom_small)
ecom_insights = generate_narrative(ecom_small, ecom_info)["insights"]
check = any("400.00" in item and "excluding explicitly cancelled" in item for item in ecom_insights)
print(("PASS " if check else "FAIL ") + "e-commerce: cancelled amount excluded from sales headline")
passed += int(check); failed += int(not check)

# Unknown domains should still receive a general analysis rather than false domain claims.
unknown = pd.DataFrame({"field_a": [1, 2, 3, 4], "field_b": ["x", "y", "x", "z"]})
unknown_info = detect_domain_and_roles(unknown)
unknown_story = generate_narrative(unknown, unknown_info)
check = unknown_info["name"] == "generic" and bool(unknown_story["insights"])
print(("PASS " if check else "FAIL ") + "unknown schema: safe generic fallback")
passed += int(check); failed += int(not check)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
