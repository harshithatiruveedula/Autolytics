"""Logic tests for the EDA chart planner — uses _generate_auto_charts via domain_analytics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import unittest.mock as mock

# Patch streamlit before importing app so cache_data / session_state are mocked
st_mock = mock.MagicMock()
st_mock.cache_data = lambda **kw: (lambda f: f)   # pass-through decorator
st_mock.session_state = mock.MagicMock()
st_mock.session_state.__contains__ = lambda self, key: False
st_mock.session_state.__getitem__ = lambda self, key: None
st_mock.session_state.__setitem__ = lambda self, key, val: None
st_mock.session_state.get = lambda key, default=None: default
sys.modules.setdefault('streamlit', st_mock)

from domain_analytics import detect_domain_and_roles, build_domain_charts


def make_hospital(n=200):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Medical Condition': rng.choice(['Cancer','Diabetes','Asthma','Hypertension','Obesity'], n),
        'Medication':        rng.choice(['Aspirin','Ibuprofen','Metformin','Lipitor','Penicillin'], n),
        'Blood Type':        rng.choice(['A+','B+','O+','AB+'], n),
        'Gender':            rng.choice(['Male','Female'], n),
        'Test Results':      rng.choice(['Normal','Abnormal','Inconclusive'], n),
        'Billing Amount':    rng.uniform(1000, 50000, n),
        'Age':               rng.integers(18, 90, n).astype(float),
    })


def make_education(n=150):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Student_ID':       range(n),
        'Gender':           rng.choice(['Male','Female'], n),
        'Grade_Level':      rng.choice(['6','7','8','9','10'], n),
        'Math_Score':       rng.integers(40, 100, n).astype(float),
        'Reading_Score':    rng.integers(40, 100, n).astype(float),
        'Extracurriculars': rng.choice(['Art','Music','Sport','None'], n),
    })


TESTS = [
    (make_hospital,   'healthcare', ['Medical Condition', 'Medication']),
    (make_education,  'education',  ['Grade_Level', 'Math_Score']),
]

passed = failed = 0

for df_fn, expected_domain, expected_cols in TESTS:
    df = df_fn()
    domain_info = detect_domain_and_roles(df)
    charts = build_domain_charts(df, domain_info, max_charts=20)
    got_domain = domain_info.get('name', 'generic')

    if got_domain == expected_domain:
        print(f'PASS  domain={got_domain}')
        passed += 1
    else:
        print(f'FAIL  expected domain={expected_domain}, got={got_domain}')
        failed += 1

    # Verify at least one chart uses the expected columns
    found = any(
        all(c in (item.get('cols_used') or []) for c in expected_cols)
        for item in charts
    )
    if found:
        print(f'PASS  chart uses {expected_cols}')
        passed += 1
    else:
        chart_cols = [item.get('cols_used') for item in charts]
        print(f'FAIL  no chart uses {expected_cols}. Got: {chart_cols[:5]}')
        failed += 1

print(f'\n{passed} passed, {failed} failed')
sys.exit(0 if failed == 0 else 1)
