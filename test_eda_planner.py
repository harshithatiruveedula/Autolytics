"""Logic test for the EDA chart planner — verifies combination charts are generated."""
import sys, os
sys.path.insert(0, r'e:\Autolytics')
os.chdir(r'e:\Autolytics')

import uuid
import pandas as pd
import numpy as np

# Patch streamlit so it doesn't error when imported outside a session
import unittest.mock as mock
import streamlit as _st
_st.warning = lambda *a, **k: None
_st.error   = lambda *a, **k: None

from app import _build_charts_for_selection

def make_hospital(n=200):
    rng = np.random.default_rng(42)
    conditions = ['Cancer','Diabetes','Asthma','Hypertension','Arthritis','Obesity']
    medications = ['Aspirin','Ibuprofen','Penicillin','Paracetamol','Metformin','Lipitor']
    blood_types = ['A+','A-','B+','B-','O+','O-','AB+','AB-']
    genders = ['Male','Female']
    test_results = ['Normal','Abnormal','Inconclusive']
    return pd.DataFrame({
        'Medical Condition': rng.choice(conditions, n),
        'Medication':        rng.choice(medications, n),
        'Blood Type':        rng.choice(blood_types, n),
        'Gender':            rng.choice(genders, n),
        'Test Results':      rng.choice(test_results, n),
        'Billing Amount':    rng.uniform(1000, 50000, n),
        'Age':               rng.integers(18, 90, n),
    })

def make_education(n=150):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Gender':          rng.choice(['Male','Female'], n),
        'Grade Level':     rng.choice(['6','7','8','9','10'], n),
        'Extracurriculars':rng.choice(['Art','Music','Sport','None'], n),
        'Math Score':      rng.integers(40, 100, n).astype(float),
        'Reading Score':   rng.integers(40, 100, n).astype(float),
    })

def make_ecommerce(n=200):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Category': rng.choice(['Electronics','Clothing','Home','Books'], n),
        'Status':   rng.choice(['Shipped','Cancelled','Pending','Returned'], n),
        'Amount':   rng.uniform(10, 500, n),
        'Region':   rng.choice(['North','South','East','West'], n),
    })

def make_finance(n=150):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Card Type':  rng.choice(['Debit','Credit','Prepaid'], n),
        'Card Brand': rng.choice(['Visa','Mastercard','Amex'], n),
        'Credit Limit':     rng.uniform(1000, 20000, n),
        'Num Cards Issued': rng.integers(1, 5, n).astype(float),
    })

TESTS = [
    # (name, df_fn, selected_cols, expected_section_types)
    ('Hospital: Med Condition × Medication',
     make_hospital,
     ['Medical Condition','Medication'],
     ['cat_cat']),

    ('Hospital: Med Condition × Blood Type',
     make_hospital,
     ['Medical Condition','Blood Type'],
     ['cat_cat']),

    ('Hospital: Med Condition × Billing Amount',
     make_hospital,
     ['Medical Condition','Billing Amount'],
     ['groupbar']),

    ('Hospital: Gender × Test Results',
     make_hospital,
     ['Gender','Test Results'],
     ['cat_cat']),

    ('Hospital: Age × Billing Amount',
     make_hospital,
     ['Age','Billing Amount'],
     ['scatter']),

    ('Education: Gender × Math Score',
     make_education,
     ['Gender','Math Score'],
     ['groupbar']),

    ('Education: Gender × Extracurriculars',
     make_education,
     ['Gender','Extracurriculars'],
     ['cat_cat']),

    ('Ecommerce: Category × Status',
     make_ecommerce,
     ['Category','Status'],
     ['cat_cat']),

    ('Finance: Card Type × Card Brand',
     make_finance,
     ['Card Type','Card Brand'],
     ['cat_cat']),

    ('Finance: Credit Limit × Num Cards Issued',
     make_finance,
     ['Credit Limit','Num Cards Issued'],
     ['scatter']),
]

passed = 0
failed = 0
for name, df_fn, cols, expected_sections in TESTS:
    df = df_fn()
    try:
        charts = _build_charts_for_selection(df, cols, 'Auto (recommended)')
        got_sections = [c['section'] for c in charts]
        ok = any(s in got_sections for s in expected_sections)
        if ok:
            print(f'PASS  {name}')
            print(f'      sections: {got_sections}')
            passed += 1
        else:
            print(f'FAIL  {name}')
            print(f'      expected one of {expected_sections}, got {got_sections}')
            failed += 1
    except Exception as e:
        print(f'ERROR {name}: {e}')
        failed += 1

print(f'\n{passed} passed, {failed} failed')
sys.exit(0 if failed == 0 else 1)
