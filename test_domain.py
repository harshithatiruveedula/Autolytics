"""Tests for domain.py — semantic mapping and hospital chart planner."""
import sys, os
sys.path.insert(0, r'e:\Autolytics')
os.chdir(r'e:\Autolytics')

import pandas as pd
import numpy as np

# Patch streamlit
import unittest.mock as mock
sys.modules.setdefault('streamlit', mock.MagicMock())

from domain import detect_domain, map_columns, plan_hospital_charts

def make_hospital_std(n=300):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Medical Condition': rng.choice(['Cancer','Diabetes','Asthma','Hypertension','Obesity'], n),
        'Medication':        rng.choice(['Aspirin','Ibuprofen','Metformin','Lipitor','Penicillin'], n),
        'Department':        rng.choice(['Cardiology','Oncology','General','Neurology'], n),
        'Blood Type':        rng.choice(['A+','B+','O+','AB+','A-','B-'], n),
        'Test Results':      rng.choice(['Normal','Abnormal','Inconclusive'], n),
        'Gender':            rng.choice(['Male','Female'], n),
        'Billing Amount':    rng.uniform(1000, 50000, n),
        'Age':               rng.integers(18, 90, n).astype(float),
        'Date of Admission': pd.date_range('2022-01-01', periods=n, freq='D'),
        'Discharge Status':  rng.choice(['Recovered','Deceased','Transferred'], n),
        'Admission Type':    rng.choice(['Emergency','Elective','Urgent'], n),
    })

def make_hospital_renamed(n=300):
    """Same data but with renamed headers — must still detect healthcare."""
    df = make_hospital_std(n)
    return df.rename(columns={
        'Medical Condition': 'Diagnosis',
        'Medication':        'Drug Name',
        'Billing Amount':    'Treatment Cost',
        'Date of Admission': 'Admit Date',
        'Test Results':      'Lab Results',
    })

def make_education(n=200):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Student_ID':    range(n),
        'Gender':        rng.choice(['Male','Female'], n),
        'Grade_Level':   rng.choice(['6','7','8','9','10'], n),
        'Math_Score':    rng.integers(40,100,n).astype(float),
        'Reading_Score': rng.integers(40,100,n).astype(float),
        'Extracurriculars': rng.choice(['Art','Music','Sport','None'], n),
    })

passed = failed = 0

def check(name, condition, detail=''):
    global passed, failed
    if condition:
        print(f'PASS  {name}')
        passed += 1
    else:
        print(f'FAIL  {name}' + (f' — {detail}' if detail else ''))
        failed += 1

# --- Hospital standard columns ---
df_std = make_hospital_std()
domain_std = detect_domain(df_std)
check('Hospital std — is_healthcare', domain_std['is_healthcare'],
      f"confidence={domain_std['confidence']}, concepts={domain_std['detected_concepts']}")
mappings_std = domain_std['mappings']
check('Hospital std — condition mapped', 'condition' in mappings_std,
      str(mappings_std.get('condition')))
check('Hospital std — medication mapped', 'medication' in mappings_std,
      str(mappings_std.get('medication')))
check('Hospital std — billing mapped', 'billing' in mappings_std,
      str(mappings_std.get('billing')))

plan_std = plan_hospital_charts(df_std, mappings_std, max_charts=8)
plan_titles = [p['title'] for p in plan_std]
print(f'  Plan titles: {plan_titles}')
check('Hospital std — Med×Med chart planned',
      any('Medication' in t and 'Medical Condition' in t or
          'Medication Usage' in t for t in plan_titles),
      str(plan_titles))
check('Hospital std — has combination charts',
      any(p['kind'] == 'cat_cat' for p in plan_std),
      str([p['kind'] for p in plan_std]))

# --- Hospital renamed columns ---
df_ren = make_hospital_renamed()
domain_ren = detect_domain(df_ren)
check('Hospital renamed — is_healthcare', domain_ren['is_healthcare'],
      f"confidence={domain_ren['confidence']}, concepts={domain_ren['detected_concepts']}")
mappings_ren = domain_ren['mappings']
check('Hospital renamed — condition (Diagnosis) mapped', 'condition' in mappings_ren,
      str(mappings_ren.get('condition')))
check('Hospital renamed — medication (Drug Name) mapped', 'medication' in mappings_ren,
      str(mappings_ren.get('medication')))
plan_ren = plan_hospital_charts(df_ren, mappings_ren, max_charts=8)
plan_ren_titles = [p['title'] for p in plan_ren]
check('Hospital renamed — Med×Cond chart planned',
      any(p['kind'] == 'cat_cat' and
          (mappings_ren.get('condition', {}).get('col', '') in p['cols'] or
           mappings_ren.get('medication', {}).get('col', '') in p['cols'])
          for p in plan_ren),
      str(plan_ren_titles))

# --- Education — should NOT be healthcare ---
df_edu = make_education()
domain_edu = detect_domain(df_edu)
check('Education — NOT healthcare', not domain_edu['is_healthcare'],
      f"confidence={domain_edu['confidence']}, concepts={domain_edu['detected_concepts']}")

print(f'\n{passed} passed, {failed} failed')
sys.exit(0 if failed == 0 else 1)
