AUTOLYTICS - Domain-aware EDA update

What's included
- domain_analytics.py: shared semantic column recognition for E-commerce, Healthcare, Finance, and Education; domain-specific single-metric charts; deterministic Key Insights, Suggestions, and Data Storytelling.
- app.py: integrates the new analysis into EDA & Charts, renders the three narrative sections below charts, passes findings into the workspace data payload, and exports CSV/TXT/PDF reports.
- reports.py: optional shared report builder support for narrative sections and chart references.
- test_domain_analytics.py: regression tests for all four domains, alternate column names, unknown-schema fallback, and finance credit-limit semantics.
- Existing healthcare domain planner, query engine, cleaning, frontend/sidebar, and existing tests are retained.

Deployment
1. Replace the matching files in your existing AUTOLYTICS repository with the files in this ZIP.
2. Keep your existing secrets and environment variables.
3. Commit and push the changes to GitHub.
4. Streamlit Cloud will reinstall requirements.txt and redeploy. reportlab was added for PDF export.
5. Upload a dataset and click Generate EDA. Key Insights, Suggestions, and Data Storytelling appear beneath the generated charts.

Analysis safeguards
- Domain detection uses normalized names, aliases, data types, and conservative value signatures; uncertain datasets use generic EDA.
- Unsupported charts are skipped individually; available charts continue to render.
- E-commerce amount charts exclude explicitly cancelled rows when an order-status field is recognized. They describe recorded amounts rather than assuming every row is a completed sale.
- Finance credit limits are not treated as income or profit. Fraud-like flags are reported as flags, not proof of fraud.
- Education pass rates are not inferred unless a suitable outcome field exists; correlations are described as associations, not causes.
- Healthcare outputs are descriptive/operational and do not make clinical recommendations.
- PDF exports include aggregate insights, suggestions, storytelling, chart references, and a limited column profile, not raw records.

Tests run in the available environment
- test_domain_analytics.py: 19 passed.
- test_domain.py: 11 passed.
- test_hospital_eda.py: 10 passed.
- Python compilation: passed.
- Live Streamlit/Altair rendering was not run here because those packages are not installed in this execution environment. The deployed environment installs dependencies from requirements.txt.
