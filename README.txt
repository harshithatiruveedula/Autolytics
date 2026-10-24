AUTOLYTICS MODEL EVALUATION SAFETY UPDATE

Replace app.py and modeling.py in your existing project with these versions.
The existing UI layout is preserved. This update:
- compares regression metrics with a simple mean-prediction baseline;
- shows a warning when R² is negative and a caution when R² is limited;
- makes prediction disclaimers match regression vs classification;
- does not force a model to look accurate or change the prediction to an arbitrary value.

Important: a negative R² is a data/model result, not a display bug. If it remains negative on your dataset, the app should report low predictive performance honestly. No software change can guarantee good predictive accuracy for every uploaded dataset.
