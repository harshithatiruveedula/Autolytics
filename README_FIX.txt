AUTOLYTICS fixes

1. Replace your app.py and query.py with the provided updated files.
2. Replace frontend/index.html with the included updated file (important for Enter-to-run and faster CSV transfer).
3. Keep cleaning.py and the other project files in their existing locations.
4. Restart Streamlit after replacing the files.

Changes:
- Press Enter in the query box to run a query; Shift+Enter adds a new line.
- CSV uploads are gzip-compressed in the browser before sending to Streamlit when supported.
- SQLite loading is deferred until a query needs SQL execution, so upload is not blocked by copying the whole dataset into SQLite.
- Common natural-language category-count questions (e.g. count of Doctor, count of Medical Condition, total of Blood Type) are answered directly from pandas and return frequency summaries.
- Numeric total/sum requests still calculate a numeric sum.
