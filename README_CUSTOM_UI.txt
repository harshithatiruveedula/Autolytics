AUTOLYTICS CUSTOM UI VERSION

This version replaces Streamlit's built-in file uploader, dataframe preview,
query input, and example buttons with a custom HTML/CSS/JavaScript Streamlit
component. It is designed to more closely match the supplied AUTOLYTICS reference.

Files to copy into your existing project:
- app.py
- frontend/index.html (keep the frontend folder)

Keep your existing cleaning.py, query.py, eda.py, modeling.py, reports.py,
and requirements.txt. The provided app.py expects clean_dataset and
format_cleaning_summary from cleaning.py, and TABLE_NAME/run_user_query from query.py.

Run from the project folder:
  python -m pip install -r requirements.txt
  python -m streamlit run app.py

If Streamlit was already running, stop it with Ctrl+C and restart it.

Notes:
- The custom upload supports .csv and .xlsx.
- The custom table displays five rows per page, with previous/next controls.
- Partial rows are preserved by the existing cleaning.py imputation logic.
- Natural-language queries are limited to patterns implemented in query.py;
  arbitrary free-form language needs a configured model or a broader parser.
- Fixes in this build: component events now trigger a rerun only once (preventing the UI jump/rerun loop), and the Choose File button explicitly opens the hidden file input. The app CSS removes Streamlit outer padding so the custom UI can fill the available width. This build has been syntax-checked; test the upload in your local browser.
