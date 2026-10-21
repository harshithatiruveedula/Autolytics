AUTOLYTICS navigation and Enter-to-query fix

Replace your existing project files with the files in this folder, especially:
- app.py
- query.py
- frontend/index.html

Fixes:
- Sidebar and in-page navigation buttons explicitly dispatch navigation events.
- Enter in the query box runs the query; Shift+Enter inserts a new line.
- The query button also reads the current text directly from the query box.
- Component events include unique IDs and the JSON data type to avoid stale/repeated event handling.

Run from the project folder:
  streamlit run app.py

If Streamlit is already running, stop it with Ctrl+C and start it again after replacing the files.
