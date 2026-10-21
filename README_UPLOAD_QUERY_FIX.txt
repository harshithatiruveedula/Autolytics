AUTOLYTICS — navigation animation, upload feedback, and query-key fix

Included fixes:
1. Upload Dataset / Query Dataset navigation now gets a rotating blue border effect when active.
2. Navigation updates visually immediately when clicked, instead of waiting for Streamlit to rerender.
3. File upload shows progress text while preparing, compressing CSV files, and sending data. The upload icon pulses while processing.
4. Enter submits a query unless Shift+Enter is used for a newline. The duplicate document-level Enter listener was removed to avoid event conflicts; Run Query and suggestion chips use the same guarded submit function.
5. Repeated query submits are guarded while a query is being processed.
6. CSV upload continues to use gzip compression where supported by the browser; CSV parsing remains on pandas' fast C engine.
7. Removed an unnecessary second full DataFrame copy during missing-value imputation to reduce memory use and upload processing time for large datasets.
8. Previous query fixes and QUERY_GUIDE.md are retained.

How to apply:
- Extract this ZIP over your existing AUTOLYTICS project, keeping the folder structure.
- Ensure frontend/index.html is inside the frontend folder beside app.py.
- Restart Streamlit (stop with Ctrl+C, then run: streamlit run app.py).
- Refresh the browser tab with Ctrl+F5.

Notes:
- Large Excel (.xlsx) files can still take longer because Excel parsing is slower than CSV parsing.
- The UI displays progress states; it cannot show exact percentage progress while pandas is parsing/cleaning on the server.
