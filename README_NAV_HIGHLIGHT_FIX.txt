AUTOLYTICS — Sidebar-to-section rotating blue border fix

What changed:
- Clicking Upload Dataset on the sidebar scrolls to the upload section and animates the same rotating blue border around the upload section's outer edge.
- Clicking Query Dataset scrolls to the query section and animates the same rotating blue border around the query section's outer edge.
- Reapplies the highlight after the Streamlit component re-renders, so the effect does not disappear during navigation.
- Existing project files and query logic are preserved.

Apply:
1. Extract this ZIP over the existing AUTOLYTICS project folder.
2. Restart Streamlit (Ctrl+C, then `streamlit run app.py`).
3. Hard-refresh the browser with Ctrl+F5.

QUERY SCROLL FIX (Oct 21): The Query Dataset navigation now scrolls using both the parent document scrolling element and window, then scrolls the section into view inside the component. This avoids returning early when the parent scroll call does not move the page.
