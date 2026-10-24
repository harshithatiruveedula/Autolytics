AUTOLYTICS ImportError fix

This patch contains matching app.py and modeling.py files.

1. Extract this ZIP.
2. In the SAME GitHub repository and deployment branch used by Streamlit Cloud, replace BOTH app.py and modeling.py with these files. Put them side-by-side in the repository folder where app.py lives (the traceback shows /mount/src/autolytics/app.py, so usually repository root).
3. Commit both files in the same commit.
4. In Streamlit Cloud, check Settings -> General for the correct repository, branch, and Main file path. Confirm the commit deployed. Then reboot the app.

The included modeling.py defines get_target_candidates. If the same ImportError remains after both files are committed and deployed, the deployed app is using a different branch/root or an old module named modeling.py / modeling package is shadowing this file. Check the full error log for the exact message and module path.
