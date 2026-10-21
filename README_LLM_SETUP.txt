AUTOLYTICS: LLM natural-language query integration

FILES
- query.py: replace your existing query.py with this file.
- requirements.txt: merge the openai and python-dotenv dependencies into your current requirements.txt; do not remove your other project dependencies.
- .env.example: copy to .env and enter your own API key.

SETUP (Windows CMD, from the AUTOLYTICS project directory)
1. Stop Streamlit with Ctrl+C.
2. Replace query.py with the supplied query.py.
3. Install the new dependencies: python -m pip install openai python-dotenv
4. Copy .env.example to a file named .env in the same directory as app.py.
5. Edit .env and replace your_api_key_here with your own OpenAI API key.
6. Ensure app.py loads .env near the top, before importing/using query.py:
       from dotenv import load_dotenv
       load_dotenv()
   Place these lines before `from query import ...`.
7. Restart: python -m streamlit run app.py

HOW IT WORKS
- Existing offline patterns are tried first (no API cost for those).
- If a question is not recognized, the LLM translates it into a single SQLite SELECT query.
- The SQL is validated, then executed against the in-memory `dataset` table.
- Only column names, types, a few example values, and the first 3 rows are sent to the LLM, not the full dataset.
- Do not commit .env or share your API key. API usage may incur charges depending on your provider account.

SECURITY / LIMITS
- The engine accepts read-only SELECT/WITH SQL and rejects common write/DDL keywords.
- LLM output can still be wrong; inspect the displayed SQL and results, especially for important analysis.
- Natural language understanding requires a valid API key, internet access, an enabled model, and available API quota. ChatGPT subscription does not automatically include API credits.
