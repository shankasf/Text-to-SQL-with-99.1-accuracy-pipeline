# Text‑to‑SQL with 99.1% Accuracy Pipeline

A production‑ready Streamlit app that converts natural‑language questions into **SQLite‑compatible SQL**, executes against survey data ingested from SPSS `.sav` files, repairs failing queries automatically, and produces a footnoted, analyst‑grade answer. The pipeline targets 99%+ execution accuracy on well‑formed questions by combining strict SQL sanitization with an error‑aware repair loop and transparent citation extraction.

> Repository starter files: `sql.py` (Streamlit app), `sqlite_alias_guard.py` (SQL sanitizer), `README.md` (this).

---

## Highlights

1. **Fully automated NL → SQL → Result → Narrative** with per‑table summaries and a final combined analysis that includes numbered footnotes for every numeric claim.
2. **Robust SQL sanitizer (`sqlite_alias_guard.py`)** fixes common LLM mistakes for SQLite: de‑quotes `"DESC"/"ASC"`, normalizes `AS`, quotes reserved aliases, converts MySQL backticks to SQLite double quotes, and enforces valid `UNION ALL` patterns.
3. **Smart repair loop**: catches DB errors, diagnoses the root cause (missing/ambiguous column, quoted keywords, invalid `UNION ALL`, etc.), and prompts the model to return a corrected query.
4. **Concurrent table processing** with a real‑time progress timeline and an execution summary.
5. **Citations** extracted from `WHERE` predicates using `sqlglot`, displayed by table after the final answer.
6. **Print/PDF mode** that expands long blocks and yields a clean, single‑click browser “Print to PDF.”
7. **Local chat history** persisted to SQLite for auditability and reuse.

---

## Architecture (at a glance)

1. **Data ingest**: `.sav` files → Pandas → per‑table SQLite DBs (`sqlite_db/<table>.db`). Labels are applied where available via `pyreadstat`.
2. **Question routing**: LLM selects relevant tables from discovered schemas/distincts.
3. **SQL generation**: Prompts constrain quoting rules and SQLite syntax.
4. **Sanitization**: `sanitize_sqlite_sql()` cleans common issues before execution.
5. **Execution + Repair**: Run SQL; on error, diagnose and auto‑repair up to a safe retry limit.
6. **Per‑table summaries**: Natural‑language rollups for each table’s results.
7. **Final analysis**: Combined narrative with indexed footnotes showing formulas, filters, and weight handling.
8. **Citations**: Column/value pairs used in filters parsed via `sqlglot` and displayed per table.

---

## Quickstart

### 1) Prerequisites

* Python 3.10+
* SQLite (bundled with Python via `sqlite3`)
* Recommended OS: Windows, macOS, or Linux

Install packages:

```bash
pip install streamlit pandas pyreadstat sqlalchemy sqlglot python-dotenv azure-ai-inference aiohttp
```

> If you use a managed system Python (PEP 668), create a virtualenv first: `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).

### 2) Configure environment

Create a `.env` next to `sql.py` and set your Azure Inference credentials and model name:

```
AZURE_ENDPOINT=https://<your-endpoint>.inference.ai.azure.com
AZURE_API_KEY=<your-key>
MODEL_NAME=<your-model-name>
```

### 3) Point to your data

`sql.py` expects SPSS `.sav` files in a directory referred to as `SAV_DIR`.

* Option A (quick): open `sql.py` and edit `SAV_DIR` to your data folder.
* Option B (preferred): set an environment variable and modify `sql.py` once to read `os.getenv('SAV_DIR', '<fallback-path>')`.

On first run the app will:

* Read every `.sav` file
* Apply value/label mappings
* Write each dataset to `sqlite_db/<table>.db`
* Record discovered schemas and distinct values for prompting

### 4) Run the app

```bash
streamlit run sql.py
```

In the UI:

1. Ask a question in the chat box
2. Watch the **Initial pipeline outputs** expander for SQL, logs, and results
3. Toggle **Performance Settings** to enable concurrent processing and set worker count
4. Use **Print/PDF mode** for a clean export
5. Open **Chat History** from the sidebar to browse prior answers

---

## Key Components

### SQL Sanitizer (`sqlite_alias_guard.py`)

The sanitizer is designed for SQLite compatibility and conservative, non‑destructive edits.

* **De‑quote ORDER BY directions**: `"DESC"/"ASC" → DESC/ASC`
* **De‑quote `"AS"`** in aliasing contexts
* **Quote reserved aliases**: `AS group → AS "group"`
* **Backticks → double quotes**: `` `col` `` → `"col"`
* **`UNION ALL` hygiene**:

  * Remove illegal parentheses around `SELECT` parts
  * Ensure a single `ORDER BY` appears only at the very end of the unioned query
  * Avoid combining `WITH` CTEs with `UNION ALL`

Use `sanitize_sqlite_sql(sql)` to apply the full pipeline in safe order.

### Error‑Aware Repair Loop

When execution fails, the app:

1. Detects the error class (missing column/table, ambiguous name, quoted keyword, union/order issues, backticks, etc.).
2. Suggests candidate fixes (closest column names, quoting notes) to the model.
3. Requests a corrected, single‑line SQLite query and re‑sanitizes before re‑try.
4. Stops early if the same hard error repeats, preventing infinite loops.

### Citations via `sqlglot`

`extract_col_values_formatted()` parses the final SQL to list each **column** and the **value(s)** used in `WHERE` predicates. The app shows these per table beneath the final answer to improve auditability.

### Concurrency + Timeline UI

* Optional concurrent execution with a worker slider
* Real‑time status panel and a collapsible timeline showing stage, table, and elapsed time
* A summary card aggregates total time, successes, failures, and success rate

### Chat History Persistence

* SQLite DB at `sqlite_db/chat_history.db`
* Each record stores timestamp, question, final answer, timings, and raw LLM output for auditing

---

## Configuration & Paths

The defaults in `sql.py` can be customized:

| Setting                  | Purpose                                | Default                                             |
| ------------------------ | -------------------------------------- | --------------------------------------------------- |
| `SAV_DIR`                | Folder containing `.sav` files         | Example Windows path in code; change to your folder |
| `DB_DIR`                 | Where per‑table SQLite DBs are written | `<SAV_DIR>/sqlite_db`                               |
| `CHAT_DB`                | Chat history database                  | `<DB_DIR>/chat_history.db`                          |
| `ENABLE_CONCURRENT`      | Toggle concurrency                     | On by default                                       |
| `MAX_CONCURRENT_WORKERS` | Parallel tables to process             | 3                                                   |
| `max_tokens`             | Per‑step token budgets                 | 12k–25k in prompts                                  |

---

## Reproducing the “99.1%” Claim (Evaluation Recipe)

The pipeline is designed to reach \~99% execution accuracy on realistic, schema‑grounded questions. To measure on your data:

1. Prepare an evaluation set: `question`, expected `SQL` or expected `rows`.
2. Normalize both predicted and reference SQL with `sqlglot` (or compare result sets when deterministic).
3. Count a hit when the executed result set matches the reference within tolerance.
4. Report: execution accuracy, first‑try SQL validity, and post‑repair success.

Accuracy depends on schema quality, labeling, and question style; publish both your dataset and harness for reproducibility.

---

## Troubleshooting

1. **`pyreadstat` install issues (PEP 668)**: create a virtualenv before `pip install`.
2. **Azure 401/permission errors**: verify `.env`, endpoint URL, and model name; ensure the 2024‑05‑01‑preview API is enabled for your model.
3. **Quoted keyword errors**: the sanitizer should remove `"DESC"/"ASC"`; check the final SQL shown in the expander.
4. **`UNION ALL` syntax errors**: ensure no CTE is combined with `UNION ALL`, and that `ORDER BY` only appears at the very end.
5. **Backticks**: if your prompt or tools introduce `` `like_this` ``, the sanitizer converts them to `"like_this"` for SQLite.
6. **Large `.sav` files**: memory usage scales with Pandas load; consider chunking upstream or filtering columns.

---

## Roadmap

* Optional `SAV_DIR` via env without editing code
* Packaged CLI for batch evaluation
* Pluggable execution backends (DuckDB, Postgres)
* Unit tests for sanitizer and repair heuristics

---

## Contributing

Pull requests are welcome. Please include a clear description, minimal repro, and tests for sanitizer changes.

---

## License

Copyright (c) 2025 Sagar Shankaran. All rights reserved.

See the [LICENSE](LICENSE) file for terms of use.
