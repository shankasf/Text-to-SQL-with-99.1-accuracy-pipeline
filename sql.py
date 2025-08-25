# -*- coding: utf-8 -*-
import os
import re
import json
import time
import threading
import difflib
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from datetime import datetime
from collections import OrderedDict

import pandas as pd
import streamlit as st
import pyreadstat
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sql_metadata import Parser  # retained for other parsing needs
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import html as ihtml

# --- sqlglot for citations ---
import sqlglot
import sqlglot.expressions as exp

# --- sanitizer that dequotes "DESC"/"ASC" and quotes reserved aliases ---
from sqlite_alias_guard import sanitize_sqlite_sql

# =================== App & Config ===================

pd.set_option('future.no_silent_downcasting', True)

load_dotenv()
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')

client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY),
    api_version='2024-05-01-preview'
)

st.set_page_config(page_title='MaristPoll LLM', layout='wide')
# --- global CSS to prevent long lines from being cut ---
st.markdown("""
<style>
/* Wrap long pre/code blocks and chat message content */
div[data-testid="stMarkdownContainer"] pre,
div[data-testid="stCodeBlock"] pre { white-space: pre-wrap !important; word-break: break-word !important; }
/* Monospace for text areas showing raw blocks */
.stTextArea textarea { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important; }
</style>
""", unsafe_allow_html=True)

# Extra print CSS
st.markdown("""
<style>
.pre-wrap { white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }
@media print {
  @page { size: A4; margin: 12mm; }
  header, footer, [data-testid="stSidebar"] { display: none !important; }
  .stButton, [data-testid="stStatusWidget"], [data-testid="stToolbar"] { display: none !important; }
  .block-container, .element-container, .stMarkdown, .stTextArea, textarea, pre, code, [data-testid="stChatMessage"], .stChatMessage {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
  }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
@media screen { .print-only { display: none !important; } }
@media print  { .no-print { display: none !important; } .print-only { display: block !important; } }
</style>
""", unsafe_allow_html=True)

# Directories
st.markdown("""
<style>
.pre-wrap { white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }
@media print {
  @page { size: A4; margin: 12mm; }
  header, footer, [data-testid="stSidebar"] { display: none !important; }
  .stButton, [data-testid="stStatusWidget"], [data-testid="stToolbar"] { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

# Directories
SAV_DIR = r'D:\MaristPoll_research\maristPoll\code\azure\sav_files'
DB_DIR = os.path.join(SAV_DIR, 'sqlite_db')
os.makedirs(DB_DIR, exist_ok=True)
CHAT_DB = os.path.join(DB_DIR, 'chat_history.db')

# =================== Session State ===================

if 'chat_thread' not in st.session_state:
    st.session_state.chat_thread = []  # each: {ts, question, final_answer, timings, tables, raw_final}
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# =================== Thread-Safe Progress Tracking ===================

import threading
from queue import Queue
from collections import deque

# Global thread-safe progress tracking
progress_events = deque()
progress_lock = threading.Lock()
concurrent_start_time = None

def safe_get_progress_events():
    """Safe wrapper for get_progress_events with error handling"""
    try:
        return get_progress_events()
    except Exception as e:
        # If there's any error, return empty list
        return []

def init_concurrent_processing():
    """Initialize concurrent processing with thread-safe progress tracking"""
    global concurrent_start_time
    clear_progress()
    concurrent_start_time = time.time()

# =================== Chat History Persistence ===================

def init_chat_db():
    eng = create_engine(f"sqlite+pysqlite:///{CHAT_DB}")
    with eng.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS chats (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              question TEXT NOT NULL,
              final_answer TEXT NOT NULL,
              timing_json TEXT NOT NULL,
              tables_json TEXT NOT NULL,
              raw_final TEXT NOT NULL
            )
            """
        ))
    eng.dispose()

def save_chat(question: str, final_answer: str, timings: dict, tables: list, raw_final: str):
    init_chat_db()
    eng = create_engine(f"sqlite+pysqlite:///{CHAT_DB}")
    with eng.begin() as conn:
        conn.execute(
            text("INSERT INTO chats (ts, question, final_answer, timing_json, tables_json, raw_final) "
                 "VALUES (:ts,:q,:a,:tj,:tj2,:rf)"),
            {
                'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'q': question,
                'a': final_answer,
                'tj': json.dumps(timings),
                'tj2': json.dumps(tables),
                'rf': raw_final or ''
            }
        )
    eng.dispose()

def load_chat_history(limit: int = 500):
    init_chat_db()
    eng = create_engine(f"sqlite+pysqlite:///{CHAT_DB}")
    with eng.connect() as conn:
        rows = conn.execute(
            text("SELECT id, ts, question, final_answer, timing_json, tables_json, raw_final "
                 "FROM chats ORDER BY id DESC LIMIT :lim"),
            {"lim": limit}
        ).fetchall()
    eng.dispose()
    hist = []
    for r in rows:
        hist.append({
            'id': r[0], 'ts': r[1], 'question': r[2], 'final_answer': r[3],
            'timing': json.loads(r[4]), 'tables': json.loads(r[5]), 'raw_final': r[6]
        })
    return hist

# =================== Top UI ===================

st.title('MaristPoll LLM (Fully Automated Data Processing and Selective Survey Pipeline)')
total_time_placeholder = st.empty()  # shows after each run completes

with st.sidebar:
    st.header('Controls')
    if st.button('Chat History'):
        st.session_state.show_history = True
    PRINT_MODE = st.toggle('Print/PDF mode', value=False, help='Use for clean browser "Print to PDF"; expands long text and replaces scrollable textareas with printable blocks.')
    
    # Concurrent processing configuration
    st.markdown("---")
    st.subheader("Performance Settings")
    ENABLE_CONCURRENT = st.toggle('Enable concurrent processing', value=True, help='Process multiple tables simultaneously to speed up analysis. Recommended for 2+ tables.')
    MAX_CONCURRENT_WORKERS = st.slider('Max concurrent workers', min_value=1, max_value=5, value=3, help='Number of tables to process simultaneously. Higher values may hit API rate limits.')

# Single dropdown for all pipeline internals of the **current** run only
init_box = st.expander(
    'Initial pipeline outputs (LLM selections, SQL, execution logs, citations, raw results, per-table summaries, raw final LLM)',
    expanded=False
)

# =================== Helpers ===================

def show_area(label: str, content: str | None, *, height: int = 260, key: str | None = None, disabled: bool = True):
    """Render a scrollable text area normally; in Print/PDF mode render a full-height <pre> block that prints completely."""
    st.caption(label)
    if 'PRINT_MODE' in globals() and PRINT_MODE:
        st.markdown(f"<pre class='pre-wrap'>{ihtml.escape(str(content) if content is not None else '')}</pre>", unsafe_allow_html=True)
    else:
        st.text_area(label, value=str(content) if content is not None else "", height=height, key=key, disabled=disabled)


def extract_after_think(text: str) -> str:
    m = re.search(r'<think>.*?</think>(.*)', text, flags=re.DOTALL)
    return m.group(1).strip() if m else text

def extract_sql(text_output: str):
    cleaned = re.sub(r"<think>.*?</think>", "", text_output, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"\*+SQL_QUERY\*+\s*:", "SQL_QUERY:", cleaned, flags=re.IGNORECASE)
    
    # More robust SQL extraction - look for SQL_QUERY: followed by SQL ending with semicolon
    raw_matches = re.findall(r"SQL_QUERY\s*:\s*(.+?;)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Fallback: if no semicolon found, look for SQL_QUERY: followed by anything that looks like SQL
    if not raw_matches:
        raw_matches = re.findall(r"SQL_QUERY\s*:\s*(.+?)(?=\n\n|\n[A-Z]|$)", cleaned, flags=re.IGNORECASE | re.DOTALL)
        # Add semicolon if missing
        raw_matches = [m.strip() + ";" if not m.strip().endswith(";") else m.strip() for m in raw_matches]
    
    # Additional fallback: look for SQL_QUERY: followed by anything that starts with SELECT, WITH, etc.
    if not raw_matches:
        raw_matches = re.findall(r"SQL_QUERY\s*:\s*((?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)[^;]*;?)", cleaned, flags=re.IGNORECASE | re.DOTALL)
        # Add semicolon if missing
        raw_matches = [m.strip() + ";" if not m.strip().endswith(";") else m.strip() for m in raw_matches]
    
    valid = [
        m.strip().replace("\n", " ").rstrip(";") + ";"
        for m in raw_matches
        if re.match(r"(?i)^\s*\(?\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b", m.strip())
    ]
    
    return (valid[-1], cleaned, raw_matches) if valid else (None, cleaned, raw_matches)

# =================== Concurrent Processing Functions ===================

def process_single_table_concurrent(
    tbl: str,
    question: str,
    schema: str,
    distinct_info: str,
    db_path: str,
    table_columns: dict[str, list[str]]
):
    """Process a single table concurrently - returns (table_name, result_dict) - NO STREAMLIT CALLS"""
    try:
        # Add progress event for start
        add_progress_event('start', tbl, f"Started processing table {tbl}")
        
        # Step 2: initial SQL generation (20k) - NO STREAMLIT UI UPDATES
        sys_sql = SystemMessage(content="You are an expert SQLite assistant. Respond with DESCRIPTION and SQL_QUERY as specified in the user instructions.")
        sql_prompt = (
            f"User question: {question}\n"
            f"Schema: {schema}\n"
            f"Distinct values:\n{distinct_info}\n"
            """Instructions: 
                1. Strictly use only those column names which are present in the table schema when writing SQL.
                2. Never assume or use a column name which is not present in the schema.
                3. Even if column name is very long use it as it is.
                4. You must perform all calculations by considering each item according to its respective weight. After completing the weighted calculation, always round the final result to the nearest whole number. No fractional or decimal results should be presented in the output.\n
                5. Always cross-check if you are using quotation marks for the right strings in an SQL query. Do not use double quotes for SQL keywords, functions, or direction specifiers. Below are common mistakes to avoid (all drawn from past errors),
                    Sample examples:
                    Wrong:
                    ORDER BY ROUND(SUM(wtfactor)) "DESC"
                    Right:
                    ORDER BY ROUND(SUM(wtfactor)) DESC
                    (DESC is a keyword, not a string — no quotes needed)

                    Wrong:
                    ORDER BY ROUND(SUM(wtfactor)) "ASC"
                    Right:
                    ORDER BY ROUND(SUM(wtfactor)) ASC
                    (ASC is a keyword, not a string — no quotes needed)

                    CRITICAL: Never put quotes around DESC or ASC in ORDER BY clauses. 
                    They are SQL keywords, not string literals.
                    Examples: ORDER BY column DESC, ORDER BY column ASC
                    Never: ORDER BY column "DESC" or ORDER BY column "ASC"

                    Wrong:
                    AS "DESC" when aliasing a column
                    Right:
                    AS some_alias or AS "some_alias"
                    (Never use keywords like DESC/ASC as aliases — if unavoidable, quote a safe alias name instead)

                    Wrong:
                    AS "AS" or AS "GROUP"
                    Right:
                    AS "group"
                    (If the alias is a reserved keyword, use lowercase and keep the quotes — "group" is fine, "GROUP" is reserved)

                    Wrong:
                    GROUP BY "gender_for_weighting___gendwt" (when it's a real column name and doesn't need quoting) combined with other unquoted names incorrectly, or mixing quotes inconsistently
                    Right:
                    GROUP BY gender_for_weighting___gendwt
                    (Only quote identifiers if they contain spaces, special characters, or are reserved keywords — be consistent).\n
                6. CRITICAL SQLite Syntax Rules (MUST FOLLOW):
                    a. NEVER use CTEs (WITH clauses) combined with UNION ALL in the same query
                    b. If you need multiple SELECT statements, use ONLY UNION ALL without CTEs
                    c. If you need a CTE, use it for a single SELECT statement only
                    d. Valid patterns:
                       - Single SELECT with CTE: WITH cte AS (SELECT ...) SELECT * FROM cte
                       - Multiple SELECTs with UNION ALL: SELECT ... UNION ALL SELECT ... UNION ALL SELECT ... (NO parentheses around subqueries)
                       - NOT: WITH cte AS (...) UNION ALL SELECT ... (this will fail)
                       - NOT: (SELECT ...) UNION ALL (SELECT ...) (parentheses cause syntax error)
                    e. For complex queries requiring multiple demographic breakdowns, use UNION ALL pattern:
                       SELECT ... LIMIT 1
                       UNION ALL
                       SELECT ... LIMIT 1
                       UNION ALL
                       SELECT ... LIMIT 1
                       ORDER BY ... (single ORDER BY at the very end)
                       Note: ORDER BY clauses cannot be used within individual SELECT statements in UNION ALL
                       CRITICAL: In UNION ALL queries, ORDER BY must come at the very end of the entire query, not within subqueries
                    f. CRITICAL: Use DOUBLE QUOTES (\") for column names that need quoting, NEVER backticks (`)
                       - Wrong: `column_name` (MySQL syntax)
                       - Right: \"column_name\" (SQLite syntax)
                       - Only quote column names that start with digits or contain special characters
                       - Example: \"2024_support__with_leaners_\" (starts with digit)
                       - Example: column_name (no quotes needed for normal names)
                7. Respond exactly as:
                DESCRIPTION: <one-sentence>
                SQL_QUERY: <single-line SQL ending with semicolon>
                """
        )
        
        # Direct LLM call without Streamlit UI updates
        add_progress_event('sql_generation', tbl, f"Generating SQL for {tbl}")
        start_time = time.time()
        resp = client.complete(messages=[sys_sql, UserMessage(content=sql_prompt)], model=MODEL_NAME, max_tokens=25000)
        raw_sql_text = resp.choices[0].message.content
        t_sql = time.time() - start_time
        add_progress_event('sql_generation', tbl, f"SQL generation completed for {tbl}", t_sql)

        # Execute with SMART repair loop (repair uses 20k inside) - NO STREAMLIT UI
        sql, cleaned_text, raw_matches = extract_sql(raw_sql_text)
        attempts_raw = [raw_sql_text]
        if not sql:
            add_progress_event('error', tbl, f"No valid SQL extracted for {tbl}")
            return {
                'table': tbl,
                'success': False,
                'error': 'No valid SQL extracted',
                'rows': None,
                'final_sql': None,
                'attempts': 0,
                'summary': f"Failed to extract SQL for table {tbl}",
                'raw_summary': "",
                'timings': {'sql_generation': t_sql, 'summary_generation': 0.0},
                'citations': "No SQL to analyze."
            }

        attempts = 0
        same_error_counter = 0
        last_error_signature = None
        current_sql = sanitize_sqlite_sql(sql)
        current_sql = fix_sqlite_quotes(current_sql)
        
        while attempts <= 4:  # max_retries
            attempts += 1
            add_progress_event('sql_execution', tbl, f"Executing SQL attempt {attempts} for {tbl}")
            
            if not isinstance(current_sql, (str, bytes)) or not str(current_sql).strip():
                err = "No valid SQL text to execute."
                rows = None
            else:
                if attempts > 1:
                    current_sql = sanitize_sqlite_sql(current_sql)
                    current_sql = fix_sqlite_quotes(current_sql)
                rows, err = try_execute_sql(db_path, current_sql)

            if err is None:
                add_progress_event('sql_execution', tbl, f"SQL execution successful for {tbl} after {attempts} attempts")
                break

            add_progress_event('sql_repair', tbl, f"SQL repair attempt {attempts} for {tbl}: {err.splitlines()[0][:50]}...")

            sig = (type(err), str(err).splitlines()[0].strip())
            if sig == last_error_signature:
                same_error_counter += 1
            else:
                same_error_counter = 0
            last_error_signature = sig
            if same_error_counter >= 2:  # max_same_error
                add_progress_event('error', tbl, f"Stopping repairs for {tbl} - same error repeated")
                break

            if attempts > 4:  # max_retries
                add_progress_event('error', tbl, f"Max repair attempts reached for {tbl}")
                break

            # Repair SQL without Streamlit UI
            suggestions, hard_stop = diagnose_error(err, tbl, table_columns)
            if hard_stop:
                add_progress_event('error', tbl, f"Hard stop condition for {tbl}: {hard_stop}")
                break

            repair_messages = build_sql_repair_messages(
                question=question,
                schema=schema,
                distinct_info=distinct_info,
                previous_sql=current_sql if isinstance(current_sql, str) else str(current_sql),
                error_text=err,
                suggestions=suggestions
            )
            
            # Direct LLM call for repair
            start_repair = time.time()
            resp = client.complete(messages=repair_messages, model=MODEL_NAME, max_tokens=12000)
            raw_repair_text = resp.choices[0].message.content
            attempts_raw.append(raw_repair_text)
            repaired_sql, _, _ = extract_sql(raw_repair_text)
            current_sql = sanitize_sqlite_sql(repaired_sql) if repaired_sql else current_sql
        
        # Step 3: per-table natural language summary (20k) - NO STREAMLIT UI
        if rows is not None:
            add_progress_event('summary_generation', tbl, f"Generating summary for {tbl}")
            
            ans_sys = SystemMessage(content='You are a helpful assistant.')
            ans_user = UserMessage(content=(
                f"Original question: {question}\n"
                f"Final SQL used (after repairs):\n{current_sql}\n"
                f"Results from {tbl}: {rows}\n"
                "Provide a concise summary for this table."
            ))
            
            start_ans = time.time()
            resp = client.complete(messages=[ans_sys, ans_user], model=MODEL_NAME, max_tokens=12000)
            raw_ans = resp.choices[0].message.content
            final_ans = extract_after_think(raw_ans)
            t_ans = time.time() - start_ans
            
            add_progress_event('summary_generation', tbl, f"Summary generation completed for {tbl}", t_ans)
        else:
            final_ans = f"Failed to process table {tbl} after {attempts} attempts."
            t_ans = 0.0
            raw_ans = ""
            add_progress_event('error', tbl, f"Failed to process table {tbl} - no results")
        
        # Add completion event
        if rows is not None:
            add_progress_event('complete', tbl, f"Successfully completed processing {tbl}")
        else:
            add_progress_event('error', tbl, f"Failed to complete processing {tbl}")
        
        # Return results for this table
        return {
            'table': tbl,
            'success': rows is not None,
            'rows': rows,
            'final_sql': current_sql if isinstance(current_sql, str) else str(current_sql),
            'attempts': attempts,
            'summary': final_ans,
            'raw_summary': raw_ans,
            'timings': {
                'sql_generation': t_sql,
                'summary_generation': t_ans
            },
            'citations': extract_col_values_formatted(current_sql if isinstance(current_sql, str) else str(current_sql), dialect="sqlite")
        }
        
    except Exception as e:
        # Add error event
        add_progress_event('error', tbl, f"Exception in {tbl}: {str(e)}")
        
        # Return error result for this table
        return {
            'table': tbl,
            'success': False,
            'error': str(e),
            'rows': None,
            'final_sql': None,
            'attempts': 0,
            'summary': f"Error processing table {tbl}: {str(e)}",
            'raw_summary': "",
            'timings': {'sql_generation': 0.0, 'summary_generation': 0.0},
            'citations': "Error occurred during processing."
        }

def process_tables_concurrently(
    tables_to_use: list[str],
    question: str,
    table_schemas: dict,
    distinct_lines: list,
    table_columns: dict,
    max_workers: int = 3  # Limit concurrent requests to avoid rate limits
):
    """Process multiple tables concurrently using ThreadPoolExecutor - NO STREAMLIT UI CALLS"""
    
    # Prepare arguments for each table
    table_args = []
    for tbl in tables_to_use:
        schema = table_schemas[tbl]
        distinct_info = "\n".join(d for d in distinct_lines if d.startswith(f"{tbl}."))
        db_path = os.path.join(DB_DIR, f"{tbl}.db")
        
        table_args.append((
            tbl, question, schema, distinct_info, db_path, table_columns
        ))
    
    # Process tables concurrently
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_table = {
            executor.submit(process_single_table_concurrent, *args): args[0] 
            for args in table_args
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_table):
            table_name = future_to_table[future]
            try:
                result = future.result()
                results.append(result)
                        
            except Exception as e:
                results.append({
                    'table': table_name,
                    'success': False,
                    'error': str(e),
                    'rows': None,
                    'final_sql': None,
                    'attempts': 0,
                    'summary': f"Exception processing table {table_name}: {str(e)}",
                    'raw_summary': "",
                    'timings': {'sql_generation': 0.0, 'summary_generation': 0.0},
                    'citations': "Exception occurred during processing."
                })
    
    # Sort results by original table order
    table_order = {tbl: idx for idx, tbl in enumerate(tables_to_use)}
    results.sort(key=lambda x: table_order.get(x['table'], 999))
    
    return results

# =================== Progress Tracking Functions ===================

def add_progress_event(event_type: str, table_name: str = None, message: str = None, duration: float = None):
    """Add a progress event to the timeline - THREAD SAFE VERSION"""
    global progress_events, concurrent_start_time
    
    with progress_lock:
        event = {
            'timestamp': time.time(),
            'elapsed': time.time() - concurrent_start_time if concurrent_start_time else 0,
            'type': event_type,
            'table': table_name,
            'message': message,
            'duration': duration
        }
        progress_events.append(event)

def clear_progress():
    """Clear progress tracking"""
    global progress_events, concurrent_start_time
    with progress_lock:
        progress_events.clear()
        concurrent_start_time = None

def get_progress_events():
    """Get all progress events - thread safe"""
    global progress_events
    with progress_lock:
        return list(progress_events)

def format_duration(seconds: float) -> str:
    """Format duration in a readable way"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"

def render_progress_timeline():
    """Render the progress timeline using collapsible sections"""
    progress_data = safe_get_progress_events()
    if not progress_data:
        return
    
    # Create a collapsible section using HTML/CSS
    timeline_html = """
    <style>
    .timeline-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin: 10px 0;
        overflow: hidden;
    }
    .timeline-header {
        background-color: #e9ecef;
        padding: 12px 15px;
        cursor: pointer;
        font-weight: bold;
        color: #495057;
        border-bottom: 1px solid #dee2e6;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .timeline-header:hover {
        background-color: #dee2e6;
    }
    .timeline-content {
        padding: 15px;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease-out;
    }
    .timeline-content.expanded {
        max-height: 1000px;
        transition: max-height 0.3s ease-in;
    }
    .timeline-toggle {
        font-size: 18px;
        transition: transform 0.3s;
    }
    .timeline-toggle.expanded {
        transform: rotate(180deg);
    }
    .event-item {
        margin: 8px 0;
        padding: 8px;
        border-left: 3px solid #007bff;
        background-color: white;
        border-radius: 4px;
    }
    .event-success { border-left-color: #28a745; }
    .event-error { border-left-color: #dc3545; }
    .event-warning { border-left-color: #ffc107; }
    .event-info { border-left-color: #17a2b8; }
    </style>
    
    <div class="timeline-container">
        <div class="timeline-header" onclick="toggleTimeline()">
            <span>📊 Concurrent Processing Timeline</span>
            <span class="timeline-toggle" id="timelineToggle">▼</span>
        </div>
        <div class="timeline-content" id="timelineContent">
    """
    
    # Group events by table
    table_events = {}
    general_events = []
    
    for event in progress_data:
        if event['table']:
            if event['table'] not in table_events:
                table_events[event['table']] = []
            table_events[event['table']].append(event)
        else:
            general_events.append(event)
    
    # Show general events
    if general_events:
        timeline_html += "<h4 style='margin: 15px 0 10px 0; color: #495057;'>🚀 General Progress</h4>"
        for event in general_events:
            elapsed = format_duration(event['elapsed'])
            icon = {
                'start': '▶️',
                'complete': '✅',
                'error': '❌',
                'info': 'ℹ️'
            }.get(event['type'], '📝')
            
            event_class = {
                'start': 'event-info',
                'complete': 'event-success',
                'error': 'event-error',
                'info': 'event-info'
            }.get(event['type'], 'event-info')
            
            timeline_html += f'<div class="event-item {event_class}"><strong>{elapsed}</strong> {icon} {event["message"]}</div>'
    
    # Show table-specific events
    for table_name, events in table_events.items():
        timeline_html += f'<h4 style="margin: 20px 0 10px 0; color: #495057;">📋 Table: {table_name}</h4>'
        
        for event in events:
            elapsed = format_duration(event['elapsed'])
            icon = {
                'start': '▶️',
                'sql_generation': '🔧',
                'sql_execution': '⚡',
                'sql_repair': '🔨',
                'summary_generation': '📝',
                'complete': '✅',
                'error': '❌'
            }.get(event['type'], '📝')
            
            event_class = {
                'start': 'event-info',
                'sql_generation': 'event-warning',
                'sql_execution': 'event-info',
                'sql_repair': 'event-warning',
                'summary_generation': 'event-info',
                'complete': 'event-success',
                'error': 'event-error'
            }.get(event['type'], 'event-info')
            
            duration_text = f" ({format_duration(event['duration'])})" if event['duration'] else ""
            timeline_html += f'<div class="event-item {event_class}"><strong>{elapsed}</strong> {icon} {event["message"]}{duration_text}</div>'
    
    # Summary statistics
    if progress_data:
        total_time = progress_data[-1]['elapsed']
        successful_tables = len([e for e in progress_data if e['type'] == 'complete' and e['table']])
        failed_tables = len([e for e in progress_data if e['type'] == 'error' and e['table']])
        
        # Calculate success rate safely
        total_processed = successful_tables + failed_tables
        success_rate = (successful_tables / total_processed * 100) if total_processed > 0 else 0.0
        
        timeline_html += f"""
        <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px; border: 1px solid #dee2e6;">
            <h4 style="margin: 0 0 15px 0; color: #495057;">📈 Summary Statistics</h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 24px; font-weight: bold; color: #007bff;">{format_duration(total_time)}</div>
                    <div style="font-size: 12px; color: #6c757d;">Total Time</div>
                </div>
                <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 24px; font-weight: bold; color: #28a745;">{successful_tables}</div>
                    <div style="font-size: 12px; color: #6c757d;">Successful Tables</div>
                </div>
                <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 24px; font-weight: bold; color: #dc3545;">{failed_tables}</div>
                    <div style="font-size: 12px; color: #6c757d;">Failed Tables</div>
                </div>
                <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 24px; font-weight: bold; color: #17a2b8;">{success_rate:.1f}%</div>
                    <div style="font-size: 12px; color: #6c757d;">Success Rate</div>
                </div>
            </div>
        </div>
        """
    
    timeline_html += """
        </div>
    </div>
    
    <script>
    function toggleTimeline() {
        const content = document.getElementById('timelineContent');
        const toggle = document.getElementById('timelineToggle');
        content.classList.toggle('expanded');
        toggle.classList.toggle('expanded');
    }
    </script>
    """
    
    st.markdown(timeline_html, unsafe_allow_html=True)

def render_live_progress():
    """Render live progress updates"""
    progress_data = get_progress_events()
    if not progress_data:
        return
    
    # Get the latest events for each table
    latest_events = {}
    for event in progress_data:
        if event['table']:
            latest_events[event['table']] = event
    
    # Show current status
    if latest_events:
        st.markdown("### 🎯 Current Status")
        
        for table_name, event in latest_events.items():
            elapsed = format_duration(event['elapsed'])
            status_icon = {
                'start': '🔄',
                'sql_generation': '🔧',
                'sql_execution': '⚡',
                'sql_repair': '🔨',
                'summary_generation': '📝',
                'complete': '✅',
                'error': '❌'
            }.get(event['type'], '📝')
            
            status_color = {
                'start': 'blue',
                'sql_generation': 'orange',
                'sql_execution': 'purple',
                'sql_repair': 'red',
                'summary_generation': 'green',
                'complete': 'green',
                'error': 'red'
            }.get(event['type'], 'gray')
            
            st.markdown(f"**{table_name}**: {status_icon} {event['message']} ({elapsed})", help=f"Status: {event['type']}")

def update_real_time_progress(placeholder):
    """Update the real-time progress display"""
    progress_data = safe_get_progress_events()
    if not progress_data:
        return
    
    # Get the latest events for each table
    latest_events = {}
    for event in progress_data:
        if event['table']:
            latest_events[event['table']] = event
    
    # Create progress display
    if latest_events:
        progress_html = "<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;'>"
        progress_html += "<h4 style='margin: 0 0 10px 0; color: #1f77b4;'>🔄 Real-time Progress</h4>"
        
        for table_name, event in latest_events.items():
            elapsed = format_duration(event['elapsed'])
            status_icon = {
                'start': '🔄',
                'sql_generation': '🔧',
                'sql_execution': '⚡',
                'sql_repair': '🔨',
                'summary_generation': '📝',
                'complete': '✅',
                'error': '❌'
            }.get(event['type'], '📝')
            
            status_color = {
                'start': '#1f77b4',
                'sql_generation': '#ff7f0e',
                'sql_execution': '#9467bd',
                'sql_repair': '#d62728',
                'summary_generation': '#2ca02c',
                'complete': '#2ca02c',
                'error': '#d62728'
            }.get(event['type'], '#7f7f7f')
            
            progress_html += f"<div style='margin: 5px 0; padding: 5px; border-left: 3px solid {status_color}; padding-left: 10px;'>"
            progress_html += f"<strong>{table_name}</strong>: {status_icon} {event['message']} <span style='color: #666;'>({elapsed})</span>"
            progress_html += "</div>"
        
        progress_html += "</div>"
        
        placeholder.markdown(progress_html, unsafe_allow_html=True)

# sqlglot-based WHERE predicate extractor for citations (no exp.NotIn to avoid version issues)

def _lit_to_str(node: exp.Expression) -> str:
    if isinstance(node, exp.Literal):
        return str(node.this)
    if isinstance(node, exp.Null):
        return "NULL"
    if isinstance(node, exp.Boolean):
        return str(node.this).upper()
    return node.sql()

def _add(colmap: OrderedDict, colname: str, value: str):
    if colname not in colmap:
        colmap[colname] = []
    if value not in colmap[colname]:
        colmap[colname].append(value)

def extract_col_values_formatted(xyz: str, dialect: str | None = None) -> str:
    try:
        tree = sqlglot.parse_one(xyz, read=dialect) if dialect else sqlglot.parse_one(xyz)
    except Exception:
        return "No column-value predicates found."
    col_values = OrderedDict()
    binary_ops = (exp.EQ, exp.NEQ, exp.Is, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.ILike)
    for node in tree.walk():
        # IN (covers NOT IN in some sqlglot versions via parent Not)
        if isinstance(node, exp.In):
            col_side = node.this
            vals = node.expressions
            if isinstance(col_side, exp.Column) and vals:
                col = col_side.name
                for v in vals:
                    _add(col_values, col, _lit_to_str(v))
            continue
        if isinstance(node, binary_ops):
            left, right = node.left, node.right
            if isinstance(left, exp.Column) and isinstance(right, (exp.Literal, exp.Null, exp.Boolean)):
                _add(col_values, left.name, _lit_to_str(right))
            elif isinstance(right, exp.Column) and isinstance(left, (exp.Literal, exp.Null, exp.Boolean)):
                _add(col_values, right.name, _lit_to_str(left))
    lines = []
    for i, (col, vals) in enumerate(col_values.items(), start=1):
        lines.append(f"{i}. column name: {col}")
        lines.append(f"    column value used: {', '.join(vals) if vals else '(none)'}")
    return "\n".join(lines) if lines else "No column-value predicates found."

def _call_llm(messages, result_holder: dict, max_tokens: int):
    resp = client.complete(messages=messages, model=MODEL_NAME, max_tokens=max_tokens)
    result_holder['raw'] = resp.choices[0].message.content

def llm_complete_with_status(
    step_label: str,
    messages,
    *,
    thinking_container,
    show_raw=True,
    tick_interval=0.1,
    max_tokens: int = 12000  # <-- 20k everywhere now
):
    """Threaded LLM call with live timer and per-step status rendered at end of chat feed."""
    result_holder = {}
    thread = threading.Thread(target=_call_llm, args=(messages, result_holder, max_tokens), daemon=True)
    start = time.time()
    with thinking_container:
        with st.status(f"{step_label} – MaristPoll LLM is thinking, Please wait.", expanded=True) as status:
            timer_ph = st.empty()
            thread.start()
            while thread.is_alive():
                elapsed = time.time() - start
                timer_ph.write(f"Elapsed: {elapsed:.1f} seconds")
                time.sleep(tick_interval)
            thread.join()
            total = time.time() - start
            timer_ph.write(f"Total time: {total:.2f} seconds")
            status.update(label=f"{step_label}: Final thought for this time ({total:.2f}s)", state="complete")
    raw_text = result_holder.get('raw', '')
    cleaned_text = extract_after_think(raw_text)
    if show_raw:
        with init_box:
            st.subheader(f"Raw LLM Response — {step_label}")
            show_area("Raw LLM text", raw_text, height=300, key=f"raw_{step_label}")
            st.caption(f"Step time: {total:.2f}s")
    return raw_text, cleaned_text, total

# =================== Robust SQL execute + SMART repair loop ===================

ERROR_PATTERNS = {
    "no_column": re.compile(r'no such column: "?([A-Za-z_][A-Za-z0-9_]*)"?', re.IGNORECASE),
    "no_table": re.compile(r'no such table: "?([A-Za-z_][A-Za-z0-9_]*)"?', re.IGNORECASE),
    "ambiguous": re.compile(r'ambiguous column name: "?([A-Za-z_][A-Za-z0-9_]*)"?', re.IGNORECASE),
    "near_syntax": re.compile(r'near\s+"?([A-Za-z_][A-Za-z0-9_]*)"?\s*:\s*syntax error', re.IGNORECASE),
    "quoted_desc": re.compile(r'near\s+"?"?DESC"?\s*"?:\s*syntax error', re.IGNORECASE),
    "quoted_asc": re.compile(r'near\s+"?"?ASC"?\s*"?:\s*syntax error', re.IGNORECASE),
    "quoted_keyword": re.compile(r'near\s+"([^"]+)"\s*:\s*syntax error', re.IGNORECASE),
    "cte_union_error": re.compile(r'near\s+"UNION"\s*:\s*syntax error', re.IGNORECASE),
    "backtick_error": re.compile(r'backtick|`', re.IGNORECASE),
    "union_order_by_error": re.compile(r'ORDER BY clause should come after UNION ALL not before', re.IGNORECASE),
    "parentheses_syntax_error": re.compile(r'near\s*"\(".*syntax error', re.IGNORECASE),
}

def fix_sqlite_quotes(sql: str) -> str:
    """Convert MySQL backticks to SQLite double quotes for column names"""
    if not isinstance(sql, str):
        return sql
    
    # Replace backticks around column names with double quotes
    # Pattern: `column_name` -> "column_name"
    sql = re.sub(r'`([^`]+)`', r'"\1"', sql)
    
    return sql

def try_execute_sql(db_path: str, sql: str):
    try:
        # Fix any backticks before execution
        sql = fix_sqlite_quotes(sql)
        
        engine = create_engine(f"sqlite+pysqlite:///{db_path}")
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
        engine.dispose()
        return rows, None
    except SQLAlchemyError as e:
        return None, str(e)

def get_close_columns(missing: str, column_list: list[str], n=5, cutoff=0.6):
    if not missing or not column_list:
        return []
    return difflib.get_close_matches(missing, column_list, n=n, cutoff=cutoff)

def build_sql_repair_messages(question: str, schema: str, distinct_info: str, previous_sql: str,
                              error_text: str, suggestions: dict | None = None):
    hint_block = ""
    if suggestions:
        if suggestions.get("column"):
            hint_block += f"\nPossible column corrections for '{suggestions['column']['name']}': {', '.join(suggestions['column']['candidates'])}"
        if suggestions.get("table"):
            hint_block += f"\nPossible table corrections for '{suggestions['table']['name']}': {', '.join(suggestions['table']['candidates'])}"
        if suggestions.get("note"):
            hint_block += f"\nNote: {suggestions['note']}"
    sys_sql = SystemMessage(content="You are an expert SQLite assistant. Respond with DESCRIPTION and SQL_QUERY as specified in the user instructions.")
    user = UserMessage(content=(
        f"Original user question: {question}\n"
        f"Table schema: {schema}\n"
        f"Distinct values (per column):\n{distinct_info}\n\n"
        f"The previous SQL failed with this database error:\n{error_text}\n"
        f"{hint_block}\n\n"
        f"Previous SQL to fix:\n{previous_sql}\n\n"
        "Instructions:\n"
        "1) Return a corrected SQLite-compatible SQL that fixes the error.\n"
        "2) Use ONLY columns present in the schema and keep names EXACT.\n"
        """3)Quotation Rules for SQL (Do Not Break):
            a. Never put double quotes around SQL keywords such as DESC, ASC, SELECT, FROM, WHERE, GROUP BY, ORDER BY, LIMIT, or functions like SUM(), ROUND().
                Wrong: ORDER BY ROUND(SUM(wtfactor)) "DESC"
                Right: ORDER BY ROUND(SUM(wtfactor)) DESC
                
                CRITICAL: Never put quotes around DESC or ASC in ORDER BY clauses. 
                They are SQL keywords, not string literals.
                Examples: ORDER BY column DESC, ORDER BY column ASC
                Never: ORDER BY column "DESC" or ORDER BY column "ASC"
            b. Only use double quotes for:
                Identifiers (column or table names) that contain spaces or special characters.
                Reserved keywords when used as an alias (prefer lowercase alias inside quotes).
                Wrong: AS "DESC"
                Right: AS "group"
                Do not quote numeric values or booleans.
                Wrong: WHERE age = "30"
                Right: WHERE age = 30
            c. Always quote string literals with single quotes ' ' not double quotes.
                Wrong: WHERE name = "John"
                Right: WHERE name = 'John'
            d. Be consistent — if one column in a list is quoted, all others should follow the same quoting rule unless unnecessary.\n"""
        "4) CRITICAL SQLite Syntax Rules:\n"
        "   a. NEVER use CTEs (WITH clauses) combined with UNION ALL\n"
        "   b. For multiple SELECT statements, use ONLY UNION ALL without CTEs\n"
        "   c. Valid pattern: SELECT ... UNION ALL SELECT ... UNION ALL SELECT ... (NO parentheses around subqueries)\n"
        "   d. Invalid pattern: (SELECT ...) UNION ALL (SELECT ...) (parentheses cause syntax error)\n"
        "   e. Invalid pattern: WITH cte AS (...) UNION ALL SELECT ... (this will fail)\n"
        "   f. CRITICAL: In UNION ALL queries, ORDER BY clauses must come at the VERY END of the entire query\n"
        "      - Wrong: SELECT ... ORDER BY col DESC UNION ALL SELECT ... ORDER BY col DESC\n"
        "      - Right: SELECT ... UNION ALL SELECT ... ORDER BY col DESC\n"
        "   g. Use DOUBLE QUOTES (\") for column names that need quoting, NEVER backticks (`)\n"
        "      - Wrong: `column_name` (MySQL syntax)\n"
        "      - Right: \"column_name\" (SQLite syntax)\n"
        "      - Only quote column names that start with digits or contain special characters\n"
        "5) Respond exactly as:\n"
        "DESCRIPTION: <one-sentence>\n"
        "SQL_QUERY: <single-line SQL ending with semicolon>"
    ))
    return [sys_sql, user]

def diagnose_error(err_text: str, table: str, table_columns: dict[str, list[str]]):
    """Return (suggestions dict, hard_stop_reason or None)."""
    if not err_text:
        return None, None

    m = ERROR_PATTERNS["no_column"].search(err_text)
    if m:
        missing = m.group(1)
        candidates = get_close_columns(missing, table_columns.get(table, []))
        sugg = {"column": {"name": missing, "candidates": candidates}}
        return sugg, None

    m = ERROR_PATTERNS["ambiguous"].search(err_text)
    if m:
        amb = m.group(1)
        sugg = {"column": {"name": amb, "candidates": table_columns.get(table, [])},
                "note": "Disambiguate by using the exact column name from this table schema."}
        return sugg, None

    m = ERROR_PATTERNS["no_table"].search(err_text)
    if m:
        bad_table = m.group(1)
        sugg = {"table": {"name": bad_table, "candidates": [table]}}
        return sugg, None

    m = ERROR_PATTERNS["near_syntax"].search(err_text)
    if m:
        token = m.group(1)
        sugg = {"note": f'Quote reserved identifiers used as aliases, e.g., AS "{token}"'}
        return sugg, None

    m = ERROR_PATTERNS["quoted_desc"].search(err_text)
    if m:
        sugg = {"note": "Remove quotes around DESC/ASC keywords in ORDER BY clauses. Use DESC or ASC without quotes."}
        return sugg, None

    m = ERROR_PATTERNS["quoted_asc"].search(err_text)
    if m:
        sugg = {"note": "Remove quotes around DESC/ASC keywords in ORDER BY clauses. Use DESC or ASC without quotes."}
        return sugg, None

    m = ERROR_PATTERNS["quoted_keyword"].search(err_text)
    if m:
        keyword = m.group(1)
        if keyword.upper() in ['DESC', 'ASC']:
            sugg = {"note": f"Remove quotes around {keyword} keyword. Use {keyword} without quotes in ORDER BY clauses."}
            return sugg, None

    m = ERROR_PATTERNS["cte_union_error"].search(err_text)
    if m:
        sugg = {"note": "CTE (WITH clause) cannot be combined with UNION ALL in SQLite. Use either: 1) WITH clause for single SELECT, or 2) Multiple SELECT statements with UNION ALL (no CTE)."}
        return sugg, None

    m = ERROR_PATTERNS["backtick_error"].search(err_text)
    if m:
        sugg = {"note": "Avoid using backticks (`) for column names. Use double quotes (\") for identifiers that contain spaces or special characters, or reserved keywords."}
        return sugg, None

    m = ERROR_PATTERNS["union_order_by_error"].search(err_text)
    if m:
        sugg = {"note": "In SQLite UNION ALL queries, ORDER BY clauses must come at the very end of the entire query, not within each subquery. Remove ORDER BY from individual SELECT statements and add a single ORDER BY at the end."}
        return sugg, None

    m = ERROR_PATTERNS["parentheses_syntax_error"].search(err_text)
    if m:
        sugg = {"note": "SQLite does not support parentheses around subqueries in UNION ALL statements. Remove the parentheses around each SELECT statement in the UNION ALL."}
        return sugg, None

    return None, None

def execute_with_smart_repair(
    *,
    thinking_container,
    db_path: str,
    question: str,
    schema: str,
    distinct_info: str,
    initial_raw_sql_text: str,
    table: str,
    table_columns: dict[str, list[str]],
    max_retries: int = 4,
    max_same_error: int = 2
):
    """
    Executes SQL with:
      - sanitize_sqlite_sql (dequote ASC/DESC + alias-guard)
      - LLM repair using DB error
      - smart hints (closest columns, reserved alias note, etc.)
      - smart stop if same hard error repeats
    Returns (rows, final_sql_used, attempts_count, all_attempts_raw_text)
    """
    sql, cleaned_text, raw_matches = extract_sql(initial_raw_sql_text)
    attempts_raw = [initial_raw_sql_text]
    if not sql:
        return None, None, 0, attempts_raw

    attempts = 0
    same_error_counter = 0
    last_error_signature = None
    current_sql = sanitize_sqlite_sql(sql)
    
    # Also fix any backticks that might have been introduced
    current_sql = fix_sqlite_quotes(current_sql)
    
    # Debug: Show what sanitizer did
    with init_box:
        st.info(f"Sanitizer applied to SQL for table '{table}':")
        st.caption("Note: Any backticks have been automatically converted to double quotes")
        st.code(f"Original: {sql}")
        st.code(f"Sanitized: {current_sql}")

    while attempts <= max_retries:
        attempts += 1

        if not isinstance(current_sql, (str, bytes)) or not str(current_sql).strip():
            err = "No valid SQL text to execute."
            rows = None
        else:
            # Always sanitize right before execution too (in case LLM added issues)
            # But only if it's not already been sanitized to avoid double-processing
            if attempts == 1:
                # First attempt already sanitized above
                pass
            else:
                current_sql = sanitize_sqlite_sql(current_sql)
                # Also fix any backticks that might have been introduced
                current_sql = fix_sqlite_quotes(current_sql)
                with init_box:
                    st.info(f"Sanitizer applied to repaired SQL (attempt {attempts}):")
                    st.caption("Note: Any backticks have been automatically converted to double quotes")
                    st.code(f"Repaired: {current_sql}")
            
            rows, err = try_execute_sql(db_path, current_sql)

        if err is None:
            return rows, current_sql if isinstance(current_sql, str) else str(current_sql), attempts, attempts_raw

        with init_box:
            st.error(f"SQL attempt {attempts} failed for table '{table}':\n{err}\n\nSQL:\n{current_sql}")

        sig = (type(err), str(err).splitlines()[0].strip())
        if sig == last_error_signature:
            same_error_counter += 1
        else:
            same_error_counter = 0
        last_error_signature = sig
        if same_error_counter >= max_same_error:
            with init_box:
                st.warning(f"Stopping early: same error repeated {same_error_counter+1} times.")
            break

        if attempts > max_retries:
            break

        suggestions, hard_stop = diagnose_error(err, table, table_columns)
        if hard_stop:
            with init_box:
                st.warning(f"Stopping: {hard_stop}")
            break

        repair_messages = build_sql_repair_messages(
            question=question,
            schema=schema,
            distinct_info=distinct_info,
            previous_sql=current_sql if isinstance(current_sql, str) else str(current_sql),
            error_text=err,
            suggestions=suggestions
        )
        # Repair step with 20k tokens
        raw_repair_text, _, _ = llm_complete_with_status(
            f"SQL repair attempt {attempts} for table '{table}'",
            repair_messages,
            thinking_container=thinking_container,
            show_raw=True,
            max_tokens=12000
        )
        attempts_raw.append(raw_repair_text)
        repaired_sql, _, _ = extract_sql(raw_repair_text)
        current_sql = sanitize_sqlite_sql(repaired_sql) if repaired_sql else current_sql

    return None, current_sql if isinstance(current_sql, str) else str(current_sql), attempts, attempts_raw

# =================== Data ingest (build schemas & distincts once) ===================

table_schemas = {}           # table -> "col TYPE, col TYPE, ..."
table_columns = {}           # table -> [col1, col2, ...]  (for suggestions)
distinct_lines = []

for path in glob(os.path.join(SAV_DIR, '*.sav')):
    name = os.path.splitext(os.path.basename(path))[0]
    table = re.sub(r'[^0-9A-Za-z_]', '_', name).lower()
    df, meta = pyreadstat.read_sav(path)
    df.columns = [c.lower() for c in df.columns]

    # rename to human-friendly labels, keep uniqueness
    seen, rename_map = {}, {}
    for col in df.columns:
        lbl = meta.column_names_to_labels.get(col.upper()) or col
        base = re.sub(r'[^0-9A-Za-z_]', '_', lbl.lower())
        idx = seen.get(base, 0)
        seen[base] = idx + 1
        rename_map[col] = f"{base}_{idx}" if idx else base
    df.rename(columns=rename_map, inplace=True)

    # map code -> label values if available
    for orig, new in rename_map.items():
        label_set = meta.variable_to_label.get(orig.upper())
        if label_set:
            vals_map = meta.value_labels.get(label_set, {})
            df[new] = df[new].map(vals_map).fillna(df[new]).infer_objects(copy=False)

    df.fillna(0, inplace=True)

    db_file = os.path.join(DB_DIR, f"{table}.db")
    # Fix potential stray brace if present
    db_file = os.path.join(DB_DIR, f"{table}.db")
    engine = create_engine(f"sqlite+pysqlite:///{db_file}")
    with engine.begin() as conn:
        df.to_sql(table, conn, if_exists='replace', index=False)
        cols = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        schema = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        table_schemas[table] = schema
        table_columns[table] = [c[1] for c in cols]
        for col in df.columns:
            rows = conn.execute(text(f'SELECT DISTINCT "{col}" FROM {table}')).fetchall()
            distinct_lines.append(f"{table}.{col}: {', '.join(str(r[0]) for r in rows)}")
    engine.dispose()

# =================== Chat Feed (ONLY current session) ===================

for item in st.session_state.chat_thread:
    with st.chat_message('user'):
        st.markdown(f"**{item['ts']}**\n\n{item['question']}")
    with st.chat_message('assistant'):
        st.markdown(item['final_answer'])
        tb = item['timings']
        total = sum(tb.values())
        breakdown = " + ".join([f"{k}: {v:.2f}s" for k, v in tb.items()]) + f" = {total:.2f}s"
        st.caption(breakdown)
        if item.get('tables'):
            st.caption('Tables: ' + ", ".join(item['tables']))

# =================== Chat History Dialog ===================

def _render_history_ui():
    st.caption("Showing stored chats from previous runs.")
    hist = load_chat_history(500)
    if not hist:
        st.info("No saved chats yet.")
        return
    cols = st.columns([1, 1, 2])
    with cols[0]:
        q_filter = st.text_input("Search text", "")
    with cols[1]:
        limit = st.number_input(
            "Max items",
            min_value=1,
            max_value=1000,
            value=min(200, len(hist)) if len(hist) > 0 else 1,
            step=10
        )
    filtered = []
    qf = q_filter.lower().strip()
    for item in hist:
        if qf:
            blob = f"{item['ts']} {item['question']} {item['final_answer']}.".lower()
            if qf not in blob:
                continue
        filtered.append(item)
        if len(filtered) >= limit:
            break
    for item in filtered:
        st.markdown(f"### {item['ts']}")
        with st.chat_message('user'):
            st.markdown(item['question'])
        with st.chat_message('assistant'):
            st.markdown(item['final_answer'])
            tb = item['timing']
            try:
                total = sum(tb.values())
                breakdown = " + ".join([f"{k}: {v:.2f}s" for k, v in tb.items()]) + f" = {total:.2f}s"
                st.caption(breakdown)
                if item.get('tables'):
                    st.caption('Tables: ' + ", ".join(item['tables']))
            except Exception:
                pass
            with st.expander("Raw final LLM output"):
                show_area("Raw final LLM output", item['raw_final'], height=300, key=f"hist_raw_{item['id']}")
        st.markdown("---")
    if st.button("Close"):
        st.session_state.show_history = False
        st.rerun()

if st.session_state.show_history:
    if hasattr(st, "dialog"):
        @st.dialog("Chat History")
        def _history_dialog():
            _render_history_ui()
        _history_dialog()
    else:
        st.warning("Your Streamlit version does not support dialogs. Showing history inline here. "
                   "Upgrade to Streamlit ≥ 1.32 to get a popup.")
        _render_history_ui()

# =================== Chat Input ===================

user_msg = st.chat_input("Ask a question")
if not user_msg:
    st.stop()

question = user_msg

# Render the user's message in the feed
with st.chat_message('user'):
    ts_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"**{ts_now}**\n\n{question}")

# Dedicated assistant bubble for live thinking statuses
with st.chat_message('assistant'):
    thinking_container = st.container()
    
    # Add real-time progress display placeholder
    progress_placeholder = st.empty()

# ------------------- Pipeline for this message -------------------

answers_by_table = []
# NEW: capture per-table citation text so we can render it **below** the final footnotes
citations_map: dict[str, str] = {}

time_bucket = {
    'Relevant table selection': 0.0,
    'SQL generation (all tables)': 0.0,
    'Per-table summaries (all tables)': 0.0,
    'Final combined analysis': 0.0,
}

# Step 1: select relevant tables (20k)
sys_tbl = SystemMessage(content='You are an expert SQLite assistant. Respond ONLY with a JSON array of table names relevant to the user question. Do not output anything else.')
tables_prompt = (
    f"User question: {question}\n"
    f"Available tables: {list(table_schemas.keys())}\n"
    'Instructions: Select only the tables needed to answer the question, and respond with a JSON array of table names. If you do not find any relevant table name in the question, use all the tables.'
)
raw_tbl, clean_tbl, t_tables = llm_complete_with_status(
    'Relevant table selection',
    [sys_tbl, UserMessage(content=tables_prompt)],
    thinking_container=thinking_container,
    show_raw=True,
    max_tokens=12000
)
time_bucket['Relevant table selection'] += t_tables

json_match = re.search(r"\[.*?\]", clean_tbl, flags=re.DOTALL)
if json_match:
    try:
        tables_to_use = json.loads(json_match.group(0))
    except Exception:
        tables_to_use = list(table_schemas.keys())
else:
    tables_to_use = list(table_schemas.keys())

# Steps 2 & 3 for each table - Now with concurrent processing option
if ENABLE_CONCURRENT and len(tables_to_use) > 1:
    # Initialize progress tracking
    init_concurrent_processing()
    add_progress_event('start', None, f"Started concurrent processing of {len(tables_to_use)} tables with {MAX_CONCURRENT_WORKERS} workers")
    
    # Use concurrent processing for multiple tables
    with init_box:
        st.info(f"🔄 Processing {len(tables_to_use)} tables concurrently with {MAX_CONCURRENT_WORKERS} workers...")
    
    # Process tables concurrently - NO STREAMLIT UI CALLS IN THREADS
    with st.spinner(f"Processing {len(tables_to_use)} tables concurrently..."):
        # Process tables
        table_results = process_tables_concurrently(
            tables_to_use=tables_to_use,
            question=question,
            table_schemas=table_schemas,
            distinct_lines=distinct_lines,
            table_columns=table_columns,
            max_workers=MAX_CONCURRENT_WORKERS
        )
        
        # Show final progress update
        update_real_time_progress(progress_placeholder)
    
    # Add completion event
    add_progress_event('complete', None, f"Completed concurrent processing of {len(tables_to_use)} tables")
    
    # Show completion summary
    successful_tables = [r['table'] for r in table_results if r['success']]
    failed_tables = [r['table'] for r in table_results if not r['success']]
    
    with init_box:
        if successful_tables:
            st.success(f"✅ Successfully processed {len(successful_tables)} tables: {', '.join(successful_tables)}")
        if failed_tables:
            st.error(f"❌ Failed to process {len(failed_tables)} tables: {', '.join(failed_tables)}")
        
        # Show the progress timeline
        render_progress_timeline()
    
    # Process results and update time bucket - ALL STREAMLIT UI CALLS HERE
    for result in table_results:
        tbl = result['table']
        
        # Update time bucket
        time_bucket['SQL generation (all tables)'] += result['timings']['sql_generation']
        time_bucket['Per-table summaries (all tables)'] += result['timings']['summary_generation']
        
        # Show per-table results
        with init_box:
            st.markdown(f"### Table: {tbl}")
            st.caption("Generated/Final SQL (after repairs if any)")
            show_area("Final SQL", result['final_sql'] or "(no valid SQL)", height=160, key=f"sql_{tbl}")

            if not result['success']:
                st.error(f"All attempts failed for table '{tbl}'. Skipping this table.")
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
            else:
                st.markdown(
                    f"<span style='color:green;'>{tbl}: SQL executed successfully after {result['attempts']} attempt(s)</span>",
                    unsafe_allow_html=True
                )

                st.caption('Raw SQL Output')
                try:
                    if result['rows']:
                        df = pd.DataFrame(result['rows'], columns=[col for col in result['rows'][0].keys()])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.write(result['rows'])
                except Exception:
                    st.write(result['rows'])

                st.caption("Table-wise Summary")
                st.markdown(f"#### Summary for table: {tbl}")
                st.markdown(result['summary'])
                st.caption(f"Summary generation time: {result['timings']['summary_generation']:.2f}s")
                st.markdown('---')

                # Collect for combined analysis
                answers_by_table.append((tbl, result['summary']))

        # Capture citation text for this table
        citations_map[tbl] = result['citations']
        
else:
    # Sequential processing (original code)
    for tbl in tables_to_use:
        schema = table_schemas[tbl]
        distinct_info = "\n".join(d for d in distinct_lines if d.startswith(f"{tbl}."))
        db_path = os.path.join(DB_DIR, f"{tbl}.db")

        # Step 2: initial SQL generation (20k)
        sys_sql = SystemMessage(content="You are an expert SQLite assistant. Respond with DESCRIPTION and SQL_QUERY as specified in the user instructions.")
        sql_prompt = (
            f"User question: {question}\n"
            f"Schema: {schema}\n"
            f"Distinct values:\n{distinct_info}\n"
            """Instructions: 
                1. Strictly use only those column names which are present in the table schema when writing SQL.
                2. Never assume or use a column name which is not present in the schema.
                3. Even if column name is very long use it as it is.
                4. You must perform all calculations by considering each item according to its respective weight. After completing the weighted calculation, always round the final result to the nearest whole number. No fractional or decimal results should be presented in the output.\n
                5. Always cross-check if you are using quotation marks for the right strings in an SQL query. Do not use double quotes for SQL keywords, functions, or direction specifiers. Below are common mistakes to avoid (all drawn from past errors),
                    Sample examples:
                    Wrong:
                    ORDER BY ROUND(SUM(wtfactor)) "DESC"
                    Right:
                    ORDER BY ROUND(SUM(wtfactor)) DESC
                    (DESC is a keyword, not a string — no quotes needed)

                    Wrong:
                    ORDER BY ROUND(SUM(wtfactor)) "ASC"
                    Right:
                    ORDER BY ROUND(SUM(wtfactor)) ASC
                    (ASC is a keyword, not a string — no quotes needed)

                    CRITICAL: Never put quotes around DESC or ASC in ORDER BY clauses. 
                    They are SQL keywords, not string literals.
                    Examples: ORDER BY column DESC, ORDER BY column ASC
                    Never: ORDER BY column "DESC" or ORDER BY column "ASC"

                    Wrong:
                    AS "DESC" when aliasing a column
                    Right:
                    AS some_alias or AS "some_alias"
                    (Never use keywords like DESC/ASC as aliases — if unavoidable, quote a safe alias name instead)

                    Wrong:
                    AS "AS" or AS "GROUP"
                    Right:
                    AS "group"
                    (If the alias is a reserved keyword, use lowercase and keep the quotes — "group" is fine, "GROUP" is reserved)

                    Wrong:
                    GROUP BY "gender_for_weighting___gendwt" (when it's a real column name and doesn't need quoting) combined with other unquoted names incorrectly, or mixing quotes inconsistently
                    Right:
                    GROUP BY gender_for_weighting___gendwt
                    (Only quote identifiers if they contain spaces, special characters, or are reserved keywords — be consistent).\n
                6. CRITICAL SQLite Syntax Rules (MUST FOLLOW):
                    a. NEVER use CTEs (WITH clauses) combined with UNION ALL in the same query
                    b. If you need multiple SELECT statements, use ONLY UNION ALL without CTEs
                    c. If you need a CTE, use it for a single SELECT statement only
                    d. Valid patterns:
                       - Single SELECT with CTE: WITH cte AS (SELECT ...) SELECT * FROM cte
                       - Multiple SELECTs with UNION ALL: SELECT ... UNION ALL SELECT ... UNION ALL SELECT ... (NO parentheses around subqueries)
                       - NOT: WITH cte AS (...) UNION ALL SELECT ... (this will fail)
                       - NOT: (SELECT ...) UNION ALL (SELECT ...) (parentheses cause syntax error)
                    e. For complex queries requiring multiple demographic breakdowns, use UNION ALL pattern:
                       SELECT ... LIMIT 1
                       UNION ALL
                       SELECT ... LIMIT 1
                       UNION ALL
                       SELECT ... LIMIT 1
                       ORDER BY ... (single ORDER BY at the very end)
                       Note: ORDER BY clauses cannot be used within individual SELECT statements in UNION ALL
                       CRITICAL: In UNION ALL queries, ORDER BY must come at the very end of the entire query, not within subqueries
                    f. CRITICAL: Use DOUBLE QUOTES (\") for column names that need quoting, NEVER backticks (`)
                       - Wrong: `column_name` (MySQL syntax)
                       - Right: \"column_name\" (SQLite syntax)
                       - Only quote column names that start with digits or contain special characters
                       - Example: \"2024_support__with_leaners_\" (starts with digit)
                       - Example: column_name (no quotes needed for normal names)
                7. Respond exactly as:
                DESCRIPTION: <one-sentence>
                SQL_QUERY: <single-line SQL ending with semicolon>
                """
        )
        raw_sql_text, _, t_sql = llm_complete_with_status(
            f"SQL generation for table '{tbl}'",
            [sys_sql, UserMessage(content=sql_prompt)],
            thinking_container=thinking_container,
            show_raw=True,
            max_tokens=25000
        )
        time_bucket['SQL generation (all tables)'] += t_sql

        # Execute with SMART repair loop (repair uses 20k inside)
        rows, final_sql_used, attempts_count, attempts_raw = execute_with_smart_repair(
            thinking_container=thinking_container,
            db_path=db_path,
            question=question,
            schema=schema,
            distinct_info=distinct_info,
            initial_raw_sql_text=raw_sql_text,
            table=tbl,
            table_columns=table_columns,
            max_retries=4,
            max_same_error=2
        )
        
        # ---- Show per-table SQL + output + summary inside the main init_box (unchanged) ----
        with init_box:
            st.markdown(f"### Table: {tbl}")
            st.caption("Generated/Final SQL (after repairs if any)")
            show_area("Final SQL", final_sql_used or "(no valid SQL)", height=160, key=f"sql_{tbl}")

            if rows is None:
                st.error(f"All attempts failed for table '{tbl}'. Skipping this table.")
            else:
                st.markdown(
                    f"<span style='color:green;'>{tbl}: SQL executed successfully after {attempts_count} attempt(s)</span>",
                    unsafe_allow_html=True
                )

                # Step 3: per-table natural language summary (20k)
                ans_sys = SystemMessage(content='You are a helpful assistant.')
                ans_user = UserMessage(content=(
                    f"Original question: {question}\n"
                    f"Final SQL used (after repairs):\n{final_sql_used}\n"
                    f"Results from {tbl}: {rows}\n"
                    "Provide a concise summary for this table."
                ))
                raw_ans, final_ans, t_ans = llm_complete_with_status(
                    f"Summary for '{tbl}'",
                    [ans_sys, ans_user],
                    thinking_container=thinking_container,
                    show_raw=True,
                    max_tokens=12000
                )
                time_bucket['Per-table summaries (all tables)'] += t_ans

                st.caption('Raw SQL Output')
                try:
                    df = pd.DataFrame(rows, columns=[col for col in rows[0].keys()])
                    st.dataframe(df, use_container_width=True)
                except Exception:
                    st.write(rows)

                st.caption("Table-wise Summary")
                st.markdown(f"#### Summary for table: {tbl}")
                st.markdown(final_ans)
                st.caption(f"Summary generation time: {t_ans:.2f}s")
                st.markdown('---')

                # Collect for combined analysis
                answers_by_table.append((tbl, final_ans))

        # Capture citation text for this table (to be shown **after** the final footnotes)
        citations_map[tbl] = extract_col_values_formatted(final_sql_used or "", dialect="sqlite")

# NOTE: Removed the earlier per-table citation expanders here to avoid duplication.
# Citations will be rendered below the final combined analysis (i.e., after footnotes), as requested.

# Step 4: final combined analysis (20k)
summaries = "\n".join(f"{tbl}: {ans}" for tbl, ans in answers_by_table)
combined_prompt = (
    f"Original question: {question}\n"
    'Here are table summaries:\n' + summaries + '\n'
    'You are MipoLLM, an expert political and social survey analyst. For any user question, produce a comprehensive, evidence-backed analysis grounded only in the provided dataset. When numerical values appear in your analysis (e.g., 20% vs 40%), make the reasoning behind those values fully transparent by including an indexed footnote for each, explaining exactly how it was computed (formula, numerator/denominator counts, filters, weighting, and which values used). Use clear labels for all statistics.'
)
raw_combined, cleaned_combined, t_combined = llm_complete_with_status(
    'Final combined analysis',
    [SystemMessage(content='You are a helpful assistant.'), UserMessage(content=combined_prompt)],
    thinking_container=thinking_container,
    show_raw=True,
    max_tokens=25000
)
time_bucket['Final combined analysis'] += t_combined

with init_box:
    st.subheader('Raw LLM Response for final analysis')
    show_area("Raw final analysis", raw_combined, height=300, key="raw_final_analysis")
    st.caption(f"Final analysis time: {t_combined:.2f}s")

# Post final assistant answer below the statuses
with st.chat_message('assistant'):
    # Final narrative + footnotes from the combined step
    st.markdown(cleaned_combined)

    # Timing + tables meta
    total = sum(time_bucket.values())
    breakdown = " + ".join([f"{k}: {v:.2f}s" for k, v in time_bucket.items()]) + f" = {total:.2f}s"
    st.caption(breakdown)
    try:
        st.caption('Tables: ' + ", ".join(tables_to_use))
    except Exception:
        pass

    # NEW: Render citations for each table **below** the footnotes
    st.markdown('---')
    st.markdown('### Citations by table')
    for tbl in tables_to_use:
        if 'PRINT_MODE' in globals() and PRINT_MODE:
            st.markdown(f"#### citation for {tbl}")
            st.caption('Column names and their values used (parsed from SQL WHERE predicates)')
            show_area("Citation details", citations_map.get(tbl, "No column-value predicates found."), height=180, key=f"cit_{tbl}")
        else:
            with st.expander(f"citation for {tbl}", expanded=False):
                st.caption('Column names and their values used (parsed from SQL WHERE predicates)')
                show_area("Citation details", citations_map.get(tbl, "No column-value predicates found."), height=180, key=f"cit_{tbl}")

# Dedicated print-friendly replica outside chat bubbles (prevents chat container clipping when printing)
if 'PRINT_MODE' in globals() and PRINT_MODE:
    st.markdown('---')
    st.markdown('## Print View (Full Content)', help=None)
    st.markdown(cleaned_combined)
    st.markdown('### Citations by table (Print View)')
    for tbl in tables_to_use:
        st.markdown(f"#### citation for {tbl}")
        st.markdown(f"<pre class='pre-wrap'>{ihtml.escape(citations_map.get(tbl, 'No column-value predicates found.'))}</pre>", unsafe_allow_html=True)

# Save to in-memory thread and DB
st.session_state.chat_thread.append({
    'ts': ts_now,
    'question': question,
    'final_answer': cleaned_combined,
    'timings': time_bucket,
    'tables': tables_to_use if 'tables_to_use' in locals() else [],
    'raw_final': raw_combined,
})
save_chat(question, cleaned_combined, time_bucket, tables_to_use if 'tables_to_use' in locals() else [], raw_combined)

# Show total thought time at top for THIS run
total_thinking_seconds = sum(time_bucket.values())
mins = int(total_thinking_seconds // 60)
secs = total_thinking_seconds % 60
with total_time_placeholder:
    st.markdown(f"### Total thought for {mins} min {secs:.2f} sec")
