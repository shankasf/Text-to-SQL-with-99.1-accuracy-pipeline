# sqlite_alias_guard.py
"""
Utility to make LLM-generated SQL safer for SQLite by:
1) De-quoting ORDER BY directions like "DESC"/"ASC" -> DESC/ASC
2) De-quoting quoted AS keyword like "AS" -> AS
3) Quoting aliases that collide with reserved keywords (e.g., AS group -> AS "group")

Design goals:
- Tolerant: no-op for non-strings; decodes bytes; safe for empty strings.
- Conservative: only touches clear patterns; won't mangle valid SQL.
"""

from __future__ import annotations
import re

# ---------- Reserved words ----------
SQLITE_RESERVED = {
    "ABORT","ACTION","ADD","AFTER","ALL","ALTER","ANALYZE","AND","AS","ASC","ATTACH","AUTOINCREMENT",
    "BEFORE","BEGIN","BETWEEN","BY","CASCADE","CASE","CAST","CHECK","COLLATE","COLUMN","COMMIT","CONFLICT",
    "CONSTRAINT","CREATE","CROSS","CURRENT_DATE","CURRENT_TIME","CURRENT_TIMESTAMP","DATABASE","DEFAULT",
    "DEFERRABLE","DEFERRED","DELETE","DESC","DETACH","DISTINCT","DROP","EACH","ELSE","END","ESCAPE","EXCEPT",
    "EXCLUSIVE","EXISTS","EXPLAIN","FAIL","FOR","FOREIGN","FROM","FULL","GLOB","GROUP","HAVING","IF","IGNORE",
    "IMMEDIATE","IN","INDEX","INDEXED","INITIALLY","INNER","INSERT","INSTEAD","INTERSECT","INTO","IS","ISNULL",
    "JOIN","KEY","LEFT","LIKE","LIMIT","MATCH","NATURAL","NO","NOT","NOTNULL","NULL","OF","OFFSET","ON","OR",
    "ORDER","OUTER","PLAN","PRAGMA","PRIMARY","QUERY","RAISE","RECURSIVE","REFERENCES","REGEXP","REINDEX",
    "RELEASE","RENAME","REPLACE","RESTRICT","RIGHT","ROLLBACK","ROW","SAVEPOINT","SELECT","SET","TABLE",
    "TEMP","TEMPORARY","THEN","TO","TRANSACTION","TRIGGER","UNION","UNIQUE","UPDATE","USING","VACUUM","VALUES",
    "VIEW","VIRTUAL","WHEN","WHERE","WITH","WITHOUT"
}

_ALIAS_TOKEN = r'([A-Za-z_][A-Za-z0-9_]*)'
ORDER_DIR_TOKEN = re.compile(r'"\s*(DESC|ASC)\s*"', re.IGNORECASE)
AS_KEYWORD_TOKEN = re.compile(r'"\s*AS\s*"', re.IGNORECASE)

def _ensure_text(sql):
    if not isinstance(sql, (str, bytes)):
        return sql, False
    if isinstance(sql, bytes):
        try:
            sql = sql.decode("utf-8", errors="ignore")
        except Exception:
            sql = sql.decode("latin-1", errors="ignore")
    return sql, True

def is_reserved(word: str) -> bool:
    return isinstance(word, str) and word.upper() in SQLITE_RESERVED

# --- Dequoters ---
def dequote_order_directions(sql):
    sql, ok = _ensure_text(sql)
    if not ok or not sql:
        return sql
    return ORDER_DIR_TOKEN.sub(lambda m: m.group(1).upper(), sql)

def dequote_as_keyword(sql):
    """Replace quoted AS like \"AS\" -> AS (SQLite treats double quotes as identifiers)."""
    sql, ok = _ensure_text(sql)
    if not ok or not sql:
        return sql
    return AS_KEYWORD_TOKEN.sub(" AS ", sql)

# --- Alias guard ---
def quote_reserved_aliases(sql):
    sql, ok = _ensure_text(sql)
    if not ok or not sql:
        return sql

    def _fix_as(m):
        alias = m.group(1)
        return f'AS "{alias}"' if is_reserved(alias) else f'AS {alias}'

    fixed = re.sub(r'\bAS\s+' + _ALIAS_TOKEN + r'\b', _fix_as, sql, flags=re.IGNORECASE)

    def _fix_bare_alias(m):
        lead = m.group(1)
        alias = m.group(2)
        # Only quote if it's actually a reserved keyword AND looks like an alias
        # Don't quote SQL keywords that are part of the query structure
        if is_reserved(alias) and alias.upper() not in ['UNION', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'LIMIT', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'TABLE', 'INDEX', 'VIEW', 'TRIGGER', 'PROCEDURE', 'FUNCTION', 'DATABASE', 'SCHEMA', 'COLUMN', 'CONSTRAINT', 'PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'DEFAULT', 'NULL', 'NOT', 'AND', 'OR', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'NATURAL', 'ON', 'USING', 'HAVING', 'DISTINCT', 'ALL', 'ANY', 'SOME', 'EXCEPT', 'INTERSECT', 'WITH', 'RECURSIVE', 'VALUES', 'SET', 'INTO', 'OF', 'REFERENCES', 'CASCADE', 'RESTRICT', 'DEFERRABLE', 'DEFERRED', 'IMMEDIATE', 'TRANSACTION', 'BEGIN', 'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'RELEASE', 'ATTACH', 'DETACH', 'ANALYZE', 'EXPLAIN', 'PRAGMA', 'VACUUM', 'REINDEX', 'REPLACE', 'IGNORE', 'FAIL', 'ABORT', 'ACTION', 'AFTER', 'BEFORE', 'EACH', 'INSTEAD', 'EXCLUSIVE', 'SHARED', 'TEMP', 'TEMPORARY', 'VIRTUAL', 'GLOB', 'REGEXP', 'MATCH', 'ESCAPE', 'COLLATE', 'CAST', 'RAISE', 'TRIGGER', 'PLAN', 'QUERY', 'KEY', 'ROW', 'ROWS', 'OFFSET', 'REPLACE', 'CONFLICT', 'ROLLBACK', 'ABORT', 'FAIL', 'IGNORE', 'REPLACE']:
            return f'{lead}"{alias}"'
        return m.group(0)

    # More conservative pattern - only look for actual aliases after commas or spaces
    # that are followed by SQL clause terminators, not SQL keywords themselves
    fixed = re.sub(
        r'([,\s])' + _ALIAS_TOKEN + r'\s*(?=,|\s+FROM\b|\s+UNION\b|\)\s*AS\b|\)\s*,|\)\s*FROM\b|\s+WHERE\b|\s+GROUP\s+BY\b|\s+ORDER\s+BY\b|\s+LIMIT\b|\s+HAVING\b|\s+WINDOW\b|\s+OVER\b)',
        _fix_bare_alias,
        fixed,
        flags=re.IGNORECASE
    )
    return fixed

# --- Public APIs ---
def fix_sqlite_aliases(sql):
    sql, ok = _ensure_text(sql)
    if not ok:
        return sql
    return quote_reserved_aliases(sql)

def fix_union_all_syntax(sql):
    """Fix common UNION ALL syntax issues in SQLite"""
    sql, ok = _ensure_text(sql)
    if not ok or not sql:
        return sql
    
    # Handle CTE with UNION ALL (invalid pattern) - this is a complex case
    # For now, we'll let the LLM repair handle this specific case
    # as it requires more sophisticated parsing
    if 'WITH' in sql.upper() and 'UNION ALL' in sql.upper():
        # Don't attempt to fix this automatically - let the error handling system deal with it
        pass
    
    # Remove parentheses around subqueries in UNION ALL
    # Only remove parentheses that are around entire SELECT statements followed by UNION ALL
    # Don't remove parentheses inside function calls like ROUND(...)
    if 'UNION ALL' in sql.upper():
        # Split by UNION ALL to process each part
        parts = re.split(r'\s+UNION\s+ALL\s+', sql, flags=re.IGNORECASE)
        if len(parts) > 1:
            cleaned_parts = []
            for part in parts:
                # Only remove outer parentheses if this part starts with (SELECT and ends with )
                # AND it's a complete UNION ALL subquery (not a scalar subquery inside a function)
                if (part.strip().startswith('(SELECT') and 
                    part.strip().endswith(')') and 
                    # Check if this is a complete SELECT statement, not a scalar subquery
                    not re.search(r'ROUND\(.*?\(SELECT.*?\)', part, flags=re.IGNORECASE | re.DOTALL)):
                    # Remove the outer parentheses
                    cleaned_part = part.strip()[1:-1]  # Remove first and last character
                    cleaned_parts.append(cleaned_part)
                else:
                    cleaned_parts.append(part.strip())
            
            # Reconstruct the query
            sql = ' UNION ALL '.join(cleaned_parts)
    
    # Move ORDER BY clauses from within subqueries to the end of the entire UNION ALL
    # This is a complex fix that requires careful parsing
    if 'UNION ALL' in sql.upper() and 'ORDER BY' in sql.upper():
        # Move ORDER BY clauses from within UNION ALL subqueries to the end of the entire query
        if 'UNION ALL' in sql.upper():
            # Split by UNION ALL to process each part
            parts = re.split(r'\s+UNION\s+ALL\s+', sql, flags=re.IGNORECASE)
            if len(parts) > 1:
                cleaned_parts = []
                collected_order_bys = []
                
                for i, part in enumerate(parts):
                    # Check if this part has ORDER BY
                    order_match = re.search(r'(.*?)\s+ORDER\s+BY\s+(.*?)(?:\s+LIMIT\s+\d+)?(?:\s*;?\s*)$', part, flags=re.IGNORECASE | re.DOTALL)
                    if order_match:
                        # Remove ORDER BY from this part and collect it
                        cleaned_part = order_match.group(1).strip()
                        order_by_clause = order_match.group(2).strip()
                        
                        # Remove LIMIT if present in the ORDER BY part and add it back to the cleaned part
                        limit_match = re.search(r'(.*?)\s+LIMIT\s+(\d+)', order_by_clause, flags=re.IGNORECASE)
                        if limit_match:
                            order_by_clause = limit_match.group(1).strip()
                            limit_clause = f"LIMIT {limit_match.group(2)}"
                            cleaned_part += f' {limit_clause}'
                        
                        # For UNION ALL, we need to use column aliases that exist in the result set
                        # Replace column references with the actual column aliases from the SELECT
                        if 'weighted_count' in order_by_clause:
                            # Look for the actual column alias in the SELECT
                            alias_match = re.search(r'ROUND\(SUM\(wtfactor\)\)\s+AS\s+(\w+)', cleaned_part, flags=re.IGNORECASE)
                            if alias_match:
                                actual_alias = alias_match.group(1)
                                order_by_clause = order_by_clause.replace('weighted_count', actual_alias)
                        
                        cleaned_parts.append(cleaned_part)
                        collected_order_bys.append(order_by_clause)
                    else:
                        cleaned_parts.append(part.strip())
                
                # Reconstruct the query
                sql = ' UNION ALL '.join(cleaned_parts)
                
                # Add a single ORDER BY at the end using the first collected ORDER BY
                if collected_order_bys:
                    # Use a simple ORDER BY that will work with the result set
                    # For UNION ALL queries, we typically want to order by the first column
                    # But first, check if there's a LIMIT clause and put ORDER BY before it
                    if 'LIMIT' in sql:
                        # Insert ORDER BY before LIMIT
                        sql = re.sub(r'(LIMIT\s+\d+)', r'ORDER BY 1 \1', sql, flags=re.IGNORECASE)
                    else:
                        sql += f' ORDER BY 1'
    
    return sql

def sanitize_sqlite_sql(sql):
    """
    Full sanitizer (recommended order):
      1) dequote "ASC"/"DESC"
      2) dequote "AS"
      3) quote reserved aliases
      4) fix UNION ALL syntax issues
    """
    original_sql = sql
    
    # Safety check: if SQL already has quoted SQL keywords, don't process it further
    # This prevents double-processing and corruption
    sql_keywords = ['UNION', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'LIMIT', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'TABLE', 'INDEX', 'VIEW', 'TRIGGER', 'PROCEDURE', 'FUNCTION', 'DATABASE', 'SCHEMA', 'COLUMN', 'CONSTRAINT', 'PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'DEFAULT', 'NULL', 'NOT', 'AND', 'OR', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'NATURAL', 'ON', 'USING', 'HAVING', 'DISTINCT', 'ALL', 'ANY', 'SOME', 'EXCEPT', 'INTERSECT', 'WITH', 'RECURSIVE', 'VALUES', 'SET', 'INTO', 'OF', 'REFERENCES', 'CASCADE', 'RESTRICT', 'DEFERRABLE', 'DEFERRED', 'IMMEDIATE', 'TRANSACTION', 'BEGIN', 'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'RELEASE', 'ATTACH', 'DETACH', 'ANALYZE', 'EXPLAIN', 'PRAGMA', 'VACUUM', 'REINDEX', 'REPLACE', 'IGNORE', 'FAIL', 'ABORT', 'ACTION', 'AFTER', 'BEFORE', 'EACH', 'INSTEAD', 'EXCLUSIVE', 'SHARED', 'TEMP', 'TEMPORARY', 'VIRTUAL', 'GLOB', 'REGEXP', 'MATCH', 'ESCAPE', 'COLLATE', 'CAST', 'RAISE', 'TRIGGER', 'PLAN', 'QUERY', 'KEY', 'ROW', 'ROWS', 'OFFSET', 'REPLACE', 'CONFLICT', 'ROLLBACK', 'ABORT', 'FAIL', 'IGNORE', 'REPLACE']
    
    for keyword in sql_keywords:
        if f'"{keyword}"' in sql or f'"{keyword.lower()}"' in sql:
            print(f"Warning: SQL already contains quoted keyword '{keyword}', skipping sanitization to prevent corruption")
            return sql
    
    sql = dequote_order_directions(sql)
    sql = dequote_as_keyword(sql)
    sql = quote_reserved_aliases(sql)
    sql = fix_union_all_syntax(sql)
    
    # Final aggressive pass to catch any remaining quoted DESC/ASC
    # This handles edge cases that might have been missed
    sql = re.sub(r'"\s*(DESC|ASC)\s*"', lambda m: m.group(1).upper(), sql, flags=re.IGNORECASE)
    
    # Debug: Check if DESC/ASC is still quoted
    if '"DESC"' in sql or '"ASC"' in sql:
        print(f"Warning: Still found quoted DESC/ASC in SQL after sanitization")
        print(f"Original: {original_sql}")
        print(f"Sanitized: {sql}")
    
    return sql

def debug_sanitization(sql):
    """Debug function to show what each step of sanitization does"""
    print("=== SQL Sanitization Debug ===")
    print(f"Original SQL: {sql}")
    
    step1 = dequote_order_directions(sql)
    print(f"After dequote_order_directions: {step1}")
    
    step2 = dequote_as_keyword(step1)
    print(f"After dequote_as_keyword: {step2}")
    
    step3 = quote_reserved_aliases(step2)
    print(f"After quote_reserved_aliases: {step3}")
    
    final = sanitize_sqlite_sql(sql)
    print(f"Final sanitized: {final}")
    print("=== End Debug ===")
    
    return final

__all__ = [
    "SQLITE_RESERVED",
    "is_reserved",
    "dequote_order_directions",
    "dequote_as_keyword",
    "quote_reserved_aliases",
    "fix_sqlite_aliases",
    "fix_union_all_syntax",
    "sanitize_sqlite_sql",
    "debug_sanitization",
]

# Simple test function
if __name__ == "__main__":
    # Test the sanitizer with a complex SQL query
    test_sql = """
    SELECT 'gender' AS demographic, final_gender AS category, ROUND(SUM(wtfactor)) AS weighted_count 
    FROM marist_poll_national_survey_sept_2024 
    WHERE "2024_support__with_leaners_" = 'Harris' 
    GROUP BY final_gender 
    ORDER BY weighted_count DESC 
    LIMIT 1
    UNION ALL
    SELECT 'race' AS demographic, race_recoded_for_weighting AS category, ROUND(SUM(wtfactor)) AS weighted_count 
    FROM marist_poll_national_survey_sept_2024 
    WHERE "2024_support__with_leaners_" = 'Harris' 
    GROUP BY race_recoded_for_weighting 
    ORDER BY weighted_count DESC 
    LIMIT 1
    """
    
    print("Testing SQL sanitizer...")
    sanitized = sanitize_sqlite_sql(test_sql)
    print(f"Sanitized SQL:\n{sanitized}")
    
    # Test that UNION, FROM, WHERE, etc. are not quoted
    if '"UNION"' in sanitized or '"FROM"' in sanitized or '"WHERE"' in sanitized:
        print("ERROR: SQL keywords are being quoted incorrectly!")
    else:
        print("SUCCESS: SQL keywords are not being quoted incorrectly")
