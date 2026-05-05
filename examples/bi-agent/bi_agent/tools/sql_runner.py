"""SQL runner tool for the BI agent pipeline.

This tool is responsible for Step 4 of the BI query pipeline: executing a SQL
statement against the configured database and returning the results as a JSON
array of row objects.

The runner is intentionally thin — it delegates connection management and
query execution to SQLAlchemy so that any database supported by SQLAlchemy
can be used without code changes.

Configuration
-------------
The database URL is taken from the ``DATABASE_URL`` environment variable.  If
not set, the runner falls back to an in-memory SQLite database populated with
sample BI data for demo purposes.

Design notes
------------
- Results are serialised to JSON strings (with Python types coerced to
  JSON-safe equivalents) so the agent can pass them directly to
  ``result_analyzer_tool`` without additional parsing.
- Row count is capped at ``MAX_ROWS`` to avoid overwhelming the context window.
- Replace or subclass the ``_get_engine`` helper to add connection pooling,
  read-replica routing, or per-tenant database switching.
"""

from __future__ import annotations

import datetime
import json
import os
from decimal import Decimal

import sqlalchemy as sa
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Maximum rows returned to avoid context-window overflow.
MAX_ROWS = 500


class SQLRunnerInput(BaseModel):
    """Input schema for the SQL runner tool."""

    sql: str = Field(description="The SQL SELECT statement to execute.")
    database_url: str = Field(
        default="",
        description=(
            "Optional SQLAlchemy database URL (e.g. 'sqlite:///./mydb.db'). "
            "Falls back to the DATABASE_URL environment variable, then to an "
            "in-memory SQLite demo database."
        ),
    )


def _get_engine(database_url: str) -> sa.Engine:
    """Resolve and create a SQLAlchemy engine.

    Args:
        database_url: Explicit URL override (may be empty string).

    Returns:
        A connected SQLAlchemy ``Engine``.
    """
    url = database_url or os.getenv("DATABASE_URL", "")
    if not url:
        # Fall back to in-memory SQLite populated with demo data.
        url = "sqlite://"
    return sa.create_engine(url)


def _seed_demo_db(engine: sa.Engine) -> None:
    """Populate an in-memory SQLite engine with sample BI data.

    Only called when no external DATABASE_URL is provided.

    Args:
        engine: In-memory SQLite engine to seed.
    """
    with engine.connect() as conn:
        conn.execute(sa.text(
            """CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY,
                date TEXT,
                region TEXT,
                product TEXT,
                amount REAL,
                quantity INTEGER
            )"""
        ))
        conn.execute(sa.text(
            """CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category TEXT,
                price REAL
            )"""
        ))
        # Insert sample rows only when tables are empty.
        count = conn.execute(sa.text("SELECT COUNT(*) FROM sales")).scalar()
        if count == 0:
            conn.execute(sa.text(
                "INSERT INTO sales VALUES "
                "(1,'2024-01-15','North','Widget A',1500.00,10),"
                "(2,'2024-01-20','South','Widget B',2300.50,15),"
                "(3,'2024-02-05','North','Widget A',1800.00,12),"
                "(4,'2024-02-18','East','Widget C', 950.75, 7),"
                "(5,'2024-03-01','South','Widget B',3100.00,20),"
                "(6,'2024-03-15','West','Widget A',2200.00,14)"
            ))
            conn.execute(sa.text(
                "INSERT INTO products VALUES "
                "(1,'Widget A','Electronics',150.00),"
                "(2,'Widget B','Electronics',230.00),"
                "(3,'Widget C','Accessories', 95.00)"
            ))
        conn.commit()


def _json_default(obj: object) -> object:
    """Coerce non-JSON-serialisable types for ``json.dumps``.

    Args:
        obj: Object that failed default serialisation.

    Returns:
        JSON-serialisable equivalent.

    Raises:
        TypeError: If the type is not handled.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    msg = f"Object of type {type(obj).__name__} is not JSON serialisable"
    raise TypeError(msg)


@tool(args_schema=SQLRunnerInput)
def sql_runner_tool(sql: str, database_url: str = "") -> str:
    """Execute a SQL statement and return results as a JSON array.

    Connects to the configured database, executes the provided SQL, and
    returns up to ``MAX_ROWS`` rows as a JSON array of objects.

    Args:
        sql: The SQL SELECT statement to execute.
        database_url: Optional SQLAlchemy URL override.

    Returns:
        JSON string: ``{"rows": [...], "row_count": N}`` on success, or
        ``{"error": "<message>"}`` on failure.
    """
    try:
        engine = _get_engine(database_url)

        # Seed demo data if using the in-memory fallback.
        if str(engine.url) == "sqlite://":
            _seed_demo_db(engine)

        with engine.connect() as conn:
            result = conn.execute(sa.text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row, strict=True)) for row in result.fetchmany(MAX_ROWS)]

        return json.dumps(
            {"rows": rows, "row_count": len(rows)},
            default=_json_default,
            ensure_ascii=False,
        )

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
