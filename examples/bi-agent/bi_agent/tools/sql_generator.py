"""SQL generator tool for the BI agent pipeline.

This tool is responsible for Step 3 of the BI query pipeline: translating the
structured field/filter context produced by ``field_selector_tool`` into an
executable SQL statement.

The generator receives an explicit structured description of what the query
should express (tables, fields, filters, groupings, limits) rather than a
free-form natural-language prompt.  This keeps the generated SQL deterministic
and auditable.

Design notes
------------
- The tool builds a *draft* SQL from the structured inputs.  When the LLM
  calls this tool it supplies the fields/filters it selected; the tool
  assembles the SQL mechanically and returns it for inspection before
  execution.
- Complex expressions or dialect-specific syntax (window functions,
  CTEs, etc.) can be handled by extending the ``QuerySpec`` model and
  the assembly logic below.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class QuerySpec(BaseModel):
    """Structured SQL query specification."""

    table: str = Field(description="Primary table name (FROM clause).")
    selected_fields: list[str] = Field(
        description=(
            "Columns to SELECT, as 'table.column' or plain 'column' strings. "
            "Use '*' for all columns."
        )
    )
    filters: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "WHERE conditions as {column: value} pairs. "
            "Values may include operators, e.g. '>= 2024-01-01'."
        ),
    )
    joins: list[str] = Field(
        default_factory=list,
        description="Optional JOIN clauses as raw SQL strings.",
    )
    group_by: list[str] = Field(
        default_factory=list,
        description="Columns for GROUP BY.",
    )
    order_by: list[str] = Field(
        default_factory=list,
        description="Columns for ORDER BY (prefix with '-' for DESC).",
    )
    limit: int | None = Field(
        default=None,
        description="Optional LIMIT value.",
    )
    dialect: str = Field(
        default="standard",
        description=(
            "SQL dialect hint. Supported values: 'standard', 'mysql', "
            "'postgresql', 'sqlite', 'bigquery'."
        ),
    )


@tool(args_schema=QuerySpec)
def sql_generator_tool(
    table: str,
    selected_fields: list[str],
    filters: dict[str, str],
    joins: list[str],
    group_by: list[str],
    order_by: list[str],
    limit: int | None,
    dialect: str,
) -> str:
    """Generate a SQL SELECT statement from a structured query specification.

    Assembles a syntactically correct SQL query from the provided components.
    The generated SQL is returned as a plain string so the agent can review
    it before passing it to ``sql_runner_tool``.

    Args:
        table: Primary table name (FROM clause).
        selected_fields: Columns to SELECT.
        filters: WHERE conditions as column -> value/expression pairs.
        joins: Optional JOIN clauses as raw SQL strings.
        group_by: Columns for GROUP BY.
        order_by: Columns for ORDER BY (prefix with ``-`` for DESC).
        limit: Optional LIMIT value.
        dialect: SQL dialect hint (informational, used for quoting style).

    Returns:
        JSON string with ``{"sql": "<generated SQL>"}`` on success or
        ``{"error": "<message>"}`` on failure.
    """
    try:
        # SELECT clause
        fields_clause = ", ".join(selected_fields) if selected_fields else "*"
        sql = f"SELECT {fields_clause}\nFROM {table}"

        # JOIN clauses
        for join in joins:
            sql += f"\n{join}"

        # WHERE clause
        if filters:
            conditions = []
            for col, val in filters.items():
                # Treat the value as a raw SQL expression when it already
                # starts with a comparison operator (>, <, !=, >=, <=, =)
                # or a SQL keyword (IN, LIKE, NOT, BETWEEN, IS, EXISTS).
                # Otherwise wrap the value in single quotes for a simple
                # equality check.
                stripped = val.strip()
                is_expression = (
                    stripped[:1] in {">", "<", "!", "="}
                    or stripped.upper().startswith((
                        "IN ", "IN(",
                        "LIKE ", "NOT ", "BETWEEN ",
                        "IS ", "EXISTS",
                    ))
                )
                if is_expression:
                    conditions.append(f"{col} {stripped}")
                else:
                    conditions.append(f"{col} = '{stripped}'")
            sql += "\nWHERE " + "\n  AND ".join(conditions)

        # GROUP BY clause
        if group_by:
            sql += "\nGROUP BY " + ", ".join(group_by)

        # ORDER BY clause
        if order_by:
            order_parts = []
            for col in order_by:
                if col.startswith("-"):
                    order_parts.append(f"{col[1:]} DESC")
                else:
                    order_parts.append(col)
            sql += "\nORDER BY " + ", ".join(order_parts)

        # LIMIT clause
        if limit is not None:
            sql += f"\nLIMIT {limit}"

        return json.dumps({"sql": sql}, ensure_ascii=False)

    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
