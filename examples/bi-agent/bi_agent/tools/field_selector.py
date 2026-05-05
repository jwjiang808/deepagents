"""Field selector tool for the BI agent pipeline.

This tool is responsible for Step 2 of the BI query pipeline: given the
target table(s) selected by ``table_selector_tool`` and the user question,
identify which columns are required and what filter values (if any) apply.

Return value is a structured JSON object that the downstream
``sql_generator_tool`` consumes directly.

Design notes
------------
- Separating field/filter selection from SQL generation keeps each tool
  focused and easier to test and tune independently.
- The tool surfaces column metadata so the LLM has enough context to reason
  about data types and typical value ranges.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class FieldSelectorInput(BaseModel):
    """Input schema for the field selector tool."""

    question: str = Field(description="The user's natural-language BI question.")
    tables_json: str = Field(
        description=(
            "JSON array of selected table names from the table_selector_tool, "
            'e.g. \'["sales"]\' or \'["sales", "products"]\'.'
        )
    )
    db_schema_json: str = Field(
        description=(
            "Full schema JSON string (same format as passed to table_selector_tool). "
            "Used to surface column definitions and sample values."
        )
    )


@tool(args_schema=FieldSelectorInput)
def field_selector_tool(question: str, tables_json: str, db_schema_json: str) -> str:
    """Select required fields and filter conditions for a BI query.

    Given the user question and selected table(s), this tool returns a
    structured JSON object that describes:

    - ``selected_fields``: list of ``"table.column"`` references to include in
      the SELECT clause.
    - ``filters``: dict mapping ``"table.column"`` to a filter value or
      expression (e.g., ``{"sales.region": "North", "sales.date": ">= 2024-01-01"}``).
    - ``instructions``: guidance for the LLM to populate the above fields.

    The orchestrating LLM fills in ``selected_fields`` and ``filters`` based on
    its analysis of the question and the schema metadata returned here.

    Args:
        question: The user's natural-language BI question.
        tables_json: JSON array of selected table names.
        db_schema_json: Full schema JSON string describing all tables.

    Returns:
        JSON string with column metadata and selection instructions.
    """
    try:
        tables: list[str] = json.loads(tables_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid tables_json: {e}"})

    try:
        schema: dict = json.loads(db_schema_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid db_schema_json: {e}"})

    # Collect column info for each selected table to give the model context.
    table_columns: dict[str, list[dict]] = {}
    for table in tables:
        table_info = schema.get(table, {})
        columns = table_info.get("columns", [])
        # Support both simple list-of-strings and list-of-dicts column formats.
        table_columns[table] = (
            [{"name": c} if isinstance(c, str) else c for c in columns]
        )

    return json.dumps(
        {
            "selected_tables": tables,
            "table_columns": table_columns,
            "instructions": (
                "Based on the question and the table columns listed above, "
                "populate the following fields in your response:\n"
                "- selected_fields: list of 'table.column' references for SELECT\n"
                "- filters: dict of 'table.column' -> filter value/expression\n"
                "- group_by: optional list of 'table.column' for GROUP BY\n"
                "- order_by: optional list of 'table.column' for ORDER BY\n"
                "- limit: optional integer row limit"
            ),
            "question": question,
        },
        ensure_ascii=False,
    )
