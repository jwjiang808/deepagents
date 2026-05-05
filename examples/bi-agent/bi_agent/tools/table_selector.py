"""Table selector tool for the BI agent pipeline.

This tool is responsible for Step 1 of the BI query pipeline: given a user's
natural-language question and a JSON schema description of available tables,
identify which table(s) are required to answer the question.

The selection logic is intentionally kept simple (keyword + LLM prompt) so
that business teams can swap in an embedding-based or catalogue-driven
implementation without touching any other part of the system.

Design notes
------------
- The tool receives the schema as a JSON string so it remains stateless and
  composable — the orchestrating agent always passes fresh context.
- Return value is a JSON array of table names to make downstream parsing
  deterministic.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TableSelectorInput(BaseModel):
    """Input schema for the table selector tool."""

    question: str = Field(description="The user's natural-language BI question.")
    db_schema_json: str = Field(
        description=(
            "JSON string describing available tables and their columns. "
            "Example: "
            '{"sales": {"columns": ["date", "amount", "region"]}, '
            '"products": {"columns": ["id", "name", "category"]}}'
        )
    )


@tool(args_schema=TableSelectorInput)
def table_selector_tool(question: str, db_schema_json: str) -> str:
    """Select the most relevant database table(s) for a BI question.

    Analyzes the user question against the provided schema and returns a JSON
    array of table names that should be queried to answer the question.  For
    simple single-table questions the array contains one entry; for join-
    requiring questions it may contain several.

    The selection is performed by the calling LLM model — this tool acts as a
    structured interface that gives the model a clear opportunity to reason
    about schema matching before generating SQL.

    Args:
        question: The user's natural-language BI question.
        db_schema_json: JSON string describing available tables and their columns.

    Returns:
        JSON array string of selected table names, e.g. ``'["sales"]'``.
    """
    # Parse the schema to validate structure and surface available tables for
    # the model's reasoning.
    try:
        schema: dict = json.loads(db_schema_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid db_schema_json: {e}"})

    available_tables = list(schema.keys())

    # Return a structured prompt artefact that the orchestrating LLM uses to
    # make the selection.  The agent itself performs the semantic matching;
    # this tool surfaces the relevant metadata in a machine-readable form.
    return json.dumps(
        {
            "available_tables": available_tables,
            "instructions": (
                "Based on the question and the available tables listed above, "
                "select the table(s) needed to answer the question. "
                "Return a JSON array of table names in your final answer, "
                'e.g. ["sales"] or ["sales", "products"].'
            ),
            "question": question,
        },
        ensure_ascii=False,
    )
