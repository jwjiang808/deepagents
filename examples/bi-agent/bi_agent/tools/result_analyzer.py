"""Result analyzer tool for the BI agent pipeline.

This tool is responsible for Step 5 (and final step) of the BI query pipeline:
interpreting the raw query results produced by ``sql_runner_tool`` and
generating a human-readable analytical summary.

The analyzer computes basic descriptive statistics over numeric columns
(count, sum, min, max, mean) and returns them alongside an ``insights``
list that the orchestrating LLM can use to compose a natural-language
answer for the user.

Design notes
------------
- The tool is intentionally statistics-only — it does not produce prose.
  Prose generation is the responsibility of the LLM that invokes the tool
  and reads the structured output.
- Add additional analysis steps (trend detection, outlier flagging, ranking)
  by extending the ``_compute_stats`` helper below.
- For large result sets consider a sampling strategy before computing stats.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ResultAnalyzerInput(BaseModel):
    """Input schema for the result analyzer tool."""

    results_json: str = Field(
        description=(
            "JSON string of query results as returned by sql_runner_tool. "
            'Expected format: {"rows": [...], "row_count": N}.'
        )
    )
    question: str = Field(
        default="",
        description="Original user question (used to contextualise the analysis).",
    )


def _compute_stats(rows: list[dict]) -> dict:
    """Compute descriptive statistics for numeric columns.

    Args:
        rows: List of row dicts from the SQL result.

    Returns:
        Dict mapping column name to a stats dict
        (``count``, ``sum``, ``min``, ``max``, ``mean``).
    """
    if not rows:
        return {}

    numeric_cols: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for col, val in row.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_cols[col].append(float(val))

    stats: dict[str, dict] = {}
    for col, values in numeric_cols.items():
        n = len(values)
        total = sum(values)
        stats[col] = {
            "count": n,
            "sum": round(total, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(total / n, 4) if n else None,
        }
    return stats


def _extract_insights(rows: list[dict], stats: dict) -> list[str]:
    """Derive a short list of factual insights from stats.

    Args:
        rows: Query result rows.
        stats: Descriptive statistics from ``_compute_stats``.

    Returns:
        List of plain-English insight strings.
    """
    insights: list[str] = []
    if not rows:
        insights.append("The query returned no rows.")
        return insights

    insights.append(f"The query returned {len(rows)} row(s).")

    for col, s in stats.items():
        insights.append(
            f"Column '{col}': sum={s['sum']}, mean={s['mean']}, "
            f"min={s['min']}, max={s['max']}."
        )

    # Detect potential data quality issue: NaN/Inf in numeric values.
    for col, s in stats.items():
        for key in ("sum", "mean", "min", "max"):
            val = s.get(key)
            if val is not None and (math.isnan(val) or math.isinf(val)):
                insights.append(
                    f"Warning: column '{col}' contains non-finite values."
                )
                break

    return insights


@tool(args_schema=ResultAnalyzerInput)
def result_analyzer_tool(results_json: str, question: str = "") -> str:
    """Analyze SQL query results and return descriptive statistics and insights.

    Parses the raw JSON output of ``sql_runner_tool``, computes descriptive
    statistics over numeric columns, and surfaces a list of factual insights
    for the LLM to use when composing a natural-language answer.

    Args:
        results_json: JSON string of query results from ``sql_runner_tool``.
        question: Original user question for context.

    Returns:
        JSON string containing:

        - ``row_count``: number of rows in the result set.
        - ``columns``: list of column names.
        - ``stats``: descriptive statistics per numeric column.
        - ``insights``: list of plain-English observations.
        - ``sample_rows``: first 5 rows for spot-checking.
    """
    try:
        data = json.loads(results_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid results_json: {e}"})

    if "error" in data:
        return json.dumps({"error": data["error"]})

    rows: list[dict] = data.get("rows", [])
    row_count: int = data.get("row_count", len(rows))
    columns: list[str] = list(rows[0].keys()) if rows else []

    stats = _compute_stats(rows)
    insights = _extract_insights(rows, stats)

    return json.dumps(
        {
            "row_count": row_count,
            "columns": columns,
            "stats": stats,
            "insights": insights,
            "sample_rows": rows[:5],
            "question": question,
        },
        ensure_ascii=False,
    )
