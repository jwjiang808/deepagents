"""BI agent tool implementations.

Each tool in this package corresponds to one step in the BI query pipeline:

1. ``table_selector``   – choose the relevant tables for a user question.
2. ``field_selector``   – select required fields (and field values) per table.
3. ``sql_generator``    – generate a SQL statement from the gathered context.
4. ``sql_runner``       – execute the SQL and return raw results.
5. ``result_analyzer``  – produce a human-readable analysis of query results.

Import all pre-built tools via::

    from bi_agent.tools import get_all_tools
"""

from bi_agent.tools.field_selector import field_selector_tool
from bi_agent.tools.result_analyzer import result_analyzer_tool
from bi_agent.tools.sql_generator import sql_generator_tool
from bi_agent.tools.sql_runner import sql_runner_tool
from bi_agent.tools.table_selector import table_selector_tool


def get_all_tools() -> list:
    """Return the default set of BI pipeline tools.

    Returns:
        List of LangChain ``BaseTool`` instances covering every step of the
        BI query pipeline.
    """
    return [
        table_selector_tool,
        field_selector_tool,
        sql_generator_tool,
        sql_runner_tool,
        result_analyzer_tool,
    ]


__all__ = [
    "field_selector_tool",
    "get_all_tools",
    "result_analyzer_tool",
    "sql_generator_tool",
    "sql_runner_tool",
    "table_selector_tool",
]
