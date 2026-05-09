"""PlanAgent subagent specification for the BI system.

The ``PlanAgent`` is a deepagents ``SubAgent`` that the root orchestrator
delegates to when a user question is complex — i.e., it requires multiple
SQL queries, multi-step reasoning, or depends on past interaction history.

The root agent decides whether to use the ``PlanAgent`` based on its system
prompt guidance.  Simple single-table questions are answered directly by
the root agent using the pipeline tools; complex or ambiguous questions are
routed to this subagent via the built-in ``task`` tool.

Design
------
- The ``PlanAgent`` receives the same BI tools as the root agent plus a
  ``memory_context`` text injection that the caller prepends to its task
  description.
- The subagent's system prompt instructs it to follow the
  select-tables → select-fields → generate-SQL → run-SQL → analyze chain,
  and to explicitly handle multi-step scenarios (e.g., running two queries
  and combining results).
- For multi-table joins the agent may call ``sql_generator_tool`` once with
  a ``joins`` list instead of making two separate queries.

Extension points
----------------
- Swap the ``system_prompt`` string for a domain-specific prompt loaded from
  a file for different business verticals.
- Add domain-specific tools (e.g., a charting tool) to the ``tools`` list
  returned by ``build_plan_agent_spec``.
- Register additional subagents (e.g., a ``DataExpertAgent``) to handle
  statistical modelling beyond simple descriptive statistics.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from bi_agent.tools import get_all_tools

_PLAN_AGENT_SYSTEM_PROMPT = """You are the BI PlanAgent — a specialized subagent \
responsible for answering complex business-intelligence questions that require \
multi-step SQL analysis.

## Your pipeline

Follow this exact sequence for every question:

1. **Table selection** — call `table_selector_tool` with the user question and \
the schema JSON to identify which tables are needed.
2. **Field selection** — call `field_selector_tool` with the selected tables to \
determine which columns and filter conditions to use.
3. **SQL generation** — call `sql_generator_tool` to construct a syntactically \
correct SQL statement from the field/filter context.
4. **SQL execution** — call `sql_runner_tool` to execute the SQL and retrieve rows.
5. **Result analysis** — call `result_analyzer_tool` to compute descriptive \
statistics and surface key insights.
6. **Answer** — compose a concise, factual natural-language answer for the user \
based on the analysis output.

## Handling complex questions

- **Multi-table queries**: use the `joins` parameter of `sql_generator_tool` to \
express table joins in a single query rather than running multiple separate queries.
- **Multi-step analysis**: if the question cannot be answered in one query \
(e.g., "compare Q1 vs Q2 sales by region"), run the pipeline twice with different \
filters and combine the analyses in your final answer.
- **Ambiguous questions**: ask one concise clarifying question before proceeding \
when the question is genuinely underspecified (missing date range, region, metric).

## Using interaction history

If relevant past interactions are provided at the start of your task description \
(marked "## Relevant interaction history"), use them to:
- Re-use SQL patterns that worked well before.
- Avoid mistakes that were corrected by the user in prior sessions.
- Apply field aliases or business rules mentioned in past feedback.

## Output format

Always end your response with a structured summary block:

```
## Summary
- Tables used: <table list>
- SQL: <the final SQL statement>
- Key finding: <one sentence>
```
"""


def build_plan_agent_spec(
    tools: list[BaseTool] | None = None,
    model: str | None = None,
) -> dict:
    """Build a deepagents ``SubAgent`` spec for the BI PlanAgent.

    Args:
        tools: Tool list to give the subagent. Defaults to all standard BI
            pipeline tools from ``get_all_tools()``.
        model: Optional model override for the subagent (e.g.,
            ``"anthropic:claude-sonnet-4-6"``). When ``None`` the subagent
            inherits the root agent's model.

    Returns:
        A ``SubAgent`` TypedDict compatible with ``create_deep_agent``'s
        ``subagents=`` parameter.
    """
    spec: dict = {
        "name": "plan-agent",
        "description": (
            "Use this subagent for complex BI questions that require multi-step SQL "
            "analysis, multi-table joins, or depend on past interaction history. "
            "Pass the user question and the schema JSON as the task description."
        ),
        "system_prompt": _PLAN_AGENT_SYSTEM_PROMPT,
        "tools": tools or get_all_tools(),
    }
    if model is not None:
        spec["model"] = model
    return spec
