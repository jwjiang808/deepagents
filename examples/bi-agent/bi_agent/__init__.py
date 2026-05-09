"""BI Agent — main entry point module.

``create_bi_agent`` is the single public factory that assembles the complete
BI multi-agent system from its constituent tools, subagents, and memory
configuration.  It delegates the agent graph construction entirely to
``deepagents.create_deep_agent`` so all deepagents features (planning,
reflection, built-in filesystem tools, summarisation middleware, etc.) are
available out of the box.

Usage::

    from bi_agent import create_bi_agent
    import json

    schema = json.loads(Path("bi_agent/schemas/sample_schema.json").read_text())
    agent = create_bi_agent(schema=schema)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "What were total sales in Q1 2024?"}]
    })
    print(result["messages"][-1].content)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.tools import BaseTool

from bi_agent.agents.plan_agent import build_plan_agent_spec
from bi_agent.memory.memory_manager import MemoryManager
from bi_agent.registry import ToolRegistry
from bi_agent.tools import get_all_tools

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_PATH = Path(__file__).parent.parent / "memory" / "AGENTS.md"

_BI_SYSTEM_PROMPT = """You are an expert BI (Business Intelligence) assistant \
powered by a multi-agent pipeline.

## Your capabilities

You have access to the following BI pipeline tools:

| Tool | Purpose |
|------|---------|
| `table_selector_tool` | Choose which table(s) to query based on the user question |
| `field_selector_tool` | Identify required columns and filter conditions |
| `sql_generator_tool` | Construct a SQL SELECT statement |
| `sql_runner_tool` | Execute the SQL and retrieve results |
| `result_analyzer_tool` | Compute descriptive statistics and surface insights |

For **simple, single-table questions** execute the pipeline steps above \
sequentially in one turn.

For **complex questions** (multi-table, multi-step, requiring past context) \
delegate to the `plan-agent` subagent via the `task` tool. Pass the user \
question AND the schema JSON as the task description so the subagent has full \
context.

## Schema awareness

Always use the schema provided in your memory/context. Never invent table or \
column names that are not in the schema.

## Learning from feedback

When the user corrects a query or provides new business rules:
1. Acknowledge the correction.
2. Update your `memory/AGENTS.md` file immediately using `edit_file` so the \
rule is remembered in future sessions.
3. Re-run the corrected query.

## Response style

- Be concise and factual.
- Show the generated SQL when answering.
- Highlight the most relevant metric or finding.
- If the result set is empty, say so and suggest possible reasons.
"""


def create_bi_agent(
    schema: dict[str, Any],
    *,
    model: str = "anthropic:claude-sonnet-4-6",
    memory_path: str | Path | None = None,
    memory_manager: MemoryManager | None = None,
    extra_tools: list[BaseTool] | None = None,
    registry: ToolRegistry | None = None,
    plan_agent_model: str | None = None,
    debug: bool = False,
) -> Any:
    """Create the BI multi-agent system.

    Assembles a deepagents graph with:

    - The five BI pipeline tools (table selector → field selector →
      SQL generator → SQL runner → result analyzer).
    - A ``plan-agent`` subagent for complex multi-step queries.
    - File-based long-term memory (``memory/AGENTS.md``) so the agent can
      learn from user feedback across sessions.
    - A ``ToolRegistry`` for dynamic tool management.

    Args:
        schema: Dict describing available tables and columns.  This is
            injected into the system prompt so every tool call has access to
            the schema definition.

            Example::

                {
                    "sales": {
                        "columns": ["date", "region", "amount"],
                        "description": "Daily sales transactions"
                    }
                }

        model: Model identifier string in ``provider:model`` format.
            Defaults to ``"anthropic:claude-sonnet-4-6"``.
        memory_path: Path to the ``AGENTS.md`` file used for persistent memory.
            Defaults to ``memory/AGENTS.md`` inside the project directory.
        memory_manager: Optional custom ``MemoryManager`` instance.  When
            provided, ``get_context_for_question`` output is NOT automatically
            injected (the caller is responsible for passing context).  Present
            for extension scenarios where callers manage the interaction loop.
        extra_tools: Additional ``BaseTool`` instances to register beyond the
            default BI pipeline tools.
        registry: Pre-configured ``ToolRegistry``.  When provided, its tools
            and subagents are merged with the defaults.
        plan_agent_model: Optional model override for the ``plan-agent``
            subagent.  Defaults to inheriting the root agent model.
        debug: Enable LangGraph debug logging.

    Returns:
        A compiled deepagents graph (``CompiledStateGraph``) ready to invoke.
    """
    # ------------------------------------------------------------------
    # 1. Resolve tools
    # ------------------------------------------------------------------
    reg = registry or ToolRegistry()

    # Register the default BI pipeline tools.
    for t in get_all_tools():
        reg.register_tool(t)

    # Register any caller-supplied extra tools.
    for t in extra_tools or []:
        reg.register_tool(t)

    all_tools = reg.get_tools()

    # ------------------------------------------------------------------
    # 2. Resolve subagents
    # ------------------------------------------------------------------
    plan_spec = build_plan_agent_spec(
        tools=all_tools,
        model=plan_agent_model,
    )
    reg.register_subagent(plan_spec)
    all_subagents = reg.get_subagents()

    # ------------------------------------------------------------------
    # 3. Build system prompt with schema context
    # ------------------------------------------------------------------
    schema_block = (
        "\n\n## Available schema\n\n"
        "```json\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n```"
    )
    system_prompt = _BI_SYSTEM_PROMPT + schema_block

    # ------------------------------------------------------------------
    # 4. Resolve memory sources
    # ------------------------------------------------------------------
    mem_path = Path(memory_path) if memory_path else _DEFAULT_MEMORY_PATH
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    if not mem_path.exists():
        _init_agents_md(mem_path, schema)

    memory_sources = [str(mem_path)]

    # ------------------------------------------------------------------
    # 5. Assemble the deep agent
    # ------------------------------------------------------------------
    agent = create_deep_agent(
        model=model,
        tools=all_tools,
        system_prompt=system_prompt,
        subagents=all_subagents,
        memory=memory_sources,
        debug=debug,
    )

    logger.info(
        "BI agent created — tools: %s, subagents: %s",
        reg.list_tools(),
        reg.list_subagents(),
    )
    return agent


def _init_agents_md(path: Path, schema: dict[str, Any]) -> None:
    """Write an initial ``AGENTS.md`` memory file with schema context.

    Args:
        path: File path to create.
        schema: Schema dict to embed in the file.
    """
    schema_summary = "\n".join(
        f"- **{table}**: {', '.join(info.get('columns', []) if isinstance(info.get('columns', []), list) else list(info.get('columns', {}).keys()))}"
        for table, info in schema.items()
    )
    content = f"""# BI Agent Memory

## Schema overview

{schema_summary}

## Business rules and learned corrections

*(This section is updated automatically when the user provides corrections or
business rules during a session.  Do not edit manually.)*

"""
    path.write_text(content, encoding="utf-8")
    logger.info("Initialised AGENTS.md at %s", path)
