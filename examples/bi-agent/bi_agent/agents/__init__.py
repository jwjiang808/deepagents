"""BI agent subagent definitions.

Exports the ``SubAgent`` specs and the ``get_all_subagents`` factory used by
``create_bi_agent`` to build the deepagents subagent stack.
"""

from bi_agent.agents.plan_agent import build_plan_agent_spec

__all__ = ["build_plan_agent_spec"]
