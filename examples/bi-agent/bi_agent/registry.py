"""Tool and subagent registry for the BI agent system.

The registry provides a centralized, dynamic catalogue of all LangChain
``BaseTool`` instances and deepagents ``SubAgent`` specs available in the
system. It decouples tool/subagent creation from orchestration logic so that
new capabilities can be added or removed without touching the main entry point.

Usage example::

    from bi_agent.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(my_custom_tool)
    registry.register_subagent(my_custom_subagent)

    tools = registry.get_tools()
    subagents = registry.get_subagents()
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for BI agent tools and subagents.

    Supports dynamic registration so new tools and specialized subagents can
    be plugged in without modifying the core orchestration code.

    Tools must be LangChain ``BaseTool`` instances (or anything accepted by
    ``create_deep_agent``'s ``tools=`` parameter).  Subagents must be
    deepagents ``SubAgent`` TypedDicts (or ``CompiledSubAgent``).
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: dict[str, BaseTool] = {}
        self._subagents: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool under its ``name`` attribute.

        Re-registering under an existing name silently replaces the prior
        entry so callers can hot-swap implementations at runtime.

        Args:
            tool: A LangChain ``BaseTool`` instance to register.
        """
        if tool.name in self._tools:
            logger.debug("Replacing existing tool: %s", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def unregister_tool(self, name: str) -> None:
        """Remove a tool from the registry by name.

        Args:
            name: The tool's ``name`` attribute value.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        if name not in self._tools:
            msg = f"No tool named '{name}' is registered"
            raise KeyError(msg)
        del self._tools[name]
        logger.info("Unregistered tool: %s", name)

    def get_tool(self, name: str) -> BaseTool:
        """Retrieve a registered tool by name.

        Args:
            name: The tool's ``name`` attribute value.

        Returns:
            The registered ``BaseTool`` instance.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        if name not in self._tools:
            msg = f"No tool named '{name}' is registered"
            raise KeyError(msg)
        return self._tools[name]

    def get_tools(self) -> list[BaseTool]:
        """Return all registered tools as an ordered list.

        Returns:
            List of registered ``BaseTool`` instances.
        """
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # Subagent management
    # ------------------------------------------------------------------

    def register_subagent(self, subagent: Any) -> None:
        """Register a deepagents ``SubAgent`` or ``CompiledSubAgent`` spec.

        The ``name`` key of the spec dict (or ``.name`` attribute for
        compiled agents) is used as the registry key.

        Args:
            subagent: A deepagents ``SubAgent`` TypedDict or
                ``CompiledSubAgent`` instance.
        """
        name: str = subagent.get("name") if isinstance(subagent, dict) else subagent.name
        if name in self._subagents:
            logger.debug("Replacing existing subagent: %s", name)
        self._subagents[name] = subagent
        logger.info("Registered subagent: %s", name)

    def unregister_subagent(self, name: str) -> None:
        """Remove a subagent from the registry by name.

        Args:
            name: The subagent's name.

        Raises:
            KeyError: If no subagent with that name is registered.
        """
        if name not in self._subagents:
            msg = f"No subagent named '{name}' is registered"
            raise KeyError(msg)
        del self._subagents[name]
        logger.info("Unregistered subagent: %s", name)

    def get_subagents(self) -> list[Any]:
        """Return all registered subagents as an ordered list.

        Returns:
            List of registered subagent specs.
        """
        return list(self._subagents.values())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        """Return names of all registered tools.

        Returns:
            Sorted list of tool names.
        """
        return sorted(self._tools.keys())

    def list_subagents(self) -> list[str]:
        """Return names of all registered subagents.

        Returns:
            Sorted list of subagent names.
        """
        return sorted(self._subagents.keys())

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ToolRegistry(tools={self.list_tools()}, "
            f"subagents={self.list_subagents()})"
        )
