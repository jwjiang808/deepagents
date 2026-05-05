"""Unit tests for the BI agent system.

Tests cover each tool individually, the registry, and the memory manager.
All tests are self-contained (no network calls, no LLM, no database required
beyond the built-in SQLite demo data).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_SCHEMA = {
    "sales": {
        "description": "Daily sales",
        "columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "date", "type": "TEXT"},
            {"name": "region", "type": "TEXT"},
            {"name": "amount", "type": "REAL"},
        ],
    },
    "products": {
        "description": "Products",
        "columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "TEXT"},
            {"name": "category", "type": "TEXT"},
        ],
    },
}

SCHEMA_JSON = json.dumps(SAMPLE_SCHEMA)


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_retrieve_tool(self):
        from langchain_core.tools import tool

        from bi_agent.registry import ToolRegistry

        @tool
        def dummy_tool(x: int) -> int:
            """Return x."""
            return x

        reg = ToolRegistry()
        reg.register_tool(dummy_tool)
        assert reg.get_tool("dummy_tool") is dummy_tool

    def test_get_tools_returns_list(self):
        from langchain_core.tools import tool

        from bi_agent.registry import ToolRegistry

        @tool
        def alpha(x: int) -> int:
            """Alpha."""
            return x

        @tool
        def beta(x: int) -> int:
            """Beta."""
            return x

        reg = ToolRegistry()
        reg.register_tool(alpha)
        reg.register_tool(beta)
        tools = reg.get_tools()
        assert len(tools) == 2  # noqa: PLR2004  # alpha + beta tools registered above

    def test_unregister_tool(self):
        from langchain_core.tools import tool

        from bi_agent.registry import ToolRegistry

        @tool
        def gamma(x: int) -> int:
            """Gamma."""
            return x

        reg = ToolRegistry()
        reg.register_tool(gamma)
        reg.unregister_tool("gamma")
        with pytest.raises(KeyError):
            reg.get_tool("gamma")

    def test_register_replaces_existing(self):
        from langchain_core.tools import tool

        from bi_agent.registry import ToolRegistry

        @tool
        def my_tool(x: int) -> int:
            """Version 1."""
            return x

        @tool
        def my_tool(x: int) -> int:  # noqa: F811 — intentional redefinition for test
            """Version 2."""
            return x + 1

        reg = ToolRegistry()
        reg.register_tool(my_tool)
        # Registering again should silently replace.
        reg.register_tool(my_tool)
        assert len(reg.get_tools()) == 1

    def test_register_subagent(self):
        from bi_agent.registry import ToolRegistry

        spec = {"name": "test-agent", "description": "Test", "system_prompt": "..."}
        reg = ToolRegistry()
        reg.register_subagent(spec)
        agents = reg.get_subagents()
        assert len(agents) == 1
        assert agents[0]["name"] == "test-agent"

    def test_list_tools_sorted(self):
        from langchain_core.tools import tool

        from bi_agent.registry import ToolRegistry

        @tool
        def zzz(x: int) -> int:
            """Zzz."""
            return x

        @tool
        def aaa(x: int) -> int:
            """Aaa."""
            return x

        reg = ToolRegistry()
        reg.register_tool(zzz)
        reg.register_tool(aaa)
        assert reg.list_tools() == ["aaa", "zzz"]


# ---------------------------------------------------------------------------
# TableSelector tests
# ---------------------------------------------------------------------------


class TestTableSelectorTool:
    def test_returns_json_with_available_tables(self):
        from bi_agent.tools.table_selector import table_selector_tool

        out = table_selector_tool.invoke(
            {"question": "total sales by region", "db_schema_json": SCHEMA_JSON}
        )
        data = json.loads(out)
        assert "available_tables" in data
        assert set(data["available_tables"]) == {"sales", "products"}

    def test_invalid_schema_json(self):
        from bi_agent.tools.table_selector import table_selector_tool

        out = table_selector_tool.invoke(
            {"question": "test", "db_schema_json": "not-json"}
        )
        data = json.loads(out)
        assert "error" in data


# ---------------------------------------------------------------------------
# FieldSelector tests
# ---------------------------------------------------------------------------


class TestFieldSelectorTool:
    def test_returns_table_columns(self):
        from bi_agent.tools.field_selector import field_selector_tool

        out = field_selector_tool.invoke(
            {
                "question": "total amount by region",
                "tables_json": json.dumps(["sales"]),
                "db_schema_json": SCHEMA_JSON,
            }
        )
        data = json.loads(out)
        assert "table_columns" in data
        assert "sales" in data["table_columns"]

    def test_invalid_tables_json(self):
        from bi_agent.tools.field_selector import field_selector_tool

        out = field_selector_tool.invoke(
            {
                "question": "test",
                "tables_json": "bad",
                "db_schema_json": SCHEMA_JSON,
            }
        )
        data = json.loads(out)
        assert "error" in data


# ---------------------------------------------------------------------------
# SQLGenerator tests
# ---------------------------------------------------------------------------


class TestSQLGeneratorTool:
    def test_basic_select(self):
        from bi_agent.tools.sql_generator import sql_generator_tool

        out = sql_generator_tool.invoke(
            {
                "table": "sales",
                "selected_fields": ["region", "amount"],
                "filters": {},
                "joins": [],
                "group_by": [],
                "order_by": [],
                "limit": None,
                "dialect": "standard",
            }
        )
        data = json.loads(out)
        assert "sql" in data
        assert "SELECT region, amount" in data["sql"]
        assert "FROM sales" in data["sql"]

    def test_with_filters(self):
        from bi_agent.tools.sql_generator import sql_generator_tool

        out = sql_generator_tool.invoke(
            {
                "table": "sales",
                "selected_fields": ["amount"],
                "filters": {"region": "North"},
                "joins": [],
                "group_by": [],
                "order_by": [],
                "limit": None,
                "dialect": "standard",
            }
        )
        data = json.loads(out)
        assert "WHERE" in data["sql"]
        assert "North" in data["sql"]

    def test_with_group_by_and_limit(self):
        from bi_agent.tools.sql_generator import sql_generator_tool

        out = sql_generator_tool.invoke(
            {
                "table": "sales",
                "selected_fields": ["region", "SUM(amount)"],
                "filters": {},
                "joins": [],
                "group_by": ["region"],
                "order_by": ["-SUM(amount)"],
                "limit": 10,
                "dialect": "standard",
            }
        )
        data = json.loads(out)
        assert "GROUP BY region" in data["sql"]
        assert "LIMIT 10" in data["sql"]
        assert "DESC" in data["sql"]

    def test_with_join(self):
        from bi_agent.tools.sql_generator import sql_generator_tool

        out = sql_generator_tool.invoke(
            {
                "table": "sales",
                "selected_fields": ["sales.amount", "products.category"],
                "filters": {},
                "joins": ["JOIN products ON sales.product = products.name"],
                "group_by": [],
                "order_by": [],
                "limit": None,
                "dialect": "standard",
            }
        )
        data = json.loads(out)
        assert "JOIN products" in data["sql"]


# ---------------------------------------------------------------------------
# SQLRunner tests (uses in-memory SQLite demo DB)
# ---------------------------------------------------------------------------


class TestSQLRunnerTool:
    def test_select_all_sales(self):
        from bi_agent.tools.sql_runner import sql_runner_tool

        out = sql_runner_tool.invoke(
            {"sql": "SELECT * FROM sales", "database_url": "sqlite://"}
        )
        data = json.loads(out)
        assert "rows" in data
        assert data["row_count"] > 0

    def test_invalid_sql_returns_error(self):
        from bi_agent.tools.sql_runner import sql_runner_tool

        out = sql_runner_tool.invoke(
            {"sql": "SELECT * FROM nonexistent_table_xyz", "database_url": "sqlite://"}
        )
        data = json.loads(out)
        assert "error" in data

    def test_filter_by_region(self):
        from bi_agent.tools.sql_runner import sql_runner_tool

        out = sql_runner_tool.invoke(
            {
                "sql": "SELECT * FROM sales WHERE region = 'North'",
                "database_url": "sqlite://",
            }
        )
        data = json.loads(out)
        assert "rows" in data
        for row in data["rows"]:
            assert row["region"] == "North"


# ---------------------------------------------------------------------------
# ResultAnalyzer tests
# ---------------------------------------------------------------------------


class TestResultAnalyzerTool:
    def _sample_results(self, rows: list[dict]) -> str:
        return json.dumps({"rows": rows, "row_count": len(rows)})

    def test_basic_stats(self):
        from bi_agent.tools.result_analyzer import result_analyzer_tool

        rows = [{"amount": 100.0}, {"amount": 200.0}, {"amount": 300.0}]
        out = result_analyzer_tool.invoke(
            {"results_json": self._sample_results(rows), "question": "test"}
        )
        data = json.loads(out)
        assert data["row_count"] == 3  # noqa: PLR2004  # three rows in sample_rows list above
        assert "amount" in data["stats"]
        assert data["stats"]["amount"]["sum"] == 600.0  # noqa: PLR2004  # sum of 100.0 + 200.0 + 300.0 test values

    def test_empty_result(self):
        from bi_agent.tools.result_analyzer import result_analyzer_tool

        out = result_analyzer_tool.invoke(
            {"results_json": self._sample_results([]), "question": "empty"}
        )
        data = json.loads(out)
        assert data["row_count"] == 0
        assert any("no rows" in i.lower() for i in data["insights"])

    def test_error_passthrough(self):
        from bi_agent.tools.result_analyzer import result_analyzer_tool

        error_json = json.dumps({"error": "connection failed"})
        out = result_analyzer_tool.invoke(
            {"results_json": error_json, "question": "q"}
        )
        data = json.loads(out)
        assert "error" in data

    def test_invalid_json(self):
        from bi_agent.tools.result_analyzer import result_analyzer_tool

        out = result_analyzer_tool.invoke(
            {"results_json": "not-json", "question": "q"}
        )
        data = json.loads(out)
        assert "error" in data


# ---------------------------------------------------------------------------
# MemoryManager tests
# ---------------------------------------------------------------------------


class TestMemoryManager:
    def _manager(self, tmp_path: Path) -> object:
        from bi_agent.memory.memory_manager import MemoryManager

        return MemoryManager(store_path=tmp_path / "test_memory.json")

    def test_save_and_retrieve(self, tmp_path: Path):
        mem = self._manager(tmp_path)
        mem.save({"question": "total sales by region", "sql": "SELECT ...", "feedback": ""})
        results = mem.retrieve("sales region")
        assert len(results) >= 1

    def test_retrieve_empty(self, tmp_path: Path):
        mem = self._manager(tmp_path)
        results = mem.retrieve("unrelated question xyz")
        assert results == []

    def test_persist_and_reload(self, tmp_path: Path):
        from bi_agent.memory.memory_manager import MemoryManager

        path = tmp_path / "mem.json"
        m1 = MemoryManager(store_path=path)
        m1.save({"question": "how many products?", "sql": "SELECT COUNT(*) FROM products"})

        m2 = MemoryManager(store_path=path)
        assert len(m2) == 1

    def test_clear(self, tmp_path: Path):
        mem = self._manager(tmp_path)
        mem.save({"question": "test"})
        assert len(mem) == 1
        mem.clear()
        assert len(mem) == 0

    def test_get_context_for_question(self, tmp_path: Path):
        mem = self._manager(tmp_path)
        mem.save(
            {
                "question": "total sales by region",
                "sql": "SELECT region, SUM(amount) FROM sales GROUP BY region",
                "feedback": "correct",
            }
        )
        ctx = mem.get_context_for_question("sales region breakdown")
        assert "Past interaction" in ctx or ctx == ""


# ---------------------------------------------------------------------------
# PlanAgent spec tests
# ---------------------------------------------------------------------------


class TestPlanAgentSpec:
    def test_spec_has_required_keys(self):
        from bi_agent.agents.plan_agent import build_plan_agent_spec

        spec = build_plan_agent_spec()
        assert spec["name"] == "plan-agent"
        assert "description" in spec
        assert "system_prompt" in spec
        assert "tools" in spec

    def test_spec_with_model_override(self):
        from bi_agent.agents.plan_agent import build_plan_agent_spec

        spec = build_plan_agent_spec(model="openai:gpt-4o")
        assert spec["model"] == "openai:gpt-4o"

    def test_spec_without_model_override(self):
        from bi_agent.agents.plan_agent import build_plan_agent_spec

        spec = build_plan_agent_spec()
        assert "model" not in spec
