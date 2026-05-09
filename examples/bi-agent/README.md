# BI Agent Demo

An extensible **Business Intelligence multi-agent system** built on top of
[deepagents](https://github.com/jwjiang808/deepagents).  It answers natural-
language BI questions by orchestrating a five-step pipeline of specialised
tools, supports complex multi-step analysis via a planning subagent, and learns
from user feedback through a persistent long-term memory file.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-tool pipeline** | Five focused tools — table selector, field selector, SQL generator, SQL runner, result analyzer — each with a single clear responsibility |
| **Dynamic registry** | Add, replace, or remove tools and subagents at runtime without changing core orchestration code |
| **PlanAgent subagent** | Handles complex or multi-table questions via deepagents' built-in `task` delegation mechanism |
| **Long-term memory** | Persistent `AGENTS.md` memory file; the agent edits it when the user provides corrections, so rules survive across sessions |
| **Extensible memory** | `MemoryManager` (JSON log) is designed to be subclassed with a vector-DB backend for semantic retrieval |
| **Any LLM** | Provider-agnostic — pass any `provider:model` string or pre-initialised `BaseChatModel` |

---

## Project structure

```text
examples/bi-agent/
├── main.py                        # CLI entry point
├── pyproject.toml
├── .env.example
├── memory/
│   └── AGENTS.md                  # Persistent long-term memory (editable by agent)
└── bi_agent/
    ├── __init__.py                # create_bi_agent() — main factory
    ├── registry.py                # ToolRegistry — dynamic tool/subagent management
    ├── tools/
    │   ├── __init__.py            # get_all_tools()
    │   ├── table_selector.py      # Step 1: choose relevant tables
    │   ├── field_selector.py      # Step 2: choose columns and filters
    │   ├── sql_generator.py       # Step 3: assemble SQL statement
    │   ├── sql_runner.py          # Step 4: execute SQL, return rows
    │   └── result_analyzer.py     # Step 5: descriptive stats + insights
    ├── agents/
    │   ├── __init__.py
    │   └── plan_agent.py          # PlanAgent subagent for complex questions
    ├── memory/
    │   ├── __init__.py
    │   └── memory_manager.py      # JSON interaction log with keyword retrieval
    └── schemas/
        └── sample_schema.json     # Demo BI schema (sales + products tables)
```

---

## Quick start

### 1. Install dependencies

```bash
cd examples/bi-agent
uv pip install -e .
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

### 3. Run a single question

```bash
python main.py "What were total sales in Q1 2024?"
```

### 4. Interactive mode

```bash
python main.py
```

### 5. Use a different model

```bash
python main.py --model openai:gpt-4o "Top 5 products by revenue?"
```

### 6. Use your own database

```bash
DATABASE_URL=sqlite:///./mydb.db python main.py
```

---

## How it works

### Simple questions

The root agent runs the pipeline tools directly in sequence:

```
User question
  → table_selector_tool   (which tables?)
  → field_selector_tool   (which columns / filters?)
  → sql_generator_tool    (build SQL)
  → sql_runner_tool       (execute SQL)
  → result_analyzer_tool  (stats + insights)
  → natural-language answer
```

### Complex questions

The root agent delegates to `plan-agent` via deepagents' `task` tool.  The
`PlanAgent` follows the same pipeline but can chain multiple SQL queries,
apply joins, and consult past-interaction history before forming its plan.

### Long-term memory

The `memory/AGENTS.md` file is loaded into the agent's system prompt at
startup via deepagents' built-in `MemoryMiddleware`.  When the user corrects
a query ("that's wrong — use the `products` table instead"), the agent calls
`edit_file` to update `AGENTS.md` immediately.  The next session starts with
the corrected rule already in memory.

---

## Extending the system

### Add a custom tool

```python
from langchain_core.tools import tool
from bi_agent import create_bi_agent
from bi_agent.registry import ToolRegistry

@tool
def forecast_tool(metric: str, periods: int) -> str:
    """Forecast a BI metric for the next N periods."""
    ...  # your implementation

registry = ToolRegistry()
registry.register_tool(forecast_tool)

agent = create_bi_agent(schema=schema, registry=registry)
```

### Add a custom subagent

```python
from bi_agent.registry import ToolRegistry

anomaly_agent = {
    "name": "anomaly-detector",
    "description": "Detect statistical anomalies in query results.",
    "system_prompt": "You are a statistical anomaly detection agent ...",
    "tools": [],
}

registry = ToolRegistry()
registry.register_subagent(anomaly_agent)

agent = create_bi_agent(schema=schema, registry=registry)
```

### Swap in a vector-DB memory backend

Subclass `MemoryManager` and override `save` and `retrieve`:

```python
from bi_agent.memory import MemoryManager
import chromadb

class ChromaMemoryManager(MemoryManager):
    def __init__(self):
        self._client = chromadb.Client()
        self._col = self._client.get_or_create_collection("bi_memory")

    def save(self, record):
        self._col.add(
            documents=[record["question"]],
            metadatas=[record],
            ids=[record.get("timestamp", str(len(self._col.get()["ids"])))],
        )

    def retrieve(self, question, *, top_k=3):
        results = self._col.query(query_texts=[question], n_results=top_k)
        return results["metadatas"][0] if results["metadatas"] else []
```

Then pass your custom manager to `create_bi_agent`:

```python
agent = create_bi_agent(schema=schema, memory_manager=ChromaMemoryManager())
```

### Use your own schema

Create a JSON file following the `sample_schema.json` structure and pass it
at startup:

```bash
python main.py --schema path/to/my_schema.json "Monthly revenue by category?"
```

---

## API / Web service integration

`create_bi_agent` returns a standard LangGraph `CompiledStateGraph`.  Wrap it
in any ASGI framework:

```python
from fastapi import FastAPI
from bi_agent import create_bi_agent
import json

app = FastAPI()
schema = json.loads(open("bi_agent/schemas/sample_schema.json").read())
agent = create_bi_agent(schema=schema)

@app.post("/ask")
async def ask(question: str):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return {"answer": result["messages"][-1].content}
```
