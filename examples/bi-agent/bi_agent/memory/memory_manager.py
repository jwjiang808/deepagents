"""Memory management for the BI agent system.

The ``MemoryManager`` persists interaction history and user-provided business
rules as a JSON store, and exposes retrieval methods that the orchestrating
agent uses to inject relevant past context before answering a new question.

Extensibility
-------------
The class is designed to be subclassed.  Override ``save`` and ``retrieve``
to plug in a vector-database backend (e.g., Chroma, FAISS, Weaviate) for
semantic similarity search.  The JSON-based default is zero-dependency and
suitable for development and low-traffic deployments.

Relationship to deepagents ``MemoryMiddleware``
-----------------------------------------------
deepagents natively supports file-based memory via ``MemoryMiddleware`` and
the ``memory=["path/to/AGENTS.md"]`` parameter of ``create_deep_agent``.
The ``AGENTS.md`` file in this project's ``memory/`` directory is where the
agent learns persistent rules from user feedback (via its built-in
``edit_file`` tool).

``MemoryManager`` complements that mechanism by providing:
- A queryable interaction log (which questions were asked, what SQL was run,
  what corrections were given).
- A ``get_context_for_question`` method that the ``PlanAgent`` subagent uses
  to surface relevant history as additional context in its system prompt.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum number of records to scan during a keyword-based retrieval.
_MAX_SCAN = 1000


class MemoryManager:
    """Persistent JSON interaction log with keyword-based retrieval.

    Each record stored by ``save`` captures one complete BI interaction:
    question, selected tables/fields, generated SQL, execution result, and
    any user correction or feedback.

    Args:
        store_path: Path to the JSON file used for persistent storage.
            Created automatically if it does not exist.
    """

    def __init__(self, store_path: str | Path = "bi_memory.json") -> None:
        """Initialize the memory manager.

        Args:
            store_path: File path for the JSON interaction log.
        """
        self._path = Path(store_path)
        self._records: list[dict[str, Any]] = self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        """Load records from disk.

        Returns:
            List of interaction records, or an empty list if the file does
            not exist or is empty.
        """
        if not self._path.exists():
            return []
        try:
            content = self._path.read_text(encoding="utf-8").strip()
            return json.loads(content) if content else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load memory store %s: %s", self._path, exc)
            return []

    def _persist(self) -> None:
        """Write the in-memory records list to disk."""
        try:
            self._path.write_text(
                json.dumps(self._records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to persist memory store: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, record: dict[str, Any]) -> None:
        """Append an interaction record to the store.

        Args:
            record: Dict containing interaction data.  Common keys:

                - ``question`` (str): user's question.
                - ``tables`` (list[str]): selected tables.
                - ``fields`` (dict): field selection result.
                - ``sql`` (str): generated SQL.
                - ``result_summary`` (str): brief summary of query result.
                - ``feedback`` (str): user correction or positive confirmation.
        """
        record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._records.append(record)
        # Keep the scan window manageable.
        if len(self._records) > _MAX_SCAN:
            self._records = self._records[-_MAX_SCAN:]
        self._persist()
        logger.debug("Saved memory record for question: %s", record.get("question", ""))

    def retrieve(self, question: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve the most relevant past interactions for a question.

        Uses simple keyword overlap for relevance scoring.  Override this
        method to use embedding-based similarity instead.

        Args:
            question: The current user question.
            top_k: Maximum number of records to return.

        Returns:
            Up to ``top_k`` records ordered by relevance (descending).
        """
        q_tokens = set(question.lower().split())

        def _score(record: dict) -> int:
            text = record.get("question", "").lower()
            return sum(1 for t in q_tokens if t in text)

        scored = sorted(
            ((r, _score(r)) for r in self._records),
            key=lambda x: x[1],
            reverse=True,
        )
        return [r for r, score in scored[:top_k] if score > 0]

    def get_context_for_question(self, question: str, *, top_k: int = 3) -> str:
        """Build a plain-text context snippet for the given question.

        This string is injected into the ``PlanAgent`` system prompt so the
        LLM can leverage past interactions when forming its plan.

        Args:
            question: The current user question.
            top_k: Maximum number of historical records to include.

        Returns:
            Multi-line string summarising relevant past interactions, or an
            empty string if no relevant history exists.
        """
        records = self.retrieve(question, top_k=top_k)
        if not records:
            return ""

        lines = ["## Relevant interaction history\n"]
        for i, rec in enumerate(records, start=1):
            lines.append(f"### Past interaction {i}")
            lines.append(f"**Question**: {rec.get('question', '(unknown)')}")
            if sql := rec.get("sql"):
                lines.append(f"**SQL used**: `{sql}`")
            if feedback := rec.get("feedback"):
                lines.append(f"**User feedback / correction**: {feedback}")
            if summary := rec.get("result_summary"):
                lines.append(f"**Result summary**: {summary}")
            lines.append("")
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all stored records and delete the backing file."""
        self._records = []
        if self._path.exists():
            self._path.unlink()
        logger.info("Memory store cleared.")

    def __len__(self) -> int:  # noqa: D105
        return len(self._records)

    def __repr__(self) -> str:  # noqa: D105
        return f"MemoryManager(store={self._path!r}, records={len(self)})"
