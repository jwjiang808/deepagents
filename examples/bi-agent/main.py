"""BI Agent Demo — interactive CLI entry point.

Launches an interactive question-and-answer session with the BI multi-agent
system.  The agent persists learned rules across sessions via the
``memory/AGENTS.md`` file.

Usage::

    # Install dependencies first
    uv pip install -e .

    # Set your LLM API key
    export ANTHROPIC_API_KEY=sk-ant-...

    # Run a single question
    python main.py "What were total sales in Q1 2024?"

    # Run in interactive mode
    python main.py

    # Use a custom database
    DATABASE_URL=sqlite:///./mydb.db python main.py

    # Use a different model
    python main.py --model openai:gpt-4o "Top 5 products by revenue?"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Load .env before importing the agent so API keys are available.
load_dotenv()

from bi_agent import create_bi_agent  # noqa: E402 — must follow load_dotenv()

console = Console()
logger = logging.getLogger(__name__)

# Default schema bundled with the demo.
_DEFAULT_SCHEMA_PATH = (
    Path(__file__).parent / "bi_agent" / "schemas" / "sample_schema.json"
)


def _load_schema(schema_path: str | Path) -> dict:
    """Load and parse a JSON schema file.

    Args:
        schema_path: Path to the JSON schema file.

    Returns:
        Parsed schema dict.

    Raises:
        SystemExit: If the file does not exist or contains invalid JSON.
    """
    path = Path(schema_path)
    if not path.exists():
        console.print(f"[bold red]Schema file not found:[/bold red] {path}")
        sys.exit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]Invalid JSON in schema file:[/bold red] {exc}")
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="BI Multi-Agent Demo powered by deepagents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What were total sales in Q1 2024?"
  python main.py --model openai:gpt-4o "Top 5 products by revenue?"
  python main.py --schema path/to/custom_schema.json
  python main.py  # interactive mode
        """,
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="BI question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("BI_AGENT_MODEL", "anthropic:claude-sonnet-4-6"),
        help="LLM model in 'provider:model' format (default: anthropic:claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--schema",
        default=str(_DEFAULT_SCHEMA_PATH),
        help="Path to the JSON schema file",
    )
    parser.add_argument(
        "--memory",
        default=None,
        help="Path to the AGENTS.md memory file (default: memory/AGENTS.md)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable LangGraph debug logging",
    )
    return parser.parse_args()


def _ask(agent: object, question: str) -> str:
    """Invoke the agent with a question and return the final answer.

    Args:
        agent: Compiled deepagents graph.
        question: User's natural-language question.

    Returns:
        The agent's final response text.
    """
    result = agent.invoke(  # type: ignore[attr-defined]
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 50},
    )
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


def _run_interactive(agent: object) -> None:
    """Run an interactive Q&A loop until the user exits.

    Args:
        agent: Compiled deepagents graph.
    """
    console.print(
        Panel(
            "[bold cyan]BI Multi-Agent Interactive Session[/bold cyan]\n"
            "Type your BI question and press Enter.  "
            "Type [bold]exit[/bold] or press Ctrl-C to quit.",
            border_style="cyan",
        )
    )
    while True:
        try:
            question = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        with console.status("[dim]Thinking…[/dim]"):
            try:
                answer = _ask(agent, question)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[bold red]Error:[/bold red] {exc}")
                continue

        console.print(
            Panel(answer, title="[bold blue]BI Agent[/bold blue]", border_style="blue")
        )


def main() -> None:
    """Main entry point."""
    if "--debug" not in sys.argv:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.DEBUG)

    args = _parse_args()
    schema = _load_schema(args.schema)

    console.print("[dim]Initialising BI agent…[/dim]")
    agent = create_bi_agent(
        schema=schema,
        model=args.model,
        memory_path=args.memory,
        debug=args.debug,
    )
    console.print("[dim]Agent ready.[/dim]\n")

    if args.question:
        # Single-shot mode.
        console.print(
            Panel(
                f"[bold cyan]Question:[/bold cyan] {args.question}",
                border_style="cyan",
            )
        )
        try:
            answer = _ask(agent, args.question)
        except Exception as exc:  # noqa: BLE001
            console.print(
                Panel(f"[bold red]Error:[/bold red]\n{exc}", border_style="red")
            )
            sys.exit(1)

        console.print(
            Panel(
                f"[bold green]Answer:[/bold green]\n\n{answer}",
                border_style="green",
            )
        )
    else:
        # Interactive mode.
        _run_interactive(agent)


if __name__ == "__main__":
    main()
