"""Run the DAG agent against the LiteLLM gateways.

    python run_dag_agent.py "What is the weather in Tokyo and what is 23 times 19?"
    python run_dag_agent.py --demo
"""

from __future__ import annotations

import argparse
import sys
import uuid

from dag_agent import build_agent
from dag_agent.observability import flush

# Windows consoles default to cp1252 and mangle degree signs in agent replies.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEMOS = [
    "What is 23 multiplied by 19, and what is the weather in Tokyo?",
    "Convert 250 USD to Japanese yen, and tell me the local time in Tokyo.",
    "Add 17 and 25, then multiply that result by 3.",
    "Who wrote the novel Dune?",
]


def run(question: str, thread_id: str | None = None) -> None:
    print("=" * 78)
    print("Q:", question)
    print("=" * 78)

    agent = build_agent()
    final = agent.invoke(
        {"question": question, "trace": []},
        config={"configurable": {"thread_id": thread_id or f"cli-{uuid.uuid4().hex[:8]}"}},
    )

    print("\n--- trace ---")
    for line in final.get("trace", []):
        print(" ", line)

    dag = final.get("dag") or []
    if dag:
        print("\n--- DAG ---")
        for s in dag:
            dep = f" after {s['depends_on']}" if s.get("depends_on") else ""
            print(f"  {s['id']}{dep}: {s['description']}")
            print(f"      -> {s['executor']}:{s.get('target')} (jev {s.get('jev_confidence', 0):.2f})")

    print("\n--- answer ---")
    print(final.get("answer", "<none>"))
    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*", help="the question to answer")
    p.add_argument("--demo", action="store_true", help="run the built-in demo questions")
    p.add_argument("--thread", help="conversation id; reuses history stored in LiteLLM memory")
    args = p.parse_args()

    if args.demo:
        for q in DEMOS:
            run(q, args.thread)
    elif args.question:
        run(" ".join(args.question), args.thread)
    else:
        p.print_help()
        return 1
    flush()  # short-lived process: push buffered spans before exiting
    return 0


if __name__ == "__main__":
    sys.exit(main())
