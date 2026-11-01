"""
Lesson 11 — Offline evals in LangSmith for the Bookstore Support Desk
=====================================================================
Lesson 10 built the graph. This file answers the question you should ask next:
*is the routing decision actually any good?*

Offline eval means: a fixed list of questions whose right answer we already
know, run through the real app, then checked by code. Nothing here is fuzzy.

Three metrics, all deterministic — no judge model, no API cost:

    routing_correct       Did Jev put the message in the lane we labelled?
                          -> tells you whether the HITL gate is load-bearing.
    tool_choice_correct   Did the run touch the capability the answer needs?
                          -> tells you whether the tool descriptions are clear.
    exited_cleanly        Did the subgraph finish, or did the breaker fire?
                          -> tells you whether MAX_TOOL_STEPS is big enough.

A note on what we are measuring. Every example is answered with the human
replying "approve", so the score reflects Jev ALONE. If we let the human
override, we would be grading the human and learn nothing about the model.

Prerequisites (same as lesson 10):
    python bookstore/seed.py
    python -m uvicorn bookstore.orders_api:app  --port 8071
    python -m uvicorn bookstore.reviews_api:app --port 8072

Run:
    python 11_langsmith_evals.py
"""

from __future__ import annotations

import truststore

truststore.inject_into_ssl()

import asyncio
import importlib.util
import os
import pathlib
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, aevaluate

HERE = pathlib.Path(__file__).parent
DATASET_NAME = "bookstore-support-desk"


# ---------------------------------------------------------------------------
# Import the lesson 10 app
# ---------------------------------------------------------------------------
# The module name starts with a digit, so `import 10_...` is a syntax error.
# importlib loads a file by path instead, which sidesteps the naming rule.
def load_lesson10():
    path = HERE / "10_langgraph_jev_HITL_langfuse_tracing.py"
    spec = importlib.util.spec_from_file_location("lesson10", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: lesson 10 uses `from __future__ import annotations`,
    # so LangGraph resolves its TypedDict hints later via sys.modules. Skip this
    # and StateGraph fails with NameError on the module's own imports.
    sys.modules["lesson10"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The labelled dataset
# ---------------------------------------------------------------------------
# Expected tools are written as CAPABILITIES, not tool names, because the two
# lanes name the same capability differently: the prebuilt agent sees the MCP
# tool `get_order`, while laya sees the label `orders_db`. An eval should test
# "did it look at the order records", not "did it call this exact function" —
# otherwise every refactor breaks the eval and teaches you nothing.
CAPABILITY = {
    "get_order": "orders",
    "list_orders_for_customer": "orders",
    "orders_db": "orders",
    "get_book_reviews": "reviews",
    "top_rated_books": "reviews",
    "reviews_db": "reviews",
    "shipping_estimate": "shipping",
    "shipping_calc": "shipping",
    "web_search": "web",
}

EXAMPLES = [
    {
        "question": "Where is my order 1002 and when will it arrive?",
        "lane": "order_help",
        "capabilities": ["orders"],
        "note": "Plain order status. The easy case; Jev should be certain.",
    },
    {
        "question": "Is 'Clean Code' worth reading? What do your readers say?",
        "lane": "book_research",
        "capabilities": ["reviews"],
        "note": "Opinion about a book, answerable from our own review data.",
    },
    {
        "question": "Who wrote 'Sapiens' and what else is that author known for?",
        "lane": "book_research",
        "capabilities": ["web"],
        "note": "We store no author facts, so this must reach the public web.",
    },
    {
        "question": "Please cancel my order 1004.",
        "lane": "order_help",
        "capabilities": ["orders"],
        "note": "Cancellation. Named explicitly in the order_help criteria.",
    },
    {
        "question": "How long does delivery take to Mumbai?",
        "lane": "order_help",
        "capabilities": ["shipping"],
        "note": "Delivery timing with no order id. A genuinely borderline case.",
    },
    {
        "question": "Which books do your readers rate highest?",
        "lane": "book_research",
        "capabilities": ["reviews"],
        "note": "Aggregate over our review data, not about one order.",
    },
    # The six above are all clear-cut, and a set of clear-cut cases will score
    # 100% and teach you nothing. These two are deliberately hard: each one
    # contains wording that pulls towards BOTH lanes. Put cases like this in
    # every dataset — an eval only has value if it is capable of failing.
    {
        "question": "I loved the book from order 1001 — what similar books do your readers recommend?",
        "lane": "book_research",
        "capabilities": ["reviews"],
        "note": "Mentions an order id but wants recommendations. Order words, research intent.",
    },
    {
        "question": "Has Yuval Noah Harari won any awards for his writing?",
        "lane": "book_research",
        "capabilities": ["web"],
        "note": "Pure external fact, no book title to match against our review data.",
    },
]


def ensure_dataset(client: Client) -> None:
    """Create the dataset, or add any examples it is missing.

    Written to be idempotent so the script is safe to re-run. Examples are
    matched by question text, so the existing ones keep their history and
    experiments stay comparable across runs.
    """
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        known = {
            example.inputs.get("question")
            for example in client.list_examples(dataset_id=dataset.id)
        }
        print(f"Dataset '{DATASET_NAME}' exists with {len(known)} examples.")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Bookstore support messages labelled with the lane Jev should pick "
                "and the capability needed to answer them."
            ),
        )
        known = set()
        print(f"Created dataset '{DATASET_NAME}'.")

    new = [example for example in EXAMPLES if example["question"] not in known]
    if not new:
        return

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": example["question"]},
                "outputs": {
                    "lane": example["lane"],
                    "capabilities": example["capabilities"],
                },
                "metadata": {"note": example["note"]},
            }
            for example in new
        ],
    )
    print(f"Added {len(new)} new example(s).")


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------
# A LangSmith evaluator is a plain function. It receives the app's `outputs`
# and the dataset's `reference_outputs`, and returns a key plus a score.
def routing_correct(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["lane"]
    actual = outputs.get("lane")
    return {
        "key": "routing_correct",
        "score": float(actual == expected),
        "comment": f"expected {expected}, Jev chose {actual}",
    }


def tool_choice_correct(outputs: dict, reference_outputs: dict) -> dict:
    """Fraction of the required capabilities that the run actually touched."""
    expected = set(reference_outputs["capabilities"])
    used = {CAPABILITY.get(name, name) for name in outputs.get("tools_used", [])}
    hit = expected & used
    missing = expected - used
    return {
        "key": "tool_choice_correct",
        "score": len(hit) / len(expected) if expected else 1.0,
        "comment": f"used {sorted(used) or ['none']}; missing {sorted(missing) or ['none']}",
    }


def exited_cleanly(outputs: dict) -> dict:
    """The circuit breaker firing is a safety net, not a success."""
    reason = outputs.get("stop_reason", "")
    return {
        "key": "exited_cleanly",
        "score": float(reason != "circuit_breaker"),
        "comment": f"stop_reason={reason or 'unknown'}",
    }


# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------
async def main() -> None:
    client = Client()
    ensure_dataset(client)

    print("\nLoading MCP tools and building the lesson 10 graph...")
    lesson10 = load_lesson10()
    mcp_tools = await lesson10.load_mcp_tools()
    graph = lesson10.build_support_graph(mcp_tools)

    async def target(inputs: dict) -> dict:
        """What LangSmith runs for each dataset example."""
        state = await lesson10.answer_message(
            graph,
            question=inputs["question"],
            human_reply="approve",  # measure Jev, not the human
            thread_id=f"eval-{uuid.uuid4().hex[:8]}",
        )
        return {
            "lane": state["approved_route"],
            "answer": state["answer"],
            "tools_used": state["tool_trail"],
            "findings": state["findings"],
            "stop_reason": state.get("stop_reason", ""),
            "jev_confidence": state["jev_confidence"],
        }

    print("Running experiment...\n")
    results = await aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[routing_correct, tool_choice_correct, exited_cleanly],
        experiment_prefix="lesson10-jev-routing",
        # laya runs locally and the MCP servers are stdio subprocesses, so keep
        # this at 1. Serial runs also make the printed log readable.
        max_concurrency=1,
        metadata={
            "jev_model": lesson10.JEV_MODEL,
            "laya_model": lesson10.LAYA_MODEL,
            "chat_model": lesson10.CHAT_MODEL,
            "max_tool_steps": lesson10.MAX_TOOL_STEPS,
        },
    )

    # LangSmith averages each metric across the run for you, so there is no
    # aggregation code here. Lesson 12 shows the other style, where you write
    # the run-level roll-up yourself.
    print(f"\nExperiment: {results.experiment_name}")
    print(f"Project   : {os.environ.get('LANGSMITH_PROJECT')}")
    print("Open LangSmith -> Datasets -> "
          f"{DATASET_NAME} -> Experiments to compare runs.")


if __name__ == "__main__":
    asyncio.run(main())
