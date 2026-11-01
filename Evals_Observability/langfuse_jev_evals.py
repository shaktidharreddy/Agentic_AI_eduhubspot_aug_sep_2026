"""
Lesson 12 — Langfuse experiments judged by Jev
==============================================
Lesson 11 checked the things code can check: the lane, the tools, the exit.
Two questions are left over, and neither one is a string comparison:

    answer_grounded       Does the answer stick to what the tools returned,
                          or did the chat model fill a gap from memory?
    answer_completeness   Did it answer everything the customer asked, or
                          only part of it?

These need judgement, so we need a judge. The usual choice is a big chat model
("LLM-as-a-judge"), but for a narrow, repeated question with a known answer set
that is the wrong tool. Jev is built for exactly this shape, and its input
tokens cost a fraction of a frontier model's, which is what makes it realistic
to score ALL production traffic instead of a 1% sample.

    https://langfuse.com/blog/2026-09-22-running-evals-with-jev

Two Jev primitives do the work:

    noul   -> yes/no. Returns P(true) and NO confidence field. Reading
              answer["confidence"] on a noul will KeyError; that is the one
              gotcha when you mix primitives.
    score  -> an ordered rubric. `criteria` is a LIST, lowest level first, and
              you get back a probability-weighted value plus the distribution.

Note the shape of each question below: one atomic thing to decide, and an
explicit rule for every verdict. Jev judges the words you wrote, not the
intent behind them, so vague criteria give you vague scores.

Prerequisites (same as lesson 10):
    python bookstore/seed.py
    python -m uvicorn bookstore.orders_api:app  --port 8071
    python -m uvicorn bookstore.reviews_api:app --port 8072

Run:
    python 12_langfuse_jev_evals.py
"""

from __future__ import annotations

import truststore

truststore.inject_into_ssl()

import asyncio
import importlib.util
import pathlib
import sys
import uuid

from dotenv import load_dotenv

# result.format() prints emoji. The default Windows console codepage cannot
# encode them, which would crash the script after the experiment had run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()  # Langfuse must be imported AFTER the keys are loaded.

from langfuse import Evaluation, get_client

HERE = pathlib.Path(__file__).parent
DATASET_NAME = "bookstore-support-desk"

langfuse = get_client()


def load_lesson10():
    """Load lesson 10 by path; its filename starts with a digit."""
    path = HERE / "10_langgraph_jev_HITL_langfuse_tracing.py"
    spec = importlib.util.spec_from_file_location("lesson10", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: lesson 10 uses `from __future__ import annotations`,
    # so LangGraph resolves its TypedDict hints later via sys.modules.
    sys.modules["lesson10"] = module
    spec.loader.exec_module(module)
    return module


lesson10 = load_lesson10()


# ---------------------------------------------------------------------------
# The evaluation dataset, hosted in Langfuse
# ---------------------------------------------------------------------------
# Same questions as lesson 11, so the two backends are measuring one app.
# Here the label is the lane only; grounding and completeness need no label,
# which is exactly why they can also run on live production traffic.
EVAL_ITEMS = [
    ("Where is my order 1002 and when will it arrive?", "order_help"),
    ("Is 'Clean Code' worth reading? What do your readers say?", "book_research"),
    ("Who wrote 'Sapiens' and what else is that author known for?", "book_research"),
    ("Please cancel my order 1004.", "order_help"),
    ("How long does delivery take to Mumbai?", "order_help"),
    ("Which books do your readers rate highest?", "book_research"),
    # The last two are deliberately hard, and the grounding score is the reason
    # they matter: when the tools cannot answer, a good reply says so, and a
    # bad one quietly invents the missing fact. That is precisely the failure
    # a judge catches and an exact-match check cannot.
    ("I loved the book from order 1001 — what similar books do your readers recommend?", "book_research"),
    ("Has Yuval Noah Harari won any awards for his writing?", "book_research"),
]


def ensure_dataset() -> None:
    """Create the Langfuse dataset and its items, idempotently."""
    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
        known = {item.input.get("question") for item in dataset.items}
        print(f"Langfuse dataset '{DATASET_NAME}' exists with {len(known)} items.")
    except Exception:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="Bookstore support messages with the expected support lane.",
        )
        known = set()
        print(f"Created Langfuse dataset '{DATASET_NAME}'.")

    for question, lane in EVAL_ITEMS:
        if question in known:
            continue
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={"question": question},
            expected_output={"lane": lane},
        )
        print(f"  + added: {question[:56]}")


# ---------------------------------------------------------------------------
# Jev as the judge
# ---------------------------------------------------------------------------
GROUNDED_QUESTION = {
    "type": "noul",
    "instructions": (
        "Does the assistant's answer stay within the tool findings, making no "
        "factual claim that the findings do not contain?"
    ),
    "criteria": {
        "true": (
            "Every order detail, rating, title, name and date in the answer also "
            "appears in the findings. Saying the findings are thin or missing counts as true."
        ),
        "false": (
            "The answer asserts at least one specific fact — a status, a rating, an "
            "author, a date — that does not appear anywhere in the findings."
        ),
    },
}

COMPLETENESS_QUESTION = {
    "type": "score",
    "instructions": "How much of what the customer asked for does the answer actually provide?",
    # A score's criteria is an ORDERED LIST: index 0 is the worst level.
    "criteria": [
        "None of it. The answer is off-topic, or refuses without giving any information.",
        "Some of it. It addresses part of the question but leaves something asked for unanswered.",
        "All of it. Every part of the customer's question is answered.",
    ],
}


def judge_with_jev(question: str, findings: list[str], answer: str) -> dict:
    """Ask Jev both judging questions in ONE call.

    Concurrent questions over a shared state are a real advantage here: the two
    judgements are computed independently, so neither one contaminates the
    other the way a single chat-model prompt asking for both would.
    """
    state = {
        "customer_question": question,
        "tool_findings": findings or ["(no tool findings were gathered)"],
        "assistant_answer": answer,
    }
    body = lesson10.ask_jev(
        state,
        {"grounded": GROUNDED_QUESTION, "completeness": COMPLETENESS_QUESTION},
    )
    return body["answers"]


# ---------------------------------------------------------------------------
# Item-level evaluators
# ---------------------------------------------------------------------------
# A Langfuse evaluator is a keyword-only function returning an Evaluation.
# The name becomes the score name in the UI, so name it after what is
# measured (`answer_grounded`), never after the evaluator itself.
def jev_judge(*, input, output, expected_output=None, metadata=None, **kwargs):
    """One Jev call, two scores. Returning a list is allowed and avoids paying twice."""
    answers = judge_with_jev(
        question=input["question"],
        findings=output["findings"],
        answer=output["answer"],
    )

    # noul gives P(true) directly and carries no confidence field.
    grounded = float(answers["grounded"]["noul"])
    # score gives a probability-weighted value over the rubric; divide by the
    # top index to land in 0..1 so it charts next to the other metrics.
    completeness = answers["completeness"]
    top = len(COMPLETENESS_QUESTION["criteria"]) - 1
    normalised = float(completeness["score"]) / top

    return [
        Evaluation(
            name="answer_grounded",
            value=grounded,
            comment=f"P(grounded)={grounded:.2f} over {len(output['findings'])} findings",
        ),
        Evaluation(
            name="answer_completeness",
            value=normalised,
            comment=(
                f"raw={completeness['score']:.2f}/{top} "
                f"confidence={completeness['confidence']:.2f} "
                f"dist={completeness['probabilities']}"
            ),
        ),
    ]


def routing_correct(*, input, output, expected_output=None, **kwargs):
    """Kept here too, so one Langfuse run shows both judged and exact metrics."""
    expected = (expected_output or {}).get("lane")
    return Evaluation(
        name="routing_correct",
        value=float(output["lane"] == expected),
        comment=f"expected {expected}, Jev chose {output['lane']}",
    )


# ---------------------------------------------------------------------------
# Run-level evaluators
# ---------------------------------------------------------------------------
# These see every item at once and attach their scores to the run itself.
# Useful for the numbers you would actually put in a release decision.
def _values(item_results, name: str) -> list[float]:
    return [
        evaluation.value
        for result in item_results
        for evaluation in result.evaluations
        if evaluation.name == name and evaluation.value is not None
    ]


def mean_grounded(*, item_results, **kwargs):
    values = _values(item_results, "answer_grounded")
    if not values:
        return Evaluation(name="mean_grounded", value=None)
    mean = sum(values) / len(values)
    return Evaluation(name="mean_grounded", value=mean, comment=f"{mean:.2%} over {len(values)} items")


def ungrounded_count(*, item_results, **kwargs):
    """The number that matters operationally: how many answers we cannot trust.

    A mean hides a single bad answer among good ones. A count does not, which
    is why a release gate should read this one.
    """
    values = _values(item_results, "answer_grounded")
    bad = [value for value in values if value < 0.5]
    return Evaluation(
        name="ungrounded_answers",
        value=len(bad),
        comment=f"{len(bad)} of {len(values)} answers made an unsupported claim",
    )


def mean_completeness(*, item_results, **kwargs):
    values = _values(item_results, "answer_completeness")
    if not values:
        return Evaluation(name="mean_completeness", value=None)
    mean = sum(values) / len(values)
    return Evaluation(name="mean_completeness", value=mean, comment=f"{mean:.2%} of the rubric")


# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------
async def main() -> None:
    ensure_dataset()

    print("\nLoading MCP tools and building the lesson 10 graph...")
    mcp_tools = await lesson10.load_mcp_tools()
    graph = lesson10.build_support_graph(mcp_tools)

    async def task(*, item, **kwargs) -> dict:
        """What Langfuse runs per dataset item. It traces itself automatically."""
        state = await lesson10.answer_message(
            graph,
            question=item.input["question"],
            human_reply="approve",  # measure Jev, not the human
            thread_id=f"lf-eval-{uuid.uuid4().hex[:8]}",
        )
        return {
            "lane": state["approved_route"],
            "answer": state["answer"],
            "tools_used": state["tool_trail"],
            "findings": state["findings"],
            "stop_reason": state.get("stop_reason", ""),
        }

    dataset = langfuse.get_dataset(DATASET_NAME)
    print(f"Running experiment over {len(dataset.items)} items...\n")

    try:
        result = dataset.run_experiment(
            name="Jev-judged support answers",
            description="Grounding and completeness judged by Jev; routing checked exactly.",
            task=task,
            evaluators=[jev_judge, routing_correct],
            run_evaluators=[mean_grounded, ungrounded_count, mean_completeness],
            # laya is local and the MCP servers are stdio subprocesses: keep it serial.
            max_concurrency=1,
            metadata={
                "judge_model": lesson10.JEV_MODEL,
                "router_model": lesson10.JEV_MODEL,
                "tool_router_model": lesson10.LAYA_MODEL,
                "chat_model": lesson10.CHAT_MODEL,
            },
        )
        print("\n" + result.format())
    finally:
        # Short-lived script: without this the scores never leave the process.
        # In a finally block so a crash while reporting cannot lose the results.
        langfuse.flush()

    print("Open Langfuse -> Datasets -> "
          f"{DATASET_NAME} -> Runs to compare experiments.")


if __name__ == "__main__":
    asyncio.run(main())
