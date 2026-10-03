"""
LangGraph + Jev: "prod, not god" document review
================================================
Follows:
  https://www.langchain.com/blog/building-prod-with-jev-and-langgraph

Core idea from the article
--------------------------
Frontier LLMs get treated like "god": one model does routing, classification,
AND text generation. That is expensive when all you needed was a yes/no.

Jev (TypeSafe, via OpenRouter) is a *decision model*, not a chat model.
You send application state + typed questions. It returns typed answers with
probabilities. Your code owns the workflow; Jev only sits at the branches.

LangGraph pieces this file makes visible
----------------------------------------
  * State  = accumulating context each later step (and each Jev call) sees
  * Nodes  = units of work (ingest, redact with an LLM, set aside, produce, HITL)
  * Edges  = what runs next. EVERY conditional edge is decided by Jev, not an LLM

Graph (same topology as the article's discovery-review example)
--------------------------------------------------------------
  START -> ingest
              |  Jev noul: is this page responsive to the request?
              +-- no  --> set_aside -----------------------------> END
              +-- yes --> privilege_gate
                            |  Jev noul: might this be privileged?
                            +-- yes --> attorney_review ---------> END
                            +-- no  --> pii_gate
                                          |  Jev noul: contains PII?
                                          +-- yes --> redact_pii (LLM) --> produce --> END
                                          +-- no  --> produce -----------------------> END

Jev is invoked *inside each routing function* passed to add_conditional_edges.
That is the drop-in replacement for "ask an LLM which node to go to".

The LLM (openai/gpt-5.6-luna on OpenRouter) is used ONLY when we need generated
text: redacting PII. That is "cheap by default, frontier on exception".

Both models share OPENROUTER_API_KEY, but they are different APIs:
  * Jev  is NOT chat-compatible. OpenAI SDK / ChatOpenAI cannot call it.
         POST https://openrouter.ai/api/alpha/decisions  (~typesafe/jev-latest)
  * Luna IS OpenAI-compatible. Point ChatOpenAI / init_chat_model at OpenRouter.
         base_url=https://openrouter.ai/api/v1  model=openai/gpt-5.6-luna
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

# ---------------------------------------------------------------------------
# Models (both through OpenRouter, one key)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_CHAT_BASE = "https://openrouter.ai/api/v1"

JEV_MODEL = "~typesafe/jev-latest"
LLM_MODEL = "openai/gpt-5.6-luna"

# Same OpenAI-compatible client the rest of this course uses, pointed at OpenRouter.
# Jev cannot use this path: it has no messages/completions surface.
llm = init_chat_model(
    LLM_MODEL,
    model_provider="openai",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_CHAT_BASE,
    default_headers={
        "HTTP-Referer": "https://localhost/langgraph-crash-course",
        "X-OpenRouter-Title": "langgraph-crash-course-jev",
    },
)

# A noul is P(yes) in [0, 1]. Code — not a prompt — picks the threshold.
# Conservative on privilege (article: "might it be privileged?").
RESPONSIVE_THRESHOLD = 0.50
PRIVILEGE_THRESHOLD = 0.35
PII_THRESHOLD = 0.50

# Path functions cannot update graph state. Each Jev route records here;
# the next node flushes the rows into decision_log via the reducer.
_jev_journal: dict[str, list[dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# State = context
# ---------------------------------------------------------------------------
def _append_log(existing: list | None, new: list | None) -> list:
    return (existing or []) + (new or [])


class ReviewState(TypedDict):
    page_id: str
    page_text: str
    discovery_request: str
    decision_log: Annotated[list[dict[str, Any]], _append_log]
    redacted_text: str
    outcome: str


# ---------------------------------------------------------------------------
# Jev via OpenRouter Decisions API (not OpenAI-compatible)
# ---------------------------------------------------------------------------
# Jev returns typed answers + probabilities, not chat tokens. Pointing the
# OpenAI SDK at /api/v1/chat/completions with ~typesafe/jev-latest will fail.
def _openrouter_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/langgraph-crash-course",
            "X-OpenRouter-Title": "langgraph-crash-course-jev",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 402:
            raise RuntimeError(
                "OpenRouter 402: this API key has no credits. "
                "Add credits at https://openrouter.ai/settings/credits "
                "or switch OPENROUTER_API_KEY in .env to a funded key."
            ) from exc
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc


# Jev primitives:
#   noul   — P(yes) for a proposition          -> answers[name]["noul"]
#   choice — pick one labelled option          -> answers[name]["choice"]
#   score  — position on an ordered rubric     -> answers[name]["score"]
# All questions in one request are answered in parallel against the same state.
def ask_jev(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    body = _openrouter_post(
        OPENROUTER_DECISIONS_URL,
        {"model": JEV_MODEL, "state": state, "questions": questions},
    )
    if "answers" not in body:
        raise RuntimeError(f"Unexpected Jev response: {body}")
    return body


def ask_jev_noul(
    *,
    state: Any,
    name: str,
    instructions: str,
    true_meaning: str,
    false_meaning: str,
) -> tuple[float, dict[str, Any]]:
    """One yes/no judgment. Returns (P(yes), raw OpenRouter body)."""
    body = ask_jev(
        state,
        {
            name: {
                "type": "noul",
                "instructions": instructions,
                "criteria": {"true": true_meaning, "false": false_meaning},
            }
        },
    )
    return float(body["answers"][name]["noul"]), body


def _record_noul(
    page_id: str,
    name: str,
    probability: float,
    threshold: float,
    route: str,
    body: dict[str, Any],
) -> None:
    print(
        f"  [Jev] {page_id} | {name}: noul={probability:.3f} "
        f"(threshold={threshold:.2f}) -> {route}"
    )
    _jev_journal.setdefault(page_id, []).append(
        {
            "question": name,
            "type": "noul",
            "noul": probability,
            "threshold": threshold,
            "route": route,
            "model": body.get("model"),
            "usage": body.get("usage"),
        }
    )


def _flush_jev_log(state: ReviewState) -> dict[str, list]:
    rows = _jev_journal.pop(state["page_id"], [])
    return {"decision_log": rows} if rows else {}


# ---------------------------------------------------------------------------
# Nodes — units of work
# ---------------------------------------------------------------------------
def ingest_page(state: ReviewState) -> dict:
    print(f"\n=== ingest {state['page_id']} ===")
    print(f"  request : {state['discovery_request']}")
    preview = state["page_text"][:160]
    suffix = "..." if len(state["page_text"]) > 160 else ""
    print(f"  page    : {preview}{suffix}")
    return {}


def privilege_gate(state: ReviewState) -> dict:
    # Anchor node so the next Jev conditional edge has a source.
    return _flush_jev_log(state)


def pii_gate(state: ReviewState) -> dict:
    return _flush_jev_log(state)


def set_aside(state: ReviewState) -> dict:
    print(f"  -> SET ASIDE (not responsive): {state['page_id']}")
    return {**_flush_jev_log(state), "outcome": "set_aside"}


def attorney_review(state: ReviewState) -> dict:
    # Article: privilege pauses the graph for a human. Use interrupt() for HITL.
    print(f"  -> ATTORNEY REVIEW (possible privilege): {state['page_id']}")
    return {**_flush_jev_log(state), "outcome": "attorney_review"}


def redact_pii(state: ReviewState) -> dict:
    """LLM is used here because we need generated text, not a decision."""
    print(f"  -> REDACT PII with {LLM_MODEL}: {state['page_id']}")
    message = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Redact personal information (names of private individuals, emails, "
                    "phone numbers, SSNs, street addresses). Replace each span with a "
                    "tag like [REDACTED_NAME]. Keep everything else verbatim. "
                    "Return only the redacted page."
                ),
            },
            {"role": "user", "content": state["page_text"]},
        ]
    )
    redacted = message.content if isinstance(message.content, str) else str(message.content)
    preview = redacted[:200]
    suffix = "..." if len(redacted) > 200 else ""
    print(f"  redacted: {preview}{suffix}")
    return {**_flush_jev_log(state), "redacted_text": redacted}


def produce_page(state: ReviewState) -> dict:
    produced = state.get("redacted_text") or state["page_text"]
    print(f"  -> PRODUCE: {state['page_id']}")
    return {**_flush_jev_log(state), "outcome": "produce", "redacted_text": produced}


# ---------------------------------------------------------------------------
# Conditional edges — Jev, not an LLM, chooses the next node
# ---------------------------------------------------------------------------
def _jev_state(state: ReviewState) -> dict[str, str]:
    return {
        "discovery_request": state["discovery_request"],
        "page_id": state["page_id"],
        "page_text": state["page_text"],
    }


def route_responsive(state: ReviewState) -> Literal["set_aside", "privilege_gate"]:
    probability, body = ask_jev_noul(
        state=_jev_state(state),
        name="is_responsive",
        instructions=(
            "Is this page responsive to the discovery request? "
            "Responsive means it discusses, mentions, or would help a reviewer "
            "understand the subject of the request."
        ),
        true_meaning="The page is about the requested subject and should be reviewed further.",
        false_meaning="The page is unrelated (personal, admin, or a different topic).",
    )
    route: Literal["set_aside", "privilege_gate"] = (
        "privilege_gate" if probability >= RESPONSIVE_THRESHOLD else "set_aside"
    )
    _record_noul(state["page_id"], "is_responsive", probability, RESPONSIVE_THRESHOLD, route, body)
    return route


def route_privilege(state: ReviewState) -> Literal["attorney_review", "pii_gate"]:
    probability, body = ask_jev_noul(
        state=_jev_state(state),
        name="might_be_privileged",
        instructions=(
            "Might this page be attorney-client privileged or work product? "
            "Look for counsel, legal advice, 'privileged and confidential', "
            "or discussion of litigation strategy."
        ),
        true_meaning="A lawyer could reasonably claim privilege; a human must review.",
        false_meaning="Ordinary business content with no legal-advice marker.",
    )
    route: Literal["attorney_review", "pii_gate"] = (
        "attorney_review" if probability >= PRIVILEGE_THRESHOLD else "pii_gate"
    )
    _record_noul(
        state["page_id"], "might_be_privileged", probability, PRIVILEGE_THRESHOLD, route, body
    )
    return route


def route_pii(state: ReviewState) -> Literal["redact_pii", "produce"]:
    probability, body = ask_jev_noul(
        state=_jev_state(state),
        name="contains_pii",
        instructions=(
            "Does this page contain personal information that should be redacted "
            "before production (individual names, emails, phones, SSNs, home addresses)?"
        ),
        true_meaning="Personal identifiers are present and need redaction.",
        false_meaning="No personal identifiers, or only public company/role names.",
    )
    route: Literal["redact_pii", "produce"] = (
        "redact_pii" if probability >= PII_THRESHOLD else "produce"
    )
    _record_noul(state["page_id"], "contains_pii", probability, PII_THRESHOLD, route, body)
    return route


# ---------------------------------------------------------------------------
# Graph — topology encodes domain knowledge (not a mega-prompt)
# ---------------------------------------------------------------------------
def build_review_graph():
    builder = StateGraph(ReviewState)

    builder.add_node("ingest", ingest_page)
    builder.add_node("privilege_gate", privilege_gate)
    builder.add_node("pii_gate", pii_gate)
    builder.add_node("set_aside", set_aside)
    builder.add_node("attorney_review", attorney_review)
    builder.add_node("redact_pii", redact_pii)
    builder.add_node("produce", produce_page)

    builder.add_edge(START, "ingest")

    builder.add_conditional_edges(
        "ingest",
        route_responsive,
        {"set_aside": "set_aside", "privilege_gate": "privilege_gate"},
    )
    builder.add_conditional_edges(
        "privilege_gate",
        route_privilege,
        {"attorney_review": "attorney_review", "pii_gate": "pii_gate"},
    )
    builder.add_conditional_edges(
        "pii_gate",
        route_pii,
        {"redact_pii": "redact_pii", "produce": "produce"},
    )

    builder.add_edge("redact_pii", "produce")
    builder.add_edge("set_aside", END)
    builder.add_edge("attorney_review", END)
    builder.add_edge("produce", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Demo pages — four paths through the same graph
# ---------------------------------------------------------------------------
DISCOVERY_REQUEST = (
    "All documents related to the Q3 2025 pricing decision for the Apex product, "
    "including internal emails, decks, and analyses."
)

SAMPLE_PAGES = [
    {
        "page_id": "page-01-lunch",
        "page_text": (
            "Hey team, I am grabbing sandwiches at 12:30. Anyone want the "
            "usual from the deli on 4th? I will also pick up oat milk."
        ),
        "expect": "set_aside",
    },
    {
        "page_id": "page-02-pricing",
        "page_text": (
            "Q3 2025 Apex pricing recommendation: drop list price 8% in EMEA "
            "to match Competitor Z. Finance model attached. No customer PII. "
            "Approved in the Monday revenue committee."
        ),
        "expect": "produce",
    },
    {
        "page_id": "page-03-pii",
        "page_text": (
            "Apex Q3 pricing win/loss notes. Customer contact Jane Doe "
            "(jane.doe@example.com, +1-202-555-0147, SSN 078-05-1120) said "
            "the new Apex list price lost the deal. Home address 14 Oak Lane."
        ),
        "expect": "produce after redact",
    },
    {
        "page_id": "page-04-legal",
        "page_text": (
            "PRIVILEGED AND CONFIDENTIAL — ATTORNEY-CLIENT COMMUNICATION\n"
            "To: counsel@company.com\n"
            "Please advise whether our Q3 2025 Apex pricing emails are "
            "work product and should be withheld from the upcoming production."
        ),
        "expect": "attorney_review",
    },
]


def main() -> None:
    graph = build_review_graph()

    print("Graph topology (mermaid):\n")
    print(graph.get_graph().draw_mermaid())
    print("\nJev model :", JEV_MODEL, "via", OPENROUTER_DECISIONS_URL)
    print("LLM model :", LLM_MODEL, "via", OPENROUTER_CHAT_BASE, "(OpenAI-compatible, redaction only)")

    for page in SAMPLE_PAGES:
        try:
            result = graph.invoke(
                {
                    "page_id": page["page_id"],
                    "page_text": page["page_text"],
                    "discovery_request": DISCOVERY_REQUEST,
                    "decision_log": [],
                    "redacted_text": "",
                    "outcome": "",
                }
            )
        except RuntimeError as exc:
            print(f"\nStopped: {exc}")
            return
        print(f"  outcome : {result['outcome']}  (lesson expects: {page['expect']})")
        print("  jev log :")
        for row in result.get("decision_log") or []:
            print(f"            {row}")


if __name__ == "__main__":
    main()
