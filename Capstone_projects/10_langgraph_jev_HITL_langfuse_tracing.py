"""
Lesson 10 — Bookstore Support Desk
==================================
One small, readable graph that shows six ideas at once:

  1. A DECISION MODEL on a conditional edge    -> Jev (~typesafe/jev-latest via OpenRouter)
  2. HUMAN IN THE LOOP after that decision     -> interrupt() before the path is taken
  3. A PREBUILT react agent                    -> langchain.agents.create_agent
    4. A HAND-BUILT react subgraph               -> Jev routing + Responses API
  5. FOUR tool kinds                           -> web search, python function, 2 MCP servers
  6. TRACING to two backends at once           -> Langfuse + LangSmith

The use case (deliberately tiny):
    A customer writes to a bookstore. Two kinds of message arrive.
      "Where is my order 1002?"            -> order_help     (fast, prebuilt agent)
      "Is Clean Code worth reading?"       -> book_research  (multi-tool research)
    Jev reads the message and picks the lane. A human confirms. Then the graph runs.

Graph
-----
    START -> triage ------- Jev picks the lane (choice + probabilities)
               |
             human_review -- interrupt(): approve or override
               |
        [conditional edge on the APPROVED lane]
               |
       +-------+---------------------------+
       |                                   |
   order_desk                        research_desk
   (prebuilt create_agent)           (hand-built react subgraph)
       |                                   |
       +-------------+---------------------+
                     |
                  wrap_up -> END

Why two models?
    Jev is a DECISION model: state + typed questions -> typed answers with
    probabilities. It never writes prose, so it is predictable, which is
    exactly what a router needs.
    gpt-5.6-luna is a CHAT model: it is used only where words are needed.

Run as a script:
    python bookstore/seed.py
    python -m uvicorn bookstore.orders_api:app  --port 8071
    python -m uvicorn bookstore.reviews_api:app --port 8072
    python 10_langgraph_jev_HITL_langfuse_tracing.py

Run in LangGraph Studio (same two services must be up first):
    langgraph dev --allow-blocking

    --allow-blocking is required, not optional. The dev server refuses
    synchronous I/O on the event loop, and Jev's urllib request is
    unavoidably blocking. Without the flag every run fails immediately with
    BlockingError.

    Studio serves EVERY graph in langgraph.json; pick this one
    ("bookstore_support_desk") from the graph selector. Then send
    {"question": "Where is my order 1002?"}, and Studio will stop at the
    human_review interrupt and let you approve or override before continuing.
"""

from __future__ import annotations

# truststore makes httpx/openai trust the OS certificate store. Needed behind a
# corporate TLS proxy, and harmless otherwise. It must run before any client.
import truststore

truststore.inject_into_ssl()

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()  # Langfuse must be imported AFTER the keys are loaded.

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from langsmith import traceable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_CHAT_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

JEV_MODEL = "~typesafe/jev-latest"      # hosted decision model (OpenRouter)
CHAT_MODEL = "openai/gpt-5.6-luna"      # OpenAI-compatible chat model

# The hand-built agent stops after this many tool steps, no matter what the
# decision model says.
MAX_TOOL_STEPS = 4

# The research loop ends when Jev is this sure the question is already
# answerable. Tuned by reading real gate values rather than guessed: answered
# questions came back around 0.90 and unanswered ones at 0.01-0.22, while one
# genuinely-finished case sat at 0.47. Any threshold you pick should be backed
# by numbers you have actually looked at.
ANSWERABLE_THRESHOLD = 0.45

langfuse = get_client()

# One chat model, reused. use_responses_api=True switches langchain-openai to
# POST /v1/responses, which is what lets us use OpenRouter's server-side tools.
chat_model = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_CHAT_BASE,
    use_responses_api=True,
)


# ---------------------------------------------------------------------------
# Decision model #1: Jev, called over OpenRouter's Decisions API
# ---------------------------------------------------------------------------
# Jev is NOT OpenAI-compatible. It has no messages/completions endpoint, so the
# OpenAI SDK cannot call it. You POST a `state` plus typed `questions` and get
# back typed answers: noul (P(yes)), choice (one option), score (a rubric).
@traceable(run_type="llm", name="jev-decision")
def ask_jev(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = {"model": JEV_MODEL, "state": state, "questions": questions}
    request = urllib.request.Request(
        OPENROUTER_DECISIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Jev/OpenRouter HTTP {exc.code}: {detail}") from exc
    if "answers" not in body:
        raise RuntimeError(f"Unexpected Jev response: {body}")
    return body


def _trace_decision(
    name: str, model: str, state: Any, body: dict[str, Any], **metadata: Any
) -> None:
    """Record a decision-model call in Langfuse as a generation.

    Jev is a plain HTTP call, so the LangChain CallbackHandler never sees it.
    Without this it would be invisible in Langfuse — and the
    routing decision is the most important thing in this whole graph.

    It lands inside the right trace because answer_message is wrapped in
    @observe, which opens the parent this generation attaches to.

    One honest limitation. A manual observation attaches to whatever
    OpenTelemetry context is current, and the LangChain handler does not make
    its per-node span current while the node body runs. So these generations
    end up as siblings of the graph span rather than nested under the node that
    asked the question. The `graph_node` metadata below is how you recover that
    link: filter or group by it in the UI.
    """
    usage = body.get("usage") or {}
    with langfuse.start_as_current_observation(
        as_type="generation", name=name, model=model
    ) as generation:
        generation.update(
            input=state,
            output=body.get("answers"),
            usage_details={
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
            },
            # Langfuse has no price list for these model names, so it cannot
            # derive cost itself. Jev reports the real figure per call. Passing
            # it explicitly makes the decision-model cost checkable against
            # your own dashboard instead of taken on faith.
            cost_details={"total": float(usage.get("cost", 0.0))},
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# The four tools
# ---------------------------------------------------------------------------
# Tool 1 — an ordinary python function.
@tool
def shipping_estimate(city: str) -> str:
    """Estimate delivery time in days to an Indian city for a bookstore parcel."""
    days = {"mumbai": 2, "bengaluru": 3, "chennai": 3, "delhi": 4}.get(city.strip().lower(), 5)
    return f"Standard delivery to {city} takes about {days} working days."


# Tool 2 — a server-side tool. OpenRouter runs the search during the request,
# so there is no client-side tool loop for it.
WEB_SEARCH_TOOL = {"type": "openrouter:web_search"}

# Tools 3 and 4 — two MCP servers, each launched over stdio. Every MCP tool is
# a thin wrapper over one of the local FastAPI services.
MCP_SERVERS = {
    "orders": {
        "command": sys.executable,
        "args": [os.path.join(HERE, "bookstore", "orders_mcp.py")],
        "transport": "stdio",
    },
    "reviews": {
        "command": sys.executable,
        "args": [os.path.join(HERE, "bookstore", "reviews_mcp.py")],
        "transport": "stdio",
    },
}


async def load_mcp_tools() -> dict[str, list]:
    """Connect to both MCP servers and return their tools, grouped by server."""
    client = MultiServerMCPClient(MCP_SERVERS)
    return {
        "orders": await client.get_tools(server_name="orders"),
        "reviews": await client.get_tools(server_name="reviews"),
    }


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------
class SupportState(TypedDict):
    question: str
    jev_route: str            # what Jev chose
    jev_confidence: float
    jev_probabilities: dict
    approved_route: str       # what the human approved (may differ from Jev)
    human_note: str
    tool_trail: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    findings: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    stop_reason: str
    answer: str


ROUTE_CRITERIA = {
    "order_help": "The customer asks about their own order: status, delivery, refund, cancellation.",
    "book_research": "The customer asks about a book itself: is it good, what do reviews say, general info.",
}


# ---------------------------------------------------------------------------
# Node 1: triage — Jev makes the routing decision
# ---------------------------------------------------------------------------
def triage(state: SupportState) -> dict:
    print(f"\n=== customer message ===\n  {state['question']}")

    body = ask_jev(
        {"customer_message": state["question"]},
        {
            "lane": {
                "type": "choice",
                "instructions": "Which support lane should handle this message?",
                "criteria": ROUTE_CRITERIA,
            }
        },
    )
    _trace_decision("jev-triage", JEV_MODEL, state["question"], body, graph_node="triage")

    answer = body["answers"]["lane"]
    route = answer["choice"]
    confidence = float(answer.get("confidence", 0.0))
    print(f"  [Jev]  lane={route}  confidence={confidence:.2f}  {answer['probabilities']}")
    return {
        "jev_route": route,
        "jev_confidence": confidence,
        "jev_probabilities": answer["probabilities"],
    }


# ---------------------------------------------------------------------------
# Node 2: human_review — HITL, after the decision, before the path is taken
# ---------------------------------------------------------------------------
def human_review(state: SupportState) -> dict:
    """Pause the graph and show the human what Jev decided.

    interrupt() raises out of the graph. The caller inspects the payload and
    resumes with Command(resume=...). Because a checkpointer is attached, the
    graph picks up exactly here — nothing before this point is re-run.
    """
    decision = interrupt(
        {
            "question": state["question"],
            "jev_route": state["jev_route"],
            "jev_confidence": state["jev_confidence"],
            "jev_probabilities": state["jev_probabilities"],
            "prompt": "Approve Jev's lane? Reply 'approve', or name the other lane to override.",
        }
    )

    reply = str(decision).strip().lower()
    if reply in ("", "approve", "yes", "y", "ok"):
        return {"approved_route": state["jev_route"], "human_note": "approved Jev's choice"}
    if reply in ROUTE_CRITERIA:
        return {"approved_route": reply, "human_note": f"human overrode Jev to {reply}"}
    return {"approved_route": state["jev_route"], "human_note": f"unrecognised reply {reply!r}; kept Jev"}


def route_after_review(state: SupportState) -> Literal["order_desk", "research_desk"]:
    """The conditional edge. It only reads state — the thinking already happened."""
    lane = state["approved_route"]
    print(f"  [HITL] {state['human_note']} -> {lane}")
    return "order_desk" if lane == "order_help" else "research_desk"


# ---------------------------------------------------------------------------
# Node 3a: order_desk — LangChain's PREBUILT react agent
# ---------------------------------------------------------------------------
# create_agent gives you the whole model/tool loop in one line. Use it when the
# default loop is what you want.
def build_order_desk(order_tools: list):
    return create_agent(
        model=chat_model,
        tools=[*order_tools, shipping_estimate],
        system_prompt=(
            "You are a bookstore order support agent. Use the tools to look up the "
            "customer's order and answer in two sentences. Never invent order details."
        ),
    )


def make_order_desk_node(order_tools: list):
    prebuilt = build_order_desk(order_tools)

    async def order_desk(state: SupportState) -> dict:
        print("  -> order_desk (prebuilt create_agent)")
        result = await prebuilt.ainvoke(
            {"messages": [HumanMessage(content=state["question"])]}
        )
        messages = result["messages"]
        used = [
            call["name"]
            for message in messages
            if isinstance(message, AIMessage)
            for call in (message.tool_calls or [])
        ]
        # Keep what the tools actually returned. The evals in lessons 11 and 12
        # judge the answer against this, so an unsupported claim is detectable.
        findings = [f"{m.name}: {m.text}" for m in messages if m.type == "tool"]
        return {
            "answer": messages[-1].text,
            "tool_trail": used,
            "findings": findings,
            "stop_reason": "prebuilt_agent_finished",
        }

    return order_desk


# ---------------------------------------------------------------------------
# Node 3b: research_desk — a HAND-BUILT react subgraph
# ---------------------------------------------------------------------------
# Same react idea as above, but every piece is explicit so you can see it:
#
#     tool_router  -- Jev: are we done?  then which tool next?
#          |
#     [conditional edge]
#          |               \
#        act                compose (write the final answer)
#          |
#     [conditional edge] -- did the model request a client-side tool?
#          |          \
#        tools         (web search already ran server-side)
#          |          /
#     back to tool_router
#
# The circuit breaker lives in tool_router, ahead of the model: past
# MAX_TOOL_STEPS it forces "finish", so the loop can never spin forever.
class ResearchState(TypedDict):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    next_tool: str
    steps: int
    tool_trail: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    notes: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    answer: str
    stop_reason: str


# Write these descriptions like a rubric for a new colleague. A decision model
# matches the words you give it, so vague boundaries produce vague routing.
TOOL_CHOICES = {
    "reviews_db": (
        "Star ratings and opinions that this bookstore's own readers wrote about a book. "
        "Only for 'is it good / what did readers think'. Holds NO author or publishing facts."
    ),
    "orders_db": "This customer's own purchase records: order status, delivery state, what they bought.",
    "shipping_calc": "How many days delivery takes to a named city.",
    "web_search": (
        "The public internet. Use for facts the bookstore does not store: who an author is, "
        "what else they wrote, publication dates, awards, news."
    ),
}


def make_research_subgraph(mcp_tools: dict[str, list]):
    # Map each decision label to the tools it unlocks.
    tools_by_choice: dict[str, list] = {
        "orders_db": mcp_tools["orders"],
        "reviews_db": mcp_tools["reviews"],
        "shipping_calc": [shipping_estimate],
    }
    every_function_tool = [t for group in tools_by_choice.values() for t in group]
    tool_node = ToolNode(every_function_tool)

    def tool_router(state: ResearchState) -> dict:
        """Decide whether to stop, and if not, which tool runs next.

                Jev first checks whether the gathered facts answer the question. If
                they do not, Jev chooses the next unused tool from the available menu.
        """
        # Circuit breaker first, so it can override the model. That is the
        # point of a breaker: no model gets a vote on its own stopping rule.
        if state["steps"] >= MAX_TOOL_STEPS:
            print(f"  [breaker] {MAX_TOOL_STEPS} tool steps used -> finish")
            return {"next_tool": "finish", "stop_reason": "circuit_breaker"}

        gathered = state["notes"] or ["(no tool has run yet)"]

        # --- Gate: Jev decides whether we already have what we need ----------
        gate_state = {"customer_question": state["question"], "facts_gathered": gathered}
        gate_body = ask_jev(
            gate_state,
            {
                "answerable": {
                    "type": "noul",
                    "instructions": "Can the customer question be fully answered using only facts_gathered?",
                    "criteria": {
                        "true": "Every fact the customer asked for is present in facts_gathered.",
                        "false": (
                            "A fact the customer asked for is missing. A tool that ran but "
                            "returned no data supplies nothing."
                        ),
                    },
                }
            },
        )
        _trace_decision(
            "jev-answerable-gate", JEV_MODEL, gate_state, gate_body,
            graph_node="tool_router", step=state["steps"] + 1,
        )
        # A noul returns P(true) and carries NO confidence field — reading
        # ["confidence"] here would raise. Only choice and score have one.
        answerable = float(gate_body["answers"]["answerable"]["noul"])

        if answerable >= ANSWERABLE_THRESHOLD:
            print(f"  [jev]  answerable={answerable:.2f} -> finish")
            return {"next_tool": "finish", "stop_reason": "jev_says_answerable"}

        # --- Choice: Jev picks the next tool from those not yet run ----------
        # Used tools come off the menu. Running one twice cannot produce new
        # information, and a choice question ALWAYS returns one of its options,
        # so leaving a dead option in place just invites the model to pick it
        # again. Narrowing the menu to the decisions still open is the cheapest
        # reliability fix available here.
        remaining = [name for name in TOOL_CHOICES if name not in state["tool_trail"]]
        if not remaining:
            print(f"  [jev]  answerable={answerable:.2f}, but every tool has been tried -> finish")
            return {"next_tool": "finish", "stop_reason": "all_tools_used"}

        choice_state = {
            "customer_question": state["question"],
            "facts_gathered": gathered,
            "tools_already_used": state["tool_trail"] or ["none"],
        }
        choice_body = ask_jev(
            choice_state,
            {
                "next_tool": {
                    "type": "choice",
                    "instructions": "Which single tool should run next to answer the customer's question?",
                    "criteria": {name: TOOL_CHOICES[name] for name in remaining},
                }
            },
        )
        _trace_decision(
            "jev-tool-choice", JEV_MODEL, choice_state, choice_body,
            graph_node="tool_router", step=state["steps"] + 1, options=remaining,
        )

        answer = choice_body["answers"]["next_tool"]
        choice = answer["choice"]
        print(f"  [jev]  step {state['steps'] + 1}: answerable={answerable:.2f}, "
              f"next_tool={choice} (confidence={answer.get('confidence', 0):.2f}) "
              f"from {remaining}")
        return {"next_tool": choice, "stop_reason": ""}

    def route_from_router(state: ResearchState) -> Literal["act", "compose"]:
        return "compose" if state["next_tool"] == "finish" else "act"

    async def act(state: ResearchState) -> dict:
        """Give the model ONLY the tool Jev selected, then let it call it."""
        choice = state["next_tool"]

        if choice == "web_search":
            # Server-side tool: OpenRouter runs the search inside this one call.
            bound = chat_model.bind(tools=[WEB_SEARCH_TOOL])
        else:
            selected = tools_by_choice.get(choice, [])
            bound = chat_model.bind(tools=[convert_to_openai_tool(t) for t in selected])

        prompt = [
            SystemMessage(
                content=(
                    "You are researching a bookstore customer's question. "
                    f"Use the '{choice}' capability that is available to you, then stop. "
                    "Do not answer from memory when a tool can answer."
                )
            ),
            HumanMessage(content=state["question"]),
            *state["messages"],
        ]
        # run_name renames the observation in both Langfuse and LangSmith.
        # Without it these show up as "ChatOpenAI", which names the class
        # rather than the step and breaks the moment you swap models. Keep the
        # name fixed and put the varying part in metadata: a name that changes
        # per run cannot be grouped, filtered or targeted by an evaluator.
        message = await bound.ainvoke(
            prompt,
            config={"run_name": "research-act", "metadata": {"selected_tool": choice}},
        )

        update: dict = {"messages": [message], "steps": state["steps"] + 1,
                        "tool_trail": [choice]}
        if not message.tool_calls:
            # Nothing to execute locally: whatever came back IS the observation.
            text = message.text
            if text:
                update["notes"] = [f"{choice}: {text}"]
        return update

    def route_after_act(state: ResearchState) -> Literal["tools", "tool_router"]:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "tool_router"

    def collect(state: ResearchState) -> dict:
        """Copy the tool results into plain notes so the routers can read them."""
        fresh = []
        for message in reversed(state["messages"]):
            if message.type != "tool":
                break
            fresh.append(f"{message.name}: {message.text}")
        return {"notes": list(reversed(fresh))}

    async def compose(state: ResearchState) -> dict:
        """The only place in this subgraph where the chat model writes prose."""
        gathered = "\n".join(state["notes"]) if state["notes"] else "(no tool results)"
        message = await chat_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Answer the bookstore customer in two or three sentences, using "
                        "only the gathered findings. If the findings are thin, say so plainly."
                    )
                ),
                HumanMessage(content=f"Question: {state['question']}\n\nFindings:\n{gathered}"),
            ],
            config={"run_name": "compose-answer"},
        )
        return {"answer": message.text,
                "stop_reason": state["stop_reason"] or "router_finished"}

    builder = StateGraph(ResearchState)
    builder.add_node("tool_router", tool_router)
    builder.add_node("act", act)
    builder.add_node("tools", tool_node)
    builder.add_node("collect", collect)
    builder.add_node("compose", compose)

    builder.add_edge(START, "tool_router")
    builder.add_conditional_edges("tool_router", route_from_router,
                                  {"act": "act", "compose": "compose"})
    builder.add_conditional_edges("act", route_after_act,
                                  {"tools": "tools", "tool_router": "tool_router"})
    builder.add_edge("tools", "collect")
    builder.add_edge("collect", "tool_router")
    builder.add_edge("compose", END)
    # Naming the compiled subgraph keeps it from showing up as a generic
    # "LangGraph" node inside the parent trace.
    return builder.compile(name="research-subgraph")


def make_research_desk_node(mcp_tools: dict[str, list]):
    subgraph = make_research_subgraph(mcp_tools)

    async def research_desk(state: SupportState) -> dict:
        print("  -> research_desk (hand-built react subgraph, Jev gates and picks tools)")
        result = await subgraph.ainvoke(
            {
                "question": state["question"],
                "messages": [],
                "next_tool": "",
                "steps": 0,
                "tool_trail": [],
                "notes": [],
                "answer": "",
                "stop_reason": "",
            }
        )
        print(f"  [subgraph] exited because: {result['stop_reason']}")
        return {
            "answer": result["answer"],
            "tool_trail": result["tool_trail"],
            "findings": result["notes"],
            "stop_reason": result["stop_reason"],
        }

    return research_desk


# ---------------------------------------------------------------------------
# Node 4: wrap_up
# ---------------------------------------------------------------------------
def wrap_up(state: SupportState) -> dict:
    print(f"  tools used : {state['tool_trail'] or ['none']}")
    print(f"  answer     : {state['answer']}")
    return {}


# ---------------------------------------------------------------------------
# Assemble the main graph
# ---------------------------------------------------------------------------
def build_support_graph(mcp_tools: dict[str, list], *, managed_persistence: bool = False):
    """Assemble the main graph.

    managed_persistence=True is for LangGraph Studio / `langgraph dev`, which
    supply their own checkpointer. Attaching one here would be thrown away, and
    the real thing is what makes Studio able to show the interrupt and resume
    from it. Standalone scripts have no such server, so they get an
    InMemorySaver instead.
    """
    builder = StateGraph(SupportState)

    builder.add_node("triage", triage)
    builder.add_node("human_review", human_review)
    builder.add_node("order_desk", make_order_desk_node(mcp_tools["orders"]))
    builder.add_node("research_desk", make_research_desk_node(mcp_tools))
    builder.add_node("wrap_up", wrap_up)

    builder.add_edge(START, "triage")
    builder.add_edge("triage", "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {"order_desk": "order_desk", "research_desk": "research_desk"},
    )
    builder.add_edge("order_desk", "wrap_up")
    builder.add_edge("research_desk", "wrap_up")
    builder.add_edge("wrap_up", END)

    # A checkpointer is REQUIRED for interrupt()/resume to work.
    return builder.compile(
        checkpointer=None if managed_persistence else InMemorySaver(),
        name="bookstore-support-desk",
    )


# ---------------------------------------------------------------------------
# Entry point for LangGraph Studio
# ---------------------------------------------------------------------------
# Registered in langgraph.json as:
#     "bookstore_support_desk": "./10_langgraph_jev_HITL_langfuse_tracing.py:make_studio_graph"
#
# Studio needs a graph object, but ours cannot be built at import time: the MCP
# tools have to be fetched over stdio first, and that is async. A factory
# function is the supported answer — the CLI will await it. The `config`
# argument is optional and unused here, but accepting it keeps the signature
# valid whether or not the server chooses to pass one.
async def make_studio_graph(config: dict | None = None):
    return build_support_graph(await load_mcp_tools(), managed_persistence=True)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
SAMPLE_MESSAGES = [
    # 1. order lane, human approves Jev
    ("Where is my order 1002 and when will it arrive?", "approve"),
    # 2. research lane, answered from the bookstore's own review data
    ("Is 'Clean Code' worth reading? What do your readers say?", "approve"),
    # 3. research lane, needs the public web (we store no author facts)
    ("Who wrote 'Sapiens' and what else is that author known for?", "approve"),
    # 4. HITL override: Jev says order_help, but the human sends it to research
    ("Where is my order 1003?", "book_research"),
]


@observe(name="bookstore-support-desk")
async def answer_message(
    graph, question: str, human_reply: str, thread_id: str, user_id: str = "demo-student"
) -> dict:
    """Answer one customer message inside exactly one Langfuse trace.

    Run the graph, stop at the HITL interrupt, then resume.

    The tracing belongs HERE rather than in the demo loop, because this is the
    unit of work: one customer message in, one answer out. @observe opens the
    trace, so both ainvoke calls (before and after the human) and every
    Jev decisions land inside it.

    Putting it in the caller instead was a real bug worth remembering. Lessons
    11 and 12 also call this function, and with no trace open the Jev
    generations became separate ROOT traces with no session and no user — an
    unreadable pile of orphans in the UI. Instrument the unit of work, not one
    particular caller of it.
    """
    with propagate_attributes(
        session_id=thread_id,
        user_id=user_id,
        tags=["bookstore-support"],
    ):
        # Show only the customer message as trace input, not every function arg
        # (`graph` would otherwise be dumped into the trace).
        langfuse.update_current_span(input={"question": question})

        config = {
            "configurable": {"thread_id": thread_id},
            # One handler for the whole run; it inherits the trace opened above.
            # LangSmith needs no handler — the LANGSMITH_* env vars enable it.
            "callbacks": [CallbackHandler()],
            "run_name": "support-graph",
        }

        state = await graph.ainvoke({"question": question, "tool_trail": []}, config=config)

        pending = state.get("__interrupt__")
        if pending:
            payload = pending[0].value
            print(f"  [PAUSED] Jev suggests '{payload['jev_route']}'. "
                  f"Human replies: {human_reply!r}")
            state = await graph.ainvoke(Command(resume=human_reply), config=config)

        langfuse.update_current_span(
            output={
                "answer": state["answer"],
                "lane": state["approved_route"],
                "tools_used": state["tool_trail"],
                "stop_reason": state.get("stop_reason", ""),
            }
        )
        return state


async def main() -> None:
    print("Loading MCP tools from two stdio servers...")
    mcp_tools = await load_mcp_tools()
    for server, tools in mcp_tools.items():
        print(f"  {server:8s}: {[t.name for t in tools]}")

    graph = build_support_graph(mcp_tools)
    print("\nGraph topology (mermaid):\n")
    print(graph.get_graph().draw_mermaid())

    for index, (question, reply) in enumerate(SAMPLE_MESSAGES, start=1):
        await answer_message(graph, question, reply, thread_id=f"lesson10-thread-{index}")

    # Short-lived script: flush or the traces never leave the process.
    langfuse.flush()
    print("\nTraces sent to Langfuse and LangSmith.")


if __name__ == "__main__":
    asyncio.run(main())
