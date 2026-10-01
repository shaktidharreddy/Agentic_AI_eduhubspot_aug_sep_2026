"""The supervisor: a three-node LangGraph that routes over A2A.

    discover  ->  decide  ->  call_agent

  discover    asks the registry which agents claim the needed capability
  decide      hands those cards to Jev, which picks one
  call_agent  sends `message/send` to the chosen agent's URL

Two different jobs, two different models. The registry does cheap word
matching to shortlist; Jev makes the actual choice and reports how confident
it was. Neither one has any idea which framework built the agents, and the
supervisor imports neither LangChain's agent nor Google ADK. All it ever sees
is agent cards and HTTP -- which is exactly what makes A2A useful.

Every node appends to `trace`, so the UI can show its work step by step.
"""

import operator
from typing import Annotated, Any, TypedDict

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

import jev
from a2a_min import send_message
from config import HOST, REGISTRY_URL, SUPERVISOR_PORT


class NoAgentsRegistered(RuntimeError):
    """Raised when the registry is empty, which is a setup problem, not a bug."""


class SupervisorState(TypedDict):
    question: str
    cards: list[dict[str, Any]]
    registry_best_match: str
    chosen_agent: str
    answer: str
    trace: Annotated[list[dict[str, Any]], operator.add]


async def discover(state: SupervisorState) -> dict:
    """Step 1: ask the registry who can handle this, ranked by capability."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{REGISTRY_URL}/discover", params={"capability": state["question"]}
        )
        response.raise_for_status()
        ranked = response.json()

    if not ranked:
        raise NoAgentsRegistered(
            "No agents are registered yet. Start the agents and give them a "
            "few seconds to announce themselves."
        )

    cards = [row["card"] for row in ranked]
    registry_pick = ranked[0]["card"]["name"]
    print(f"[supervisor] discovered: {[card['name'] for card in cards]}")
    return {
        "cards": cards,
        "registry_best_match": registry_pick,
        "trace": [
            {
                "step": "discover",
                "summary": (
                    f"Registry ranked {len(cards)} agent card(s); "
                    f"its best keyword match is {registry_pick}"
                ),
                "data": {
                    "registry_best_match": registry_pick,
                    "asked": f"GET {REGISTRY_URL}/discover?capability={state['question']}",
                    "ranked": [
                        {
                            "name": row["card"]["name"],
                            "framework": row["card"]["framework"],
                            "keyword_score": row["score"],
                            "matched_words": ", ".join(row["matched"]) or "-",
                            "best_match": row["best_match"],
                            "url": row["card"]["url"],
                        }
                        for row in ranked
                    ],
                },
            }
        ],
    }


async def decide(state: SupervisorState) -> dict:
    """Step 2: Jev picks one card. The menu is built from the cards, not code."""
    options = {card["name"]: card["skills"][0]["description"] for card in state["cards"]}

    verdict = jev.choose(
        state={"question": state["question"]},
        options=options,
        instructions="Which agent should handle this question?",
    )

    chosen = _match_card(state["cards"], verdict["choice"])["name"]
    agreed = chosen == state["registry_best_match"]
    print(
        f"[supervisor] Jev chose {chosen} "
        f"(confidence={verdict['confidence']:.2f}) {verdict['probabilities']}"
    )
    return {
        "chosen_agent": chosen,
        "trace": [
            {
                "step": "decide",
                "summary": (
                    f"Jev chose {chosen} with {verdict['confidence']:.0%} confidence "
                    f"({'agrees with' if agreed else 'overrides'} the registry's keyword pick)"
                ),
                "data": {
                    "model": verdict["model"],
                    "registry_best_match": state["registry_best_match"],
                    "agreed_with_registry": agreed,
                    "options_offered": options,
                    "probabilities": verdict["probabilities"],
                    "confidence": verdict["confidence"],
                    "cost_usd": verdict["cost_usd"],
                },
            }
        ],
    }


async def call_agent(state: SupervisorState) -> dict:
    """Step 3: invoke the chosen agent over A2A and keep its reply."""
    card = _match_card(state["cards"], state["chosen_agent"])
    call = await send_message(card, state["question"])

    return {
        "answer": call["reply"],
        "trace": [
            {
                "step": "call_agent",
                "summary": f"Sent A2A message/send to {card['name']} at {card['url']}",
                "data": {
                    "framework": card["framework"],
                    "a2a_request": call["request"],
                    "a2a_response": call["response"],
                },
            }
        ],
    }


def _match_card(cards: list[dict], name: str) -> dict:
    """Find the card that was chosen, tolerating extra words or odd casing."""
    name = name.strip().lower()
    for card in cards:
        if card["name"].lower() in name or name in card["name"].lower():
            return card
    return cards[0]


builder = StateGraph(SupervisorState)
builder.add_node("discover", discover)
builder.add_node("decide", decide)
builder.add_node("call_agent", call_agent)
builder.add_edge(START, "discover")
builder.add_edge("discover", "decide")
builder.add_edge("decide", "call_agent")
builder.add_edge("call_agent", END)

graph = builder.compile()


# --------------------------------------------------------------------------
# A thin HTTP front door so the whole flow can be driven with one request.
# --------------------------------------------------------------------------
app = FastAPI(title="LangGraph A2A Supervisor")


class Ask(BaseModel):
    question: str


@app.post("/ask")
async def ask(request: Ask) -> dict:
    try:
        result = await graph.ainvoke({"question": request.question, "trace": []})
    except NoAgentsRegistered as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        # Most often an OpenRouter credit or rate-limit error. Pass the reason
        # through rather than letting it surface as a bare 500.
        raise HTTPException(status_code=502, detail=f"{type(error).__name__}: {error}") from error

    return {
        "question": request.question,
        "registry_best_match": result["registry_best_match"],
        "routed_to": result["chosen_agent"],
        "answer": result["answer"],
        "trace": result["trace"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=SUPERVISOR_PORT)
