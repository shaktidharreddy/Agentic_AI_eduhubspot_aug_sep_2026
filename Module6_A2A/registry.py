"""The agent registry: a phone book of agent cards.

An agent registers by sending only its base URL. The registry then fetches
that agent's card itself, so the card always comes from the agent and can
never drift out of date -- and a successful fetch doubles as proof the agent
is actually alive right now.

Three ways to read it back:

    GET /agents                    every card it holds
    GET /discover?capability=...   cards ranked by how well they match
    GET /status                    who is registered, and how fresh they are
"""

import re
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from a2a_min import AGENT_CARD_PATH, fetch_agent_card
from config import HOST, REGISTRY_PORT

app = FastAPI(title="Minimal A2A Agent Registry")

# The entire registry, held in memory in this one dict: agent name -> entry.
# Nothing is persisted, so restarting the registry empties it -- the agents
# then re-announce themselves on their next heartbeat. A real registry would
# put this in Redis or Postgres, but the shape of the data would be the same.
AGENTS: dict[str, dict[str, Any]] = {}


class Registration(BaseModel):
    base_url: str


@app.post("/register")
async def register(registration: Registration) -> dict:
    """Register (or refresh) an agent by URL; the registry pulls its card.

    Fetching the card is also the liveness check: an agent that cannot serve
    its card right now does not get in.
    """
    try:
        card = await fetch_agent_card(registration.base_url)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read an agent card from {registration.base_url}. "
                f"Expected {registration.base_url.rstrip('/')}{AGENT_CARD_PATH} "
                f"to return one. ({type(error).__name__}: {error})"
            ),
        ) from error

    name = card["name"]

    existing = AGENTS.get(name)
    AGENTS[name] = {
        "card": card,
        "first_registered": existing["first_registered"] if existing else time.time(),
        "last_seen": time.time(),
        "heartbeats": (existing["heartbeats"] + 1) if existing else 1,
    }

    if not existing:
        print(f"[registry] registered {name} ({card['framework']}) at {card['url']}")
    return {
        "registered": name,
        "framework": card["framework"],
        "skills": [skill["id"] for skill in card["skills"]],
        "heartbeats": AGENTS[name]["heartbeats"],
    }


@app.get("/agents")
def list_agents() -> list[dict]:
    """Every card we hold. This is the raw discovery feed."""
    return [entry["card"] for entry in AGENTS.values()]


@app.get("/agents/{name}")
def get_agent(name: str) -> dict:
    """One card by name."""
    if name not in AGENTS:
        raise HTTPException(status_code=404, detail=f"No agent named {name!r} is registered")
    return AGENTS[name]["card"]


@app.get("/discover")
def discover(capability: str) -> list[dict]:
    """Capability-based discovery: rank cards by how well they match a request.

    Returns `{"score", "matched", "best_match", "card"}` per agent, best first.

    The scoring is deliberately dumb word overlap against each card's
    advertised skills. It ranks but does not eliminate, so the supervisor can
    hand the whole list to Jev and you can watch the two disagree. A production
    registry would do this same job with embeddings or a skills taxonomy.
    """
    wanted = _words(capability)
    ranked = []

    for entry in AGENTS.values():
        card = entry["card"]
        skill = card["skills"][0]
        advertised = _words(
            f"{card['description']} {skill['description']} {' '.join(skill['tags'])} "
            f"{' '.join(skill.get('examples', []))}"
        )
        overlap = wanted & advertised
        ranked.append({"score": len(overlap), "matched": sorted(overlap), "card": card})

    ranked.sort(key=lambda row: row["score"], reverse=True)
    for position, row in enumerate(ranked):
        row["best_match"] = position == 0

    print(
        f"[registry] discover({capability!r}) -> "
        f"{[(row['card']['name'], row['score']) for row in ranked]}"
    )
    return ranked


@app.get("/best-match")
def best_match(capability: str) -> dict:
    """The single best-matching agent for a capability, card included.

    A client that just wants an answer can call this and skip the ranking. Be
    aware of what it means when `score` is 0: no advertised skill shared a word
    with the request, so "best" is really just first. That is exactly the case
    the supervisor hands to Jev instead of trusting.
    """
    ranked = discover(capability)
    if not ranked:
        raise HTTPException(status_code=404, detail="No agents are registered")
    return ranked[0]


@app.get("/status")
def status() -> list[dict]:
    """A flat summary for the UI: who is here, and how recently they checked in."""
    now = time.time()
    return [
        {
            "name": name,
            "framework": entry["card"]["framework"],
            "url": entry["card"]["url"],
            "skills": [skill["id"] for skill in entry["card"]["skills"]],
            "heartbeats": entry["heartbeats"],
            "seconds_since_last_seen": round(now - entry["last_seen"], 1),
        }
        for name, entry in AGENTS.items()
    ]


def _words(text: str) -> set[str]:
    """Lowercase words of three letters or more, ignoring common filler."""
    filler = {"the", "and", "for", "you", "what", "how", "many", "much", "into", "between", "given"}
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3} - filler


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=REGISTRY_PORT)
