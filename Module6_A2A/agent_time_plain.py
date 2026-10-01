"""Agent 3: a time agent with no agent framework at all.

This one exists to make a point. There is no LangChain here, no Google ADK,
and no model of any kind -- just a dict and an f-string. It publishes the same
agent card, self-registers the same way, and answers `message/send` the same
way, so the registry and the supervisor cannot tell it apart from the other
two. A2A only cares about the two endpoints, never about what is behind them.

It also stands in for an agent hosted by somebody else. Unlike the other two,
it is NOT given the registry's URL, so it never announces itself. It just sits
there serving its card, and somebody has to register it by URL -- from the
Streamlit UI, or with one POST. That is the path you would use for an agent
running on another machine that you do not control.

It is the template for adding your own agent too: copy this file, change the
card, write `answer()`, add it to run_all.py. Nothing else in the project
changes -- not the registry, and not the supervisor.
"""

from datetime import datetime, timedelta, timezone

import uvicorn

from a2a_min import build_a2a_server, build_agent_card
from config import HOST, TIME_AGENT_PORT, TIME_AGENT_URL

# Hours ahead of UTC. Deliberately fixed, so no timezone database is needed.
OFFSETS = {"london": 0, "mumbai": 5.5, "delhi": 5.5, "tokyo": 9, "new york": -5}


async def answer(question: str) -> str:
    """The same one function A2A needs -- here it is just string handling."""
    city = next((name for name in OFFSETS if name in question.lower()), None)
    if city is None:
        return f"I only know the time in: {', '.join(sorted(OFFSETS))}."

    local = datetime.now(timezone.utc) + timedelta(hours=OFFSETS[city])
    return f"It is {local:%H:%M} in {city.title()}."


card = build_agent_card(
    name="Time Agent",
    description="Tells the current local time in a city. Built with plain Python.",
    url=TIME_AGENT_URL,
    framework="Plain Python",
    skill_id="local_time",
    skill_description="Given a city name, return the current local clock time there.",
    examples=["What time is it in Tokyo?", "Current time in New York"],
)

# No registry URL, so no heartbeat: this agent waits to be registered by URL.
app = build_a2a_server(card, answer)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=TIME_AGENT_PORT)
