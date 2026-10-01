"""Agent 2: a currency agent built with Google ADK.

A completely different framework from Agent 1 -- different agent class, different
tool format, different way of running a turn. Yet the last few lines are the
same as Agent 1, so callers cannot tell the two apart.
"""

import uvicorn
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from a2a_min import build_a2a_server, build_agent_card
from config import (
    CURRENCY_AGENT_PORT,
    CURRENCY_AGENT_URL,
    CURRENCY_MODEL,
    HEARTBEAT_SECONDS,
    HOST,
    REGISTRY_URL,
)

APP_NAME = "currency_app"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount from one currency to another."""
    per_usd = {"USD": 1.0, "EUR": 0.92, "GBP": 0.78, "INR": 88.5, "JPY": 157.0}
    source, target = from_currency.upper(), to_currency.upper()

    if source not in per_usd or target not in per_usd:
        return {"error": f"I only know these currencies: {', '.join(per_usd)}"}

    converted = amount / per_usd[source] * per_usd[target]
    return {"amount": round(converted, 2), "currency": target}


currency_agent = LlmAgent(
    name="currency_agent",
    # ADK defaults to Gemini. LiteLlm is its escape hatch to any other
    # provider; it reads OPENROUTER_API_KEY from the environment.
    model=LiteLlm(model=CURRENCY_MODEL),
    instruction="You convert currencies. Call convert_currency, then reply in one short sentence.",
    tools=[convert_currency],
)

runner = InMemoryRunner(agent=currency_agent, app_name=APP_NAME)


async def answer(question: str) -> str:
    """The same one function A2A needs -- just driven by ADK's Runner instead."""
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="a2a_caller")
    message = types.Content(role="user", parts=[types.Part(text=question)])

    reply = ""
    async for event in runner.run_async(
        user_id="a2a_caller", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            reply = event.content.parts[0].text
    return reply or "Sorry, I could not answer that."


card = build_agent_card(
    name="Currency Agent",
    description="Converts money between currencies. Built with Google ADK.",
    url=CURRENCY_AGENT_URL,
    framework="Google ADK",
    skill_id="currency_conversion",
    skill_description="Convert an amount between USD, EUR, GBP, INR and JPY.",
    examples=["Convert 100 USD to INR", "How many euros is 50 pounds?"],
)

# Same two lines as the LangChain agent: announce on startup, then heartbeat.
app = build_a2a_server(card, answer, REGISTRY_URL, HEARTBEAT_SECONDS)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=CURRENCY_AGENT_PORT)
