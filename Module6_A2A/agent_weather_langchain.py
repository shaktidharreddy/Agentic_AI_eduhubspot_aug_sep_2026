"""Agent 1: a weather agent built with LangChain's prebuilt agent.

Framework-specific code stops at `answer()`. Below that line it is plain A2A.
"""

import uvicorn
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from a2a_min import build_a2a_server, build_agent_card
from config import (
    HEARTBEAT_SECONDS,
    HOST,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REGISTRY_URL,
    WEATHER_AGENT_PORT,
    WEATHER_AGENT_URL,
    WEATHER_MODEL,
)


def get_weather(city: str) -> str:
    """Look up today's weather for a city."""
    forecasts = {
        "london": "14 degrees C and drizzling",
        "mumbai": "31 degrees C and humid",
        "delhi": "27 degrees C and hazy",
        "tokyo": "19 degrees C and clear",
    }
    return forecasts.get(city.lower(), f"23 degrees C and sunny in {city}")


# OpenRouter is OpenAI-API-compatible, so the ordinary OpenAI provider works:
# just point it at OpenRouter's base URL.
model = init_chat_model(
    WEATHER_MODEL,
    model_provider="openai",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

weather_agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You report the weather. Call get_weather, then reply in one short sentence.",
)


async def answer(question: str) -> str:
    """The one function A2A needs: a question in, a reply out."""
    result = await weather_agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    # A model can end its turn with an empty message, so take the last one
    # that actually said something.
    said = [m.text for m in result["messages"] if m.type == "ai" and m.text.strip()]
    return said[-1] if said else "Sorry, I could not answer that."


card = build_agent_card(
    name="Weather Agent",
    description="Reports the current weather for a city. Built with LangChain.",
    url=WEATHER_AGENT_URL,
    framework="LangChain",
    skill_id="weather_lookup",
    skill_description="Given a city name, return today's temperature and conditions.",
    examples=["What is the weather in London?", "Is it raining in Mumbai?"],
)

# Passing the registry URL is what makes registration automatic: on startup
# this agent announces itself, then keeps doing so every few seconds.
app = build_a2a_server(card, answer, REGISTRY_URL, HEARTBEAT_SECONDS)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=WEATHER_AGENT_PORT)
