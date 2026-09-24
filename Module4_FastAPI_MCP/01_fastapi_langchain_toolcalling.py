# 1_langchain.py

import requests

from langchain.agents import create_agent


def add_via_api(a: int, b: int) -> int:
    """Add two numbers using the FastAPI calculator."""

    response = requests.get(
        "http://localhost:8000/add",
        params={
            "a": a,
            "b": b
        }
    )

    return response.json()["result"]


def multiply_via_api(a: int, b: int) -> int:
    """Multiply two numbers using the FastAPI calculator."""

    response = requests.get(
        "http://localhost:8000/multiply",
        params={
            "a": a,
            "b": b
        }
    )

    return response.json()["result"]


agent = create_agent(
    "openai:gpt-5",
    tools=[add_via_api, multiply_via_api]
)


result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is 10 + 20? And what is 10 * 20?"
        }
    ]
})


print(result["messages"][-1].content)