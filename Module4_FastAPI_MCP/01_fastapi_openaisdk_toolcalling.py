# 1_openai.py

from inspect import getdoc, signature

from openai import OpenAI
import requests
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def add_via_api(a: int, b: int) -> int:
    """Add two numbers using the calculator API."""

    response = requests.get(
        "http://localhost:8000/add",
        params={
            "a": a,
            "b": b
        }
    )

    return response.json()["result"]


def multiply_via_api(a: int, b: int) -> int:
    """Multiply two numbers using the calculator API."""

    response = requests.get(
        "http://localhost:8000/multiply",
        params={
            "a": a,
            "b": b
        }
    )

    return response.json()["result"]


# def to_function_tool(fn):
#     properties = {
#         name: {"type": "integer"}
#         for name in signature(fn).parameters
#     }
#     return {
#         "type": "function",
#         "name": fn.__name__,
#         "description": getdoc(fn) or "",
#         "parameters": {
#             "type": "object",
#             "properties": properties,
#             "required": list(properties),
#         },
#     }


# functions = [add_via_api, multiply_via_api]
# tools = [to_function_tool(fn) for fn in functions]
# tool_dict = {fn.__name__: fn for fn in functions}

tools = [
    {
        "type": "function",
        "name": "add",
        "description": "Add two numbers using the calculator API",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "type": "function",
        "name": "multiply",
        "description": "Multiply two numbers using the calculator API",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
]

tool_dict = {
    "add": add_via_api,
    "multiply": multiply_via_api,
}

response = client.responses.create(
    model="gpt-5",
    input="What is 10 + 20? And what is 10 * 20?",
    tools=tools,
    tool_choice="required",
)

print(response.output)

for item in response.output:

    if item.type == "function_call":

        args = json.loads(item.arguments)

        result = tool_dict[item.name](**args)

        print("API result:", result)
