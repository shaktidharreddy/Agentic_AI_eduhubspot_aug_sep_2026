import requests
from langchain.tools import tool

@tool #decorator for creating a tool
def get_greeting(name: str) -> str:
    """"Generate a greeting message for a user""" #docstring

    return f"Hello {name}. Welcome to the AI world"

greeting_tool = get_greeting

result = greeting_tool.invoke({"name": "John"})
print(result)

print(greeting_tool.name)
print(greeting_tool.description)
print(greeting_tool.args)


# Base URL of the local FastAPI server (see fastapiserver.py)
API_BASE_URL = "http://127.0.0.1:8000"


def _call_math_api(operation: str, a: float, b: float) -> float:
    """Helper that calls a math endpoint on the FastAPI server and returns the result."""

    response = requests.get(f"{API_BASE_URL}/{operation}", params={"a": a, "b": b}, timeout=10)
    response.raise_for_status()
    return response.json()["result"]


@tool #decorator for creating a tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers using my FastAPI /add endpoint."""
    
    return _call_math_api("add", a, b)


@tool #decorator for creating a tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract two numbers using my FastAPI /subtract endpoint."""

    return _call_math_api("subtract", a, b)


@tool #decorator for creating a tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers using my FastAPI /multiply endpoint."""

    return _call_math_api("multiply", a, b)


@tool #decorator for creating a tool
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers using my FastAPI /divide endpoint."""

    return _call_math_api("divide", a, b)


math_tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]

for math_tool in math_tools:
    print(math_tool.name, "->", math_tool.description)

# Invoke the tools -> these call the FastAPI server running in fastapiserver.py
print("add:", add_numbers.invoke({"a": 10, "b": 5}))
print("subtract:", subtract_numbers.invoke({"a": 10, "b": 5}))
print("multiply:", multiply_numbers.invoke({"a": 10, "b": 5}))
print("divide:", divide_numbers.invoke({"a": 10, "b": 5}))