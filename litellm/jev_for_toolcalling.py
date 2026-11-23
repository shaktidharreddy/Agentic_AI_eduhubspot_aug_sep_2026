"""jev-as-router: a tool-calling loop where TypeSafe `jev` (a classifier, not a
generative model) picks the next tool each iteration, and gpt-5-nano is only
used once at the very end to write the natural-language answer.

Loop: render messages + OpenAI-style tool schemas -> jev `choice` picks a tool
(or `no_tool`) -> jev `choice` resolves each enum arg -> call the Python
function -> append AIMessage(tool_calls) + ToolMessage via the `add_messages`
reducer -> repeat until jev answers `no_tool`.

Docs: https://docs.typesafe.ai/introduction/quickstart
"""

import json
import os
import ssl
import uuid
from typing import Annotated, Any, Literal, TypedDict

import httpx
import truststore
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_litellm import ChatLiteLLM
from langgraph.graph.message import add_messages

load_dotenv()

# corporate TLS interception: trust the Windows cert store instead of certifi
SSL_CTX = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
FINAL_MODEL = "gpt-5-nano"
MAX_ITERATIONS = 8  # safety net; the real exit is jev answering "no_tool"

USER_QUESTION = (
    "I'm flying to Paris on Air France flight AF1234 next week. What's the "
    "weather like there, is my flight on time, and how much baggage can I bring?"
)


# ---------------------------------------------------------------------------
# 1. Tools: @tool derives the OpenAI schema; Literal becomes an enum
# ---------------------------------------------------------------------------
# jev can only classify, so every argument must be a Literal it can choose from.
@tool(parse_docstring=True)
def get_weather(city: Literal["paris", "london", "tokyo", "new_york"]) -> str:
    """Current weather conditions for a city.

    Args:
        city: City to look up
    """
    return {
        "paris": "14C, light rain, wind 18 km/h",
        "london": "11C, overcast",
        "tokyo": "22C, clear",
        "new_york": "9C, windy",
    }[city]


@tool(parse_docstring=True)
def get_flight_status(flight_number: Literal["AF1234", "BA5678", "LH9012"]) -> str:
    """Departure status and gate for a flight number.

    Args:
        flight_number: IATA flight number
    """
    return {
        "AF1234": "on time, departs 09:40 CDG gate K32",
        "BA5678": "delayed 45 min",
        "LH9012": "cancelled",
    }[flight_number]


@tool(parse_docstring=True)
def get_baggage_policy(
    airline: Literal["air_france", "british_airways", "lufthansa"],
) -> str:
    """Cabin and checked baggage allowance for an airline.

    Args:
        airline: Airline operating the flight
    """
    return {
        "air_france": "1 cabin bag 12kg + 1 checked bag 23kg included in Economy",
        "british_airways": "1 cabin bag 23kg, checked bags from GBP 40",
        "lufthansa": "1 cabin bag 8kg, 1 checked bag 23kg included",
    }[airline]


TOOLS = [get_weather, get_flight_status, get_baggage_policy]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
SCHEMAS = [convert_to_openai_tool(t) for t in TOOLS]
PARAMS_BY_NAME = {s["function"]["name"]: s["function"]["parameters"] for s in SCHEMAS}

NO_TOOL = "no_tool"


# ---------------------------------------------------------------------------
# 2. jev client
# ---------------------------------------------------------------------------
def jev(state: str, questions: dict[str, Any]) -> dict[str, Any]:
    resp = httpx.post(
        JEV_URL,
        headers={
            "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={"state": state, "model": JEV_MODEL, "questions": questions},
        timeout=60.0,
        verify=SSL_CTX,
    )
    resp.raise_for_status()
    return resp.json()["answers"]


# ---------------------------------------------------------------------------
# 3. State with the add_messages reducer
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def render_messages(messages: list[AnyMessage]) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"TOOL_CALL: {tc['name']}({json.dumps(tc['args'])})")
        elif isinstance(m, ToolMessage):
            lines.append(f"TOOL_RESULT[{m.name}]: {m.content}")
    return "\n".join(lines)


def build_state(messages: list[AnyMessage]) -> str:
    return (
        "You are routing a tool-calling agent.\n\n"
        "AVAILABLE TOOL SCHEMAS:\n"
        f"{json.dumps(SCHEMAS, indent=2)}\n\n"
        "CONVERSATION SO FAR:\n"
        f"{render_messages(messages)}"
    )


# ---------------------------------------------------------------------------
# 4. The jev routing loop
# ---------------------------------------------------------------------------
def select_tool(messages: list[AnyMessage]) -> tuple[str, float]:
    criteria = {t.name: t.description for t in TOOLS}
    criteria[NO_TOOL] = (
        "Every part of the user's request is already covered by a TOOL_RESULT "
        "above; no further tool call is needed"
    )
    answer = jev(
        build_state(messages),
        {
            "next_tool": {
                "type": "choice",
                "instructions": (
                    "Which tool should be called next to make progress on the "
                    "user's request? Do not repeat a tool that already has a "
                    "TOOL_RESULT"
                ),
                "criteria": criteria,
            }
        },
    )["next_tool"]
    return answer["choice"], answer["confidence"]


def resolve_args(messages: list[AnyMessage], tool_name: str) -> dict[str, str]:
    """One jev choice question per required enum parameter."""
    params = PARAMS_BY_NAME[tool_name]
    questions: dict[str, Any] = {}
    for name in params["required"]:
        spec = params["properties"][name]
        questions[name] = {
            "type": "choice",
            "instructions": (
                f"Calling `{tool_name}`. Which value of `{name}` "
                f"({spec['description']}) does the user's request refer to?"
            ),
            "criteria": {v: f"{name} = {v}" for v in spec["enum"]},
        }
    if not questions:
        return {}
    answers = jev(build_state(messages), questions)
    return {name: answers[name]["choice"] for name in questions}


def run_jev_loop(question: str) -> AgentState:
    messages: list[AnyMessage] = [HumanMessage(question)]
    seen: set[str] = set()

    for i in range(1, MAX_ITERATIONS + 1):
        tool_name, confidence = select_tool(messages)
        print(f"\n[iter {i}] jev -> {tool_name} (confidence {confidence})")

        if tool_name == NO_TOOL:
            print(f"[iter {i}] circuit breaker: jev sees nothing left to call")
            break

        args = resolve_args(messages, tool_name)
        signature = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        if signature in seen:
            print(f"[iter {i}] repeat of {signature}, stopping to avoid a loop")
            break
        seen.add(signature)

        result = TOOLS_BY_NAME[tool_name].invoke(args)
        print(f"[iter {i}] {tool_name}({args}) -> {result}")

        call_id = f"call_{uuid.uuid4().hex[:12]}"
        # add_messages is typed against the loose MessageLikeRepresentation union
        messages = add_messages(  # type: ignore[assignment]
            messages,  # type: ignore[arg-type]
            [
                AIMessage(
                    content="",
                    tool_calls=[{"id": call_id, "name": tool_name, "args": args}],
                ),
                ToolMessage(content=result, name=tool_name, tool_call_id=call_id),
            ],
        )
    else:
        print(f"\nhit MAX_ITERATIONS={MAX_ITERATIONS} without jev choosing {NO_TOOL}")

    return {"messages": messages}


# ---------------------------------------------------------------------------
# 5. One generative call at the end
# ---------------------------------------------------------------------------
def summarise(state: AgentState) -> str:
    llm = ChatLiteLLM(
        model=f"litellm_proxy/{FINAL_MODEL}",
        api_key=os.environ["LITELLM_API_KEY"],
        api_base=os.environ["LITELLM_BASE_URL"],
        max_retries=2,
    )
    system = SystemMessage(
        "Answer the user's question in natural language using only the tool "
        "results in this conversation. Be concise and do not request more tools."
    )
    return str(llm.invoke([system, *state["messages"]]).content)


if __name__ == "__main__":
    print(f"USER: {USER_QUESTION}")
    final_state = run_jev_loop(USER_QUESTION)

    print("\n--- messages ---")
    for m in final_state["messages"]:
        print(f"{type(m).__name__}: {m.content or getattr(m, 'tool_calls', None)}")

    print("\n--- final answer (gpt-5-nano, no tools) ---")
    print(summarise(final_state))
