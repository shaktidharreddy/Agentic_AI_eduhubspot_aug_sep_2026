from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field


class State(TypedDict):
    messages: Annotated[list, add_messages]


class BuyDecision(BaseModel):
    """Human response for the buy interrupt.

    response_schema also accepts a typing_extensions.TypedDict, a dataclass,
    or a raw JSON Schema dict. A model, TypedDict, or dataclass is validated
    on resume; a JSON Schema dict is exposed to clients but not validated.
    Requires langgraph>=1.2.12.
    """

    approved: bool = Field(description="Whether to approve the purchase.")
    note: str | None = Field(default=None, description="Optional reviewer note.")


@tool
def get_stock_price(symbol: str) -> float:
    """Return the current price of a stock given the stock symbol"""
    return {"MSFT": 200.3, "AAPL": 100.4, "AMZN": 150.0, "RIL": 87.6}.get(symbol, 0.0)


@tool
def buy_stocks(symbol: str, quantity: int, total_price: float) -> str:
    """Buy stocks given the stock symbol and quantity"""
    decision = interrupt(
        {
            "action": "buy_stocks",
            "symbol": symbol,
            "quantity": quantity,
            "total_price": total_price,
            "question": f"Approve buying {quantity} {symbol} stocks for ${total_price:.2f}?",
        },
        response_schema=BuyDecision,
    )

    if decision.approved:
        note = f" Note: {decision.note}" if decision.note else ""
        return f"You bought {quantity} shares of {symbol} for a total price of {total_price}.{note}"
    note = f" Note: {decision.note}" if decision.note else ""
    return f"Buying declined.{note}"


tools = [get_stock_price, buy_stocks]

llm = init_chat_model("openai:gpt-5.6-luna", use_responses_api=True)
llm_with_tools = llm.bind_tools(tools +[{"type": "web_search"}])


def chatbot_node(state: State):
    msg = llm_with_tools.invoke(state["messages"] + [{"role": "system", "content": "You are a helpful assistant that can answer questions and help with tasks."}])
    return {"messages": [msg]}


checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "buy_thread"}}

# Step 1: user asks price
state = graph.invoke(
    {"messages": [{"role": "user", "content": "What is the current price of 10 MSFT stocks?"}]},
    config=config,
)
print(state["messages"][-1].content)

# Step 2: user asks to buy
state = graph.invoke(
    {"messages": [{"role": "user", "content": "Buy 10 MSFT stocks at current price."}]},
    config=config,
)
pending = state["__interrupt__"][0]
print(pending.value)
print(pending.response_schema)

approved = input("Approve (yes/no): ").strip().lower() in {"yes", "y"}
note = input("Note (optional): ").strip() or None
state = graph.invoke(
    Command(resume={"approved": approved, "note": note}),
    config=config,
)
print(state["messages"][-1].content)
