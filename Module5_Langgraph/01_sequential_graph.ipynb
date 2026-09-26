from typing import TypedDict
# from pydantic import BaseModel

class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    total_inr: float

def calc_total(state: PortfolioState) -> PortfolioState:
    state['total_usd'] = state['amount_usd'] * 1.08
    return state

def convert_to_inr(state: PortfolioState) -> PortfolioState:
    state['total_inr'] = state['total_usd'] * 85
    return state

from langgraph.graph import StateGraph, START, END

builder = StateGraph(PortfolioState)

builder.add_node("abc", calc_total)
builder.add_node("def", convert_to_inr)

builder.add_edge(START, "abc")
builder.add_edge("abc", "def")
builder.add_edge("def", END)

graph = builder.compile()

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

graph.invoke({"amount_usd": 100000, "total_usd": "dummy", "floattotal_inr": "dummy"})
