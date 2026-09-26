from typing import TypedDict, Literal
from typing_extensions import Annotated
from typing import List
from operator import add

class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    target_currency: Literal["INR", "EUR", "RUB"]
    total: Annotated[List[float], add]

def calc_total(state: PortfolioState) -> PortfolioState:
    state['total_usd'] = state['amount_usd'] * 1.08
    return state

def convert_to_inr(state: PortfolioState) -> dict:
    return {"total":  [1.2]}

def convert_to_eur(state: PortfolioState) -> dict:
    return {"total":  [2.2]}

def convert_to_rub(state: PortfolioState) -> dict:
    return {"total": [3.2]}

def choose_conversion(state: PortfolioState) -> str:
    return state["target_currency"]

from langgraph.graph import StateGraph, START, END

builder = StateGraph(PortfolioState)

builder.add_node("calc_total_node", calc_total)
builder.add_node("convert_to_inr_node", convert_to_inr)
builder.add_node("convert_to_eur_node", convert_to_eur)
builder.add_node("convert_to_rub_node", convert_to_rub)

builder.add_edge(START, "calc_total_node")


#parallel execution of all 3 nodes
builder.add_edge("calc_total_node", "convert_to_inr_node")
builder.add_edge("calc_total_node", "convert_to_eur_node")
builder.add_edge("calc_total_node", "convert_to_rub_node")


# conditional execution of all 3 nodes

# builder.add_conditional_edges(
#     "calc_total_node",
#     choose_conversion,
#     {
#         "INR": "convert_to_inr_node",
#         "EUR": "convert_to_eur_node",
#         "RUB": "convert_to_rub_node",
#     }
# )

builder.add_edge(["convert_to_inr_node", "convert_to_eur_node", "convert_to_rub_node"], END)

graph = builder.compile()

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

graph.invoke({"amount_usd": 1000, "target_currency": "EUR"})
