from typing import Annotated, Literal
from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

llm = init_chat_model("openai:gpt-5.6-luna")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def human_node(state: State) -> State:
    in_message = input("You: ")
    print("Human:", in_message)
    return {"messages": [{"role": "user", "content": in_message}]}

def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    print("Bot:", response.content)
    return {"messages": [response]}

def should_continue(state: State) -> Literal["continue", "end"]:
    last = state["messages"][-1]
    content = last["content"] if isinstance(last, dict) else last.content
    if str(content).strip().lower() in {"quit", "exit"}:
        return "end"
    return "continue"

builder = StateGraph(State)
builder.add_node("human_node", human_node)
builder.add_node("chatbot_node", chatbot)

builder.add_edge(START, "human_node")
builder.add_conditional_edges(
    "human_node",
    should_continue,
    {
        "continue": "chatbot_node",
        "end": END,
    },
)
builder.add_edge("chatbot_node", "human_node")

graph = builder.compile()

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# One invoke runs the cycle. Type quit or exit to follow the conditional edge to END.
# messages is kept in graph state across cycles by the add_messages reducer.
final_state = graph.invoke({"messages": []}, {"recursion_limit": 100})
final_state["messages"]
