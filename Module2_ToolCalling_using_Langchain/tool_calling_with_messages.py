from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from rich import print

load_dotenv()

#1 creating a tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in the text"""
    return len(text)

tools = {
    "get_text_length": get_text_length,

}

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# tool binding
llm_with_tools = llm.bind_tools([get_text_length])

messages = []
# prompt = input("Enter your prompt: ")
query = HumanMessage(content="What is the length of the text 'Hello, world!'?")
messages.append(query)

print(messages)
print("--------------------------------")

result = llm_with_tools.invoke(messages)

print(result)
messages.append(result)
print(messages)
print("*********************************")

# extracting tool name from the result
if result.tool_calls:
    tool_call = result.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id = tool_call["id"]
    print(tool_name)
    print(tool_args)
    print(tool_id)
    
    # calling the tool
    tool_result = tools[tool_name].invoke(tool_args)
    print(tool_result)
    messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))
    print(messages)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    final_response = llm.invoke(messages[-10:])
    print(final_response)
    messages.append(final_response)
    print(messages)
    print("################################")

# if result.tool_calls:
#     for tool_call in result.tool_calls:
#         tool_name = tool_call["name"]
#         tool_result = tools[tool_name].invoke(tool_call["args"])
#         print(tool_result)
#         messages.append(tool_result)
#         print(messages)

# final_response = llm.invoke(messages)
# print(final_response)
# messages.append(final_response)
# print(messages)