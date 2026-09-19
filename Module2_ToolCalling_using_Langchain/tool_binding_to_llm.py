from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from rich import print

load_dotenv()

#1 creating a tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in the text"""
    return len(text)

# tool invocation without llm
# print(get_text_length.invoke({"text": "Hello, world!"}))

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# tool binding
llm_with_tools = llm.bind_tools([get_text_length, ])

# print(llm.invoke("Hello"))
# print("--------------------------------")
# print(llm_with_tools.invoke("Hello"))

# result1 = llm.invoke("What is the number of characters in the text: 'Hello, world!'?")
# print(result1)
# print(result1.tool_calls)

print("--------------------------------")

# tool calling
result2 = llm_with_tools.invoke("What is the number of characters in the text: 'Hello, world!'?")
# print(result2)

# print the tool calls
# print(result2.tool_calls)     # unique identifier for the tool call
    
# print(result2.tool_calls[0])
# print(get_text_length.invoke({"text": "Hello, world!"}))
# print(get_text_length.invoke({"name": "get_text_length", "args": {"text": "Hello, world!"}}))
# print(get_text_length.invoke(result2.tool_calls[0]["args"]["text"]))

if result2.tool_calls:
    tool_call = result2.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id = tool_call["id"]
    print(tool_name)
    print(tool_args)
    print(tool_id)
    
    tool_result = get_text_length.invoke(tool_args["text"])
    print(tool_result)
    
    final_response = llm.invoke(f"The length of the text '{tool_args['text']}' is {tool_result}")
    print(final_response)   

# if result.tool_calls:
#     for tool_call in result.tool_calls:
#         tool_name = tool_call["name"]
#         tool_args = tool_call["args"]
#         tool_id = tool_call["id"]
#         print(tool_name)
#         print(tool_args)
#         print(tool_id)
        
#         tool_result = get_text_length.invoke(tool_args["text"])
#         print(tool_result)

#         final_response = llm.invoke(f"The length of the text '{tool_args['text']}' is {tool_result}")
#         print(final_response)
