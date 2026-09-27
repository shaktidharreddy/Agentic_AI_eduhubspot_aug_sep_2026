
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

load_dotenv()


from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's 32°C in {location}."

@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return f"The result of {a} * {b} is {a * b}."

# model = ChatOpenAI(model="gpt-5-nano", temperature=1.0, )
model = init_chat_model("openai:gpt-5.6-luna", use_responses_api=True)
tools = [get_weather, multiply]


# Bind (potentially multiple) tools to the model
model_with_tools = model.bind_tools(tools + [{"type": "web_search"}]) #Bind multiple tools to the model

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "Find the weather in Boston in °C, and then multiply the temperature by 2."}]
response = model_with_tools.invoke(messages)
messages.append(response)

# Step 2: Execute tools and collect results
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    # Execute the tool with the generated arguments using the tool name as the key
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    tool_result = tools[tool_name].invoke(tool_args)
    messages.append({"role": "tool", "name": tool_name, "content": tool_result.content})

# Step 3: Pass results back to model for final response
final_response = model_with_tools.invoke(messages)
print("************ Final RESPONSE : " + final_response.content)
# "The current weather in Boston is 72°F and sunny."
