from dotenv import load_dotenv
import os
from langchain_core.load.serializable import to_json_not_implemented
import requests
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_tool_call

load_dotenv()

# Weather tool
@tool
def get_weather(city: str) -> str:
    """Get the weather of a city"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv('OPENWEATHER_API_KEY')}"
    response = requests.get(url)
    data = response.json()
    print("DEBUG: Weather data:", data)

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"The weather in {city} is {desc} with a temperature of {temp}°C"


# tAVILY NEWS TOOL

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get the news related to a city"""
    response = tavily_client.search(query=f"news about {city}", search_depth="basic", max_results=3)
    results = response.get("results", [])

    if not results:
        return f"No news found for this {city}"
    
    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "No URL")
        snippet = r.get("content", "No snippet")

        news_list.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet[:100]}...\n")

    return f"Here are the latest news about {city}:\n" + "\n".join(news_list)

# print(get_news.invoke("New York"))

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful city assistant",
    middleware=[human_approval],
)

print("City Agent | type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={
            "middleware": [wrap_tool_call(llm)],
        }
    )
    print("DEBUG: Response:", response)
    print("BOT:", response['messages'][-1].content)

print("Exiting...")