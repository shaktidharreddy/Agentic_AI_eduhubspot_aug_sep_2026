from dotenv import load_dotenv
import os
from langchain_core.load.serializable import to_json_not_implemented
import requests
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from tavily import TavilyClient
from rich import print


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

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools([get_weather, get_news])

# Agent loop

while(True):
    prompt = input("You : ")
    if prompt == "exit":
        break
    messages = []
    query = HumanMessage(content=prompt)
    messages.append(query)
    
    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]

                #Human In The Loop
                confirm = input(f"Agent wants to call {tool_name}. Approve (Y/N): ")
                
                if confirm.lower() == "n":
                    messages.append(ToolMessage(content=f"Tool {tool_name} was rejected by the user", tool_call_id=tool_call["id"]))
                    continue

                tool_result = tools[tool_name].invoke(tool_call["args"])
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
            continue    
        else:
            print("\n Answer:")
            print(result.content)
            messages.append(AIMessage(content=result.content))
            break
    
