from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_tool = TavilySearchResults(max_results=3)

model = ChatOpenAI(model="gpt-5-nano", temperature=0)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    "Summarize the following news article in 1-2 butllet points: {article}")

chain = prompt | model | parser

news_result = search_tool.invoke("latest AI news in 2026")

# print(news_result)

result = chain.invoke({"article": news_result})

print(result)

# print(search_tool.name)
# print(search_tool.description)
# print(search_tool.args)
