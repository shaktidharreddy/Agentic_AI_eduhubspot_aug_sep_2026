from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
# from langchain.chains import LLMChain

load_dotenv()

#Two different prompts
# short_prompt = ChatPromptTemplate.from_template(
#     "Explain {topic1} in 1-2 lines."
# )

# long_prompt = ChatPromptTemplate.from_template(
#     "Explain {topic2} in 8-10 lines."
# )

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in {short_lines} lines."
)   

long_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in {long_lines} lines."
)

#2. Model
model = ChatOpenAI(model="gpt-5-nano", temperature=0)

# 3. Output Parser
parser = StrOutputParser()

#Input
topic = "Machine Learning"

# line 28 — invoke expects a dict input
# formatted_short = short_prompt.invoke({"topic": topic})
# response_short = model.invoke(formatted_short)
# output_short = parser.invoke(response_short.content)

# # line 32 — format_messages expects the variable name as kwarg
# formatted_long = long_prompt.format_messages(topic=topic)
# response_long = model.invoke(formatted_long)
# output_long = parser.invoke(response_long.content)

# print(output_short)
# print("--------------------------------")
# print(output_long)


# chain = RunnableParallel({
#     "short" :short_prompt | model | parser ,
#     "long" :long_prompt | model |parser
# })

# # result = chain.invoke({"topic" :"Machine Learning"})
# result = chain.invoke({"topic1": "Machine Learning", "topic2": "Deep Learning"})

# print(result['short'])
# print("--------------------------------")
# print(result['long'])


short_chain = RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser 
long_chain = RunnableLambda(lambda x: x["long"]) | long_prompt | model | parser 

chain = RunnableParallel({"short":short_chain, "long":long_chain}) 

result = chain.invoke({
    "short": {"topic": "Machine Learning", "short_lines": 1},
    "long": {"topic": "Deep Learning", "long_lines": 5}
})
 

print(result["short"])
print("--------------------------------")
print(result["long"])


