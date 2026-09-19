from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano", temperature=0)
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates Python code."),
    ("human", "Generate a Python function for {function_name}."),
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains Python code."),
    ("human", "Explain the following Python code in 3 lines: {code}."),
])

# seq = code_prompt | model | parser | explain_prompt | model | parser

# result = seq.invoke({"function_name": "check palindrome"})

# print(result)

seq = code_prompt | model | parser

seq2 = RunnableParallel(
    {"code": RunnablePassthrough(),
    "explanation": explain_prompt | model | parser}
)

chain = seq | seq2

result = chain.invoke({"function_name": "check palindrome"})

print(result["code"])
print("--------------------------------")
print(result["explanation"])

