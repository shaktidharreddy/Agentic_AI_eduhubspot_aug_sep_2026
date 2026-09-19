from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import LLMChain

load_dotenv()

# 1. Prompt Templates
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines."
)

# prompt2 = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that explains topics in 1-2 lines."),
#     ("user", "Explain {topic} in 1-2 lines.")
# ])

#2. Model
model = ChatOpenAI(model="gpt-5-nano", temperature=0)

# 3. Output Parser
parser = StrOutputParser()

# Step by step manual flow

# Format the prompt
formatted_prompt = prompt.invoke(input= "Machine Learning")

#Call the model manually
response = model.invoke(formatted_prompt)

output = parser.invoke(response.content)

print(output)

# fixed_chain = LLMChain(llm=model, prompt=prompt, output_parser=parser)
# dynamic_chain = prompt | model | parser

# result = dynamic_chain.invoke({"topic": "Machine Learning"})
# print(result)