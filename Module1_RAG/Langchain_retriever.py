from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = OpenAIEmbeddings()

# Must match collection_name used in create_vector_store.py (default is "langchain").
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db_abraxane",
    collection_name="shakti-index",
)

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})
# 0 - diverse
# 1 - similar

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5, })

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Use ONLY the provided context to answer the question."""),
    ("user", "Context: {context}"),
    ("user", "Question: {question}")
])


while True:
    question = input("Enter a question: ")
    if question == "exit":
        break
    context = "\n\n".join([doc.page_content for doc in retriever.invoke(question)])
    # chain = prompt | retriever | llm # or you could do retriever.invoke and prompt.invoke separately


    print("DEBUG: Context: ", context)
    print("-"*100)

    result = llm.invoke(prompt.invoke({"question": question, "context": context}))

    # result = chain.invoke({"question": question, "context": context})
    print("Answer: ", result.content)
    # print("Citation: ", result.metadata)
    # print("Page Number: ", result.metadata.get("page", "No page number"))
