from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import glob

all_docs = []
# pdf_files = glob.glob("FDA/*.pdf")
# for pdf_path in pdf_files:
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
#     all_docs.extend(docs)
#     print(f"Loaded {len(docs)} docs from {pdf_path}")

loader = PyMuPDFLoader("FDA/NSCLC_Abraxane_1.0_Historical.pdf")
all_docs = loader.load()

if all_docs:
    print(all_docs[0].page_content)
print(f"Total documents loaded: {len(all_docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(all_docs)

# Create a vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name="fda-documents"
)

# Create a retriever
retriever = vectorstore.as_retriever()

print("-"*100)
print("Query:")
print("What is the recommended dosage for Abraxane in NSCLC?")
print("-"*100)

print("-"*100)
print("Retriever:")
relevant_docs = retriever.invoke("What is the recommended dosage for Abraxane in NSCLC?")
for doc in relevant_docs:
    print(doc.page_content)
    print(doc.metadata)
    print("-"*100)
print("-"*100)  

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a helpful assistant that can answer questions about the document.
        Answer the question based on the documents provided.
        If the question is not related to the documents, say "I don't know".
    """),
    ("user", "Here are the relevant documents: {input}"),
    ("user", "Question: {question}"),
    ("assistant", "Answer: \n"),
])  

# final_prompt = prompt.format_messages(input="".join([doc.page_content for doc in relevant_docs]), question="What is the recommended dosage for Abraxane in NSCLC?")
# result = model.invoke(final_prompt)

final_prompt = prompt.invoke({"input": "".join([doc.page_content for doc in relevant_docs]), "question": "What is the recommended dosage for Abraxane in NSCLC?"})
result = model.invoke(final_prompt.to_messages())

print("-"*100)
print("Final Prompt:")
print(final_prompt.to_messages())
print("-"*100)
print("Result:")
print(result.content)
print("-"*100)

# chain = prompt | model

# result = chain.invoke({"input": "".join([doc.page_content for doc in relevant_docs]), "question": "What is the recommended dosage for Abraxane in NSCLC?"})
# print(result.content)
