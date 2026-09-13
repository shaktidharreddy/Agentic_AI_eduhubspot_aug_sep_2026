#load pdf
#split into chunks
#convert chunks to embeddings
#store embeddings in a vector database


from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
# from langchain_vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "./chroma_db_abraxane"

# texts = [
#     "Hello, how are you?",
#     "The capital of France is Paris",
#     "The capital of Germany is Berlin",
#     "The capital of Italy is Rome",
#     "The capital of Spain is Madrid",
#     "The capital of Portugal is Lisbon",
#     "The capital of Greece is Athens",
# ]
# docs = [Document(page_content=text) for text in texts]

loader = PyMuPDFLoader("FDA/NSCLC_Abraxane_1.0_Historical.pdf")
docs = loader.load()

# print("-"*100)
# print("".join([doc.page_content for doc in docs]))
# print("-"*100)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name="shakti-index",
    # persist_directory=str(PERSIST_DIR),
)

result = vectorstore.similarity_search("What are the indications for which Abraxane is approved by FDA?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
    print("-"*100)
