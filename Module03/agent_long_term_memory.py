import os
from openai import OpenAI

client = OpenAI(api_key="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")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

DB_PATH = "agent_long_term_memory_db"
COLLECTION_NAME = "long_term_memory"
VECTOR_SIZE = 1536

qdrant = QdrantClient(path=DB_PATH)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        ),
    )

print("Long-term memory store ready.")

EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

from qdrant_client.models import PointStruct

def store_memory(text: str, memory_id: int):
    vector = embed_text(text)
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=memory_id,
                vector=vector,
                payload={"text": text}
            )
        ]
    )

store_memory("Anupam works as a Senior Research Analyst.", 1)
store_memory("Anupam prefers step-by-step technical explanations.", 2)
store_memory("Anupam is learning agent memory and state management.", 3)

print("Memories stored.")

def retrieve_memory(query: str, top_k: int = 3) -> list[str]:
    query_vector = embed_text(query)
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    return [hit.payload["text"] for hit in search_result.points]

recalled = retrieve_memory("What does Anupam do professionally?")
print("\nRecalled Memories:")
for r in recalled:
    print("-", r)

CHAT_MODEL = "gpt-4o-mini"

def agent_reply(user_question: str) -> str:
    memory_context = retrieve_memory(user_question)
    memory_block = "\n".join(memory_context)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Use the provided memory to answer accurately."},
            {"role": "user", "content": f"MEMORY:\n{memory_block}\n\nQUESTION:\n{user_question}"}
        ]
    )
    return response.choices[0].message.content

answer = agent_reply("Tell me about Anupam.")
print("\nAgent Response:")
print(answer)

qdrant.close()