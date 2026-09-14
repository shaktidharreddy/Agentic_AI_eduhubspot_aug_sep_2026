import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv(OPENAI_API_KEY))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

DB_PATH = "agent_long_term_memory_db"
COLLECTION_NAME = "long_term_memory"
VECTOR_SIZE = 1536

MAX_MEMORY_ITEMS = 3

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

def retrieve_memory(query: str) -> list[str]:
    query_vector = embed_text(query)
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=MAX_MEMORY_ITEMS
    )
    return [hit.payload["text"] for hit in search_result.points]

def format_memory(memories: list[str]) -> str:
    return "\n".join(f"- {m}" for m in memories)

recalled = retrieve_memory("What does Anupam do professionally?")
print("\nRecalled Memories:")
for r in recalled:
    print("-", r)

CHAT_MODEL = "gpt-4o-mini"

def agent_reply(user_question: str) -> str:
    memories = retrieve_memory(user_question)
    memory_block = format_memory(memories)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer the question using only the relevant memory provided."
            },
            {
                "role": "user",
                "content": f"MEMORY:\n{memory_block}\n\nQUESTION:\n{user_question}"
            }
        ]
    )
    return response.choices[0].message.content

answer = agent_reply("What do you know about Anupam's work and preferences?")
print("\nAgent Response:")
print(answer)

qdrant.close()