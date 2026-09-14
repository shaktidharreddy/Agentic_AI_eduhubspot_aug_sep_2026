import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv(OPENAI_API_KEY))

EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding
    
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

DB_PATH = "semantic_memory_db"
COLLECTION_NAME = "memory_collection"
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

print("Vector collection ready.")

from qdrant_client.models import PointStruct

texts = [
    "Anupam works as a Senior Research Analyst.",
    "Anupam uses Python 3.14 for local development.",
    "Anupam prefers step-by-step technical labs.",
    "Anupam uses VS Code for coding and experimentation."
]

points = []
for idx, text in enumerate(texts, start=1):
    vector = embed_text(text)
    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={"text": text}
        )
    )

qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("Stored semantic memories:", len(points))

query = "What tools are used for development?"
query_vector = embed_text(query)

search_result = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
)

hits = search_result.points

print("\nRetrieved Semantic Memories:")
for hit in hits:
    print("-", hit.payload["text"])

qdrant.close()