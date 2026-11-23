import math
import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"],
    # proxy's deployment defaults to 2025-01-01-preview, which predates the Azure Responses API
    default_query={"api-version": "2025-04-01-preview"},
)
MODEL = "gpt-5-nano"
EMBED_MODEL = "text-embedding-3-large"

# --- Responses API ---
try:
    r = client.responses.create(model=MODEL, input="ping")
    print("responses:", r.status)
except Exception as e:
    print("responses FAILED:", e)

# --- Structured output (strict json_schema) ---
class Event(BaseModel):
    name: str
    date: str
    participants: list[str]

try:
    r = client.responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": "Alice and Bob go to a science fair Friday."}],
        text_format=Event,
    )
    print("structured:", r.output_parsed)
except Exception as e:
    print("structured FAILED:", e)

# --- Tool calling ---
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a location.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
        "additionalProperties": False,
    },
    "strict": True,
}]
try:
    r = client.responses.create(model=MODEL, input="Weather in Paris?", tools=tools)
    print("tools:", [i.type for i in r.output])
except Exception as e:
    print("tools FAILED:", e)

# --- Embeddings ---
try:
    r = client.embeddings.create(model=EMBED_MODEL, input=["hello", "world"])
    print("embeddings:", len(r.data), "x", len(r.data[0].embedding),
          "| tokens:", r.usage.total_tokens)
except Exception as e:
    print("embeddings FAILED:", e)

# --- Embeddings: reduced dimensions + cosine similarity ---
def cosine(u, v):
    dot = sum(x * y for x, y in zip(u, v))
    return dot / (math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v)))

try:
    r = client.embeddings.create(
        model=EMBED_MODEL,
        input=[
            "a cat sleeping on a couch",
            "a kitten napping on a sofa",
            "quarterly revenue report",
        ],
        dimensions=256,
    )
    cat, kitten, report = (d.embedding for d in r.data)
    print(f"embeddings(256): similar={cosine(cat, kitten):.3f} "
          f"unrelated={cosine(cat, report):.3f}")
except Exception as e:
    print("embeddings(256) FAILED:", e)
