"""Shared helpers used by every script in this course.

Every other script does `from kg import ...`, so connection details live in ONE place.
"""
import os                                   # read environment variables (our settings)

from dotenv import load_dotenv              # loads the key=value pairs from the .env file
from neo4j import GraphDatabase             # the official Neo4j Python driver

# Copy every line of .env into os.environ so os.getenv() can see it.
load_dotenv()

# --- Neo4j connection settings (second argument = default if .env does not set it) ---
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")          # where the database listens
NEO4J_AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))  # (user, password)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")                 # which database inside the server

# --- Embedding settings (used from Module 5 on) ---
EMBED_MODEL = "text-embedding-3-small"   # OpenAI model that turns text into numbers
EMBED_DIMS = 1536                        # how many numbers that model returns per text

# One driver object for the whole program. It manages a pool of connections for us,
# so we create it once here instead of connecting again for every query.
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def run(cypher, **params):
    """Run one Cypher statement and return the rows as a list of plain dicts.

    Example:  run("MATCH (s:Study {id: $id}) RETURN s.name AS name", id="ZEN-301")
              -> [{'name': 'ZEN-301'}]
    Every keyword argument becomes a $parameter inside the Cypher text.
    """
    # execute_query sends the query, waits for all rows, and returns three things:
    # the records (rows), a summary (timings, counters) and the column names.
    records, _summary, _keys = driver.execute_query(cypher, params, database_=NEO4J_DATABASE)
    # record.data() converts each row into a normal Python dict: {column_name: value}.
    return [record.data() for record in records]


def show(rows, title=None):
    """Print query rows one per line, with an optional heading."""
    if title:
        print(f"\n== {title} ==")
    if not rows:                       # an empty list means the query matched nothing
        print("(no rows)")
    for row in rows:
        print(row)


def embed(texts):
    """Turn a list of strings into a list of embedding vectors (one list of 1536 floats per string)."""
    # Imported here (not at the top) so Modules 1-4 work even without an OpenAI key installed.
    from openai import OpenAI

    client = OpenAI()           # reads OPENAI_API_KEY (and optional OPENAI_BASE_URL) from the environment
    # One API call can embed many texts at once, which is faster and cheaper than one call per text.
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # response.data has one item per input text, in the same order; .embedding is the vector.
    return [item.embedding for item in response.data]
