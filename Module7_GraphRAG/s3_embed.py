"""Module 5 – add OpenAI embeddings to Result nodes and create a vector index.

An embedding is a list of numbers that captures the MEANING of a text.
Texts with similar meaning get vectors that point in similar directions,
which is what makes "semantic" search possible.
"""
from kg import EMBED_DIMS, embed, run, show

# 1. Which results still need an embedding? Re-running the script only embeds new ones,
#    so you never pay OpenAI twice for the same text.
todo = run("MATCH (r:Result) WHERE r.embedding IS NULL RETURN r.id AS id, r.text AS text")
print(f"{len(todo)} result(s) to embed")

if todo:
    # 2. One API call for the whole batch: send all texts, get back one vector per text (same order).
    vectors = embed([row["text"] for row in todo])

    # 3. Store each vector as a list property called "embedding" on its node.
    #    zip() pairs each row with its vector; we send them all in one query via UNWIND.
    run("""
        UNWIND $rows AS row
        MATCH (r:Result {id: row.id})
        SET r.embedding = row.embedding
    """, rows=[{"id": row["id"], "embedding": vec} for row, vec in zip(todo, vectors)])

# 4. Create the vector index. Without it Neo4j would have to compare the question with
#    every node one by one; the index finds the nearest vectors quickly.
#    - vector.dimensions must equal the model's output size (1536)
#    - cosine similarity compares the direction of two vectors (recommended by OpenAI)
#    The backticks around `vector.dimensions` are needed because the name contains a dot.
run(f"""
    CREATE VECTOR INDEX result_embedding IF NOT EXISTS
    FOR (r:Result) ON (r.embedding)
    OPTIONS {{ indexConfig: {{
        `vector.dimensions`: {EMBED_DIMS},
        `vector.similarity_function`: 'cosine'
    }} }}
""")
# Indexes are built in the background; wait (max 300 s) until it is ONLINE before using it.
run("CALL db.awaitIndexes(300)")

# Sanity checks: every Result should now have a 1536-long list, and the index should be ONLINE.
show(run("MATCH (r:Result) RETURN r.id AS id, size(r.embedding) AS dims ORDER BY id"), "Embedding sizes")
show(run("""
    SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties
    WHERE type = 'VECTOR'
    RETURN name, state, labelsOrTypes, properties
"""), "Vector indexes")
