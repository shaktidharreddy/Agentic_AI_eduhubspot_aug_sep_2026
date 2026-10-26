"""Wipe all data and the vector index (constraints are kept; scripts recreate them anyway).
Deletes EVERYTHING in NEO4J_DATABASE, so it refuses to run against the default 'neo4j' database."""
from kg import NEO4J_DATABASE, run

# Safety guard: your other graphs (movies, SEC, ...) usually live in 'neo4j'.
if NEO4J_DATABASE == "neo4j":
    raise SystemExit("Refusing to wipe the default 'neo4j' database. Set NEO4J_DATABASE=gvd in .env.")

# MATCH (n) = every node; DETACH DELETE also removes the relationships attached to them.
run("MATCH (n) DETACH DELETE n")
# Drop the vector index so Module 5 starts from a clean state (IF EXISTS = no error if it is gone).
run("DROP INDEX result_embedding IF EXISTS")
print("Database emptied. Re-run s1 -> s3 -> s5 -> s6 to rebuild.")
