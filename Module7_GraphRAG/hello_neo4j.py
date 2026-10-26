from neo4j import GraphDatabase      # the official Neo4j driver (pip install neo4j)

URI = "neo4j://localhost:7687"       # Bolt address of your local database (port 7687)
AUTH = ("neo4j", "password123")      # <- your instance password

# "with" closes the driver automatically at the end of the block, even if an error happens.
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()     # raises an error if the DB is not reachable
    print("Connected!")

    # Write: create a node with properties
    # MERGE = create the Person only if one with this name doesn't exist yet.
    # $name and $role are parameters, filled from the dict on the next line.
    driver.execute_query(
        "MERGE (p:Person {name: $name}) SET p.role = $role",
        {"name": "Ada", "role": "analyst"},
        database_="neo4j",            # which database inside the server to use
    )

    # Read: query it back
    # execute_query returns (records, summary, keys): the rows, statistics, and column names.
    records, summary, keys = driver.execute_query(
        "MATCH (p:Person) RETURN p.name AS name, p.role AS role",
        database_="neo4j",
    )
    for record in records:
        # Access columns by the alias used after AS in the query.
        print(record["name"], "-", record["role"])
    print("Query took", summary.result_available_after, "ms")

    # Clean up the test node (DETACH DELETE also removes any relationships it has)
    driver.execute_query("MATCH (p:Person {name: 'Ada'}) DETACH DELETE p", database_="neo4j")
