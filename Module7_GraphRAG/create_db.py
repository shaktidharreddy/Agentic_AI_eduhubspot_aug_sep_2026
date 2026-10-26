from neo4j import GraphDatabase

with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password123")) as driver:
    # Admin commands like CREATE DATABASE always run against the special 'system' database.
    # IF NOT EXISTS = no error if it already exists; WAIT = return only when the new database is online.
    driver.execute_query("CREATE DATABASE gvd IF NOT EXISTS WAIT", database_="system")
    # List all databases and whether they are online.
    records, _, _ = driver.execute_query("SHOW DATABASES YIELD name, currentStatus", database_="system")
    for r in records:
        print(r["name"], r["currentStatus"])
