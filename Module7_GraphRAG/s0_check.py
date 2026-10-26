"""Module 1 – confirm Python can talk to the local Neo4j database."""
from kg import driver, run, show   # our shared helpers (see kg.py)

# Ask the driver to open a test connection. Raises a clear error if Neo4j is down
# or the password is wrong, so you find problems before running real queries.
driver.verify_connectivity()

# The smallest possible query: no MATCH, just RETURN a literal value.
show(run("RETURN 'connected' AS status"), "Connection")

# dbms.components() is a built-in procedure that reports the server version.
# CALL runs a procedure, YIELD picks which of its output columns we want.
show(run("CALL dbms.components() YIELD name, versions RETURN name, versions[0] AS version"), "Server")

# MATCH (n) matches every node; count(n) counts them. A fresh 'gvd' database should say 0.
show(run("MATCH (n) RETURN count(n) AS nodes"), "Nodes currently in the database")

# Release the connection pool when the script is done.
driver.close()
