"""Module 4 – Cypher pattern queries with value filters (no vectors yet).

Values are passed as keyword arguments (max_p=0.05) and used in Cypher as $max_p.
Never paste values into the query text yourself: parameters are safer and faster.
"""
from kg import run, show

# Query 1: filter on node properties with WHERE.
show(run("""
    MATCH (r:Result)                                  // every Result node, called r
    WHERE r.pValue < $max_p AND r.value >= $min_value // keep only significant, high-response results
    RETURN r.id AS id, r.value AS value, r.pValue AS p
    ORDER BY value DESC                               // biggest response first
""", max_p=0.05, min_value=75), "Significant results with value >= 75")

# Query 2: a multi-hop pattern. The comma joins two patterns that share the variable r.
show(run("""
    MATCH (s:Study)-[:REPORTS]->(r:Result)-[:VERSUS]->(cmp:Product {name: $comparator}),
          (r)-[:MEASURES]->(e:Endpoint {category: $category})
    RETURN s.id AS study, e.name AS endpoint, r.value AS zentrax, r.comparatorValue AS comparator
""", comparator="Adalimumab", category="efficacy"), "Efficacy vs Adalimumab")

# Query 3: filter on a RELATIONSHIP property. Giving the relationship a variable (x) lets us use x.page.
show(run("""
    MATCH (r:Result)-[x:EXTRACTED_FROM]->(d:SourceDocument)
    WHERE x.page < $max_page
    RETURN r.id AS id, d.title AS document, x.page AS page
    ORDER BY document, page
""", max_page=50), "Results on early pages")

# Query 4: OPTIONAL MATCH keeps products even when no study evaluates them (count becomes 0).
show(run("""
    MATCH (p:Product)
    OPTIONAL MATCH (s:Study)-[:EVALUATES]->(p)
    RETURN p.name AS product, count(s) AS studies
"""), "Studies per product")
