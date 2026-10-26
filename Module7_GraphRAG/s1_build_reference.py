"""Module 3 – ingest the REFERENCE layer: evidence that is true regardless of any dossier.
Reference nodes never get a projectId.

How the script is organised:
  1. NODES          – the data for every node, grouped by label
  2. RELS           – the relationships between nodes (with their properties)
  3. three functions that write constraints, nodes and relationships into Neo4j
"""
from kg import run, show

# ---------------------------------------------------------------------------
# 1. NODES: a dict of  label -> list of property dicts.
#    Every node gets a unique, human-readable "id" so we can find it again later.
#    All other keys become node properties (name, phase, pValue, ...).
# ---------------------------------------------------------------------------
NODES = {
    "Product": [                                   # drugs: the one we study + comparators
        {"id": "P-ZEN", "name": "Zentrax", "type": "investigational"},
        {"id": "P-PBO", "name": "Placebo", "type": "comparator"},
        {"id": "P-ADA", "name": "Adalimumab", "type": "comparator"},
    ],
    "Indication": [                                # the disease the drug treats
        {"id": "IND-PSO", "name": "Moderate-to-severe plaque psoriasis"},
    ],
    "Study": [                                     # clinical trials
        {"id": "ZEN-301", "name": "ZEN-301", "phase": 3, "design": "RCT vs placebo", "year": 2024, "n": 600},
        {"id": "ZEN-302", "name": "ZEN-302", "phase": 3, "design": "RCT vs active comparator", "year": 2025, "n": 450},
    ],
    "Endpoint": [                                  # what was measured; category is used for filtering later
        {"id": "E-PASI75", "name": "PASI 75 at week 16", "category": "efficacy"},
        {"id": "E-DLQI", "name": "DLQI 0/1 at week 16", "category": "quality_of_life"},
        {"id": "E-SAE", "name": "Serious adverse events", "category": "safety"},
    ],
    "Population": [                                # which patients a result is about
        {"id": "POP-ITT", "name": "Intention-to-treat"},
        {"id": "POP-BIONAIVE", "name": "Biologic-naive"},
    ],
    "SourceDocument": [                            # where the numbers were found
        {"id": "DOC-CSR-301", "title": "ZEN-301 Clinical Study Report", "docType": "CSR", "year": 2025},
        {"id": "DOC-PUB-302", "title": "Zentrax vs adalimumab head-to-head trial (J Derm Trials)", "docType": "publication", "year": 2025},
    ],
    # Results are the heart of the graph. "text" is a readable sentence (embedded in Module 5);
    # value / comparatorValue / pValue are numbers we can filter on with WHERE.
    "Result": [
        {"id": "R1", "value": 78.0, "comparatorValue": 6.0, "unit": "%", "pValue": 0.0001,
         "text": "In ZEN-301, 78% of patients on Zentrax achieved PASI 75 at week 16 versus 6% on placebo."},
        {"id": "R2", "value": 62.0, "comparatorValue": 9.0, "unit": "%", "pValue": 0.0001,
         "text": "Quality of life improved: 62% of Zentrax patients reported DLQI 0/1 at week 16 compared with 9% on placebo."},
        {"id": "R3", "value": 2.1, "comparatorValue": 1.8, "unit": "%", "pValue": 0.81,
         "text": "Serious adverse events occurred in 2.1% of Zentrax patients and 1.8% of placebo patients through week 16."},
        {"id": "R4", "value": 81.0, "comparatorValue": 70.0, "unit": "%", "pValue": 0.003,
         "text": "In the head-to-head ZEN-302 trial, PASI 75 at week 16 was reached by 81% with Zentrax versus 70% with adalimumab."},
        {"id": "R5", "value": 84.0, "comparatorValue": 71.0, "unit": "%", "pValue": 0.01,
         "text": "Among biologic-naive patients in ZEN-302, PASI 75 response was 84% for Zentrax and 71% for adalimumab."},
        {"id": "R6", "value": 1.2, "comparatorValue": 1.5, "unit": "%", "pValue": 0.62,
         "text": "Rates of serious infections were low and similar: 1.2% with Zentrax and 1.5% with adalimumab."},
    ],
}

# ---------------------------------------------------------------------------
# 2. RELS: one tuple per relationship.
#    (from_label, from_id, REL_TYPE, to_label, to_id, {relationship properties})
#    The arrow always goes from the first node to the second node.
# ---------------------------------------------------------------------------
RELS = [
    ("Study", "ZEN-301", "EVALUATES", "Product", "P-ZEN", {"role": "investigational arm"}),
    ("Study", "ZEN-302", "EVALUATES", "Product", "P-ZEN", {"role": "investigational arm"}),
    ("Study", "ZEN-301", "IN_INDICATION", "Indication", "IND-PSO", {}),        # {} = no properties
    ("Study", "ZEN-302", "IN_INDICATION", "Indication", "IND-PSO", {}),
    # A node can point at another node of the SAME label: Biologic-naive is a subgroup of ITT.
    ("Population", "POP-BIONAIVE", "SUBGROUP_OF", "Population", "POP-ITT", {"criterion": "No prior biologic therapy"}),
]

# Every Result connects to the same five kinds of things, so we describe them as one row each
# instead of writing 30 tuples by hand:
# result, study,    endpoint,   population,     versus,  document,      page, table,    primary
RESULT_LINKS = [
    ("R1", "ZEN-301", "E-PASI75", "POP-ITT",      "P-PBO", "DOC-CSR-301", 45, "14.2.1", True),
    ("R2", "ZEN-301", "E-DLQI",   "POP-ITT",      "P-PBO", "DOC-CSR-301", 52, "14.2.4", False),
    ("R3", "ZEN-301", "E-SAE",    "POP-ITT",      "P-PBO", "DOC-CSR-301", 88, "14.3.1", False),
    ("R4", "ZEN-302", "E-PASI75", "POP-ITT",      "P-ADA", "DOC-PUB-302", 4,  "2",      True),
    ("R5", "ZEN-302", "E-PASI75", "POP-BIONAIVE", "P-ADA", "DOC-PUB-302", 6,  "3",      False),
    ("R6", "ZEN-302", "E-SAE",    "POP-ITT",      "P-ADA", "DOC-PUB-302", 7,  "4",      False),
]
# Expand each row into five relationship tuples and append them to RELS.
for rid, study, endpoint, pop, versus, doc, page, table, primary in RESULT_LINKS:
    RELS += [
        ("Study", study, "REPORTS", "Result", rid, {"primary": primary}),          # property on the relationship
        ("Result", rid, "MEASURES", "Endpoint", endpoint, {}),
        ("Result", rid, "FOR_POPULATION", "Population", pop, {}),
        ("Result", rid, "VERSUS", "Product", versus, {}),
        # page/table live on the relationship: they describe WHERE in the document this result is.
        ("Result", rid, "EXTRACTED_FROM", "SourceDocument", doc, {"page": page, "table": table}),
    ]


# ---------------------------------------------------------------------------
# 3. Functions that write to Neo4j
# ---------------------------------------------------------------------------
def create_constraints():
    """One uniqueness constraint per label: no two Study nodes may share an id (and lookups by id get fast)."""
    for label in NODES:
        # IF NOT EXISTS makes the statement safe to run again and again.
        run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")


def load_nodes():
    """Create (or update) every node with a single query per label."""
    # Labels cannot be parameters in Cypher, so the label goes in the f-string; values stay as $parameters.
    # {{ and }} are how you write a literal { } inside a Python f-string.
    for label, rows in NODES.items():
        # UNWIND turns the list $rows into one row per dict, called "row".
        # MERGE finds the node with that id, or creates it if missing (so no duplicates on re-run).
        # SET n += row copies every key of the dict onto the node as properties.
        run(f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row", rows=rows)


def load_relationships():
    """Create (or update) every relationship."""
    for from_label, from_id, rel_type, to_label, to_id, props in RELS:
        run(f"""
            // 1) find the two existing nodes by their id
            MATCH (a:{from_label} {{id: $fid}}), (b:{to_label} {{id: $tid}})
            // 2) create the arrow between them only if it is not there yet
            MERGE (a)-[r:{rel_type}]->(b)
            // 3) copy the property dict onto the relationship
            SET r += $props
        """, fid=from_id, tid=to_id, props=props)


# This block only runs when you execute the file directly (python s1_build_reference.py),
# not when another script imports it.
if __name__ == "__main__":
    create_constraints()
    load_nodes()
    load_relationships()
    # labels(n) returns a list of labels; [0] takes the first. count(*) counts rows per label.
    show(run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label"), "Nodes")
    # ()-[r]->() matches every relationship; type(r) gives its type name, e.g. 'REPORTS'.
    show(run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"), "Relationships")
