"""Module 7 – PROJECT layer: dossier skeleton + claims written from retrieval results.
Every node created here carries a projectId and links INTO the reference layer.

Flow:  create_project()  ->  write_claim() = search() + WRITE_CLAIM query
"""
from kg import run, show
from s4_vector_search import search          # the hybrid search from Module 6

PROJECT_LABELS = ["Dossier", "Section", "ValueMessage", "Claim"]


def ensure_project_schema():
    """Constraints + indexes for the project-layer labels (safe to run every time)."""
    for label in PROJECT_LABELS:
        # Unique ids, exactly like the reference layer.
        run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
        # An ordinary index on projectId makes "give me everything of project X" fast.
        run(f"CREATE INDEX {label.lower()}_project IF NOT EXISTS FOR (n:{label}) ON (n.projectId)")


def create_project(pid, name, market, sections, value_messages):
    """Create the dossier, its sections and its value messages. All get projectId = pid."""
    run("""
        // the dossier node itself; its id IS the project id
        MERGE (d:Dossier {id: $pid})
        SET d.projectId = $pid, d.name = $name, d.market = $market
        // WITH passes d on to the next part of the query
        WITH d
        // one row per section dict in the $sections list
        UNWIND $sections AS sec
        // ids are prefixed with the project id ('GVD-DE-2026:S-EFF') so they are unique across projects
        MERGE (s:Section {id: $pid + ':' + sec.key})
        SET s.projectId = $pid, s.title = sec.title, s.order = sec.order
        MERGE (s)-[:PART_OF]->(d)
    """, pid=pid, name=name, market=market, sections=sections)

    run("""
        UNWIND $vms AS vm
        MERGE (v:ValueMessage {id: $pid + ':' + vm.key})
        SET v.projectId = $pid, v.text = vm.text
    """, pid=pid, vms=value_messages)


# Writes ONE claim plus all its links in a single query.
WRITE_CLAIM = """
// 1) Find the value message and section. Requiring projectId = $pid means a claim
//    can never be attached to another project's section by mistake.
MATCH (vm:ValueMessage {id: $vmId, projectId: $pid}),
      (sec:Section     {id: $secId, projectId: $pid})
// 2) Create/update the claim node (project layer, so it gets projectId).
MERGE (c:Claim {id: $claimId})
SET c.projectId = $pid, c.text = $text, c.status = 'draft', c.updatedAt = datetime()
// 3) Project-layer links.
MERGE (c)-[:SUPPORTS]->(vm)
MERGE (c)-[a:APPEARS_IN]->(sec)
SET a.order = $order
// 4) The BRIDGE into the reference layer: one BASED_ON per retrieved result.
//    The retrieval score, method and question are stored ON the relationship for auditing.
WITH c
UNWIND $evidence AS ev
MATCH (r:Result {id: ev.id})
MERGE (c)-[b:BASED_ON]->(r)
SET b.score = ev.score, b.method = $method, b.query = $question, b.linkedAt = datetime()
RETURN c.id AS claim, collect(r.id) AS basedOn
"""


def write_claim(pid, key, text, vm_key, section_key, question,
                order=1, top_n=2, min_score=0.7, search_fn=search, method="neo4j-vector+filter", **filters):
    """Retrieve evidence for a claim, then write the claim and its BASED_ON links.
    search_fn / method let Module 7P plug in pgvector retrieval instead of the Neo4j vector index.

    pid / key        – project id and short claim key; the stored id becomes 'pid:key'
    question         – what we search for in the evidence
    top_n            – how many of the best hits become BASED_ON links
    **filters        – any extra keyword (category=..., versus=...) is passed on to the search
    """
    # RETRIEVAL: ask for 10 candidates, keep the best top_n.
    hits = search_fn(question, k=10, min_score=min_score, **filters)[:top_n]
    if not hits:
        # No good evidence: better to write nothing than an unsupported claim.
        print(f"No evidence above {min_score} for {question!r}; claim NOT written")
        return []
    print(f"\n{pid}:{key}  retrieved -> {[(h['id'], h['score']) for h in hits]}")
    # INGESTION: write the claim and its links back into the graph.
    rows = run(WRITE_CLAIM, pid=pid, claimId=f"{pid}:{key}", text=text,
               vmId=f"{pid}:{vm_key}", secId=f"{pid}:{section_key}", order=order,
               question=question, method=method,
               evidence=[{"id": h["id"], "score": h["score"]} for h in hits])
    if not rows:
        # The MATCH in step 1 found nothing (wrong key or another project's section).
        print("Section or value message not found in this project; nothing written")
    return rows


if __name__ == "__main__":
    ensure_project_schema()

    # ---------- Project 1: German GVD ----------
    DE = "GVD-DE-2026"
    create_project(DE, "Zentrax GVD - Germany", "Germany",
                   sections=[{"key": "S-EFF", "title": "Clinical efficacy", "order": 1},
                             {"key": "S-SAF", "title": "Safety", "order": 2}],
                   value_messages=[{"key": "VM-EFF", "text": "Zentrax delivers superior skin clearance"},
                                   {"key": "VM-SAF", "text": "Zentrax has a favourable safety profile"},
                                   {"key": "VM-QOL", "text": "Zentrax improves patients' quality of life"}])

    # category="efficacy" leaves only the three PASI 75 results, so top_n=3 takes all of them.
    show(write_claim(DE, "C1", "Zentrax achieved higher PASI 75 response rates than placebo and adalimumab at week 16.",
                     vm_key="VM-EFF", section_key="S-EFF", order=1,
                     question="PASI 75 skin clearance response at week 16",
                     top_n=3, category="efficacy"), "DE claim C1")

    show(write_claim(DE, "C2", "Serious adverse event rates with Zentrax were comparable to placebo and adalimumab.",
                     vm_key="VM-SAF", section_key="S-SAF", order=1,
                     question="serious adverse events and infections",
                     top_n=2, category="safety"), "DE claim C2")
    # Note: VM-QOL gets no claim on purpose – Module 8 finds it as a "gap".

    # ---------- Project 2: UK GVD reuses the SAME evidence ----------
    UK = "GVD-UK-2026"
    create_project(UK, "Zentrax GVD - United Kingdom", "United Kingdom",
                   sections=[{"key": "S-EFF", "title": "Comparative effectiveness", "order": 1}],
                   value_messages=[{"key": "VM-EFF", "text": "Zentrax outperforms adalimumab in biologic-naive patients"}])

    show(write_claim(UK, "C1", "In biologic-naive patients, Zentrax outperformed adalimumab on PASI 75.",
                     vm_key="VM-EFF", section_key="S-EFF", order=1,
                     question="PASI 75 response in biologic-naive patients compared with adalimumab",
                     top_n=1, category="efficacy", versus="Adalimumab"), "UK claim C1")

    # coalesce() returns the first non-null value: reference nodes have no projectId, so they show as 'reference'.
    show(run("""
        MATCH (n)
        RETURN coalesce(n.projectId, 'reference') AS layer, labels(n)[0] AS label, count(*) AS count
        ORDER BY layer, label
    """), "Nodes per layer")
