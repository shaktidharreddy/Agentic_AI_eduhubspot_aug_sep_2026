"""Module 8 – capability queries that cross the reference and project layers.

Each question below is answered by ONE graph pattern, because the answer is literally a path:
  1. Traceability  Claim -> Result -> SourceDocument (page)
  2. Reuse         Result <- Claims from several projects
  3. Gaps          ValueMessage with no incoming SUPPORTS
  4. Impact        new Result SUPERSEDES old Result <- Claim -> Section -> Dossier
  5. Isolation     delete one project without touching shared evidence
"""
from kg import embed, run, show
from s5_project_layer import create_project

# 1. TRACEABILITY: claim -> result -> study + document page
#    Two MATCH clauses: the first walks claim -> result -> document, the second adds the study.
show(run("""
    MATCH (c:Claim {id: $claimId})-[b:BASED_ON]->(r:Result)-[x:EXTRACTED_FROM]->(d:SourceDocument)
    MATCH (s:Study)-[:REPORTS]->(r)
    RETURN r.id AS result, b.score AS score, s.id AS study, d.title AS document, x.page AS page, x.table AS tbl
    ORDER BY result
""", claimId="GVD-DE-2026:C1"), "1. Traceability for GVD-DE-2026:C1")

# 2. REUSE: reference results used by more than one project
#    collect(DISTINCT ...) gathers the project ids per result into a list; size() counts them.
#    WITH works like a pipe: aggregate first, then filter on the aggregate.
show(run("""
    MATCH (c:Claim)-[:BASED_ON]->(r:Result)
    WITH r, collect(DISTINCT c.projectId) AS projects
    WHERE size(projects) > 1
    RETURN r.id AS result, projects, r.text AS text
"""), "2. Evidence reused across projects")

# 3. GAPS: value messages with no supporting claim, per project
#    NOT EXISTS { pattern } is true when the pattern cannot be found.
show(run("""
    MATCH (vm:ValueMessage)
    WHERE NOT EXISTS { (vm)<-[:SUPPORTS]-(:Claim) }
    RETURN vm.projectId AS project, vm.id AS valueMessage, vm.text AS text
"""), "3. Value messages without evidence")

# 4. IMPACT: a final analysis supersedes R1 -> which claims are affected?
#    First simulate ingestion of a corrected result (reference layer, so no projectId).
new_result = {
    "id": "R1-FINAL", "value": 79.0, "comparatorValue": 5.0, "unit": "%", "pValue": 0.0001,
    "text": "Final analysis of ZEN-301: 79% of patients on Zentrax achieved PASI 75 at week 16 versus 5% on placebo.",
}
run("""
    // find the old result and everything it is connected to
    MATCH (old:Result {id: 'R1'})<-[:REPORTS]-(s:Study),
          (old)-[:MEASURES]->(e:Endpoint),
          (old)-[:FOR_POPULATION]->(pop:Population),
          (old)-[:VERSUS]->(cmp:Product),
          (old)-[:EXTRACTED_FROM]->(d:SourceDocument)
    // create the new result (with its embedding so the Neo4j vector index can find it)
    MERGE (new:Result {id: $row.id})
    SET new += $row, new.embedding = $embedding
    // give it the same connections as the old one
    MERGE (s)-[:REPORTS {primary: true}]->(new)
    MERGE (new)-[:MEASURES]->(e)
    MERGE (new)-[:FOR_POPULATION]->(pop)
    MERGE (new)-[:VERSUS]->(cmp)
    MERGE (new)-[:EXTRACTED_FROM {page: 112, table: '14.2.1a'}]->(d)
    // and record that it replaces the old one
    MERGE (new)-[sup:SUPERSEDES]->(old)
    SET sup.reason = 'final analysis', sup.date = date('2026-03-01')
""", row=new_result, embedding=embed([new_result["text"]])[0])

#    Now the impact query: follow SUPERSEDES back to old results, then to the claims using them,
#    their sections and dossiers. SET flags each affected claim for human review.
show(run("""
    MATCH (new:Result)-[:SUPERSEDES]->(old:Result)<-[:BASED_ON]-(c:Claim)-[:APPEARS_IN]->(sec:Section)-[:PART_OF]->(d:Dossier)
    SET c.status = 'needs_review'
    RETURN d.projectId AS project, sec.title AS section, c.id AS claim,
           old.id AS outdated, new.id AS replacement, c.status AS status
"""), "4. Impact of superseded evidence")

# 5. PROJECT ISOLATION: delete a whole project without touching reference evidence
#    Create a throw-away project, count reference nodes, delete the project, count again.
create_project("GVD-SANDBOX", "Throw-away dossier", "Nowhere",
               sections=[{"key": "S1", "title": "Scratch", "order": 1}],
               value_messages=[{"key": "VM1", "text": "temporary"}])
before = run("MATCH (n) WHERE n.projectId IS NULL RETURN count(n) AS referenceNodes")[0]
# DETACH DELETE removes the nodes AND their relationships. Only nodes with this projectId match.
run("MATCH (n {projectId: $pid}) DETACH DELETE n", pid="GVD-SANDBOX")
after = run("MATCH (n) WHERE n.projectId IS NULL RETURN count(n) AS referenceNodes")[0]
show([{"before": before["referenceNodes"], "after": after["referenceNodes"]}],
     "5. Reference nodes before/after deleting GVD-SANDBOX")
