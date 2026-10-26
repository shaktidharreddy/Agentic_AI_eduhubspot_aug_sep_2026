"""Module 6 – vector search, hybrid (vector + filter) search, and graph-expanded context.

Three ideas, from simple to powerful:
  plain_search()  – "which results MEAN something similar to my question?"
  search()        – the same, plus exact filters on the graph (category, comparator, p-value)
  context_for()   – follow relationships from the hits to their study and source page
"""
import os

from kg import embed, run, show

# The SEARCH clause (Neo4j 2026.01+, Cypher 25) asks the vector index 'result_embedding'
# for the k Result nodes whose embeddings are closest to the query vector ($qvec).
# SCORE AS score adds the similarity (0..1, higher = closer).
# "CYPHER 25" on the first line makes sure the new syntax is used even if the database
# defaults to the older Cypher 5.
PLAIN_SEARCH = """
CYPHER 25
MATCH (r:Result)
  SEARCH r IN (VECTOR INDEX result_embedding FOR $qvec LIMIT $k) SCORE AS score
RETURN r.id AS id, round(score, 3) AS score, r.text AS text
ORDER BY score DESC
"""

# Hybrid search = vector search first, then ordinary Cypher on the hits:
#   1. SEARCH lets the vector index propose the k most similar Result nodes (named r)
#   2. the second MATCH walks from r to its study, endpoint, comparator and population
#   3. WHERE keeps only the rows that pass the filters
# "$category IS NULL OR ..." means: if Python passed None, the filter is switched off.
HYBRID_SEARCH = """
CYPHER 25
MATCH (r:Result)
  SEARCH r IN (VECTOR INDEX result_embedding FOR $qvec LIMIT $k) SCORE AS score
MATCH (s:Study)-[:REPORTS]->(r)-[:MEASURES]->(e:Endpoint),
      (r)-[:VERSUS]->(cmp:Product),
      (r)-[:FOR_POPULATION]->(pop:Population)
WHERE score >= $minScore
  AND ($category IS NULL OR e.category = $category)
  AND ($versus   IS NULL OR cmp.name   = $versus)
  AND ($maxP     IS NULL OR r.pValue  <= $maxP)
  AND NOT EXISTS { (:Result)-[:SUPERSEDES]->(r) }   // skip results replaced by a newer one (Module 8)
RETURN r.id AS id, round(score, 3) AS score, s.id AS study, e.name AS endpoint,
       cmp.name AS versus, pop.name AS population, r.text AS text
ORDER BY score DESC
"""

# Graph expansion: given result ids, collect the extra facts an LLM needs to cite properly.
CONTEXT = """
MATCH (s:Study)-[:REPORTS]->(r:Result)-[x:EXTRACTED_FROM]->(d:SourceDocument)
WHERE r.id IN $ids
RETURN r.id AS id, r.text AS text, s.id AS study, s.design AS design, s.n AS n,
       d.title AS source, x.page AS page
"""


def plain_search(question, k=3):
    """Vector similarity only."""
    # embed() takes a LIST of texts and returns a list of vectors; [0] picks the one vector we need.
    return run(PLAIN_SEARCH, k=k, qvec=embed([question])[0])


def search(question, k=10, min_score=0.0, category=None, versus=None, max_p=None):
    """Vector similarity + optional graph/value filters. Pass None to switch a filter off.

    k         – how many candidates the vector index returns BEFORE filtering (keep it generous)
    min_score – drop weak matches (0.7 is a sensible start for this model)
    """
    return run(HYBRID_SEARCH, k=k, qvec=embed([question])[0], minScore=min_score,
               category=category, versus=versus, maxP=max_p)


def context_for(result_ids):
    """Follow relationships from matched results to their study and source page."""
    return run(CONTEXT, ids=result_ids)


def answer(question, search_fn=search):
    """Optional bonus: GraphRAG – retrieve, expand via the graph, then ask an LLM.
    search_fn lets Module 6P swap in pgvector retrieval without changing this function."""
    from openai import OpenAI

    # Step 1 – RETRIEVE: best 3 results above the threshold.
    hits = search_fn(question, k=5, min_score=0.7)[:3]
    # Step 2 – EXPAND: fetch study design, sample size and page for each hit, and format
    #          them as bullet lines the LLM can quote.
    evidence = "\n".join(
        f"- {c['text']} (Study {c['study']}, {c['design']}, n={c['n']}; source: {c['source']}, p.{c['page']})"
        for c in context_for([h["id"] for h in hits])
    )
    # Step 3 – GENERATE: the system message restricts the model to our evidence (reduces hallucination).
    response = OpenAI().chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "Answer only from the evidence and cite the source and page. "
                                          "If the evidence is insufficient, say you don't know."},
            {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    question = "How does Zentrax compare with adalimumab for skin clearance?"
    # Pure similarity: notice a SAFETY result can rank first because it mentions both drug names.
    show(plain_search(question), f"Plain vector search: {question}")

    # Same idea, but only safety endpoints are allowed through.
    show(search("Is the drug safe?", category="safety"), "Hybrid: safety results only")

    # Two filters at once: comparator must be Placebo AND p-value at most 0.05.
    show(search("skin clearance response", versus="Placebo", max_p=0.05),
         "Hybrid: skin clearance vs placebo, p <= 0.05")

    # No graph filters, only a minimum similarity score.
    show(search("How well does Zentrax work in patients new to biologics?", min_score=0.7),
         "Hybrid: score threshold only")

    # Retrieve, then walk the graph to get study + page for each hit.
    hits = search("quality of life", k=3)
    show(context_for([h["id"] for h in hits]), "Graph-expanded context for 'quality of life'")

    print("\n== LLM answer ==")
    print(answer(question))
