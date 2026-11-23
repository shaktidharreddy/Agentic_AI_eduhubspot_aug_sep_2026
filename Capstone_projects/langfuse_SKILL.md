 now we need to re-integrate litellm as we did earlier into the new backend folder Trustform-AI and on top of it 

on top of it integrate langfuse for tracing all the possible events which langfuse can along with llm calls by integrating with litellm

below is langfuse integration with litellm samples:
[Langfuse Integration | mikeplavsky/litellm | DeepWiki](https://deepwiki.com/mikeplavsky/litellm/4.1-langfuse-integration)
[Langfuse | liteLLM](https://docs.litellm.ai/docs/observability/langfuse_integration)

below are other metrics along with sso i want to track inform of langfuse trace/span/event:
## What Langfuse shd measure
Three metric families over four dimensions. Everything below is derived from the same observations table, which is why one instrumentation effort feeds dashboards, alerts and the Metrics API.

### Quality

- Score averages & distributions
- LLM-as-a-judge (hallucination, relevance, toxicity)
- User feedback (thumbs, ratings)
- Code evaluators (JSON valid, regex, schema)
- Human annotation queues

### Cost & Latency

- Cost per trace / user / feature / model
- Input vs output token cost split
- p50 / p95 / p99 latency
- Time-to-first-token
- Per-observation latency (which node is slow)

### Volume

- Traces & observations per hour
- Tokens consumed
- Tool invocation counts
- Active users / sessions
- Error & warning level counts

### Metric catalogue you will actually build
MetricDefinitionWhere it comes fromAlert ideaTrace volumecount(traces) per intervalingestiondrop > 50% vs 7d baselinep95 latencypercentile(observation.duration, 95)span timingsp95 > 8 s for 15 minTime-to-first-tokencompletionStartTime − startTimegenerationTTFT > 2 sCost / tracesum(costDetails.total) / tracesmodel pricing × usage> $0.05 avgToken burnsum(usageDetails.input+output)generation usagedaily budgetError rateobservations[level=ERROR] / allspan status> 2%Tool success ratetool spans without ERROR / tool spanstool observations< 95%Retrieval qualityavg(score "context_relevance")evaluator on retriever spans< 0.7Hallucination rateshare of judge score = 0LLM-as-a-judge> 5%User feedbackavg(score "user-feedback")frontend → /api/public/scores< 0.8Cost per usersum(cost) group by userIdtrace dimensionrunaway tenantSteps per tracecount(observations)/traceagent depthloop detection

how can langfuse track non-llm events:
Nothing about Langfuse requires an LLM. You get a request waterfall, error rates, latency percentiles and custom business scores — in the same project as your agents, so a single trace can span both.

### B1 · Decorator style (fastest to adopt)

```
from fastapi import FastAPI
from langfuse import get_client, observe, propagate_attributes

app = FastAPI()
langfuse = get_client()

@observe(name="db.load_customer")
def load_customer(cid: str) -> dict:
    return db.fetch_one("SELECT * FROM customers WHERE id = %s", (cid,))

@observe(name="billing.charge")
def charge(cid: str, amount: float) -> dict:
    return payments.charge(cid, amount)

@app.post("/checkout/{cid}")
@observe(name="POST /checkout")            # becomes the root observation of the trace
def checkout(cid: str, amount: float):
    with propagate_attributes(trace_name="checkout", user_id=cid,
                              tags=["billing"], metadata={"channel": "web"}):
        customer = load_customer(cid)
        receipt  = charge(cid, amount)
        langfuse.update_current_trace(output={"receipt_id": receipt["id"]})
        return receipt
```

### B2 · Explicit spans for I/O you care about

```
from langfuse import get_client
langfuse = get_client()

with langfuse.start_as_current_observation(as_type="span", name="mongo.insert_order") as span:
    span.update(input={"collection": "orders", "op": "insert_one"})
    res = orders.insert_one(doc)
    span.update(output={"inserted_id": str(res.inserted_id)},
                metadata={"database": "genai_proddb"})

# error handling shows up as level=ERROR in the UI
try:
    call_upstream()
except Exception as e:
    span.update(level="ERROR", status_message=str(e))
    raise
finally:
    langfuse.flush()      # important in serverless / short-lived processes
```

below are the langfuse details to be used in .env
LANGFUSE_SECRET_KEY="sk-lf-258f2832-6472-48a2-b366-99738a840717" 
LANGFUSE_PUBLIC_KEY="pk-lf-2088931d-6b7e-429a-a222-493f9708a8ec" 
LANGFUSE_BASE_URL="http://10.166.133.180:3000"

LANGFUSE_SECRET_KEY="sk-lf-38d6275b-5082-46d2-b691-58b90fbf3da7"

LANGFUSE_PUBLIC_KEY="pk-lf-5a48e8fb-cee1-4305-8741-ef348fff097a"

LANGFUSE_BASE_URL="`http://10.166.133.180:3000"`


