# Minimal A2A demo: two frameworks, one protocol

Three agents talk to each other over Google's **A2A protocol**, and no two of them
share a framework:

| Piece | Framework | Port |
| --- | --- | --- |
| Agent registry | plain FastAPI | 7000 |
| Weather agent | LangChain prebuilt agent | 7001 |
| Currency agent | Google ADK | 7002 |
| Time agent | no framework at all | 7004 |
| Supervisor / router | LangGraph + Jev | 7003 |
| Streamlit UI | Streamlit | 8501 |

The time agent is there to make the point bluntly: it has no framework and no
model, just a dict and an f-string, and neither the registry nor the supervisor
can tell it apart from the other two.

The point of the demo: the supervisor never imports LangChain's agent or Google
ADK. It only reads **agent cards** and makes HTTP calls. That is what A2A buys
you — agents written by different teams in different frameworks can be swapped
in and out without the caller changing a line.

## Who does what

The three responsibilities are deliberately kept apart, because conflating them
is the thing that confuses people about agent registries:

| Question | Answer |
| --- | --- |
| Does the registry store all the agent cards, and where? | Yes — in one in-memory dict, `AGENTS` in `registry.py`, keyed by agent name. Restart it and it empties; the agents refill it on their next heartbeat. |
| Does it keep skills, endpoint and metadata? | Yes — the whole card: `url`, `preferredTransport`, `protocolVersion`, `capabilities`, and the `skills` array with tags and examples. Plus `first_registered`, `last_seen` and a heartbeat count per agent. |
| Who publishes a card? | Each **agent**, at its own `/.well-known/agent-card.json`. The registry never invents a card, it fetches it. |
| Is the registry updated by the agents automatically? | Yes — the weather and currency agents POST their own URL on startup and again every 15s. The time agent shows the other route: register any URL from the UI at runtime. |
| Does it enable capability-based discovery? | Yes — `GET /discover?capability=...` ranks every card by word overlap against its advertised skills, and says which words matched. |
| Does it return the best matching agent? | Yes — `GET /best-match?capability=...` returns that single card, and `/discover` flags it with `best_match: true`. |
| Does the supervisor discover dynamically, using Jev? | Yes — `jev.py` builds its options menu at runtime from whatever cards came back, so registering a third agent changes the routing with no code change. |
| Does the chosen card drive the A2A call? | Yes — `send_message(card, question)` POSTs one `message/send` to that card's `url`. |

The division of labour between the last three rows is the interesting part. The
registry's keyword score is a **hint**; Jev makes the call and reports how
confident it was. Ask *"Should I take an umbrella in Tokyo?"* and you can watch
them disagree:

```
registry keyword pick : Currency Agent     <- 0 words matched either card, so "best" is just first
jev pick              : Weather Agent  (OVERRODE)
probabilities         : {'Currency Agent': 0, 'Weather Agent': 1}
answer                : No umbrella is needed in Tokyo today; it's clear and 19°C.
```

That is the whole argument for not routing with string matching.

## The protocol, in two endpoints

Every A2A agent publishes exactly two things, and `a2a_min.py` implements both
in about 40 lines:

1. **`GET /.well-known/agent-card.json`** — the agent card, a public résumé
   listing the agent's address and skills. (Same well-known path that a hosted
   LangGraph Platform deployment serves.)
2. **`POST /`** — the JSON-RPC method `message/send`, which runs one turn.

An agent card looks like this:

```json
{
  "protocolVersion": "0.3.0",
  "name": "Weather Agent",
  "description": "Reports the current weather for a city. Built with LangChain.",
  "url": "http://127.0.0.1:7001/",
  "preferredTransport": "JSONRPC",
  "version": "1.0.0",
  "framework": "LangChain",
  "capabilities": { "streaming": false, "pushNotifications": false },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [
    {
      "id": "weather_lookup",
      "name": "Weather Agent",
      "description": "Given a city name, return today's temperature and conditions.",
      "tags": ["weather_lookup", "langchain"],
      "examples": ["What is the weather in London?", "Is it raining in Mumbai?"]
    }
  ]
}
```

`framework` is the one field here that is **not** in the A2A spec. Nothing in
the protocol needs it — it exists only so the UI can show that these agents
really were built with different toolkits.

And one call to it looks like this:

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "message/send",
  "params": {
    "message": {
      "kind": "message",
      "role": "user",
      "messageId": "abc123",
      "parts": [{ "kind": "text", "text": "What is the weather in London?" }]
    }
  }
}
```

## Setup

```powershell
cd A2A\A2A_cep
uv venv
uv pip install -r requirements.txt
```

Copy `sample.env` to `.env` and add one key, from
[openrouter.ai/settings/keys](https://openrouter.ai/settings/keys):

```
OPENROUTER_API_KEY=...
```

### One key, two APIs

Everything goes through OpenRouter, but over two different APIs, because the two
jobs are different:

| Who | Needs | API | Model |
| --- | --- | --- | --- |
| Weather agent (LangChain) | generated text | `/api/v1` (OpenAI-compatible) | `openai/gpt-5.6-luna` |
| Currency agent (Google ADK) | generated text | `/api/v1` via LiteLLM | `openrouter/openai/gpt-5.6-luna` |
| Supervisor | one decision | `/api/alpha/decisions` | `~typesafe/jev-latest` |

OpenRouter is OpenAI-API-compatible, so the LangChain agent just points the
ordinary OpenAI provider at a different base URL:

```python
model = init_chat_model(
    WEATHER_MODEL,
    model_provider="openai",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)
```

Google ADK defaults to Gemini, so it reaches the same model through its LiteLLM
escape hatch — which is why `requirements.txt` asks for `google-adk[extensions]`:

```python
currency_agent = LlmAgent(model=LiteLlm(model=CURRENCY_MODEL), ...)
```

Routing with Jev instead of a chat model costs about **$0.000015 per decision**.
That is most of the reason to use a decision model for a branch: you are not
paying frontier prices to answer "which of these two agents".

## Run it

```powershell
.venv\Scripts\python.exe run_all.py                    # terminal 1: the four servers
.venv\Scripts\python.exe -m streamlit run app_streamlit.py   # terminal 2: the UI
```

Then open <http://localhost:8501>. Or skip the UI and use the console walkthrough:

```powershell
.venv\Scripts\python.exe demo.py
```

`run_all.py` waits for each port to answer before telling you it is ready —
importing Google ADK takes a good 20 seconds, so give it a moment. The agents
announce themselves about a second after they boot, so if the UI shows an empty
registry, wait and hit **Refresh**.

## What the UI shows

Two halves, matching the two ideas:

**The registry.** A box to register any agent by URL, then a table of every
agent currently known: its name, which framework built it, its address, its
advertised skill, how many heartbeats it has sent and how long ago. Below that,
each full agent card as raw JSON — exactly what the agent serves at its
well-known path.

**A question.** Type one (or click an example) and the supervisor's three steps
are laid out with their interim output:

1. `discover` — the ranked candidates from the registry, with the keyword score
   and which words matched, so the shortlist is not a black box.
2. `decide` — Jev's probability for *every* agent as a bar chart, its confidence,
   the cost of the decision, whether it agreed with the registry's keyword pick,
   and the exact options menu it was handed.
3. `call_agent` — the raw A2A `message/send` request and the response envelope,
   side by side, plus which framework the agent on the other end was built with.

The example worth demoing is **"Should I take an umbrella in Tokyo?"**. Neither
card contains the word "umbrella" or "Tokyo", so keyword overlap scores both
agents zero and the registry's best match is simply whichever agent it happens
to hold first — the currency one. Jev overrides it and routes to the weather
agent with probability 1. One click makes the case for why the cheap filter only
shortlists and a decision model decides.

`demo.py` prints the same story as the UI, in text:

```
1. REGISTRATION -- the agents announced themselves
Currency Agent   built with Google ADK   at http://127.0.0.1:7002/  (2 heartbeats, last 13.3s ago)
Weather Agent    built with LangChain    at http://127.0.0.1:7001/  (2 heartbeats, last 10.5s ago)

2. DISCOVERY -- cards come from the agents, ranked by capability
'Convert 100 USD to INR'
  score 3  Currency Agent   matched on ['convert', 'inr', 'usd']
  score 0  Weather Agent    matched on -

3. INVOCATION -- Jev chooses, then the supervisor calls over A2A
Q: What is the weather in London?
   discover    Registry ranked 2 agent card(s) by capability
   decide      Jev chose Weather Agent with 100% confidence
               probabilities: {'Currency Agent': 0, 'Weather Agent': 1}
   call_agent  Sent A2A message/send to Weather Agent at http://127.0.0.1:7001/
   answer      It is currently 14 degrees Celsius and drizzling in London.
```

Watch terminal 1 at the same time — you can see the request hop from supervisor
to registry to agent:

```
[registry] registered Weather Agent (LangChain) at http://127.0.0.1:7001/
[registry] discover('What is the weather in London?') -> [('Weather Agent', 2), ('Currency Agent', 0)]
[supervisor] Jev chose Weather Agent (confidence=1.00) {'Weather Agent': 1, 'Currency Agent': 0}
[Weather Agent] asked: What is the weather in London?
[Weather Agent] replied: It is currently 14 degrees Celsius and drizzling in London.
```

## Poking at it by hand

Each server also has interactive Swagger docs — open
<http://127.0.0.1:7000/docs> (or `7001`, `7002`, `7003`) and click through the
endpoints. Or use PowerShell:

```powershell
# read an agent card straight from the agent (this one also works in a browser)
Invoke-RestMethod http://127.0.0.1:7001/.well-known/agent-card.json | ConvertTo-Json -Depth 5

# what the registry holds, and how fresh it is
Invoke-RestMethod http://127.0.0.1:7000/status | Format-Table
Invoke-RestMethod http://127.0.0.1:7000/agents | ConvertTo-Json -Depth 5

# capability-based discovery, ranked
Invoke-RestMethod "http://127.0.0.1:7000/discover?capability=convert%20100%20USD%20to%20INR" |
  ConvertTo-Json -Depth 4

# just the best-matching agent
Invoke-RestMethod "http://127.0.0.1:7000/best-match?capability=weather%20in%20London" |
  ConvertTo-Json -Depth 4

# call an agent directly over A2A, bypassing the supervisor
Invoke-RestMethod -Method Post http://127.0.0.1:7001/ -ContentType application/json -Body '{
  "jsonrpc": "2.0", "id": "1", "method": "message/send",
  "params": {"message": {"kind": "message", "role": "user", "messageId": "abc",
                         "parts": [{"kind": "text", "text": "weather in Tokyo?"}]}}}' |
  ConvertTo-Json -Depth 5

# or go through the supervisor and let it choose
Invoke-RestMethod -Method Post http://127.0.0.1:7003/ask `
  -ContentType application/json -Body '{"question": "Convert 50 GBP to EUR"}'
```

Calling an agent directly returns the raw A2A envelope, which is worth showing
students next to the card:

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "kind": "message",
    "role": "agent",
    "messageId": "ca430995565e484b85d2841f4e7c18cd",
    "parts": [{ "kind": "text", "text": "It is 19 degrees C and clear in Tokyo." }]
  }
}
```

## The files

| File | What it holds |
| --- | --- |
| `a2a_min.py` | The whole protocol layer: build a card, serve an agent, heartbeat, discover, call |
| `config.py` | Ports, model names and the API keys, in one place |
| `registry.py` | The registry: `/register`, `/agents`, `/discover`, `/best-match`, `/status` |
| `agent_weather_langchain.py` | Agent 1 — `create_agent(...)` plus one tool |
| `agent_currency_adk.py` | Agent 2 — `LlmAgent` + `InMemoryRunner` plus one tool |
| `agent_time_plain.py` | Agent 3 — no framework, no model. The template for your own |
| `jev.py` | One function: ask Jev to choose one labelled option |
| `supervisor.py` | A 3-node LangGraph: `discover` -> `decide` -> `call_agent` |
| `app_streamlit.py` | The UI |
| `demo.py` | The same walkthrough in the console |
| `run_all.py` | Starts the four servers together |

The two agent files are worth reading side by side. Everything above `answer()`
is framework-specific and completely different; everything below it — the card,
the server, the heartbeat — is identical.

## Adding your own agent

There are two ways in, depending on whether you own the agent.

### If you wrote the agent: let it register itself

Copy `agent_time_plain.py`, which is the smallest possible example — no
framework, no model, just a dict and an f-string. Three things to change:

```python
async def answer(question: str) -> str:      # 1. what your agent does
    return "..."

card = build_agent_card(                     # 2. what it advertises
    name="Invoice Agent",
    description="Looks up invoice status. Built with CrewAI.",
    url=INVOICE_AGENT_URL,
    framework="CrewAI",
    skill_id="invoice_lookup",
    skill_description="Given an invoice number, return its payment status.",
    examples=["Is invoice 4471 paid?"],
)

# 3. passing REGISTRY_URL is what makes registration automatic
app = build_a2a_server(card, answer, REGISTRY_URL, HEARTBEAT_SECONDS)
```

Then add a port to `config.py` and a line to `SERVICES` in `run_all.py`. That
is all. **Nothing else in the project changes** — not the registry, not the
supervisor, not the UI. On startup your agent announces itself, the registry
fetches its card, and Jev starts including it in the menu because the menu is
built from whatever cards came back.

### If someone else runs the agent: register the URL

You do not need any code in this project at all. If an A2A agent is running
anywhere and serves `/.well-known/agent-card.json`, paste its URL into the
**Register an agent by URL** box at the top of the UI and press Register. It is
routable immediately — no restart, no redeploy.

The time agent is set up to demonstrate exactly this. It runs on port 7004 but
is deliberately *not* given the registry's URL, so it never announces itself and
starts life unregistered, standing in for an agent on someone else's machine.
Register it from the UI and then ask "What time is it in Tokyo?".

The same thing from a shell:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:7000/register `
  -ContentType application/json -Body '{"base_url": "http://127.0.0.1:7004/"}'
```

```
POST /register -> {'registered': 'Time Agent', 'framework': 'Plain Python',
                   'skills': ['local_time'], 'heartbeats': 1}
```

Note what you did *not* send: no skills, no description, no endpoint list. The
registry takes the URL and fetches the card from the agent itself, which is why
a registered card can never be stale or a lie. The trade-off is that the agent
must be reachable at that moment — registration doubles as a liveness check, and
a URL that cannot serve a card is rejected with a `400` explaining what was
expected where:

```
Could not read an agent card from http://127.0.0.1:9999/. Expected
http://127.0.0.1:9999/.well-known/agent-card.json to return one.
(ConnectError: All connection attempts failed)
```

Re-POST the same URL to refresh it, which is exactly what the self-registering
agents do on their heartbeat. That also means the **Heartbeats** column in the UI
tells you which route an agent took: it climbs on its own for a self-registering
agent and stays at 1 for one you registered by hand.

### What you get for free

With three agents registered, routing gets genuinely interesting — and the
keyword filter starts being actively wrong:

```
What time is it in Tokyo?              registry: Time Agent     jev: Time Agent
Should I take an umbrella in Tokyo?    registry: Time Agent     jev: Weather Agent
How many yen is 50 dollars?            registry: Time Agent     jev: Currency Agent
```

The registry favours the Time Agent on the umbrella question because "Tokyo"
appears in *its* examples, and wins the third by default because nothing matched
at all. Jev gets all three right. This is the case for keeping shortlisting and
deciding as separate jobs.

## If you are on a corporate network

Two different things go wrong behind a corporate proxy, and they look similar
enough to be confusing.

**Certificate errors.** A message like

```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
self-signed certificate in certificate chain
```

means the proxy is intercepting HTTPS and re-signing it with a private company
root CA. Windows trusts that CA; Python does not, because it ignores the OS
certificate store and ships its own `certifi` bundle. `config.py` calls
`truststore.inject_into_ssl()` before anything opens a connection, which points
Python at the OS store and fixes every library at once. Certificate checking
stays on — never "fix" this by turning verification off.

**A blocked site.** If instead you see

> openrouter.ai is blocked by your network's web filter

then TLS was fine and the proxy refused the request on policy grounds, usually
because the site is categorised as artificial intelligence. No code change can
get around that. Either request an exception for the site, or run the demo on a
network that allows it.

Only the two model-backed agents and the supervisor's Jev call need the
internet. The registry, agent-card discovery, the A2A message exchange and the
Time Agent all run entirely on localhost, so they keep working either way.

## Things left out on purpose

This is a teaching slice of A2A, not a compliant implementation. A real one adds
tasks and task IDs for long-running work, `message/stream` over SSE for token
streaming, `contextId` so a conversation has memory across calls, push
notifications, and authentication via the card's `securitySchemes`. For those,
use the official [`a2a-sdk`](https://github.com/a2aproject/a2a-python).

The registry is also simpler than a real one: it holds everything in memory,
has no authentication, never evicts an agent that has gone quiet (it only
records how long ago it was last seen), and matches capabilities with word
overlap rather than embeddings.

Spec version note: this demo targets A2A **0.3.0**, where an agent's address is
the card's `url` + `preferredTransport`. Spec **1.0** replaces that pair with a
`supportedInterfaces` list. The well-known path and `message/send` are unchanged.
