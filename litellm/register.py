"""Register MCP servers and A2A agents over REST.

Use this when you cannot edit the LiteLLM server's config.yaml. Requires
STORE_MODEL_IN_DB=True on that server, otherwise registrations do not survive a
restart.

    python register.py --list
    python register.py --apply
    python register.py --remove

Registrations created here live in the database and ARE editable from the Admin
UI. Config-file registrations are not. If a name exists in both, the database
record wins and the config entry is skipped.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

PROXY = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000").rstrip("/")
KEY = os.getenv("LITELLM_MASTER_KEY") or os.getenv("LITELLM_API_KEY")
if not KEY:
    sys.exit("set LITELLM_MASTER_KEY (or LITELLM_API_KEY) in the environment or .env")

HEADERS = {"Authorization": f"Bearer {KEY}"}

MCP_HOST = os.getenv("MCP_PUBLIC_HOST", "mcp-host.internal")
A2A_HOST = os.getenv("A2A_ADVERTISED_HOST", "a2a-host.internal")

# Skills live on the proxy itself; without this they are proxied to Anthropic,
# which needs ANTHROPIC_API_KEY and is a different registry entirely.
_SKILL_Q = {"custom_llm_provider": "litellm_proxy"}

MCP_SERVERS = [
    {
        "server_name": "calc_http",
        "alias": "calc_http",
        "description": "FastMCP Calculator over streamable HTTP",
        "transport": "http",
        "url": f"http://{MCP_HOST}:8001/mcp",
    },
    {
        "server_name": "calc_sse",
        "alias": "calc_sse",
        "description": "FastMCP Calculator over SSE",
        "transport": "sse",
        "url": f"http://{MCP_HOST}:8003/mcp",
    },
    {
        "server_name": "calc_openapi",
        "alias": "calc_openapi",
        "description": "Tools generated from the Calculator API OpenAPI spec",
        "transport": "http",
        "url": f"http://{MCP_HOST}:8002/mcp",
    },
]

AGENTS = [
    {
        "agent_name": "weather-agent",
        "agent_card_params": {
            "name": "Weather Agent",
            "description": "Reports the current weather for a city. Built with LangChain.",
            "url": f"http://{A2A_HOST}:7001/",
            "protocolVersion": "0.3",
        },
    },
    {
        "agent_name": "currency-agent",
        "agent_card_params": {
            "name": "Currency Agent",
            "description": "Converts money between currencies. Built with Google ADK.",
            "url": f"http://{A2A_HOST}:7002/",
            "protocolVersion": "0.3",
        },
    },
    {
        "agent_name": "time-agent",
        "agent_card_params": {
            "name": "Time Agent",
            "description": "Tells the current local time in a city. Built with plain Python.",
            "url": f"http://{A2A_HOST}:7004/",
            "protocolVersion": "0.3",
        },
    },
    {
        "agent_name": "langgraph-dag-agent",
        "agent_card_params": {
            "name": "LangGraph DAG Planner",
            "description": (
                "Decomposes a question into a subtask DAG, routes each subtask to the best "
                "MCP tool or A2A agent, and synthesizes a cited answer. Built with LangGraph."
            ),
            "url": f"http://{A2A_HOST}:7005/",
            "protocolVersion": "0.3",
            # Omitting skills makes LiteLLM store a generic "chat" skill, which is
            # what agent_search embeds, so the planner ranks on the wrong text.
            "skills": [
                {
                    "id": "dag-planning",
                    "name": "Multi-step planning and orchestration",
                    "description": (
                        "Answers questions that need several capabilities combined: decomposition, "
                        "dependency-aware planning, tool and agent routing, and synthesis with citations."
                    ),
                    "tags": ["planning", "orchestration", "dag", "langgraph", "multi-step"],
                }
            ],
        },
    },
]

# Skills are Anthropic Agent Skills archives hosted by the proxy: a zip with
# SKILL.md at the root. The proxy injects that file into the system prompt when
# a call passes container.skills. Only the archive content reaches the model --
# the `instructions` form field is stored but never returned or injected.
POSTMORTEM_SKILL_MD = """---
name: blameless-incident-postmortem
description: House format and rules for writing or reviewing a blameless incident postmortem.
---

# Blameless Incident Postmortem

When asked to write or review an incident postmortem, always produce these sections in order:

1. **Summary** - one paragraph, plain language, no jargon.
2. **Impact** - who was affected, for how long, and how it was measured.
3. **Timeline** - UTC timestamps, one line per event, detection before mitigation.
4. **Root cause** - the contributing conditions, not a person.
5. **What went well / What went poorly**
6. **Action items** - each with an owner and a due date, phrased as a change to a system.

Rules:
- Never name individuals. Refer to roles ("the on-call engineer").
- Never write "human error" as a root cause; describe the system that allowed it.
- Every action item must be verifiable and must prevent recurrence, not just detect it.
- Always end with a line: "Reviewed by: <role>".
"""

# The agent card's `dag-planning` skill only advertises what the planner can do;
# it is A2A metadata and cannot be attached to a model call. This is the
# reusable form of the same know-how, usable by any agent via container.skills.
DAG_PLANNING_SKILL_MD = """---
name: dag-task-planning
description: Method for decomposing a request into subtasks, declaring dependencies and ordering them into parallel levels.
---

# DAG Task Planning

Turn a request into the smallest set of subtasks that can be executed and verified
independently, then order them.

## 1. Decompose
- Produce the smallest useful subtasks. Prefer 1-4; never invent work that was not asked for.
- Each subtask must be independently checkable: someone reading only its description
  and its result can tell whether it succeeded.
- Give each an id: s1, s2, s3 ...

## 2. Declare dependencies
- `depends_on` lists only the subtasks whose **result** this one consumes.
- Shared topic is not a dependency. If s2 would run correctly without s1's output,
  it does not depend on s1.
- Keep the dependency set minimal: every edge removes parallelism.

## 3. Order into levels
- Level 0 is every subtask with no dependencies; they run in parallel.
- A subtask enters the next level once all of its dependencies have completed.
- A cycle is a decomposition error. Break it by merging the subtasks involved.

## 4. Reference results, never restate them
- When a subtask needs an earlier result, reference it as `${s1}` rather than
  guessing or copying a value.

## Output
```json
{"subtasks": [{"id": "s1", "description": "...", "depends_on": []}]}
```

Rules:
- Never put two unrelated goals in one subtask; split them so they can run in parallel.
- A subtask that only reformats or summarises another subtask's output is usually
  not worth its own node; fold it into the consumer.
"""

SKILLS = [
    {
        "display_title": "Blameless Incident Postmortem",
        "description": (
            "House format and rules for writing or reviewing a blameless incident postmortem: "
            "required sections, timeline conventions and action-item rules."
        ),
        "files": {"SKILL.md": POSTMORTEM_SKILL_MD},
    },
    {
        "display_title": "DAG Task Planning",
        "description": (
            "Method for decomposing a request into the smallest useful subtasks, declaring "
            "dependencies between them, and ordering them into parallel execution levels."
        ),
        "files": {"SKILL.md": DAG_PLANNING_SKILL_MD},
    },
]


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


def _client() -> httpx.Client:
    return httpx.Client(base_url=PROXY, headers=HEADERS, timeout=120)


def list_registered() -> None:
    with _client() as c:
        servers = c.get("/v1/mcp/server").json()
        print(f"MCP servers ({len(servers)}):")
        for s in servers:
            name = s.get("alias") or s.get("server_name")
            print(f"  {name:16} {s.get('transport'):6} {s.get('url') or s.get('command', '')}")
            print(f"     server_id={s.get('server_id')}  is_config={s.get('is_config')}")

        agents = c.get("/v1/agents").json()
        agents = agents if isinstance(agents, list) else agents.get("data", [])
        print(f"\nA2A agents ({len(agents)}):")
        for a in agents:
            card = a.get("agent_card_params") or {}
            print(f"  {a.get('agent_name'):16} {card.get('url')}  v{card.get('protocolVersion')}")

        health = c.get("/v1/mcp/server/health")
        print(f"\nMCP health: {health.text[:400]}")

        skills = c.get("/v1/skills", params=_SKILL_Q).json().get("data", [])
        print(f"\nSkills ({len(skills)}):")
        for s in skills:
            print(f"  {s.get('display_title'):34} {s.get('id')}")


def apply() -> None:
    with _client() as c:
        existing = {(s.get("alias") or s.get("server_name")) for s in c.get("/v1/mcp/server").json()}
        for payload in MCP_SERVERS:
            if payload["server_name"] in existing:
                print(f"skip   mcp   {payload['server_name']} (already registered)")
                continue
            r = c.post("/v1/mcp/server", json=payload)
            print(f"{'ok  ' if r.status_code < 300 else 'FAIL'}   mcp   "
                  f"{payload['server_name']} -> {r.status_code} {r.text[:120]}")

        raw = c.get("/v1/agents").json()
        raw = raw if isinstance(raw, list) else raw.get("data", [])
        have = {a.get("agent_name") for a in raw}
        for payload in AGENTS:
            if payload["agent_name"] in have:
                print(f"skip   agent {payload['agent_name']} (already registered)")
                continue
            r = c.post("/v1/agents", json=payload)
            print(f"{'ok  ' if r.status_code < 300 else 'FAIL'}   agent "
                  f"{payload['agent_name']} -> {r.status_code} {r.text[:120]}")

        titles = {s.get("display_title") for s in c.get("/v1/skills", params=_SKILL_Q).json().get("data", [])}
        for skill in SKILLS:
            if skill["display_title"] in titles:
                print(f"skip   skill {skill['display_title']} (already registered)")
                continue
            r = c.post("/v1/skills", params=_SKILL_Q,
                       data={"display_title": skill["display_title"], "description": skill["description"]},
                       files={"files[]": ("skill.zip", _zip(skill["files"]), "application/zip")})
            print(f"{'ok  ' if r.status_code < 300 else 'FAIL'}   skill "
                  f"{skill['display_title']} -> {r.status_code} {r.text[:120]}")


def remove() -> None:
    wanted_mcp = {p["server_name"] for p in MCP_SERVERS}
    wanted_agents = {p["agent_name"] for p in AGENTS}
    with _client() as c:
        for s in c.get("/v1/mcp/server").json():
            name = s.get("alias") or s.get("server_name")
            if name in wanted_mcp and not s.get("is_config"):
                r = c.delete(f"/v1/mcp/server/{s['server_id']}")
                print(f"delete mcp   {name} -> {r.status_code}")

        raw = c.get("/v1/agents").json()
        raw = raw if isinstance(raw, list) else raw.get("data", [])
        for a in raw:
            if a.get("agent_name") in wanted_agents:
                ident = a.get("agent_id") or a["agent_name"]
                r = c.delete(f"/v1/agents/{ident}")
                print(f"delete agent {a['agent_name']} -> {r.status_code}")

        wanted_skills = {s["display_title"] for s in SKILLS}
        for s in c.get("/v1/skills", params=_SKILL_Q).json().get("data", []):
            if s.get("display_title") in wanted_skills:
                r = c.delete(f"/v1/skills/{s['id']}", params=_SKILL_Q)
                print(f"delete skill {s['display_title']} -> {r.status_code}")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="show what is registered")
    g.add_argument("--apply", action="store_true", help="register anything missing")
    g.add_argument("--remove", action="store_true", help="delete what this script registered")
    args = p.parse_args()

    print(f"proxy: {PROXY}\n")
    if args.list:
        list_registered()
    elif args.apply:
        apply()
    else:
        remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
