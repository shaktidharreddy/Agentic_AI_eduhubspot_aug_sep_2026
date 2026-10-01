"""A hand-written, minimal slice of Google's A2A protocol (spec v0.3.0).

Every A2A agent, whatever framework it was built with, exposes just two things:

  1. GET  /.well-known/agent-card.json   -> "who am I and what can I do"
  2. POST /                              -> JSON-RPC method "message/send"

That is the whole contract. Because it is plain HTTP + JSON, a LangChain agent
and a Google ADK agent look identical from the outside, and a LangGraph
supervisor can call either one without knowing how it was written.

Everything below is either a *server* helper (used by the two agents) or a
*client* helper (used by the registry and the supervisor).
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

import httpx
from fastapi import FastAPI, Request

# The well-known path where every A2A agent publishes its agent card.
AGENT_CARD_PATH = "/.well-known/agent-card.json"


# --------------------------------------------------------------------------
# Messages: the tiny JSON shape A2A uses to carry text both ways.
# --------------------------------------------------------------------------
def text_message(role: str, text: str) -> dict[str, Any]:
    """Build an A2A Message carrying a single text part."""
    return {
        "kind": "message",
        "role": role,  # "user" when asking, "agent" when answering
        "messageId": uuid.uuid4().hex,
        "parts": [{"kind": "text", "text": text}],
    }


def read_text(message: dict[str, Any]) -> str:
    """Pull the plain text back out of an A2A Message."""
    return " ".join(p["text"] for p in message["parts"] if p.get("kind") == "text")


# --------------------------------------------------------------------------
# Server side: describe an agent, then serve it.
# --------------------------------------------------------------------------
def build_agent_card(
    *,
    name: str,
    description: str,
    url: str,
    framework: str,
    skill_id: str,
    skill_description: str,
    examples: list[str],
) -> dict[str, Any]:
    """Build the agent card that will be published at AGENT_CARD_PATH.

    The card is an agent's public resume: its address, and the skills it
    advertises. The supervisor reads these skills to decide who to ask.

    `framework` is our own addition, not part of the A2A spec. Nothing in the
    protocol needs it -- it exists only so the UI can show that these agents
    really were written with different toolkits.
    """
    return {
        "protocolVersion": "0.3.0",
        "name": name,
        "description": description,
        "url": url,  # where message/send requests should be POSTed
        "preferredTransport": "JSONRPC",
        "version": "1.0.0",
        "framework": framework,
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": skill_id,
                "name": name,
                "description": skill_description,
                "tags": [skill_id, framework.lower()],
                "examples": examples,
            }
        ],
    }


async def announce_forever(card: dict[str, Any], registry_url: str, every: float) -> None:
    """Tell the registry we exist, then keep saying so.

    This is what makes registration automatic: nobody registers an agent on its
    behalf. Repeating it means the registry recovers on its own if it restarts,
    and its "last seen" timestamps double as a liveness signal.
    """
    # Let uvicorn finish binding first: the registry answers /register by
    # fetching our card, so we must be able to serve it before we ask.
    await asyncio.sleep(1)

    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(f"{registry_url}/register", json={"base_url": card["url"]})
        except Exception as error:  # registry not up yet, or restarting
            print(f"[{card['name']}] registry unreachable ({error}); will retry")
        await asyncio.sleep(every)


def build_a2a_server(
    card: dict[str, Any],
    answer: Callable[[str], Awaitable[str]],
    registry_url: str | None = None,
    heartbeat_seconds: float = 15.0,
) -> FastAPI:
    """Turn any `async answer(question) -> reply` function into an A2A server.

    This is the adapter that hides the framework. `answer` can be backed by
    LangChain, Google ADK, or anything else; the HTTP surface is the same.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        heartbeat = None
        if registry_url:
            heartbeat = asyncio.create_task(
                announce_forever(card, registry_url, heartbeat_seconds)
            )
        yield
        if heartbeat:
            heartbeat.cancel()

    app = FastAPI(title=card["name"], lifespan=lifespan)

    @app.get(AGENT_CARD_PATH)
    def serve_agent_card() -> dict[str, Any]:
        """Discovery endpoint: hand out the agent card."""
        return card

    @app.post("/")
    async def handle_json_rpc(request: Request) -> dict[str, Any]:
        """Execution endpoint: run one A2A `message/send` call."""
        rpc = await request.json()

        if rpc.get("method") != "message/send":
            return {
                "jsonrpc": "2.0",
                "id": rpc.get("id"),
                "error": {"code": -32601, "message": "Only message/send is supported"},
            }

        question = read_text(rpc["params"]["message"])
        print(f"[{card['name']}] asked: {question}")

        try:
            reply = await answer(question)
        except Exception as error:
            # A2A failures belong in the JSON-RPC error field, not an HTTP 500,
            # so the caller learns what actually went wrong (usually a quota).
            print(f"[{card['name']}] failed: {type(error).__name__}: {error}")
            return {
                "jsonrpc": "2.0",
                "id": rpc.get("id"),
                "error": {"code": -32603, "message": f"{type(error).__name__}: {error}"},
            }

        print(f"[{card['name']}] replied: {reply}")

        return {
            "jsonrpc": "2.0",
            "id": rpc.get("id"),
            "result": text_message("agent", reply),
        }

    return app


# --------------------------------------------------------------------------
# Client side: discover an agent, then call it.
# --------------------------------------------------------------------------
async def fetch_agent_card(base_url: str) -> dict[str, Any]:
    """Discovery: read an agent's card straight off its well-known URL."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(base_url.rstrip("/") + AGENT_CARD_PATH)
        response.raise_for_status()
        return response.json()


async def send_message(card: dict[str, Any], question: str) -> dict[str, Any]:
    """Invocation: send one `message/send` JSON-RPC call to a discovered agent.

    Returns the reply text together with the raw request and response, so the
    UI can show students the actual protocol traffic rather than just the
    answer that came out of it.
    """
    request_payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {"message": text_message("user", question)},
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(card["url"], json=request_payload)
        response.raise_for_status()
        rpc = response.json()

    if "error" in rpc:
        raise RuntimeError(rpc["error"]["message"])

    return {
        "reply": read_text(rpc["result"]),
        "request": request_payload,
        "response": rpc,
    }
