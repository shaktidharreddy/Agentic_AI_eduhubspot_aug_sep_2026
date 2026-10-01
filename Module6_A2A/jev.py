"""Jev: the decision model that picks which agent should handle a question.

Jev is not a chat model. You POST application state plus typed questions to
OpenRouter's Decisions API and get typed answers back, each with probabilities
and a confidence. There is no messages/completions surface, so ChatOpenAI and
the OpenAI SDK cannot call it.

    https://openrouter.ai/api/alpha/decisions

Choosing between agents is a decision, not a writing task, which is exactly the
shape Jev is built for. The menu of options is not hardcoded here: it is built
at runtime from whatever agent cards the registry happens to be holding, so
registering a new agent changes the routing with no code change.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any

JEV_MODEL = "~typesafe/jev-latest"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


def choose(
    *,
    state: dict[str, Any],
    options: dict[str, str],
    instructions: str,
) -> dict[str, Any]:
    """Ask Jev to pick one of `options` (label -> what that option is for).

    Returns the choice plus the probability it assigned to every option, which
    is the part worth showing a student: you can see the runner-up.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set -- add it to your .env")

    payload = {
        "model": JEV_MODEL,
        "state": state,
        "questions": {
            "agent": {
                "type": "choice",
                "instructions": instructions,
                "criteria": options,
            }
        },
    }

    request = urllib.request.Request(
        DECISIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "a2a-cep-supervisor",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 402:
            raise RuntimeError(
                "OpenRouter 402: this key has no credits. Top up at "
                "https://openrouter.ai/settings/credits"
            ) from exc
        # A corporate web filter answers with an HTML block page rather than
        # the JSON error OpenRouter would send, which is worth naming outright
        # because no amount of retrying will help.
        if "Web Page Blocked" in detail:
            raise RuntimeError(
                "openrouter.ai is blocked by your network's web filter, so Jev "
                "cannot be reached. Request an exception for the site, or run "
                "the demo off the corporate network."
            ) from exc
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach openrouter.ai: {exc.reason}") from exc

    answer = body["answers"]["agent"]
    return {
        "choice": answer["choice"],
        "confidence": float(answer.get("confidence", 0.0)),
        "probabilities": answer.get("probabilities", {}),
        "model": body.get("model", JEV_MODEL),
        "cost_usd": body.get("usage", {}).get("cost"),
    }
