"""Shared settings, loaded once and imported by every other file.

Change a port here if one is already taken on your machine.

Everything runs through OpenRouter on a single API key, but over two different
APIs, because the two jobs are different:

  * The two agents need generated text, so they use OpenAI chat models over
    OpenRouter's OpenAI-compatible endpoint (/api/v1).
  * The supervisor only needs to pick an agent, which is a decision, so it uses
    Jev over OpenRouter's Decisions API (/api/alpha/decisions). See jev.py.
"""

import os
import sys

import truststore
from dotenv import load_dotenv

# On a corporate network, HTTPS is usually intercepted and re-signed with a
# private root CA. That CA is trusted by Windows, but Python ignores the
# Windows store and ships its own certifi bundle, so every outbound call dies
# with CERTIFICATE_VERIFY_FAILED. This makes Python use the OS trust store
# instead, which fixes urllib, httpx, OpenAI and LiteLLM in one go. It must run
# before anything opens a connection, which is why it lives at the top of the
# one module every other file imports.
truststore.inject_into_ssl()

# Windows consoles default to cp1252, where printing a reply containing a
# rupee sign or a degree symbol raises UnicodeEncodeError and takes the whole
# request down with it. Every file imports config, so fixing it once here
# covers all four servers.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Copy sample.env to .env and add your key "
        "from https://openrouter.ai/settings/keys"
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Both agents happen to use the same model; they are separate settings so you
# can point one agent somewhere else and watch nothing else change.
WEATHER_MODEL = "openai/gpt-5.6-luna"
CURRENCY_MODEL = "openrouter/openai/gpt-5.6-luna"  # LiteLLM wants the provider prefix

# How often each agent re-announces itself to the registry.
HEARTBEAT_SECONDS = 15.0

REGISTRY_PORT = 7000
WEATHER_AGENT_PORT = 7001
CURRENCY_AGENT_PORT = 7002
SUPERVISOR_PORT = 7003
TIME_AGENT_PORT = 7004

HOST = "127.0.0.1"

REGISTRY_URL = f"http://{HOST}:{REGISTRY_PORT}"
WEATHER_AGENT_URL = f"http://{HOST}:{WEATHER_AGENT_PORT}/"
CURRENCY_AGENT_URL = f"http://{HOST}:{CURRENCY_AGENT_PORT}/"
TIME_AGENT_URL = f"http://{HOST}:{TIME_AGENT_PORT}/"
SUPERVISOR_URL = f"http://{HOST}:{SUPERVISOR_PORT}"
