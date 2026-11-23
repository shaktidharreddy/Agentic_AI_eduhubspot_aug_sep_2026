"""TypeSafe `jev` model invocation via the raw System One REST API.

Docs: https://docs.typesafe.ai/introduction/quickstart
"""

import json
import os
import ssl

import httpx
import truststore
from dotenv import load_dotenv

load_dotenv()

# corporate TLS interception: trust the Windows cert store instead of certifi
SSL_CTX = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

TICKET = (
    "Hi, I've been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing. I'm losing sales. Please help ASAP."
)

# Three question types: choice (pick one), score (ordinal scale), noul (0..1 truth value)
payload = {
    "state": TICKET,
    "model": MODEL,
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "criteria": [
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency or time-sensitivity",
        },
    },
}


def invoke_jev(body: dict) -> dict:
    resp = httpx.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=60.0,
        verify=SSL_CTX,
    )
    resp.raise_for_status()
    return resp.json()


try:
    result = invoke_jev(payload)
except httpx.HTTPStatusError as e:
    print("jev FAILED:", e.response.status_code, e.response.text)
    raise SystemExit(1)

print("model:", result["model"])
print("raw:", json.dumps(result, indent=2))

answers = result["answers"]
print("\n--- parsed ---")
print("department:", answers["department"]["choice"],
      "| confidence:", answers["department"]["confidence"])
print("frustration:", answers["frustration"]["score"],
      "->", answers["frustration"]["legend"][str(int(answers["frustration"]["score"]))])
print("is_urgent:", answers["is_urgent"]["noul"])
print("usage:", result["usage"])
