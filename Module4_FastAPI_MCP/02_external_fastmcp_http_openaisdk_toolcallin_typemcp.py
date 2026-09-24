# 4_openai.py

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

github_token = os.environ.get("GITHUB_TOKEN")
if not github_token:
    raise RuntimeError("Set GITHUB_TOKEN in .env (GitHub PAT with MCP/repo scopes).")

response = client.responses.create(
    model="gpt-5",
    input="Who am I on GitHub? Reply with my login and a short profile summary.",
    tools=[
        {
            "type": "mcp",
            "server_label": "github",
            "server_url": "https://api.githubcopilot.com/mcp/",
            "headers": {
                "Authorization": f"Bearer {github_token}",
            },
            "require_approval": "never",
        }
    ],
)

print(response.output_text)

for item in response.output:
    if item.type == "mcp_call":
        print(f"MCP call: {item.name}")
        print(item.output)
