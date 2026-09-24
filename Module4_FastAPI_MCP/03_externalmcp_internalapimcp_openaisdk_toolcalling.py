# 5_openai.py

import asyncio
import json
import os

from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI

load_dotenv()

client = OpenAI()

github_token = os.environ.get("GITHUB_TOKEN")
if not github_token:
    raise RuntimeError("Set GITHUB_TOKEN in .env (GitHub PAT with MCP/repo scopes).")

# FastMCP wrapper around the local FastAPI calculator (03_add_fastmcp_fastapi.py)
INTERNAL_MCP_URL = "http://127.0.0.1:8002/mcp"

SEQUENTIAL_MCP = {
    "mcpServers": {
        "sequential-thinking": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-sequential-thinking",
            ],
        }
    }
}


def to_function_tools(mcp_tools):
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        }
        for tool in mcp_tools
    ]


async def main():
    async with (
        Client(INTERNAL_MCP_URL) as calculator,
        Client(SEQUENTIAL_MCP) as sequential,
    ):
        calculator_tools = await calculator.list_tools()
        sequential_tools = await sequential.list_tools()

        local_clients = {
            **{tool.name: calculator for tool in calculator_tools},
            **{tool.name: sequential for tool in sequential_tools},
        }

        tools = [
            {
                "type": "mcp",
                "server_label": "github",
                "server_url": "https://api.githubcopilot.com/mcp/",
                "headers": {
                    "Authorization": f"Bearer {github_token}",
                },
                "require_approval": "never",
            },
            *to_function_tools(calculator_tools),
            *to_function_tools(sequential_tools),
        ]

        input_list = [
            {
                "role": "user",
                "content": (
                    "Who am I on GitHub? Use sequential thinking to reason about "
                    "my public profile, then use the calculator MCP to compute 10 + 20. "
                    "Give a short summary that includes both the GitHub identity and the sum."
                ),
            }
        ]

        for _ in range(8):
            response = client.responses.create(
                model="gpt-5",
                input=input_list,
                tools=tools,
            )

            input_list += response.output

            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            for item in response.output:
                if item.type == "mcp_call":
                    print(f"GitHub MCP call: {item.name}")
                    print(item.output)
                elif item.type == "function_call":
                    args = json.loads(item.arguments)
                    mcp = local_clients[item.name]
                    result = await mcp.call_tool(item.name, args)
                    source = (
                        "Calculator MCP"
                        if mcp is calculator
                        else "Sequential MCP"
                    )
                    print(f"{source} call: {item.name}")
                    print(result.data)
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(result.data, default=str),
                        }
                    )

            if not function_calls:
                print(response.output_text)
                break


asyncio.run(main())
