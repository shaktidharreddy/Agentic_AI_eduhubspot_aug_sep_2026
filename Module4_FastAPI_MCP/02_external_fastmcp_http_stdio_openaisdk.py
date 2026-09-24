
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

# Remote GitHub MCP is called by OpenAI. sequential-thinking is a local stdio
# process, so this client starts it with npx and adds its tools in the same
# Responses request. OpenAI type:"mcp" cannot spawn local commands.
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


async def main():
    async with Client(SEQUENTIAL_MCP) as sequential:
        sequential_tools = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            }
            for tool in await sequential.list_tools()
        ]

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
            *sequential_tools,
        ]

        input_list = [
            {
                "role": "user",
                "content": (
                    "Who am I on GitHub? Use sequential thinking to reason "
                    "about what my public profile suggests, then give a short summary."
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
                    result = await sequential.call_tool(item.name, args)
                    print(f"Sequential MCP call: {item.name}")
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
