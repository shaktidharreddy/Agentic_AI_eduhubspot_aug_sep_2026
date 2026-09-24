# 5_langchain.py

import asyncio
import json
import os

from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

load_dotenv()

github_token = os.environ.get("GITHUB_TOKEN")
if not github_token:
    raise RuntimeError("Set GITHUB_TOKEN in .env (GitHub PAT with MCP/repo scopes).")

INTERNAL_MCP_URL = "http://127.0.0.1:8002/mcp"

GITHUB_MCP = {
    "mcpServers": {
        "github": {
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {
                "Authorization": f"Bearer {github_token}",
            },
        }
    }
}

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


def tool_payload(result):
    if result.data is not None:
        return result.data
    if result.structured_content is not None:
        return result.structured_content
    texts = [
        block.text
        for block in (result.content or [])
        if getattr(block, "text", None)
    ]
    if not texts:
        return None
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    return texts


def to_langchain_tools(mcp, mcp_tools, prefix: str, label: str):
    tools = []
    tool_map = {}

    for mcp_tool in mcp_tools:
        name = f"{prefix}_{mcp_tool.name}"

        async def _call(tool_name=mcp_tool.name, **kwargs):
            result = await mcp.call_tool(tool_name, kwargs)
            payload = tool_payload(result)
            print(f"{label} call: {tool_name}")
            print(payload)
            return payload

        lc_tool = StructuredTool.from_function(
            name=name,
            description=mcp_tool.description or "",
            coroutine=_call,
            args_schema=mcp_tool.input_schema,
        )
        tools.append(lc_tool)
        tool_map[name] = lc_tool

    return tools, tool_map


async def main():
    async with (
        Client(GITHUB_MCP) as github,
        Client(INTERNAL_MCP_URL) as calculator,
        Client(SEQUENTIAL_MCP) as sequential,
    ):
        github_tools, github_map = to_langchain_tools(
            github, await github.list_tools(), "github", "GitHub MCP"
        )
        calculator_tools, calculator_map = to_langchain_tools(
            calculator, await calculator.list_tools(), "calculator", "Calculator MCP"
        )
        sequential_tools, sequential_map = to_langchain_tools(
            sequential,
            await sequential.list_tools(),
            "sequential",
            "Sequential MCP",
        )

        tools = github_tools + calculator_tools + sequential_tools
        tool_map = {**github_map, **calculator_map, **sequential_map}

        model = ChatOpenAI(model="gpt-5").bind_tools(tools)
        messages = [
            HumanMessage(
                content=(
                    "Who am I on GitHub? Use sequential thinking to reason about "
                    "my public profile, then use the calculator MCP to compute 10 + 20. "
                    "Give a short summary that includes both the GitHub identity and the sum."
                )
            )
        ]

        for _ in range(8):
            response = await model.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                print(response.content)
                break

            for tool_call in response.tool_calls:
                result = await tool_map[tool_call["name"]].ainvoke(tool_call["args"])
                messages.append(
                    ToolMessage(
                        content=json.dumps(result, default=str),
                        tool_call_id=tool_call["id"],
                    )
                )


asyncio.run(main())
