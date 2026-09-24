# 4_openai.py

import asyncio
import json

from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI

load_dotenv()

client = OpenAI()
MCP_URL = "http://127.0.0.1:8001/mcp"


async def main():
    async with Client(MCP_URL) as mcp:
        mcp_tools = await mcp.list_tools()
        tools = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            }
            for tool in mcp_tools
        ]

        response = client.responses.create(
            model="gpt-5",
            input="What is 10 + 20?",
            tools=tools,
            tool_choice="required",
        )

        print(response.output)

        for item in response.output:
            if item.type == "function_call":
                args = json.loads(item.arguments)
                result = await mcp.call_tool(item.name, args)
                print("MCP result:", result.data)


asyncio.run(main())
