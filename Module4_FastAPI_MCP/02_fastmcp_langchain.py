# 4_langchain.py
# Not working, because langchain_mcp_adapters is not compatible with the latest version of FastMCP.

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


async def main():

    client = MultiServerMCPClient({
        "calculator": {
            "transport": "http",
            "url": "http://127.0.0.1:8001/mcp",
        }
    })

    tools = await client.get_tools()

    agent = create_agent(
        "openai:gpt-5",
        tools
    )

    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "What is 10 + 20?"
            }
        ]
    })

    print(result["messages"][-1].content)


asyncio.run(main())