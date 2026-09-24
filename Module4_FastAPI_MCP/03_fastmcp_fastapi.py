# 5_mcp.py

import httpx

from fastmcp import FastMCP


API_URL = "http://127.0.0.1:8000"


# Get FastAPI's OpenAPI specification
openapi = httpx.get(
    f"{API_URL}/openapi.json"
).json()


# HTTP client used to call FastAPI
client = httpx.AsyncClient(
    base_url=API_URL
)


# Generate MCP interface from OpenAPI
mcp = FastMCP.from_openapi(
    openapi_spec=openapi,
    client=client,
    name="Calculator API"
)


if __name__ == "__main__":

    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8002
    )