# 04_fastmcp_only.py

from fastmcp import FastMCP

mcp = FastMCP("Calculator")


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


if __name__ == "__main__":
    mcp.run(
        transport="sse",
        host="127.0.0.1",
        port=8003,
        path="/mcp",
    )
