"""LangChain <-> LiteLLM integration snippets.

Docs: https://docs.langchain.com/oss/python/integrations/providers/litellm
Install: uv add langchain-litellm
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_litellm import (
    ChatLiteLLM,
    ChatLiteLLMRouter,
    LiteLLMEmbeddings,
    LiteLLMEmbeddingsRouter,
)
from litellm import Router
from pydantic import BaseModel, Field

load_dotenv()

API_KEY = os.environ["LITELLM_API_KEY"]
BASE_URL = os.environ["LITELLM_BASE_URL"]
MODEL = "gpt-5-nano"
EMBED_MODEL = "text-embedding-3-large"


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. ChatLiteLLM - talking to a LiteLLM proxy
# ---------------------------------------------------------------------------
# The "litellm_proxy/" prefix tells LiteLLM to forward to a proxy instead of
# calling a provider SDK directly. Without a proxy, use a plain provider model
# string such as "openai/gpt-4.1" or "anthropic/claude-sonnet-4-5".

llm = ChatLiteLLM(
    model=f"litellm_proxy/{MODEL}",
    api_key=API_KEY,
    api_base=BASE_URL,
    temperature=0.1,
    max_retries=2,
)


def basic_invoke() -> None:
    section("1. invoke")
    messages = [
        SystemMessage("You are a terse assistant. Answer in one sentence."),
        HumanMessage("Why is the sky blue?"),
    ]
    response = llm.invoke(messages)
    print(response.content)
    print("usage:", response.usage_metadata)


# ---------------------------------------------------------------------------
# 2. Streaming
# ---------------------------------------------------------------------------
def streaming() -> None:
    section("2. stream")
    for chunk in llm.stream("Count from 1 to 5, comma separated."):
        print(chunk.content, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 3. Async + async streaming
# ---------------------------------------------------------------------------
async def async_calls() -> None:
    section("3. ainvoke / astream")
    response = await llm.ainvoke(
        "Classify the sentiment as neutral, negative or positive. "
        "Text: I think the food was okay. Sentiment:"
    )
    print("ainvoke:", response.content)

    print("astream: ", end="")
    async for chunk in llm.astream("Name three primary colors."):
        print(chunk.content, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 4. Tool calling
# ---------------------------------------------------------------------------
@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current temperature for a location."""
    return f"It is 22 degrees {unit} and sunny in {location}."


def tool_calling() -> None:
    section("4. bind_tools")
    llm_with_tools = llm.bind_tools([get_weather])

    ai_msg = llm_with_tools.invoke("What's the weather in Paris?")
    print("tool_calls:", ai_msg.tool_calls)

    # Feed the tool result back so the model can produce a final answer.
    history = [HumanMessage("What's the weather in Paris?"), ai_msg]
    for call in ai_msg.tool_calls:
        history.append(get_weather.invoke(call))
    print("final:", llm_with_tools.invoke(history).content)


# ---------------------------------------------------------------------------
# 5. Structured output
# ---------------------------------------------------------------------------
class Event(BaseModel):
    """A calendar event extracted from free text."""

    name: str = Field(description="Short name of the event")
    date: str = Field(description="Date of the event, ISO 8601 if possible")
    participants: list[str] = Field(description="People attending")


def structured_output() -> None:
    section("5. with_structured_output")
    structured_llm = llm.with_structured_output(Event)
    event = structured_llm.invoke("Alice and Bob go to a science fair on Friday.")
    print(event)


# ---------------------------------------------------------------------------
# 6. LCEL chain with a prompt template
# ---------------------------------------------------------------------------
def chain() -> None:
    section("6. prompt | llm chain")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Translate the user text to {language}. Reply with the translation only."),
            ("human", "{text}"),
        ]
    )
    result = (prompt | llm).invoke({"language": "French", "text": "Good morning"})
    print(result.content)


# ---------------------------------------------------------------------------
# 7. ChatLiteLLMRouter - load balancing and fallbacks
# ---------------------------------------------------------------------------
# The router owns credentials; several deployments can share one model_name and
# LiteLLM load balances between them. `fallbacks` kick in when a group fails.
def router_chat() -> None:
    section("7. ChatLiteLLMRouter")
    model_list = [
        {
            "model_name": "fast",
            "litellm_params": {
                "model": "litellm_proxy/gpt-5-nano",
                "api_key": API_KEY,
                "api_base": BASE_URL,
            },
        },
        {
            "model_name": "fast",
            "litellm_params": {
                "model": "litellm_proxy/gpt-5-mini",
                "api_key": API_KEY,
                "api_base": BASE_URL,
            },
        },
        {
            "model_name": "smart",
            "litellm_params": {
                "model": "litellm_proxy/gpt-4.1",
                "api_key": API_KEY,
                "api_base": BASE_URL,
            },
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",
        fallbacks=[{"fast": ["smart"]}],
        num_retries=2,
    )

    router_llm = ChatLiteLLMRouter(router=router, model_name="fast", temperature=0.1)
    response = router_llm.invoke("Give me a one-line fun fact about octopuses.")
    print("fast:", response.content)
    print("served by:", response.response_metadata.get("model_name"))

    # Switch model groups on an existing instance.
    router_llm.set_default_model("smart")
    print("smart:", router_llm.invoke("Give me a one-line fun fact about squids.").content)


# ---------------------------------------------------------------------------
# 8. LiteLLMEmbeddings
# ---------------------------------------------------------------------------
# All provider config is explicit - no environment variables required.
#   OpenAI:  LiteLLMEmbeddings(model="openai/text-embedding-3-large", api_key=...)
#   Cohere:  LiteLLMEmbeddings(model="cohere/embed-english-v3.0", api_key=...,
#                              document_input_type="search_document",
#                              query_input_type="search_query")
#   Azure:   LiteLLMEmbeddings(model="azure/my-deployment", api_key=..., api_base=...,
#                              api_version="2024-02-01")
#   Bedrock: LiteLLMEmbeddings(model="bedrock/amazon.titan-embed-text-v1")
def embeddings() -> None:
    section("8. LiteLLMEmbeddings")
    emb = LiteLLMEmbeddings(
        model=f"litellm_proxy/{EMBED_MODEL}",
        api_key=API_KEY,
        api_base=BASE_URL,
        dimensions=256,
    )
    vectors = emb.embed_documents(["hello", "world"])
    print("docs:", len(vectors), "x", len(vectors[0]))
    print("query:", len(emb.embed_query("hello")))


async def async_embeddings() -> None:
    section("8b. LiteLLMEmbeddings async")
    emb = LiteLLMEmbeddings(
        model=f"litellm_proxy/{EMBED_MODEL}",
        api_key=API_KEY,
        api_base=BASE_URL,
    )
    print("query:", len(await emb.aembed_query("hello")))


# ---------------------------------------------------------------------------
# 9. LiteLLMEmbeddingsRouter - load balance across embedding deployments
# ---------------------------------------------------------------------------
def router_embeddings() -> None:
    section("9. LiteLLMEmbeddingsRouter")
    router = Router(
        model_list=[
            {
                "model_name": EMBED_MODEL,
                "litellm_params": {
                    "model": f"litellm_proxy/{EMBED_MODEL}",
                    "api_key": API_KEY,
                    "api_base": BASE_URL,
                },
            },
            {
                "model_name": EMBED_MODEL,
                "litellm_params": {
                    "model": f"litellm_proxy/{EMBED_MODEL}",
                    "api_key": API_KEY,
                    "api_base": BASE_URL,
                },
            },
        ]
    )
    emb = LiteLLMEmbeddingsRouter(router=router, model=EMBED_MODEL)
    print("query:", len(emb.embed_query("hello")))


def run(label: str, fn) -> None:
    try:
        asyncio.run(fn()) if asyncio.iscoroutinefunction(fn) else fn()
    except Exception as e:  # keep the rest of the snippets running
        print(f"{label} FAILED: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    run("invoke", basic_invoke)
    run("stream", streaming)
    run("async", async_calls)
    run("tools", tool_calling)
    run("structured", structured_output)
    run("chain", chain)
    run("router", router_chat)
    run("embeddings", embeddings)
    run("embeddings-async", async_embeddings)
    run("embeddings-router", router_embeddings)
