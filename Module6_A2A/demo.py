"""Walk through the three ideas end to end: registry, discovery, invocation.

Run this once the servers are up (see run_all.py). Step 1 shows both ways an
agent gets into the registry: the weather and currency agents announce
themselves, and the time agent -- which stands in for an agent hosted
elsewhere -- gets registered here by URL.
"""

import asyncio
import json

import httpx

from config import REGISTRY_URL, SUPERVISOR_URL, TIME_AGENT_URL

QUESTIONS = [
    "What is the weather in London?",    # should land on the LangChain agent
    "Convert 100 USD to INR",            # should land on the Google ADK agent
    "What time is it in Tokyo?",         # should land on the plain-Python agent
]


def banner(step: str, title: str) -> None:
    print(f"\n{'=' * 70}\n{step}. {title}\n{'=' * 70}")


async def show_registry(client: httpx.AsyncClient) -> None:
    for row in (await client.get(f"{REGISTRY_URL}/status")).json():
        print(
            f"{row['name']:<16} built with {row['framework']:<14} at {row['url']:<24}"
            f" ({row['heartbeats']} heartbeats)"
        )


async def main() -> None:
    async with httpx.AsyncClient(timeout=300) as client:

        banner("1a", "SELF-REGISTRATION -- these agents announced themselves")
        await show_registry(client)

        banner("1b", "REGISTRATION BY URL -- for an agent hosted elsewhere")
        print(f"POST {REGISTRY_URL}/register  {{'base_url': '{TIME_AGENT_URL}'}}")
        registered = await client.post(
            f"{REGISTRY_URL}/register", json={"base_url": TIME_AGENT_URL}
        )
        print(f"  -> {registered.json()}")
        print("\nNo skills or endpoints were sent; the registry fetched the card itself.\n")
        await show_registry(client)

        banner("2", "DISCOVERY -- cards come from the agents, ranked by capability")
        for question in QUESTIONS:
            ranked = (
                await client.get(f"{REGISTRY_URL}/discover", params={"capability": question})
            ).json()
            print(f"{question!r}")
            for row in ranked:
                print(
                    f"  score {row['score']}  {row['card']['name']:<16}"
                    f" matched on {row['matched'] or '-'}"
                )

        print("\nOne full card, exactly as its agent serves it:")
        cards = (await client.get(f"{REGISTRY_URL}/agents")).json()
        print(json.dumps(cards[0], indent=2))

        banner("3", "INVOCATION -- Jev chooses, then the supervisor calls over A2A")
        for question in QUESTIONS:
            response = await client.post(f"{SUPERVISOR_URL}/ask", json={"question": question})
            if response.status_code != 200:
                print(f"\nQ: {question}\n   FAILED ({response.status_code}): {response.text}")
                continue
            result = response.json()

            print(f"\nQ: {result['question']}")
            for entry in result["trace"]:
                print(f"   {entry['step']:<11} {entry['summary']}")
                if entry["step"] == "decide":
                    print(f"   {'':<11} probabilities: {entry['data']['probabilities']}")
            print(f"   answer      {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
