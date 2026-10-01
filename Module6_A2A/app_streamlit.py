"""A small Streamlit UI over the running demo.

Two halves:

  Top     what the registry holds right now, and how it got there
  Bottom  ask a question, and watch the supervisor's three steps

Run the servers first (`python run_all.py`), then:

    streamlit run app_streamlit.py
"""

import httpx
import pandas as pd
import streamlit as st

from config import REGISTRY_URL, SUPERVISOR_URL, TIME_AGENT_URL

EXAMPLES = [
    "What is the weather in London?",
    "Convert 100 USD to INR",
    # These two share no words with the right agent's card, so the registry's
    # keyword score is 0 for everyone and its "best match" is really just the
    # first one it happens to hold. Jev gets both right anyway, and on the
    # umbrella question it overrides the registry outright.
    "2000 yen in rupees?",
    "Should I take an umbrella in Tokyo?",
    # Same city as the umbrella question, different agent.
    "What time is it in Tokyo?",
]

st.set_page_config(page_title="A2A agent registry demo", layout="wide")
st.title("Agent-to-agent communication across two frameworks")
st.caption(
    "Three agents -- one LangChain, one Google ADK, one plain Python -- register "
    "themselves with a shared registry. A LangGraph supervisor discovers them from "
    "their agent cards and calls one over Google's A2A protocol. It imports none of "
    "the three."
)


def get(url: str, **params):
    """GET some JSON, or show why we could not."""
    try:
        response = httpx.get(url, params=params or None, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        st.error(f"Could not reach {url} -- is `python run_all.py` running?\n\n{error}")
        return None


# --------------------------------------------------------------------------
# 1. The registry
# --------------------------------------------------------------------------
st.header("1. The agent registry")

# Handled before the table is fetched below, so a newly registered agent shows
# up in the same run rather than after a second refresh.
st.subheader("Register an agent by URL")
st.caption(
    "Anything that serves `/.well-known/agent-card.json` can join, wherever it is "
    "hosted. You send only the URL — the registry fetches the card from the agent "
    "itself, so a registered card can never be stale or a lie."
)

with st.form("register", clear_on_submit=False):
    new_url = st.text_input(
        "Agent base URL",
        value=TIME_AGENT_URL,
        help="The Time Agent is running but never announces itself. Register it here.",
    )
    submitted = st.form_submit_button("Register", type="primary")

if submitted and new_url.strip():
    try:
        registered = httpx.post(
            f"{REGISTRY_URL}/register", json={"base_url": new_url.strip()}, timeout=30
        )
    except Exception as error:
        st.error(f"Could not reach the registry.\n\n{error}")
    else:
        if registered.status_code == 200:
            body = registered.json()
            st.success(
                f"Registered **{body['registered']}** (built with {body['framework']}) — "
                f"skills: {', '.join(body['skills'])}. It is now routable."
            )
        else:
            st.error(registered.json().get("detail", registered.text))

if st.button("Refresh", type="secondary"):
    st.rerun()

status = get(f"{REGISTRY_URL}/status")
cards = get(f"{REGISTRY_URL}/agents")

if status is not None and cards is not None:
    if not status:
        st.warning(
            "Nothing registered yet. The agents announce themselves a second after "
            "they boot, and Google ADK takes ~20s to import -- wait, then Refresh."
        )
    else:
        st.write(
            f"**{len(status)} agents registered.** The weather and currency agents put "
            "themselves here: each POSTs its own URL on startup and again every 15 "
            "seconds. Watch the **Heartbeats** column to tell the two routes apart — it "
            "climbs on its own for a self-registering agent, and stays put for one that "
            "was registered by hand above."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Agent": row["name"],
                        "Built with": row["framework"],
                        "Address": row["url"],
                        "Skill": ", ".join(row["skills"]),
                        "Heartbeats": row["heartbeats"],
                        "Last seen": f"{row['seconds_since_last_seen']:.0f}s ago",
                    }
                    for row in status
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        st.subheader("The agent cards themselves")
        st.caption(
            "This is the JSON each agent serves at /.well-known/agent-card.json -- the "
            "same well-known path a hosted LangGraph Platform deployment uses."
        )
        for card in cards:
            with st.expander(f"{card['name']}  ·  built with {card['framework']}"):
                st.json(card)

# --------------------------------------------------------------------------
# 2. Capability discovery
# --------------------------------------------------------------------------
st.header("2. Ask a question")
st.caption(
    "The registry shortlists by capability, Jev decides which agent wins, and the "
    "supervisor calls it over A2A."
)

if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]

columns = st.columns(len(EXAMPLES))
for column, example in zip(columns, EXAMPLES):
    if column.button(example, width="stretch"):
        st.session_state.question = example

question = st.text_input("Question", key="question")

if st.button("Send to supervisor", type="primary"):
    with st.spinner("Supervisor working..."):
        try:
            response = httpx.post(
                f"{SUPERVISOR_URL}/ask", json={"question": question}, timeout=300
            )
        except Exception as error:
            st.error(f"Could not reach the supervisor: {error}")
            st.stop()

    if response.status_code != 200:
        st.error(f"Supervisor returned {response.status_code}: {response.text}")
        st.stop()

    result = response.json()

    st.success(f"**{result['answer']}**")
    st.caption(f"answered by {result['routed_to']}")

    st.subheader("What the supervisor did")
    labels = {
        "discover": "Step 1 · discover — who claims this capability?",
        "decide": "Step 2 · decide — Jev picks one",
        "call_agent": "Step 3 · call_agent — talk A2A to the winner",
    }

    for entry in result["trace"]:
        st.markdown(f"**{labels.get(entry['step'], entry['step'])}**")
        st.write(entry["summary"])
        data = entry["data"]

        if entry["step"] == "discover":
            st.dataframe(
                pd.DataFrame(data["ranked"]).rename(
                    columns={
                        "name": "Agent",
                        "framework": "Built with",
                        "keyword_score": "Keyword score",
                        "matched_words": "Matched on",
                        "best_match": "Registry's best match",
                        "url": "Address",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        elif entry["step"] == "decide":
            left, right = st.columns([2, 1])
            left.bar_chart(pd.Series(data["probabilities"], name="probability"))
            right.metric("Confidence", f"{data['confidence']:.0%}")
            right.metric("Cost of this decision", f"${data['cost_usd']:.6f}")
            right.metric(
                "Registry's keyword pick",
                data["registry_best_match"],
                delta="Jev agrees" if data["agreed_with_registry"] else "Jev overrode it",
                delta_color="normal" if data["agreed_with_registry"] else "inverse",
            )
            right.caption(f"model: {data['model']}")
            with st.expander("The options Jev was given (built from the cards above)"):
                st.json(data["options_offered"])

        elif entry["step"] == "call_agent":
            st.caption(f"This agent was built with {data['framework']}.")
            left, right = st.columns(2)
            left.markdown("A2A request — `message/send`")
            left.json(data["a2a_request"])
            right.markdown("A2A response")
            right.json(data["a2a_response"])

        st.divider()
