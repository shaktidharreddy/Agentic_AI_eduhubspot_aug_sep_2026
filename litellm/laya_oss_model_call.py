"""
Demo: calling the Laya model (convaiinnovations/laya).

Laya is a multilingual, non-autoregressive "System 1" decision model. Given a
state (text/email/ticket/JSON) and a set of typed questions, it returns typed
answers with calibrated probabilities in a single forward pass (no text
generation, no hallucination).

Install first:
    pip install laya
"""

import truststore

# corporate TLS interception: trust the Windows cert store instead of certifi.
# Must run before laya/huggingface_hub build their HTTP clients.
truststore.inject_into_ssl()

from laya import Router  # noqa: E402


def main() -> None:
    router = Router()

    # The "state" is the raw context Laya reasons over.
    state = (
        "Hi, we were billed twice for March. Please refund the duplicate "
        "today or we will cancel our plan."
    )

    # Typed questions Laya should answer about the state.
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
        "churn_risk": {
            "type": "noul",  # yes/no/unsure-style question
            "instructions": "Does the user threaten to cancel or leave?",
        },
    }

    result = router.predict(state, questions)

    print("Department:", result["answers"]["department"]["choice"])
    print("Churn risk:", result["answers"]["churn_risk"]["noul"])
    print("Routed model:", result["routing"]["model"])

    # Laya auto-detects language/script and routes non-English text to the
    # multilingual checkpoint.
    # Free the English weights first: holding both checkpoints resident exhausts
    # the Windows commit limit ("paging file is too small").
    router.unload()

    multilingual_examples = [
        "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।",
        "La aplicación se cierra cada vez que abro la configuración.",
    ]
    for text in multilingual_examples:
        r = router.predict(text, {"department": questions["department"]})
        print(r["routing"]["model"], "->", r["answers"]["department"]["choice"])


if __name__ == "__main__":
    main()
