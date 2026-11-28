from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "openai:gpt-5.6-luna",
    use_responses_api=True,
    output_version="responses/v1",
    use_previous_response_id=True, # server-side conversation state
)

# What those flags do:

# use_responses_api=True — call /v1/responses instead of /v1/chat/completions. If you omit it, LangChain still switches automatically when you use Responses-only features (built-in tools, previous_response_id, etc.). Set it explicitly so routing is not inferred from the model name.
# output_version="responses/v1" — put reasoning summaries and built-in tool results on AIMessage.content instead of additional_kwargs. In langchain-openai 1.x this is already the default.
# use_previous_response_id=True drops older messages from the payload and continues the thread from the last response ID.


response = model.invoke("You're a sarcastic agent who replies to every question with wit. Question: Why do parrots talk?")
