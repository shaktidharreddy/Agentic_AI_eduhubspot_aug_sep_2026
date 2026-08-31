import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conversation_memory = []

def chat_with_memory(user_input):
    conversation_memory.append(
        {"role": "user", "content": user_input}
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_memory
    )

    assistant_reply = response.choices[0].message.content

    conversation_memory.append(
        {"role": "assistant", "content": assistant_reply}
    )

    return assistant_reply

reply_1 = chat_with_memory(
    "My name is Anupam and I work as a Senior Research Analyst."
)

print(reply_1)

reply_2 = chat_with_memory(
    "What is my profession?"
)
print(reply_2)

print("\n--- Conversation Memory ---")
for message in conversation_memory:
    print(message)
