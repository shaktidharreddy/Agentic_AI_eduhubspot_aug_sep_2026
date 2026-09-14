# first example using pydantic

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

# 📝 Define the exact schema blueprint
class ProductDetails(BaseModel):
    item: str
    price: int

# Fixed: Using the dedicated '.parse()' endpoint to handle Pydantic objects properly
completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "I bought a vintage leather jacket for 150 dollars."}
    ],
    # 🛠️ Activates Structured Outputs via response_format
    response_format=ProductDetails
)

# 🔄 Automatically parsed; no json.loads needed
parsed_data = completion.choices[0].message.parsed

print(parsed_data.item)   # Output: vintage leather jacket
print(parsed_data.price)  # Output: 150


# second example

import json
from openai import OpenAI

client = OpenAI()

# Fixed: Filled out cut-off property strings, types, and booleans
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "product_schema",
        "strict": True,  # Required for Structured Outputs
        "schema": {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "price": {"type": "integer"}
            },
            "required": ["item", "price"],
            "additionalProperties": False
        }
    }
}

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "I bought a vintage leather jacket for 150 dollars."}
    ],
    response_format=response_format
)

# Raw string output must be manually parsed into a python dictionary
raw_json_string = completion.choices[0].message.content
parsed_data = json.loads(raw_json_string)

print(parsed_data["item"])   # Output: vintage leather jacket
print(parsed_data["price"])  # Output: 150



# for responses api counterpart

# response = client.responses.create(
#   model="gpt-4o",
#   input="I bought a vintage leather jacket for 150 dollars.",
#   text={
#      "format": response_format # Reuses the fixed dictionary from Version 2
#   }
#)
#print(response.output_text)


# third example

# This is the shared raw JSON Schema map we will pass to both APIs
event_schema = {
    "name": "calendar_event",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "date": {"type": "string"},
            "location": {"type": "string"}
        },
        "required": ["name", "date", "location"],
        "additionalProperties": False
    }
}

from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Alice is attending a tech conference in Seattle on Friday."}
    ],
    # Configuration container is at the top-level
    response_format={
        "type": "json_schema",
        "json_schema": event_schema
    }
)

# Parsed via standard choices routing
print(response.choices[0].message.content)



# fourth example

import json
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant. You must always respond with a valid JSON object."
        },
        {
            "role": "user", 
            "content": "List 3 common programming languages and their primary use case."
        }
    ],
    # Enforces that the model's output is valid JSON syntax
    response_format={"type": "json_object"}
)

# Extract the raw JSON string
raw_json_string = response.choices.message.content
print("Raw Output:\n", raw_json_string)

# Parse it into a standard Python dictionary
parsed_data = json.loads(raw_json_string)
print("\nParsed Dictionary:", parsed_data)

