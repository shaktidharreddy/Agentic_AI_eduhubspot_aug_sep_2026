from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite", temperature = 0.7)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a specialized Medical Data Extraction Assistant for MedPulse Analytics.
        Your task is to parse unstructured clinical narratives into a precise JSON format.

        ### Extraction Rules:
        1. **Patient Info**: Identify name, age (as integer), and gender.
        2. **Encounter**: Extract the date of service, the provider name, and a list of reported symptoms.
        3. **Clinical Data**: Identify the primary diagnosis and its corresponding ICD-10 code.
        4. **Medications**: For every medication, extract the name, dosage, and frequency into a list.
        5. **Billing**: Capture the specific Insurance Claim ID and all CPT billing codes mentioned.

        ### Target JSON Schema:
        {{
          "patient_analytics": {{
            "patient_info": {{
              "name": "string",
              "age": "integer",
              "gender": "string"
            }},
            "encounter_details": {{
              "date": "string",
              "provider": "string",
              "symptoms": ["string"]
            }},
            "clinical_data": {{
              "diagnosis": "string",
              "icd_10_code": "string",
              "medications": [
                {{
                  "name": "string",
                  "dosage": "string",
                  "frequency": "string"
                }}
              ]
            }},
            "billing_claims": {{
              "claim_id": "string",
              "cpt_codes": ["string"]
            }}
          }}
        }}

        Return ONLY valid JSON. Do not include any conversational text."""
    ),
    ("human", "Please extract the data from this clinical paragraph: {input_text}")
])

para = input("Give your clinical paragraph: ")
final_prompt = prompt.invoke({"input_text": para})
response = model.invoke(final_prompt)

# chain = prompt | llm
# response = chain.invoke({"input_text": para})


print(response.content)


parser = JsonOutputParser()
print("DEBUG: Parsed JSON: ", parser.parse(response.content))
print("-"*100)

# parser = StrOutputParser()
# print("DEBUG: Parsed String: ", parser.parse(response.content))
# print("-"*100)

