import os, httpx, json
from dotenv import load_dotenv

load_dotenv()

BASE = "http://internal-cep-ore-private-non-prod-ai-106882125.us-west-2.elb.amazonaws.com:4000"
KEY  = os.environ["LITELLM_API_KEY"]
H    = {"Authorization": f"Bearer {KEY}"}

with httpx.Client(base_url=BASE, headers=H, timeout=30) as c:
    print("--- models this key can call ---")
    for m in c.get("/v1/models").json()["data"]:
        print(" ", m["id"])

    print("\n--- capabilities ---")
    r = c.get("/model/info")
    if r.is_success:
        for m in r.json()["data"]:
            info = m.get("model_info") or {}
            print(f"{m['model_name']:<28} "
                  f"tools={info.get('supports_function_calling')} "
                  f"schema={info.get('supports_response_schema')} "
                  f"vision={info.get('supports_vision')} "
                  f"tool_choice={info.get('supports_tool_choice')}")
            print("   params:", info.get("supported_openai_params"))
    else:
        print(f"unavailable ({r.status_code}): {r.text}")

    print("\n--- key scope ---")
    r = c.get("/key/info")
    print(json.dumps(r.json(), indent=2) if r.is_success
          else f"unavailable ({r.status_code}): {r.text}")
