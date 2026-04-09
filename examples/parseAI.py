import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# --- Your actual Python functions ---
def get_weather(city: str):
    return f"It's sunny and 72°F in {city}."

def add_numbers(a: int, b: int):
    return a + b

# --- Map of available functions ---
FUNCTIONS = {
    "get_weather": get_weather,
    "add_numbers": add_numbers,
}

# --- Describe tools to the model ---
TOOLS_DESCRIPTION = """
You are a helpful assistant. When the user asks something, respond ONLY with a JSON object (no extra text, no markdown) in this format:
{
  "tool": "<tool_name>",
  "args": { "<arg_name>": <value>, ... }
}

Available tools:
- get_weather(city: str) — gets the weather for a city
- add_numbers(a: int, b: int) — adds two numbers together

If no tool applies, use:
{ "tool": "none", "args": {}, "response": "your plain text answer here" }
"""

messages = [
    {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": TOOLS_DESCRIPTION,
            }
        ],
    }
]

def model_return(text):
    if not text:
        print("[ERROR] No text to send to model")
        return None
    messages.append({
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": text,
            }
        ],
    })

    print("Sending to model...")
    model = "google/gemini-2.5-flash-lite-preview-09-2025"
    url = "https://ai.hackclub.com/proxy/v1/responses"
    headers = {
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    data = {
        "model": model,
        "input": messages,
    }
    req = requests.post(url, headers=headers, json=data, timeout=30)
    result = req.json()

    # Extract text from the /responses endpoint format
    raw = result["output"][0]["content"][0]["text"]

    # Append assistant reply to history
    messages.append({
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": raw}],
    })

    # Strip markdown fences in case the model wraps JSON in ```
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    parsed = json.loads(raw)

    if parsed["tool"] == "none":
        return parsed["response"]

    # Look up and call the real function
    tool_name = parsed["tool"]
    args = parsed["args"]

    if tool_name not in FUNCTIONS:
        return f"Unknown tool: {tool_name}"

    result = FUNCTIONS[tool_name](**args)
    return str(result)


print(model_return("What's the weather in Austin?"))   # calls get_weather("Austin")
print(model_return("What is 42 plus 7?"))              # calls add_numbers(42, 7)
print(model_return("What is the capital of France?"))  # no tool, plain response