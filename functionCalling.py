import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# --- Your actual Python functions ---
def get_weather(city: str):
    return f"It's sunny and 72°F in {city}."

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    }
]

def model_return(text):
    if not text:
        print("[ERROR] No text to send to model")
        return None
    messages.append({
        "role": "user",
        "content": text
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
        "tools": tools,
    }
    req = requests.post(url, headers=headers, json=data, timeout=30)
    result = req.json()

    # Check for function call in the response
    if "tool_calls" in result:
        for tool_call in result["tool_calls"]:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            if func_name in TOOL_MAPPING:
                func_result = TOOL_MAPPING[func_name](**arguments)
                print(f"Function result: {func_result}")
                return func_result
    return result

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city for which to get the weather"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

TOOL_MAPPING = {
    "get_weather": get_weather
}

print(model_return("What's the weather in Austin?"))   # calls get_weather("Austin")