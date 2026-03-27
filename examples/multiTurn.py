
import requests
from dotenv import load_dotenv

load_dotenv()
import os

messages = [
    {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "Always respond in concise, short messages as you are a companion robot."
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
    print(result)
    messages.append({
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": result["output"][0]["content"][0]["text"],
            }
        ],
    })
    try:
        return result["output"][0]["content"][0]["text"]
    except (KeyError, IndexError) as e:
        print(f"[ERROR] Unexpected response structure: {e}")
        return None

def main():
    while True:
        text = input("You: ")
        if text.lower() in {"exit", "quit"}:
            break
        response = model_return(text)
        print(f"Assistant: {response}")
        
if __name__ == "__main__":
    main()