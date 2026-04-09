import json, requests
from openai import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

# You can use any model that supports tool calling
MODEL = "x-ai/grok-4.1-fast"

openai_client = OpenAI(
  base_url="https://ai.hackclub.com/proxy/v1/",
  api_key= os.getenv('HACKCLUB_API_KEY'),
)
def move_forward(seconds):
    return f"Robot will move forward for {seconds} seconds."

tools = [
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Move the robot forward for a set number of seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Number of seconds to move forward"
                    }
                },
                "required": ["seconds"]
            }
        }
    }
]

TOOL_MAPPING = {
    "move_forward": move_forward
}
"""
messages = [
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": task,
  }
]
"""

messages = [
  {
    "role": "system",
    "content": "You are a helpful robot assistant."
  }
]

user_input = "Move forward for 5 seconds"

messages.append({
    "role": "user",
    "content": user_input
})

response_1 = openai_client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools
)

message_1 = response_1.choices[0].message

if message_1.tool_calls:
    for tool_call in message_1.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_response = TOOL_MAPPING[tool_name](**tool_args)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_response),
        })

    # Get the final response after tool execution
    response_2 = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    print(response_2.choices[0].message.content)
else:
    print(message_1.content)