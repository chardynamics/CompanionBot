import yaml

def parse_response_to_dict(response_text):
    try:
        # yaml.safe_load is great for parsing this structure
        data = yaml.safe_load(response_text)
        return data
    except yaml.YAMLError as e:
        print(f"Error parsing response: {e}")
        return None

def execute_command(command_data):
    command = command_data.get("command")
    args = command_data.get("arguments", {})

    if command == "book_flight":
        print(f"Executing book_flight function with args: {args}")
        # Call your actual Python function here:
        # book_flight(destination=args.get('destination'), date=args.get('date'))
    elif command == "none":
        print("No specific command identified. Responding naturally.")
        # Handle the case where the model didn't find a command
    else:
        print(f"Unknown command: {command}")

# --- Main execution flow ---
user_request = "I want to book a flight to NYC for May 20th."
api_response = get_command_from_gemini(user_request)

parsed_command = parse_response_to_dict(api_response)

if parsed_command:
    execute_command(parsed_command)