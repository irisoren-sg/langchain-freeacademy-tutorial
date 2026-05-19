import json
import os
#from dotenv import load_dotenv
from openai import OpenAI

#load_dotenv()

#client = OpenAI()
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # dummy value
)

# Define the tools our agent can use
def get_weather(city: str) -> str:
    """Simulate getting weather data."""
    weather_data = {
        "Tokyo": {"temp": 18, "condition": "Rainy", "humidity": 85},
        "London": {"temp": 12, "condition": "Cloudy", "humidity": 70},
        "New York": {"temp": 25, "condition": "Sunny", "humidity": 45},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "Unknown", "humidity": "Damp"})
    return json.dumps({"city": city, **data})


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return json.dumps({"expression": expression, "result": result})
        return json.dumps({"error": "Invalid expression"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# Map function names to implementations
available_tools = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# Define tool schemas for the LLM
tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name (e.g., Tokyo, London)"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression (e.g., '2 + 2', '100 * 0.15')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


def run_agent(user_input: str, max_steps: int = 5) -> str:
    """Run the agent loop with ReACT-style reasoning."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Think step by step about what the user needs. "
                "Use tools when you need external data or calculations. "
		"The tools that you have represent the latest real time data and are completely accurate. "
                "When you have enough information, respond directly."
            )
        },
        {"role": "user", "content": user_input}
    ]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        response = client.chat.completions.create(
            model="llama3.1",#"gpt-4",
            messages=messages,
            tools=tool_schemas,
        )

        message = response.choices[0].message
        messages.append(message)

        # If the agent responds without tool calls, we are done
        if not message.tool_calls:
            print(f"Agent: {message.content}")
            return message.content

        # Process each tool call
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"Tool call: {func_name}({func_args})")

            # Execute the tool
            if func_name in available_tools:
                result = available_tools[func_name](**func_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {func_name}"})

            print(f"Result: {result}")

            # Feed the result back to the conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Agent reached maximum steps without completing the task."


def main():
    print("AI Agent Started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        run_agent(user_input)
        print()


if __name__ == "__main__":
    main()
