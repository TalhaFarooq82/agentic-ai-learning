import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def calculate(expression: str) -> str:
    """Evaluates a math expression and returns the result."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """Gets real current weather for a city using Open-Meteo."""
    
    # Step 1 - convert city name to coordinates
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    geo_response = requests.get(geo_url).json()
    
    if not geo_response.get("results"):
        return f"City '{city}' not found."
    
    lat = geo_response["results"][0]["latitude"]
    lon = geo_response["results"][0]["longitude"]
    
    # Step 2 - get weather using coordinates
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    weather_response = requests.get(weather_url).json()
    
    current = weather_response["current"]
    temp = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]
    wind = current["wind_speed_10m"]
    
    return f"{city}: {temp}°C, Humidity {humidity}%, Wind {wind} km/h"


calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculate",
            description="Evaluates a mathematical expression. Use this for any math calculation.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="The math expression to evaluate e.g. '25 * 4 + 10'"
                    )
                },
                required=["expression"]
            )
        )
    ]
)

weather_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Gets current weather for a city. Use this when user asks about weather.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "city": types.Schema(
                        type=types.Type.STRING,
                        description="The city name e.g. 'Lahore'"
                    )
                },
                required=["city"]
            )
        )
    ]
)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        tools=[calculator_tool,weather_tool],
        system_instruction="You are a helpful assistant. Use the calculator tool for math. Use web search for current information."
    )
)

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    response = chat.send_message(user_input)
    part = response.candidates[0].content.parts[0]

    if part.function_call:
        tool_name = part.function_call.name
        tool_args = part.function_call.args

        if tool_name == "calculate":
            result = calculate(tool_args["expression"])
        elif tool_name == "get_weather":
            result = get_weather(tool_args["city"])

        final = chat.send_message(
            types.Part(
                function_response=types.FunctionResponse(
                    name=tool_name,
                    response={"result": result}
                )
            )
        )
        print(f"Gemini: {final.text}\n")
    else:
        print(f"Gemini: {response.text}\n")       