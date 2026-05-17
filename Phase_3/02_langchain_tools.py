import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY")
)

@tool
def calculate(expression:str) -> str:
    """" Evaluates a mathematical expression. Use this for any math calculation."""
    return str(eval(expression))

@tool
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

agent = create_react_agent(
    model = llm,
    tools=[calculate, get_weather]
)
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    response = agent.invoke({"messages": [("user", user_input)]})
    print(f"Agent: {response['messages'][-1].content}\n")