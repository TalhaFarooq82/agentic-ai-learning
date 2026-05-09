import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

search_tool = types.Tool(
    google_search= types.GoogleSearch()
)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        tools=[search_tool],
        system_instruction="You are a helpful research assistant. Use web search to find current, accurate information."
    )
)

while True:
    user_input = input("You:")
    if user_input.lower() == "quit":
        break
    response = chat.send_message(user_input)
    print(f"Gemini: {response.text}")   
