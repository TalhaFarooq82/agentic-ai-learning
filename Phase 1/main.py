import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Exercise 1: System prompt ---
response = client.models.generate_content(
    model="gemini-2.5-flash",
    config={
        "system_instruction": "You are a teacher explaining to a 10 year old.",
    },
    contents="What is Python used for?"
)
print("=== System Prompt Test ===")
print(response.text)

# --- Exercise 2: Multi-turn chat ---
print("\n=== Multi-turn Chat ===")
chat = client.chats.create(model="gemini-2.5-flash")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chat.send_message(user_input)
    print(f"Gemini: {response.text}\n")

# --- Exercise 3: Token usage ---
print("\n=== Token Usage ===")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain what an API is in one sentence."
)
print(response.text)
print(f"Input tokens:  {response.usage_metadata.prompt_token_count}")
print(f"Output tokens: {response.usage_metadata.candidates_token_count}")