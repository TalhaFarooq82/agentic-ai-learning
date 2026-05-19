import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY")
)
search_tool = TavilySearch(
    max_results = 3,
    topic='general'
)
agent = create_react_agent(
    model = llm,
    tools = [search_tool],
    prompt = SystemMessage("You are a research assistant. Use web search to find current accurate information on any topic.")
)

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    response = agent.invoke({"messages": [("user", user_input)]})
    # Correct
    last_message = response['messages'][-1]
    if isinstance(last_message.content, list):
        for block in last_message.content:
            if isinstance(block, dict) and block.get('type') == 'text':
                print(f"Agent: {block['text']}\n")
    else:
        print(f"Agent: {last_message.content}\n")