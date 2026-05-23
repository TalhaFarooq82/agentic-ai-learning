import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_tavily import TavilySearch
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY")
)

search_tool = TavilySearch(
    max_results = 5,
    topic = "general"
)

agent = create_react_agent(
    model = llm,
    tools = [search_tool],
    prompt = SystemMessage("""
                           You are an expert research assistant. When given a research topic:

                            1. Break it into 3-4 specific search queries
                            2. Search each query separately using the search tool
                            3. Compile all findings into a structured report with:
                            - Introduction (2-3 sentences)
                            - Key Findings (5-7 bullet points)
                            - Current Developments (what's happening now)
                            - Summary (2-3 sentences)

                            Always search multiple times. Never answer from just one search.
                            """)
)

topic = input("Enter research topic: ")

print("\n Researching... this may take a moment...\n")

response = agent.invoke({"messages": [("user", f"Research this topic thoroughly: {topic}")]})

last_message = response['messages'][-1]
if isinstance(last_message.content, list):
    for block in last_message.content:
        if isinstance(block, dict) and block.get('type') == 'text':
            print(block['text'])
else:
    print(last_message.content)