import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key = os.getenv("GEMINI_API_KEY")
)

search_tool = TavilySearch(
    max_results = 5,
    topic = 'general'
)

memory = MemorySaver()
agent = create_react_agent(
    tools = [search_tool],
    model= llm,
    checkpointer=memory,
    prompt=SystemMessage(content="""
                        You are an expert research assistant. When given a research topic:
                        1. Search for it using 4 different specific queries
                        2. Combine all search results
                        3. Write a structured report with:
                        - Introduction
                        - Key Findings (5-7 bullet points)  
                        - Current Developments
                        - Summary
                        Always search multiple times before writing the report."""
                        )
)

config = {
    'configurable' : {
        'thread_id' : 'user_talha'
    }
}


print("Agent has memory! Try telling it your name, then ask it later.")
print("Type 'quit' to exit\n")

while True:
    topic = input("Enter the Research Topic: ")
    if topic.lower().strip() =='quit':
        break

    response = agent.invoke(
    {
        'messages' : [('user', f'Research this topic thoroughly: {topic}')]
    },
    config=config
    )
    
    answer = response['messages'][-1]

    if isinstance(answer.content, list):
        for block in answer.content:
            if isinstance(block, dict) and block.get('type') == 'text':
                print(f"Agent: {block['text']}\n")  

    else:
        print(f"Agent: {answer.content}")
