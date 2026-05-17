import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

response = llm.invoke("What is agentic ai in 3 lines?")
print(response)




messages = [
    SystemMessage(content="You are a teacher who explains everything in simple Urdu."),
    HumanMessage(content="What is Agentic AI?")
]

response = llm.invoke(messages)
print(response)