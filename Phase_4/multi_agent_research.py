import os
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key = os.getenv("GROQ_API_KEY")
)

search_tool = TavilySearch(
    max_results = 5,
    topic = "general",
)
class ResearchState(TypedDict):
    topic : str
    research : str
    report : str
    feedback : str
    status : str
    revisions : int


research_agent = create_react_agent(
    tools = [search_tool],
    model = llm,
    prompt = SystemMessage(content="You are a researcher. Search thoroughly")
        
    )




def planner_node(state : ResearchState):
    print(f"Planning researh for: {state['topic']}")
    return state

def researcher_node(state : ResearchState):
    response = research_agent.invoke({'messages': [('user', f'Research this: {state["topic"]}')]})
    return {"research": response["messages"][-1].content}

def writer_node(state: ResearchState):
    response = llm.invoke([
        SystemMessage(content="""You are a professional report writer.
        Write a structured report with:
        - Introduction
        - Key Findings
        - Current Developments  
        - Summary
        If feedback is provided, improve the report based on it."""),
        HumanMessage(content=f"Research: {state['research']}\nFeedback: {state['feedback']}")
    ])
    return {
        "report": response.content,
        "revisions": state["revisions"] + 1
    }

def critic_node(state: ResearchState):
    response = llm.invoke([
        SystemMessage(content="""You are a strict report reviewer.
            Review the report for quality, completeness and accuracy.
            If it is good enough end with APPROVED.
            If it needs improvement end with REJECTED and give specific feedback."""),
        HumanMessage(content=f"Review this report: {state['report']}")
    ])
    # detect status from response
    if "APPROVED" in response.content:
        status = "approved"
    else:
        status = "rejected"
    return {"feedback": response.content, "status": status}



graph = StateGraph(ResearchState)

graph.add_node('planner', planner_node)
graph.add_node('researcher', researcher_node ) 
graph.add_node('writer', writer_node)  
graph.add_node('critic', critic_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "critic")

def decide_next(state: ResearchState):
    if state["status"] == "approved":
        return "end"
    elif state["revisions"] >= 2:
        return "end"
    else:
        return "writer"


graph.add_conditional_edges(
    "critic",
    decide_next,
    {"writer": "writer", "end": END}
)


app = graph.compile()

result = app.invoke({
    "topic": "Agentic AI",
    "research": "",
    "report": "",
    "feedback": "",
    "status": "",
    "revisions": 0
})

print(result["report"])