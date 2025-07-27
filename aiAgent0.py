from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from groq import Groq
client = Groq()

load_dotenv()

class AgentState(TypedDict):
    message: List[HumanMessage]

llm = ChatOpenAI(model="gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    """
    Process the state by sending the message to the LLM and updating the state.
    """
    response = llm.invoke(state["message"])
    print(f"LLM Response: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

user_input = input("Enter your message: ")

result = app.invoke({"message": [HumanMessage(content=user_input)]})

