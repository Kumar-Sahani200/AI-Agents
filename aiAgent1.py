from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# State Schema
class AgentState(TypedDict):
    message: str

# Node
def process(state: AgentState) -> AgentState:
    """
    Process the state by sending the message to the LLM and updating the state.
    """

    client = Groq()
    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {
                "role": "user",
                "content": state['message']
            }
        ]
    )
    print(f"AI: {completion.choices[0].message.content}")
    return state

# Graph Definition
graph = StateGraph(AgentState)

# Add nodes and edges to the graph
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

print("Welcome to the AI Agent! Type 'exit' to quit.")

user_input = input("Enter: ")
app.invoke({"message": user_input})

while user_input.lower() != "exit":
    user_input = input("Enter: ")
    app.invoke({"message": user_input})


