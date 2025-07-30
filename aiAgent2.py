from typing import List, TypedDict, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()


class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]


def processNode(state: AgentState) -> AgentState:
    """ This process node is responsible for sending the message to the LLM and updating the state. while maintating the conversation history. """

    llm = init_chat_model(model="gemma2-9b-it", model_provider="groq")

    response = llm.invoke(state["message"])

    print(f"AI: {response.content}")

    state["message"].append(AIMessage(content=response.content))

    return state


graph = StateGraph(AgentState)

graph.add_node("processNode", processNode)
graph.add_edge(START, "processNode")
graph.add_edge("processNode", END)

app = graph.compile()

conversation_history = []

print("Welcome to the AI Agent! Type 'exit' to quit.")

user_input = input("Enter: ")


while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"message": conversation_history})
    conversation_history = result["message"]
    user_input = input("Enter: ")
