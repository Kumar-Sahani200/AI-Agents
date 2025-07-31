from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

# defining the state for the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define tools for the agent
@tool
def add_nums(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def subtract_nums(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

@tool
def multiply_nums(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def divide_nums(a: int, b: int) -> int:
    """Divide two numbers."""
    return a / b

tools = [add_nums, subtract_nums, multiply_nums, divide_nums]

# initialize the chat model
llm = init_chat_model(model="gemma2-9b-it", model_provider="groq").bind_tools(tools)

# creating the process node
def model_calls(state: AgentState) -> AgentState:
    """Process the message using with or without the tools and returns the updated state"""
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Conditional Node to check if the agent should continue or exit
def should_continue(state: AgentState):
    message = state["messages"]
    last_message = message[-1]

    if not last_message.tool_calls:
        return "exit"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("agent", model_calls)

tool_node = ToolNode(tools=tools)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges("agent", should_continue, {
    "exit": END, 
    "continue": "tool"
})

graph.add_edge("tool", "agent")


app = graph.compile()

def print_stream(steam):
    for s in steam:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

input = {"messages": [('user', "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(input, stream_mode="values"))

