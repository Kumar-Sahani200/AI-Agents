from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

document_content = "" 

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str):
    """Update a document with new content.
    
    Args:
        content (str): The new content to replace the document with.
    """

    global document_content 
    document_content = content
    return f"Document updated. Current content:\n{document_content}"

@tool
def save(filename: str):
    """Save the current content to a file.
    
    Args:
        filename (str): The name of the file to save the content to.
    """

    global document_content

    if not filename.endswith('.txt'):
        filename += '.txt'

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
            print(f"Content saved to {filename}")
        return f"Content saved to {filename}"
    except Exception as e:
        print(f"Error saving content: {e}")

tools = [update, save]

llm = init_chat_model(model="gemma2-9b-it", model_provider="groq").bind_tools(tools)



def agent(state: AgentState) -> AgentState:
    
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        user_input = "Hi, help me draft a document."
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("AI: What would you like to do with the document?\n")
        print(f"User: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state['messages']) + [user_message]

    response = llm.invoke(all_messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"using tool: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState): 

    if not state["messages"]:
        return "continue"
    
    for message in reversed(state['messages']):
        if (isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower()):
            return "end"
    else: 
        return "continue"



graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {
    "continue": "agent",
    "end": END
})
app = graph.compile()



def print_messages(messages):
    """Helps in printing tool messages"""
    if not messages:
        return
    else:
        for message in messages[-3:]:
            if isinstance(message, ToolMessage):
                print(f'\n Tool Result: {message.content}')


def run_drafter_agent():

    print("\n ==== Drafter === ")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n === End ====")


if __name__ == "__main__":
    run_drafter_agent()
        
        
