import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, NotRequired
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(find_dotenv())

def add(a: int | float, b: int | float) -> int | float:
    """
    Adds two numbers.

    Args:
        a (int, float): The first number.
        b (int, float): The second number.

    Returns:
        int, float: The sum of the two numbers.
    """
    return a + b

def divide(a: int | float, b: int | float) -> int | float:
    """
    Divides two numbers.
    Args:
        a (int, float): The numerator.
        b (int, float): The denominator.
    """
    return a / b

def multiply(a: int | float, b: int | float) -> int | float:
    """
    Multiplies two numbers.

    Args:
        a (int, float): The first number.
        b (int, float): The second number.

    Returns:
        int, float: The product of the two numbers.
    """
    return a * b

def subtract(a: int | float, b: int | float) -> int | float:
    """
    Subtracts two numbers.

    Args:
        a (int, float): The first number.
        b (int, float): The second number.

    Returns:
        int, float: The difference of the two numbers.
    """
    return a - b

# Create a list of the functions
tools = [add, divide, multiply, subtract]

llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools,parallel_tool_calls=False)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    session_id: str
    context: dict[str, str]

sys_msg = SystemMessage(content="You are a helpful assistant that can perform basic math operations. You have access to the following functions: add, subtract, multiply, divide. Use these functions to answer the user's questions.")

def assistant(state: MessagesState):
    return {"messages":[llm_with_tools.invoke([sys_msg] + state["messages"])]}

memory = MemorySaver()

# Build Node Structure
builder = StateGraph(MessagesState)

# Define Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools",ToolNode(tools))

# Define Edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": "Michael"
    }
}
config["configurable"]["thread_id"] = "07035995152"

# print(messages["messages"][-1:])
messages = {"messages": [HumanMessage(content="Hi")]}
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
    messages = messages["messages"] + [HumanMessage(content=user_input)]
    messages = react_graph.invoke({"messages": messages},config=config)
    print("Bountip Assistant:", messages["messages"][-1].content)