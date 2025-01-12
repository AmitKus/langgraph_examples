import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.prebuilt import tools_condition
from typing import TypedDict, Literal
from langgraph.graph.message import MessagesState
from simple_agent.utils.nodes import call_llm_with_tool, tool_node

# Define the config
class GraphConfig(TypedDict):
    vendor_name: Literal["sambanova", "groq", "together"]
    model_name: Literal["Meta-Llama-3.1-70B-Instruct", "llama-3.3-70B-specdec", "meta-llama/Llama-3.1-70B-Instruct-Turbo"]

graph_builder = StateGraph(MessagesState, config_schema=GraphConfig)
graph_builder.add_node('llm',call_llm_with_tool)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point('llm')
graph_builder.add_conditional_edges(
    "llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
graph_builder.add_edge("tools", "llm")
graph = graph_builder.compile()

from IPython.display import Image, display

# try:
#     # Generate and save the Mermaid graph image
#     png_data = graph.get_graph().draw_mermaid_png()
#     with open('graph.png', 'wb') as f:
#         f.write(png_data)
#     # Optionally still display it if in a notebook
#     display(Image(png_data))
# except Exception as e:
#     print(f"Failed to generate or save graph: {e}")
#     pass

# from langchain_core.messages import HumanMessage
# messages = [HumanMessage(content="What is 1100+600?")]
# messages = graph.invoke({"messages": messages})
# for m in messages['messages']:
#     m.pretty_print()