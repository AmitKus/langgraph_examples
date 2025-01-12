from functools import lru_cache
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from langgraph.graph.message import MessagesState
from simple_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode

@lru_cache(maxsize=4)
def _get_llm(provider="groq", model_name="llama-3.3-70B-specdec"):
    """Get LLM based on provider.
    
    Args:
        provider: LLM provider, either "groq" or "sambanova"
        model_name: Base model name to use (default: "llama-3.3-70B-specdec")
        
    Returns:
        LLM instance
    """
    if provider == "sambanova":
        model = ChatSambaNovaCloud(
            model=model_name,
            max_tokens=10000,
            temperature=0.01,
        )
    elif provider == "groq":
        model = ChatGroq(
            model=model_name,
            max_tokens=8192,
            temperature=0.01,
        )
    elif provider == "together":
        model = ChatTogether(
            model=model_name,
            max_tokens=None,
            temperature=0,
            max_retries=1
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    model = model.bind_tools(tools)
    return model


def call_llm_with_tool(state, config):
    messages = state["messages"]
    vendor_name = config.get('configurable', {}).get("model_name", "groq")
    model_name = config.get('configurable', {}).get("model_name", "llama-3.3-70B-specdec")
    llm = _get_llm(vendor_name, model_name)
    response = llm.invoke(messages)
    return {"messages":[response]}

tool_node = ToolNode(tools)