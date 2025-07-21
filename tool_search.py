#when query was out of context
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
import logging

# Optional: Tavily
# from langchain_community.tools.tavily_search import TavilySearchResults

# Setup tools
wikipedia_tool = WikipediaQueryRun(api_wrapper=None)
arxiv_tool = ArxivQueryRun()
# tavily_tool = TavilySearchResults()

def search_with_tools(query: str, min_len: int = 50) -> dict:
    """
    Search using external tools when the query is out-of-context.

    Args:
        query: The user's query.
        min_len: Minimum length of a valid result.

    Returns:
        dict: {"tool_used": tool name, "result": content}
    """
    logging.info(f"🧠 Searching external tools for query: {query}")
    
    try:
        arxiv_result = arxiv_tool.run(query).strip()
        if len(arxiv_result) > min_len:
            return {"tool_used": "arxiv", "result": arxiv_result}
    except Exception as e:
        logging.warning(f"❌ Arxiv tool failed: {e}")
    
    try:
        wiki_result = wikipedia_tool.run(query).strip()
        if len(wiki_result) > min_len:
            return {"tool_used": "wikipedia", "result": wiki_result}
    except Exception as e:
        logging.warning(f"❌ Wikipedia tool failed: {e}")
    
    # Optional Tavily
    # try:
    #     tavily_result = tavily_tool.run(query).strip()
    #     if len(tavily_result) > min_len:
    #         return {"tool_used": "tavily", "result": tavily_result}
    # except Exception as e:
    #     logging.warning(f"❌ Tavily tool failed: {e}")
    
    return {"tool_used": "none", "result": "❌ No relevant data found using external tools."}
