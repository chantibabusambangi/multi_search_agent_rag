from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
import logging

# ✅ Proper initialization
wiki_api = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)

# ✅ Arxiv tool works without arguments
arxiv_tool = ArxivQueryRun()

def search_with_tools(query: str, min_len: int = 50) -> dict:
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
    
    return {"tool_used": "none", "result": "❌ No relevant data found using external tools."}
