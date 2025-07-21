from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq  # ✅ Groq client
import os
import logging

# =============================
# ✅ Set up logging (if needed)
# =============================
logging.basicConfig(level=logging.INFO)

# =============================
# ✅ Set up external tools
# =============================

wiki_api = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)
arxiv_tool = ArxivQueryRun()

tools = [wikipedia_tool, arxiv_tool]

# =============================
# ✅ Set up Groq LLM Agent
# =============================

# 👇 Use a different model than the one used in app.py to avoid session confusion (if needed)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="gemma2-9b-it"  # ✅ Use a **different model** than llama3-70b-8192
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# =============================
# ✅ Function to use agent
# =============================

def search_with_tools(query: str) -> dict:
    logging.info(f"🔎 Agent searching tools for query: {query}")
    try:
        result = agent.run(query)
        return {
            "tool_used": "agent (LLM decided)",
            "result": result
        }
    except Exception as e:
        logging.error(f"❌ Agent tool search failed: {e}")
        
        return {
            "tool_used": "none",
            "result": "❌ Agent failed to answer the query with external tools."
        }
