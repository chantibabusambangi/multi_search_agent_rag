from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

from langchain.agents import initialize_agent, AgentType, Tool
from langchain_groq import ChatGroq
import os
import logging

logging.basicConfig(level=logging.INFO)

# ✅ Tool setup
wiki_api = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)
arxiv_tool = ArxivQueryRun()

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="Useful for answering general knowledge or factual questions using Wikipedia."
    ),
    Tool(
        name="Arxiv",
        func=arxiv_tool.run,
        description="Useful for retrieving and summarizing scientific papers from Arxiv."
    )
]

# ✅ Groq LLM setup
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="gemma2-9b-it"
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ✅ Query function
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
