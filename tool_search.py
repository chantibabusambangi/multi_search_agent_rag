from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import os
import logging

logging.basicConfig(level=logging.INFO)

# ✅ Tool setup
wiki_api = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
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
    model_name="llama-3.3-70b-versatile",  # Changed for consistency
    temperature=0.3
)

def search_with_tools(query: str, chat_history: list = None) -> dict:
    """
    Search using external tools with conversational context.
    
    Args:
        query: The user's question
        chat_history: List of LangChain message objects (HumanMessage, AIMessage)
    
    Returns:
        dict with 'tool_used' and 'result' keys
    """
    logging.info(f"🔎 Agent searching tools for query: {query}")
    
    try:
        # Create memory and populate with chat history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Add previous messages to memory
        if chat_history:
            for msg in chat_history[-6:]:  # Last 3 exchanges (6 messages)
                if isinstance(msg, HumanMessage):
                    memory.chat_memory.add_user_message(msg.content)
                elif isinstance(msg, AIMessage):
                    memory.chat_memory.add_ai_message(msg.content)
        
        # Initialize agent with memory
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        result = agent.run(query)
        
        return {
            "tool_used": "agent (Wikipedia/Arxiv)",
            "result": result
        }
        
    except Exception as e:
        logging.error(f"❌ Agent tool search failed: {e}")
        return {
            "tool_used": "none",
            "result": f"❌ Agent failed to answer the query with external tools. Error: {str(e)}"
        }
