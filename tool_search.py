from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
import logging

logging.basicConfig(level=logging.INFO)

# ✅ Tool setup
wiki_api = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)
arxiv_tool = ArxivQueryRun()

tools = [wikipedia_tool, arxiv_tool]

# ✅ Groq LLM setup
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

# ✅ Create a conversational ReAct prompt
template = """You are a helpful assistant with access to Wikipedia and Arxiv tools.

Previous conversation context:
{chat_history}

Answer the following question using the tools if needed. Be conversational and remember the context above.

Question: {input}

You have access to the following tools:

{tools}

Use the following format:

Thought: Think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

def search_with_tools(query: str, chat_history: list = None) -> dict:
    """
    Search using external tools with conversational context.
    
    Args:
        query: The user's question
        chat_history: List of LangChain message objects (HumanMessage, AIMessage) or None
    
    Returns:
        dict with 'tool_used' and 'result' keys
    """
    logging.info(f"🔎 Agent searching tools for query: {query}")
    
    try:
        # Format chat history for the prompt
        formatted_history = ""
        if chat_history and len(chat_history) > 0:
            # Take last 6 messages (3 exchanges)
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            
            for msg in recent_history:
                try:
                    # Handle both LangChain message objects and dict format
                    if hasattr(msg, 'type') and hasattr(msg, 'content'):
                        # LangChain message object
                        role = "User" if msg.type == "human" else "Assistant"
                        formatted_history += f"{role}: {msg.content}\n"
                    elif isinstance(msg, dict):
                        # Dict format from session state
                        role = "User" if msg.get("role") == "user" else "Assistant"
                        formatted_history += f"{role}: {msg.get('content', '')}\n"
                except Exception as e:
                    logging.warning(f"Could not format message: {e}")
                    continue
        
        if not formatted_history:
            formatted_history = "No previous conversation."
        
        # Create agent
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        # Run agent with chat history context
        result = agent_executor.invoke({
            "input": query,
            "chat_history": formatted_history
        })
        
        # Extract the answer
        answer = result.get("output", "No answer generated.")
        
        return {
            "tool_used": "agent (Wikipedia/Arxiv)",
            "result": answer
        }
        
    except Exception as e:
        logging.error(f"❌ Agent tool search failed: {e}", exc_info=True)
        
        # Fallback: Try direct Wikipedia search
        try:
            logging.info("Attempting fallback to direct Wikipedia search...")
            wiki_result = wikipedia_tool.run(query)
            return {
                "tool_used": "Wikipedia (fallback)",
                "result": wiki_result
            }
        except:
            return {
                "tool_used": "none",
                "result": f"❌ I couldn't find an answer using external tools. Please try rephrasing your question."
            }
