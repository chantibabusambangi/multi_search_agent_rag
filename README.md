🚀**Multi-Search Agent RAG System using Groq + LangChain for blazing-fast,accurate retrieval-augmented generation across diverse sources**.
link:https://multisearchagentragcbs.streamlit.app/

got 110+ users😍
<img width="1885" height="1055" alt="image" src="https://github.com/user-attachments/assets/a076a66a-fa56-45e1-bd1a-074d71742ca6" />
******************************about the project***************************\
RAG (Retrieval-Augmented Generation) is an architecture used to optimize the output of a Large Language Model (LLM)\
**RAG**: Retrieval Augumented Generation.\
**Retrieval**: retrieving relevant information from database/vector store. retreieving top k(4) chunks from database.\

**Augumented**: augumenting/enhancing the query by adding the context of data gotten from database.\ 

here, we augumented the user_query: 1.to formulated to a standalone question to handle follow-ups safely with injuction of chat_history.

#Contextualize question prompt - reformulates question based on chat history\
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


2.augumented the system prompt, injucted the relevant chunks(top k=4) along with enhanced_query and  dev_prompt(you are an ai assistent).

qa_system_prompt = """You are an AI assistant. First check the conversation history for relevant information.
Then use the information in the <context> to answer the question.

Important:
- Remember personal information shared by the user in previous messages (like name, location, preferences)
- If the question is personal or conversational (greetings, personal facts), answer from chat history
- If the question requires document knowledge, use the <context>
- If the answer is not in history or context, respond with "I don't know"

<context>
{context}
</context>"""\

**so basically augumentation means enhancing query or prompt by adding specific prompts & context or history, to get concise,accurate response.**\

**Generation**: LLM generates the concise answer.\
**Q1.why RAG should be used?**\
ans:
**Limitations on LLMs:**\
    1.every llm was trained with data that has **knowledge cutoff date**.\
    2.Inability to Access Proprietary Data :every company have their **proprietary data**(files,code,documentations etc).\
    3.context window(max no.of tokens it can hold) is limited.
        context window(Everything the model considers for its answer,
        the system prompt, the chat history, and any external context—must fit within this fixed limit (e.g., 8k, 100k, or more tokens)).\
    4.hallunication and factual grounding.
    LLMs are predictive engines; when they don't know an answer, they often generate hallunication.\
    5.data privacy(asking questions to chatgpt on your company's documentation) can lead data privacy issues.\
**how RAG overcame these:**\
    1.RAG have the complete access to data you given to it. RAG is independent of the model's knowledge cutoff.\
    2.rag has acccess to private docs,past docs or present the current docs any kind of docs, it have the access.\
    3.limited context window :RAG's Solution: Selective Retrieval and Chunking 🎯:
      RAG solves this by using semantic search(on vector store FAISS) to retrieve only the **most relevant "snippets/chunks"** of information, 
      making the large external corpus of data manageable for the LLM.\
    4.Cures Hallucinations with Factual Grounding.
      RAG retrieves relevant chunks of text and passes them to the LLM as explicit context.
      and The LLM is **instructed to answer only based** on this retrieved text.\
    5.it deont modify the public LLM's weights. The documents are only used for retrieval, thus ensuring data privacy.\
  
  **Q2.explain about your project?**
  firstly i will explain about this rag system,
  step1 : **data injection**\
  1.data loading, : code checks what kind of data it is. like url(webbaseloader),csv file(pandas → string → TextLoader),text file(Textloader) or pdf(pymupdf).
      All text documents are appended to all_docs as LangChain Document object.\
  2.b) Parsing / Cleaning
    Empty documents (doc.page_content.strip() == "") are filtered out.So after this step → you have a clean list of textual documents.\
  3.Text Splitting (Chunking):Your text is very large (maybe several pages long).
    Large chunks cannot be efficiently embedded or retrieved, so they’re broken down.
    Each document is split into pieces (chunks) of ~1000 characters each.
    Each chunk overlaps with the previous one by 200 characters for context continuity (to avoid splitting important sentences).\
  4. Embeddings (Turning text → vectors)
      Now we convert each text chunk into a numerical vector (embedding).these vectors capture semantic meaning of each chunk.
      embeddings  we used = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
      we do semantic similarity search on these vectors later.\
  5.Vector Database (Vector Store — FAISS):
      All embeddings/vectors + corresponding chunks are stored inside a FAISS index:
      It allows you to search for top-k most similar chunks given a query embedding.\
  6.Retriever (Finding relevant chunks):
  it get the query string as input.
  Retriever: Uses the vector database (like FAISS) to perform similarity search on the user query’s embedding and returns the top-k (default 4) most relevant chunks.\
**generation step:**\
  7.Stuff Documents Chain for Answer generation:
      question_answer_chain = create_stuff_documents_chain(llm, qa_prompt).
      It stuffs the retrieved chunks into the LLM prompt (along with your QA instructions).
        The prompt looks like: pmompt+retreved chunks + query.
    and the answer from llm is the final answer to the user.\
8.Retrieval Chain: Combine retriever + generation chain → full RAG pipeline.\
**challenges in the project:**
1.handling the follow up questiobns.
1.Initially, the system hanlded the followup questions ,it treated each query independently.
Problem: Without session memory, the model didn’t remember previous context, so follow-up questions like “What about its applications?” failed.
Solution:
I integrated session memory **create_history_aware_retriever provided by langchain**, into the RAG pipeline.
Now, the model stores previous user–assistant exchanges and passes them as part of the context for the next query — 
enabling smooth contextual understanding and follow-up answers.

-------------------------------------------------
from langchain.chains import create_history_aware_retriever  
history_aware_retriever = create_history_aware_retriever(     
    llm, retriever, contextualize_q_prompt                    
)                                                            
-------------------------------------------------

It creates a retriever that understands conversation history.
Normally, a retriever only sees the latest question, not the earlier chat.So it gets confused by follow-ups like:

create_history_aware_retriever() makes the retriever “history-aware.”
**It automatically rewrites every question using the chat history before retrieving.
**
ex.User: What is Artificial Intelligence?
User: What about its applications?
So, the LLM changes
👉 “What about its applications?”
into
👉 “What are the applications of Artificial Intelligence?”

This rewriting happens in the first stage, before the document retrieval:
1. Query Contextualization (Rewriting)
The History-Aware Retriever component uses the LLM (Groq) to reformulate the user's ambiguous follow-up question (like "What are its main features?") into a standalone, context-independent query (like "What are the main features of the LangChain RAG system discussed earlier?").
This rewritten query is then used to search the FAISS vector store to ensure accurate document retrieval, as the vector store cannot process the conversational history itself.
who rewrites the followup questions?
**llm** in the llm prompt we told/instructed to him that
**"Given a chat history and the latest user question... formulate a standalone question which can be understood without the chat history."\

where chat history stored:  st.session_state.chat_history in streamlit session state, given to llm to rewrite the query.


2.Limited Knowledge of Local Documents
The model couldn’t answer queries outside the uploaded dataset.
Solution:
**I integrated external tools like Wikipedia and ArXiv** .
If relevant information wasn’t found in FAISS store, it could fall back to these external knowledge sources — giving more complete and updated responses.
making it reliable.
✅ Final Summary:

“The main challenges were handling follow-up questions and extending knowledge beyond local documents.
I solved the first using LangChain’s create_history_aware_retriever() to make the retriever context-aware — it rewrites follow-up questions based on chat history.
For the second, I integrated external tools like Wikipedia and ArXiv as fallback sources, so the system stays accurate and complete even beyond local data.”
