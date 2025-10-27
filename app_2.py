import pandas as pd
from datetime import datetime
import uuid
import os
import streamlit as st

# ======================
# ⚡ User Count Tracking
# ======================
user_id = str(uuid.uuid4())
visits_file = "user_visits.csv"

if not os.path.exists(visits_file):
    df = pd.DataFrame(columns=["user_id", "visit_time"])
    df.to_csv(visits_file, index=False)

df = pd.read_csv(visits_file)

if "counted" not in st.session_state:
    if user_id not in df["user_id"].values:
        new_visit = pd.DataFrame([[user_id, datetime.now()]], columns=["user_id", "visit_time"])
        new_visit.to_csv(visits_file, mode="a", header=False, index=False)
    st.session_state.counted = True

st.sidebar.markdown(f"👥 **Total Visitors:** {df['user_id'].nunique()}")

# ======================
# Imports
# ======================
from tool_search import search_with_tools
import time
from dotenv import load_dotenv

# LangChain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

# ======================
# Setup
# ======================
load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

print("✅ Hugging Face Embeddings initialized successfully!")

# ======================
# ⚡ Conversational Prompt with Chat History
# ======================

# Contextualize question prompt - reformulates question based on chat history
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Answer prompt with context and history
qa_system_prompt = """You are an AI assistant. Use only the information in the <context> to answer the question.
If the answer is not explicitly stated, respond with "I don't know".

<context>
{context}
</context>"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ======================
# Streamlit UI
# ======================
st.title("🔍 Multi-Search Agent RAG System (Groq + LangChain)")

st.sidebar.header("📥 Ingest Your Data")

input_url = st.sidebar.text_input("🌐 Or enter a URL:")
pdf_files = st.sidebar.file_uploader("📄 Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
text_files = st.sidebar.file_uploader("📝 Upload Text/Markdown file(s)", type=["txt", "md"], accept_multiple_files=True)
csv_files = st.sidebar.file_uploader("📊 Upload CSV file(s)", type=["csv"], accept_multiple_files=True)

if st.sidebar.button("Ingest Data"):
    all_docs = []

    # 🔹 Load from URL
    if input_url:
        try:
            st.info("📡 Loading from URL...")
            loader = WebBaseLoader(input_url)
            docs = loader.load()
            docs = [doc for doc in docs if doc.page_content.strip()]
            if docs:
                all_docs.extend(docs)
            else:
                st.warning(f"⚠️ Empty or invalid URL")
        except Exception as e:
            st.error("❌ Failed to load URL.")
            st.exception(e)

    # 🔹 Load PDFs
    if pdf_files:
        for file in pdf_files:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                docs = [doc for doc in docs if doc.page_content.strip()]
                if docs:
                    all_docs.extend(docs)
                else:
                    st.warning(f"⚠️ No text found in {file.name}")
            except Exception as e:
                st.error(f"❌ Error loading PDF: {file.name}")
                st.exception(e)
            finally:
                os.remove(temp_path)

    # 🔹 Load Text files
    if text_files:
        for file in text_files:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())
            try:
                loader = TextLoader(temp_path)
                docs = loader.load()
                docs = [doc for doc in docs if doc.page_content.strip()]
                if docs:
                    all_docs.extend(docs)
                else:
                    st.warning(f"⚠️ No text found in {file.name}")
            except Exception as e:
                st.error(f"❌ Error loading text file: {file.name}")
                st.exception(e)
            finally:
                os.remove(temp_path)

    # 🔹 Load CSV files
    if csv_files:
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if df.empty:
                    st.warning(f"⚠️ Empty CSV: {file.name}")
                    continue
                text = df.to_string(index=False)
                temp_txt = f"temp_{file.name}.txt"
                with open(temp_txt, "w", encoding="utf-8") as f:
                    f.write(text)
                loader = TextLoader(temp_txt)
                docs = loader.load()
                if docs:
                    all_docs.extend(docs)
                else:
                    st.warning(f"⚠️ Could not extract from CSV: {file.name}")
                os.remove(temp_txt)
            except Exception as e:
                st.error(f"❌ Failed to process CSV: {file.name}")
                st.exception(e)

    if not all_docs:
        st.error("⚠️ No documents loaded. Please upload or enter a valid input.")
        st.stop()
    
    st.info("🧠 Processing documents...")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(all_docs)
    
    # Create vector store
    st.session_state.vector_store = FAISS.from_documents(documents, embedding=embeddings)
    
    # Create history-aware retriever
    retriever = st.session_state.vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Create document chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create RAG chain with history
    st.session_state.retrieval_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )
    
    # Initialize chat history if needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.success("✅ Data ingestion and vector store setup complete! You can now ask questions below.")

st.sidebar.markdown("🔹 **Built with ❤️ by chantibabusambangi@gmail.com**")

# ======================
# Initialize Session State
# ======================
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add Clear History button in sidebar
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# ======================
# Display Chat History
# ======================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================
# Chat Interface
# ======================
if (
    st.session_state.retrieval_chain is not None
    and st.session_state.vector_store is not None
    and len(st.session_state.vector_store.index_to_docstore_id) > 0
):
    user_query = st.chat_input("Ask your question:")
    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.spinner("Generating answer..."):
            start_time = time.time()
            
            # Pass chat history to the chain
            response = st.session_state.retrieval_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            
            elapsed = time.time() - start_time
        
        # Extract answer
        answer = response.get("answer") or response.get("output") or ""
        
        # Check if RAG failed
        if "i don't know" in answer.lower() or not answer.strip():
            st.warning("🛠 Out of context — switching to external tools (Arxiv/Wikipedia)...")
            
            # Call external tools with history
            tool_result = search_with_tools(user_query, st.session_state.chat_history)
            
            with st.chat_message("assistant"):
                st.markdown(tool_result["result"])
                st.caption(f"🔎 Tool used: **{tool_result['tool_used']}**")
            
            st.session_state.messages.append({"role": "assistant", "content": tool_result["result"]})
            
            # Update chat history
            st.session_state.chat_history.extend([
                HumanMessage(content=user_query),
                AIMessage(content=tool_result["result"])
            ])
            
        else:
            # RAG gave an answer
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"⚡ Response generated in {elapsed:.2f} seconds.")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Update chat history
            st.session_state.chat_history.extend([
                HumanMessage(content=user_query),
                AIMessage(content=answer)
            ])
            
            with st.expander("🔍 Full raw response (debug):"):
                st.json(response)
            
            # Show retrieved context
            if "context" in response and response["context"]:
                with st.expander("📄 Show retrieved context chunks"):
                    for doc in response["context"]:
                        st.write(doc.page_content)
                        st.write("---")
else:
    st.warning("👈 Please ingest your data first using the sidebar before asking questions.")
