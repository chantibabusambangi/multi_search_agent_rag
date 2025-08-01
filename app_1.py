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


# Step 1: Importing All Required Libraries for Multi-Search Agent RAG System
from tool_search import search_with_tools #tool search

# Frontend

# Environment management

import os
import time

# LangChain - LLM via Groq
from langchain_groq import ChatGroq

# LangChain - Local, open-source embeddings
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain - Document loaders for URLs, PDFs, TXT/MD
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader
)

# LangChain - Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# LangChain - Chains and prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# (Optional: Later when you build agent extensions)
# from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper


#step 2
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_groq import ChatGroq

#llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")

# Step 3: Hugging Face Embeddings Setup
# Initialize Hugging Face Embeddings with a recommended retrieval-optimized model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",  # You can replace with another HF model if desired
)

print("✅ Hugging Face Embeddings initialized successfully!")
# ======================
# ⚡ Multi-Search Agent RAG System - Step 4 (Corrected)
# ======================

# Create Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant. Use only the information in the <context> to answer the question. 
    If the answer is not explicitly stated, respond with "I don't know".

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Streamlit UI
st.title("🔍 Multi-Search Agent RAG System (Groq + LangChain)")

st.sidebar.header("📥 Ingest Your Data")
# ✅ Allow users to provide multiple data types
# st.sidebar.markdown("### Upload any combination of data sources:")


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
            if(docs):
                all_docs.extend(docs)
            else:
                st.warning(f"⚠️ Empty or invalid text file: {file.name}")
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
    
    from langchain_community.vectorstores import FAISS
    st.session_state.vector_store = FAISS.from_documents(documents,embedding=embeddings) #non-persistant(before 07/2025)

    #from langchain_community.vectorstores import Chroma
    #st.session_state.vector_store = Chroma.from_documents(documents, embedding=embeddings)
    
    # Create the retrieval chain
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    st.session_state.retrieval_chain = create_retrieval_chain(
        retriever=st.session_state.vector_store.as_retriever(),
        combine_docs_chain=document_chain
    )

    st.info("Loading and processing documents...")
    
    st.success("✅ Data ingestion and vector store setup complete! You can now ask questions below.")

st.sidebar.markdown("🔹 **Built with ❤️ by chantibabusambangi@gmail.com**")


# Only allow question input if retrieval_chain is ready

# Initialize session state variables if not already set
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if (
    st.session_state.retrieval_chain is not None
    and st.session_state.vector_store is not None
    and len(st.session_state.vector_store.index_to_docstore_id) > 0
):
    user_query = st.chat_input("Ask your question:")
    if user_query:
        st.chat_message("user").markdown(f"**You:** {user_query}")

       
        with st.spinner("Generating answer..."):
            start_time = time.time()
            response = st.session_state.retrieval_chain.invoke({"input": user_query})
            elapsed = time.time() - start_time
    
    
        
        # Extract answer safely
        answer = response.get("answer") or response.get("output") or ""
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Check if RAG failed to give answer
        if "i don't know" in answer.lower() or not answer.strip():
            st.warning("🛠 Out of context — switching to external tools (Arxiv/Wikipedia)...")
    
            # Call external tools
            tool_result = search_with_tools(user_query)
    
            st.subheader("📡 External Tool Response")
            st.write(tool_result["result"])
            st.caption(f"🔎 Tool used: **{tool_result['tool_used']}**")
            st.chat_message("assistant").markdown(tool_result["result"])
            #st.session_state.messages.append({"role": "assistant","content": tool_result["result"] })
            
        else:
            # RAG gave an answer — show it
            st.subheader("Answer:")
            st.write(answer)
            st.caption(f"⚡ Response generated in {elapsed:.2f} seconds.")
    
            with st.expander("🔍 Full raw response (debug):"):
                st.json(response)
    
            # Optional: show retrieved context
            if "context" in response and response["context"]:
                with st.expander("📄 Show retrieved context chunks"):
                    for doc in response["context"]:
                        st.write(doc.page_content)
                        st.write("---")
            else:
                st.info("⚠️ No retrieved context available for this query.")
else:
    st.warning("👈 Please ingest your data first using the sidebar before asking questions.")
