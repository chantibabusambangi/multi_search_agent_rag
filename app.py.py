# Step 1: Importing All Required Libraries for Multi-Search Agent RAG System

# Frontend
import streamlit as st

# Environment management
from dotenv import load_dotenv
import os
import time

# LangChain - LLM via Groq
from langchain_groq import ChatGroq

# LangChain - Local, open-source embeddings
from langchain.embeddings import HuggingFaceEmbeddings

# LangChain - Document loaders for URLs, PDFs, TXT/MD
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader
)

# LangChain - Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain - Vector DB (Chroma)
from langchain_community.vectorstores import Chroma

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

print(llm,"done")

#step3
# Step 3: Hugging Face Embeddings Setup

from langchain.embeddings import HuggingFaceEmbeddings

# Initialize Hugging Face Embeddings with a recommended retrieval-optimized model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",  # You can replace with another HF model if desired
    cache_folder="/kaggle/working/hf_cache"  # Optional: cache to persist between Kaggle sessions
)

print("✅ Hugging Face Embeddings initialized successfully!")
# ======================
# ⚡ Multi-Search Agent RAG System - Step 4 (Corrected)
# ======================

# Create Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context below only.
    If the answer is not in the context, say "I don't know".

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Streamlit UI
st.title("🔍 Multi-Search Agent RAG System (Groq + LangChain)")

st.sidebar.header("📥 Ingest Your Data")

data_source = st.sidebar.radio("Select data source:", ["URL", "PDF", "Text File"])

uploaded_file = None
input_url = None

if data_source == "URL":
    input_url = st.sidebar.text_input("Enter URL to ingest:")
elif data_source in ["PDF", "Text File"]:
    uploaded_file = st.sidebar.file_uploader(f"Upload your {data_source} file", type=["pdf", "txt", "md"])

# Initialize session state holders
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if st.sidebar.button("Ingest Data"):

    if data_source == "URL" and input_url:
        loader = WebBaseLoader(input_url)
    elif data_source == "PDF" and uploaded_file is not None:
        with open("temp_uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp_uploaded_file.pdf")
    elif data_source == "Text File" and uploaded_file is not None:
        with open("temp_uploaded_file.txt", "wb") as f:
            f.write(uploaded_file.read())
        loader = TextLoader("temp_uploaded_file.txt")
    else:
        st.error("⚠️ Please provide a valid input for the selected data source.")
        st.stop()

    st.info("Loading and processing documents...")

    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(docs)

    # Store in Chroma
    st.session_state.vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        collection_name="rag_multi_search_dynamic"
    )

    st.success("✅ Data ingestion and vector store setup complete! You can now ask questions below.")

    # Create retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    st.session_state.retrieval_chain = create_retrieval_chain(
        retriever=st.session_state.vector_store.as_retriever(),
        combine_docs_chain=document_chain
    )

# Only allow question input if retrieval_chain is ready
if st.session_state.retrieval_chain is not None:
    user_query = st.text_input("Ask your question:")

    if user_query:
        with st.spinner("Generating answer..."):
            start_time = time.time()
            response = st.session_state.retrieval_chain.invoke({"input": user_query})
            elapsed = time.time() - start_time

        st.subheader("Answer:")
        st.write(response.get('answer') or response.get('output') or response or "⚠️ No answer returned.")

        st.caption(f"⚡ Response generated in {elapsed:.2f} seconds.")

        with st.expander("🔍 Full raw response (debug):"):
            st.json(response)

        if "context" in response and response["context"]:
            with st.expander("📄 Show retrieved context chunks"):
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("---")
        else:
            st.info("⚠️ No retrieved context available for this query.")
else:
    st.warning("👈 Please ingest your data first using the sidebar before asking questions.")
