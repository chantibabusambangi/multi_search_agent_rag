import pandas as pd
from datetime import datetime
import uuid
import os
import streamlit as st

# ======================
# ⚡ User Count Tracking
# ======================
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
user_id = st.session_state.user_id

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

st.sidebar.markdown(f"👥 *Total Visitors:* {df['user_id'].nunique()}")


# Step 1: Importing All Required Libraries for Multi-Search Agent RAG System

# Frontend
import streamlit as st

# Environment management
from dotenv import load_dotenv
import os
import time

# LangChain - LLM via Groq
from langchain_groq import ChatGroq

# LangChain - Document loaders for URLs, PDFs, TXT/MD
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader
)

# LangChain - Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain - Vector DB (Chroma)
from langchain_community.vectorstores import FAISS
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

print(llm,"done")

import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceHubEmbeddings

# Load Hugging Face API key from Streamlit secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Use Hugging Face Inference API to get embeddings
huggingface_embeddings = HuggingFaceHubEmbeddings(
    repo_id="BAAI/bge-small-en-v1.5",  # or use "sentence-transformers/all-MiniLM-L6-v2" for faster response
    model_kwargs={"task": "text-embedding"}
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

data_source = st.sidebar.radio("Select data sources:", ["URL", "PDF", "Text File", "CSV File"])

uploaded_file = None
input_url = None

if data_source == "URL":
    input_url = st.sidebar.text_input("Enter URL to ingest:")
elif data_source in ["PDF", "Text File"]:
    uploaded_file = st.sidebar.file_uploader(f"Upload your {data_source} file", type=["pdf", "txt", "md"])
elif data_source == "CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Initialize session state holders
if "messages" not in st.session_state:
    st.session_state.messages = []
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
    elif data_source == "CSV File" and uploaded_file is not None:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        csv_text = df.to_string(index=False)  # convert DataFrame to plain text
        with open("temp_uploaded_file.csv.txt", "w", encoding="utf-8") as f:
            f.write(csv_text)
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader("temp_uploaded_file.csv.txt")

    else:
        st.error("⚠ Please provide a valid input for the selected data source.")
        st.stop()

    st.info("Loading and processing documents...")

    docs = loader.load()
    if not docs:
        st.error("❌ Failed to load documents. Please check your file content.")
        st.stop()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(docs)

    from langchain_community.vectorstores import FAISS

    st.session_state.vector_store = FAISS.from_documents(
        documents,
        embedding=huggingface_embeddings
    )

    st.success("✅ Data ingestion and vector store setup complete! You can now ask questions below.")

    # Create retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    st.session_state.retrieval_chain = create_retrieval_chain(
        retriever=st.session_state.vector_store.as_retriever(),
        combine_docs_chain=document_chain
    )

st.sidebar.markdown("🔹 *Built with 💓 by chantibabusambangi@gmail.com*")

# Only allow question input if retrieval_chain is ready
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
        with st.spinner("Generating answer..."):
            start_time = time.time()
            response = st.session_state.retrieval_chain.invoke({"input": user_query})
            elapsed = time.time() - start_time

        st.subheader("Answer:")
        st.write(response.get('answer') or response.get('output') or response or "⚠ No answer returned.")

        st.caption(f"⚡ Response generated in {elapsed:.2f} seconds.")

        with st.expander("🔍 Full raw response (debug):"):
            st.json(response)

        if "context" in response and response["context"]:
            with st.expander("📄 Show retrieved context chunks"):
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("---")
        else:
            st.info("⚠ No retrieved context available for this query.")
else:
    st.warning("👈 Please ingest your data first using the sidebar before asking questions.")
