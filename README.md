🚀**Multi-Search Agent RAG System using Groq + LangChain for blazing-fast,accurate retrieval-augmented generation across diverse sources**.
link:https://multisearchagentragcbs.streamlit.app/

got 110+ users😍
<img width="1885" height="1055" alt="image" src="https://github.com/user-attachments/assets/a076a66a-fa56-45e1-bd1a-074d71742ca6" />
there may some triggers
✅ 1. the app or environment was rebuilt(manually or due to auto-rebuild)
When that happened, Streamlit Cloud reinstalled all packages from scratch using your requirements.txt.

If anything in your repo changed — or **even if a dependency upstream was updated **— it can trigger new behaviors, even with the same code.
******************************about the project***************************
**Q1.why RAG should be used?**
ans:
**Limitations on LLMs:**
    1.every llm was trained with data that has **knowledge cutoff date**.
    2.Inability to Access Proprietary Data :every company have their **proprietary data**(files,code,documentations etc).
    3.context window(max no.of tokens it can hold) is limited.
        context window(Everything the model considers for its answer,
        the system prompt, the chat history, and any external context—must fit within this fixed limit (e.g., 8k, 100k, or more tokens))
    4.hallunication and factual grounding.
    LLMs are predictive engines; when they don't know an answer, they often generate hallunication.
    5.data privacy(asking questions to chatgpt on your company's documentation) can lead data privacy issues.
**how RAG overcame these:**
    1.RAG have the complete access to data you given to it. RAG is independent of the model's knowledge cutoff.
    2.rag has acccess to private docs,past docs or present the current docs any kind of docs, it have the access.
    3.limited context window :RAG's Solution: Selective Retrieval and Chunking 🎯:
      RAG solves this by using semantic search(on vector store FAISS) to retrieve only the **most relevant "snippets/chunks"** of information, 
      making the large external corpus of data manageable for the LLM.
    4.Cures Hallucinations with Factual Grounding.
      RAG retrieves relevant chunks of text and passes them to the LLM as explicit context.
      and The LLM is **instructed to answer only based** on this retrieved text.
    5.it deont modify the public LLM's weights. The documents are only used for retrieval, thus ensuring data privacy.
  q2.explain about your project?
  
