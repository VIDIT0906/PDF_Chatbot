import os
import streamlit as st
import logging

import config
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="PDF Chatbot (RAG)", layout="wide")
logging.basicConfig(level=logging.INFO)

@st.cache_resource(show_spinner="Loading models and database ...")
def load_pipeline():
    return RAGPipeline(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        llm_model_name=config.LLM_MODEL_NAME,
        db_path=config.VECTOR_DB_DIR,
        top_k_retrieval=config.TOP_K_RETRIEVAL,
    )

pipeline = load_pipeline()

st.sidebar.title("Settings")
st.sidebar.write(f"**LLM Model:** {config.LLM_MODEL_NAME}")
st.sidebar.write(f"**Embedding Model:** {config.EMBEDDING_MODEL_NAME}")
try:
    num_chunks = len(pipeline.collection.get()['documents'])
except Exception:
    num_chunks = 0
st.sidebar.write(f"**Chunks Indexed:** {num_chunks}")

if st.sidebar.button("Clear Chat", key="clear_chat"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

if st.sidebar.button("Ingest PDFs Now", key="ingest_pdfs"):
    with st.spinner("Ingesting PDF documents ..."):
        try:
            pipeline.ingest_documents(config.PDF_DIR)
            st.success("PDFs ingested and indexed.")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

if st.sidebar.button("Rebuild VectorDB From Saved Chunks"):
    with st.spinner("Rebuilding vector database..."):
        try:
            pipeline.clear_and_rebuild_collection(config.PDF_DIR, config.CHUNKS_DIR)
            st.success("Vector DB rebuilt from saved chunks/embeddings.")
        except Exception as e:
            st.error(f"Error: {e}")

# Chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("📄 PDF Chatbot — Retrieval-Augmented Generation")

user_input = st.text_input("Ask a question about the PDF(s):", key="user_input", value="", max_chars=800)
ask_button = st.button("Ask", key="ask_btn")

if ask_button and user_input.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            response_stream, source_chunks = pipeline.answer_query(user_input)
            answer_str = ""
            answer_display = st.empty()
            for partial in response_stream:
                answer_display.markdown(partial)
                answer_str = partial
            st.session_state["history"].append({"question": user_input, "answer": answer_str, "chunks": source_chunks})
        except Exception as e:
            st.error(f"Error: {e}")

for idx, turn in enumerate(st.session_state["history"][::-1]):
    st.markdown(f"**You:** {turn['question']}")
    st.markdown(f"**Bot:** {turn['answer']}")
    with st.expander("Source Chunks Used", expanded=False):
        for i, src_txt in enumerate(turn["chunks"]):
            st.markdown(f"- *Context [{i+1}]:* `{src_txt[:220].replace('`','')}`...")

st.info(f"To re-index PDF(s), upload to the `data/` folder then click 'Ingest PDFs Now'.")
