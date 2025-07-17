import os
import logging
from typing import Iterator, List, Tuple

import numpy as np

from sentence_transformers import SentenceTransformer

from . import document_processor
from . import vector_db_manager
from . import llm_interface
import config

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) Pipeline.
    """
    def __init__(self, embedding_model_name, llm_model_name, db_path, top_k_retrieval):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm_model, self.tokenizer = llm_interface.load_llm(llm_model_name)
        self.db_client = vector_db_manager.initialize_chroma_db(db_path)
        self.collection = self.db_client.get_or_create_collection("rag_documents")
        self.top_k = top_k_retrieval

    def ingest_documents(self, pdf_dir: str):
        """Processes and adds documents in pdf_dir to vector DB."""
        chunks, embeddings, metadata = document_processor.process_documents(pdf_dir, config.CHUNKS_DIR)
        vector_db_manager.add_documents_to_db(self.collection, chunks, embeddings, metadata)

    def answer_query(self, user_query: str) -> Tuple[Iterator[str], List[str]]:
        """Performs retrieval and generation, returns streaming response and source contexts."""
        # Embed query
        query_embedding = self.embedding_model.encode([user_query], normalize_embeddings=True)
        # Retrieve top-K relevant chunks
        hits = vector_db_manager.query_db(self.collection, query_embedding, self.top_k)
        top_chunks = [hit["document"] for hit in hits]
        # Create prompt from retrieved chunks and user query
        prompt = llm_interface.create_prompt_template(top_chunks, user_query)
        # Streaming generation
        response_stream = llm_interface.generate_response_stream(self.llm_model, self.tokenizer, prompt)
        return response_stream, top_chunks
    
    def save_collection(self):
        # No explicit save needed; collections are persisted automatically.
        logging.info("ChromaDB is persisted on disk.")

    def clear_and_rebuild_collection(self, pdf_dir, chunks_dir):
        """Clear the collection and rebuild from saved chunks/embeddings or fresh ingest."""
        self.db_client.delete_collection("rag_documents")
        self.collection = self.db_client.get_or_create_collection("rag_documents")
        # Try to reload from chunks/embeddings:
        chunks, embeddings, metadata = document_processor.load_chunks_and_embeddings(chunks_dir)
        if chunks is not None and embeddings is not None and metadata is not None:
            vector_db_manager.add_documents_to_db(self.collection, chunks, embeddings, metadata)
            logging.info("Vector DB rebuilt from saved chunks/embeddings.")
        else:
            logging.warning("No saved chunks found; trigger fresh ingest.")
            self.ingest_documents(pdf_dir)

