import os
import logging
from typing import List, Dict
import numpy as np

import chromadb
from chromadb.config import Settings

def initialize_chroma_db(db_path: str) -> chromadb.Client:
    """Initialize or load a ChromaDB client."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        logging.info(f"Initialized ChromaDB with path {db_path}")
    except Exception as e:
        logging.error(f"ChromaDB initialization error: {e}")
        raise
    return client

def add_documents_to_db(collection, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict] = None):
    """Add text chunks and their embeddings to a ChromaDB collection."""
    try:
        ids = [f"chunk-{i}" for i in range(len(texts))]
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas or [{} for _ in texts],
        )
        logging.info(f"Added {len(texts)} documents to ChromaDB collection.")
    except Exception as e:
        logging.error(f"Error adding to ChromaDB: {e}")
        raise

def query_db(collection, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
    """Query ChromaDB with an embedding and get top-K similar chunks."""
    try:
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        hits = [{"document": doc, "metadata": md, "distance": dist}
                for doc, md, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]
        return hits
    except Exception as e:
        logging.error(f"ChromaDB query error: {e}")
        return []
