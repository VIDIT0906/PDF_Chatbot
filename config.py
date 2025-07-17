import os

# Directory Paths
PDF_DIR = os.path.join(os.getcwd(), "data")
CHUNKS_DIR = os.path.join(os.getcwd(), "chunks")
VECTOR_DB_DIR = os.path.join(os.getcwd(), "vectordb")

# Model Names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # or "mistralai/Mistral-7B-Instruct-v0.2"

# Chunking
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

# Retrieval
TOP_K_RETRIEVAL = 5
