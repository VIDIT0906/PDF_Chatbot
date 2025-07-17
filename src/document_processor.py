import os
import logging
import numpy as np
import json
from typing import List, Tuple
from tqdm import tqdm

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

def load_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error reading PDF ({pdf_path}): {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean up headers/footers/extra whitespace/etc."""
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([^\S\r\n]{2,})', ' ', text)  # multi-whitespace
    return text.strip()

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping, sentence-aware chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", "!", "?", " "])
    chunks = splitter.split_text(text)
    return chunks

def generate_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """Generate embeddings for a list of text chunks."""
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        raise
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def process_documents(pdf_dir: str, chunks_dir: str) -> Tuple[List[str], np.ndarray]:
    """Process all PDFs in a directory, save chunk and embedding files."""
    all_chunks = []
    metadata = []
    logging.info(f"Processing documents in {pdf_dir}")
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, fname)
            text = clean_text(load_pdf(pdf_path))
            chunks = chunk_text(text, chunk_size=250, chunk_overlap=50)
            all_chunks.extend(chunks)
            metadata.extend([{"source": fname}] * len(chunks))
    if not all_chunks:
        logging.warning(f"No PDF chunks found in {pdf_dir}.")
    embeddings = generate_embeddings(all_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Save artifacts
    os.makedirs(chunks_dir, exist_ok=True)
    with open(os.path.join(chunks_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)
    np.save(os.path.join(chunks_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(chunks_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return all_chunks, embeddings, metadata

def load_chunks_and_embeddings(chunks_dir: str):
    """Load saved chunks, embeddings, and metadata from disk."""
    try:
        with open(os.path.join(chunks_dir, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(os.path.join(chunks_dir, "embeddings.npy"))
        with open(os.path.join(chunks_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return chunks, embeddings, metadata
    except Exception as e:
        logging.error(f"Failed to load chunks/embeddings: {e}")
        return None, None, None
