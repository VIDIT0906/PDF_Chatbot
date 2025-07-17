# PDF Chatbot with Retrieval-Augmented Generation (RAG)

This project is an AI-powered chatbot that answers user queries based only on a provided set of PDF documents.

- **UI:** Streamlit
- **LLM:** Mistral-7B-Instruct or TinyLlama (via Hugging Face)
- **Embeddings:** all-MiniLM-L6-v2
- **Vector DB:** ChromaDB
- **PDF Processing:** pypdf or PyMuPDF
- **Chunking:** langchain splitters

See `requirements.txt` for installation and `app.py` for entry point.

# PDF Chatbot with Retrieval-Augmented Generation (RAG)

## 1. Project Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot capable of answering queries grounded in a provided set of PDF documents.  
The solution utilizes modern NLP techniques: PDFs are processed into semantic chunks, embedded into a vector database, and relevant contexts are retrieved at query time for fact-based, streamed LLM responses.

**Key Technologies:**  
- Frontend: Streamlit (real-time, interactive UI)  
- LLM: Mistral-7B-Instruct or TinyLlama (locally, via Hugging Face Transformers)  
- Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)  
- Vector DB: ChromaDB (persistent, local)  
- PDF Processing: PyMuPDF / pypdf  
- Chunking: LangChain text splitters (context-aware)  

---

## 2. Architecture & Workflow

**Main Components**

| Component | Role |
| :-------------- | :----------------------------------------------------------------- |
| PDF Loader | Reads and extracts plaintext from PDFs |
| Chunker | Slices texts into overlapping, sentence-aligned segments |
| Embedding Model | Encodes chunks into semantic vectors for search |
| Vector Database | Stores chunk embeddings for efficient retrieval |
| RAG Pipeline | Handles retrieval and LLM-based answer generation |
| Streamlit App | Chat interface, streaming answers, displays sources |

**Workflow Diagram**  
1. Preprocessing:  
    PDFs → Text → Cleaned Text → Chunks → Embeddings
2. Indexing:  
    Embeddings + Metadata → Vector DB (ChromaDB)
3. Query:  
    User Query → Embedded → Retrieve Top-k Chunks → Contextual Prompt → LLM
4. Response:  
    Streamed answer + Source Displays

---

## 3. Getting Started
**Installation**  
```bash
git clone https://github.com/VIDIT0906/PDF_Chatbot.git
cd RAG
pip install -r requirements.txt
```

**Preprocessing & Indexing**  
1. Add PDFs: Place files in data/.  
2. Preprocess & Index (recommended from the app sidebar):  
    -Launch app (see below), then click “Ingest PDFs Now” in the Streamlit sidebar.  
    -Alternatively, directly run preprocessing via a provided notebook or a CLI script (optional).

Artifacts are created in chunks/ (chunks.json, embeddings.npy, metadata.json). The vector DB persists in vectordb/.

**Running the Chatbot**  
```bash
streamlit run app.py
```
- Open the provided URL.  
- Enter your question related to uploaded PDFs.  
- The answer will stream live, with retrieved source passages shown below.  

**Streaming Response Demo**  
- As you type a query and press "Ask", the chatbot response appears incrementally (token by token).
- Source passages appear in the output with each answer.

---

## 4. Model & Embedding Choices

| Purpose     | Model Name                           | Reason                                    |
| :---------- | :----------------------------------- | :---------------------------------------- |
| LLM         | `mistralai/Mistral-7B-Instruct-v0.2` | High performance, open weights            |
| (Alternative) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Lightweight, fast                         |
| Embedding   | `sentence-transformers/all-MiniLM-L6-v2` | Fast, robust, widely adopted              |

- LLM is loaded locally using transformers. Select your preference in config.py.
- Embeddings are auto-cached; only re-generated when new content is ingested.

---

## 5. Example Usage & Interface
**Sample Queries**  
| Query | Answer Behavior | Source Displayed |
| :---------------------------------------------- | :-------------------------------------------------------------------------------------- | :--------------- |
| "Summarize the findings in Section 3 of the paper." | Detailed, section-specific summary if present in context | Yes |
| "What methods were used for data analysis?" | Methods listed and referenced (if in docs) | Yes |
| "When was the document published?" | Date reported if present; else, fallback message | Yes/Or Fallback |
| "Who is the author?" | Author extracted from doc if present; else, fallback | Yes/Or Fallback |
| "Does this document discuss climate change policy?" | Yes/No and supporting passage(s), or fallback | Yes/Or Fallback |
| "How do black holes form?" (not mentioned in docs) | “I could not find an answer in the documents.” | (n/a) |

**Output (Screenshot Description):**  
- Left: Sidebar with model/embedding info, chunk stats, ingest/reset buttons
- Center: Chat input field, streamed chatbot responses, collapsible retrieved sources
- Bottom: Info panel for usage tips

---

## 6. Demo Link
  [![Watch the video](https://img.youtube.com/vi/gsRTDt1bPPc/maxresdefault.jpg)](https://youtu.be/gsRTDt1bPPc)
  [Watch this video on YouTube](https://youtu.be/gsRTDt1bPPc)

---

## 7. Troubleshooting
- **Ingestion Fails / ID Errors:** Ensure all lists (chunks, embeddings, metadata) have matching length.
- **Slow LLM Responses:** On low-resource hardware, use TinyLlama or reduce prompt chunk count.
- **Missing Answers:** If context doesn’t contain the answer, the bot will explicitly state so.
- **Model Hallucinations:** Minimized by enforced context-constrained prompt; still possible if context is ambiguous or missing.