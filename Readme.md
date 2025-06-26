Prompt Pilot — RAG-Powered Local LLM Chatbot

**Prompt Pilot** is a local-first, extensible assistant designed to answer your questions by retrieving and reasoning over your own documents using state-of-the-art open-source language models.

##  Current Features

-  **PDF Document Ingestion**
  - Loads and chunks PDF files into manageable sections for semantic search.

-  **Embeddings with HuggingFace**
  - Uses `all-MiniLM-L6-v2` to embed text chunks into vector space for similarity search.

-  **Vector Indexing via LlamaIndex**
  - Fast vector search with in-memory retrieval. Supports pluggable backends like FAISS/Chroma (planned).

-  **Local LLM (Mistral 7B Instruct)**
  - Integrated via `llama-cpp-python` and wrapped using LlamaIndex’s `LlamaCPP`.
  - No API keys or external services — runs fully offline.

-  **Retrieval-Augmented Generation (RAG)**
  - Retrieves relevant chunks and generates context-aware answers using a local decoder-only LLM.

-  **Multi-Turn Chat with Memory**
  - Track and retain chat history to enable contextual, ongoing conversations.
##  Upcoming Features


-  **LoRA Fine-Tuning**
  - Apply lightweight fine-tuning to control the assistant’s tone and behavior.

-  **Observability and Logging**
  - Log prompts, token usage, latency, and errors. Visualize with Prometheus + Grafana.

-  **Web-Based Chat UI**
  - Build an interactive UI using Next.js with PDF upload and real-time response streaming.

-  **Multimodal Support (Optional)**
  - Extend ingestion to images (OCR), audio (Whisper), and video (YouTube transcripts).

##  Tech Stack

- Python, HuggingFace, LlamaIndex, `llama-cpp-python`
- Mistral 7B (GGUF format, quantized)
- Redis / FAISS (planned)
- Prometheus + Grafana (planned)
- Next.js frontend (planned)

##  Vision

To build a completely offline, privacy-friendly **multimodal assistant** that:
- Understands and answers questions over private documents
- Supports RAG-based QA + memory
- Can be fine-tuned for personality/tone via LoRA
- Optionally handles multimodal data inputs

---

###  Project Status: In Progress
