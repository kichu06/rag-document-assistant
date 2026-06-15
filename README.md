# AI Document Assistant

A Retrieval-Augmented Generation (RAG) application built with Python, Streamlit, ChromaDB, and Hugging Face models.

## Features

- Upload PDF documents
- Automatic text extraction
- Semantic chunking
- BGE embeddings
- ChromaDB vector storage
- Semantic search
- LLM-powered question answering

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Hugging Face Inference API
- Qwen 2.5

## Architecture

PDF
→ Chunking
→ Embeddings
→ ChromaDB
→ Retrieval
→ LLM
→ Answer

## Run Locally

pip install -r requirements.txt
streamlit run app.py