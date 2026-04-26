# LLM-Powered Knowledge Retrieval System (RAG)

## Overview

This project is an end-to-end Retrieval-Augmented Generation (RAG) system that enables users to ask natural language questions over custom document collections (PDFs). The system retrieves relevant context using vector similarity search and generates responses using a large language model (LLM).

It combines document processing, embeddings, vector databases, and LLMs to create a practical AI-powered knowledge assistant.

---

## Features

* Load and process multiple PDF documents
* Split documents into semantically meaningful chunks
* Generate embeddings using OpenAI embedding models
* Store and retrieve vectors using ChromaDB
* Perform semantic search using similarity matching
* Generate context-aware responses using LLMs
* Simple Streamlit-based user interface

---

## Architecture

PDF Documents
↓
Text Extraction (PyPDFLoader)
↓
Text Chunking (RecursiveCharacterTextSplitter)
↓
Embedding Generation (OpenAI Embeddings)
↓
Vector Storage (ChromaDB)
↓
User Query
↓
Semantic Retrieval (Top-K Relevant Chunks)
↓
Context Augmentation
↓
LLM (GPT-4o-mini)
↓
Final Response

---

## Tech Stack

* Python 3.11
* LangChain
* OpenAI API
* ChromaDB (Vector Database)
* Streamlit (User Interface)
* PyPDFLoader

---

## Project Structure

rag_project/
├── app.py              # Streamlit user interface
├── ingest.py           # Document ingestion and vector database creation
├── rag_chain.py        # Retrieval and LLM response generation logic
├── docs/               # Input PDF documents
├── db/                 # Persisted vector database
└── requirements.txt

---

## Setup Instructions

### 1. Clone repository

```
git clone <repository-url>
cd rag_project
```

### 2. Create virtual environment

```
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Configure API key

Set your OpenAI API key as an environment variable:

```
OPENAI_API_KEY=your_api_key_here
```

### 5. Add documents

Place PDF files inside the docs directory:

```
docs/
```

### 6. Build vector database

```
py ingest.py
```

### 7. Run the application

```
streamlit run app.py
```

---

## Example Use Cases

* Question answering over technical documentation
* Internal enterprise knowledge base search
* System design and engineering notes retrieval
* Resume or report querying system

---

## Example Queries

* What is multithreading in Java?
* Explain system design fundamentals
* What is a HashMap and how does it work?
* Summarize the content of this document

---

## Key Learnings

This project demonstrates:

* Retrieval-Augmented Generation (RAG) architecture
* Vector similarity search techniques
* Embedding-based document representation
* Prompt engineering for LLMs
* Integration of external knowledge with large language models

---

## Future Improvements

* Add conversational memory for multi-turn chat
* Integrate local LLMs such as Llama or Ollama
* Improve retrieval with reranking models
* Add source highlighting for answers
* Deploy on cloud infrastructure (AWS/Azure)

---

## Author

This project was built to demonstrate practical implementation of Retrieval-Augmented Generation systems using modern LLM tooling and vector databases.
