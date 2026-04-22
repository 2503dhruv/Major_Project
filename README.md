# 📄 RAG PDF QA System with PostgreSQL & Ollama

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to:

* Upload a PDF document
* Ask questions based on the document
* Get AI-generated answers using a local LLM
* Store chat history persistently using PostgreSQL

Unlike basic chatbot systems, this project combines:

* Vector search (FAISS)
* Embeddings (Sentence Transformers)
* Local LLM (Ollama)
* Persistent memory (PostgreSQL)

---

## 🧠 Architecture

```
PDF → Text Extraction → Chunking → Embeddings → FAISS Index
                                      ↓
                               Query Embedding
                                      ↓
                           Relevant Chunk Retrieval
                                      ↓
                    Ollama (LLM) → Answer Generation
                                      ↓
                        PostgreSQL → Store History
```

---

## ✨ Features

* 📄 PDF-based question answering
* 🧠 Retrieval-Augmented Generation (RAG)
* 🤖 Local LLM using Ollama (no API cost)
* 🗄️ Persistent chat history (PostgreSQL)
* 👤 Multi-user support with session tracking
* ⚡ Fast similarity search using FAISS

---

## 🛠️ Tech Stack

* Python 3.10
* PyMuPDF (PDF processing)
* FAISS (vector search)
* Sentence Transformers (embeddings)
* Ollama (local LLM)
* PostgreSQL (database)

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/rag-pdf-system.git
cd rag-pdf-system
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup PostgreSQL

Create database:

```sql
CREATE DATABASE ragdb;
```

Update credentials in code:

```python
DB_CONFIG = {
    "dbname": "ragdb",
    "user": "postgres",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}
```

---

### 5. Install & Run Ollama

Download from: https://ollama.com/download

Run:

```bash
ollama pull llama3.2
```

Ollama runs automatically in the background.

---

## ▶️ Usage

Run the script:

```bash
python rag_postgres.py
```

Follow prompts:

```
Enter PDF path
Enter user ID (e.g., user1)
Ask questions
```

---

## 🧪 Example Queries

* What is Artificial Intelligence?
* Summarize the document
* Explain Machine Learning

---

## 🗄️ Database Schema

Table: `userchat_history`

| Column     | Type      |
| ---------- | --------- |
| id         | SERIAL    |
| user_id    | TEXT      |
| session_id | TEXT      |
| question   | TEXT      |
| answer     | TEXT      |
| timestamp  | TIMESTAMP |

---

## ⚠️ Notes

* Ensure the PostgreSQL service is running
* Ollama must be installed and running
* First-time model download may take time

---

## 🚀 Future Improvements

* Multi-PDF support
* Web UI (Streamlit / React)
* pgvector integration
* Source highlighting
* API deployment (FastAPI)

---

## 👨‍💻 Author

Dhruv Sharma

---

## 📌 License

This project is for educational purposes.
