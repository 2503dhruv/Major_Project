from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import os
import uuid
import psycopg2
from dotenv import load_dotenv
import traceback
from contextlib import asynccontextmanager

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# Stop huggingface hub from trying to reach out to the internet
os.environ['HF_HUB_OFFLINE'] = '1'

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def initialize_db():
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS userchat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    finally:
        conn.close()

# Global state for indexing
model = None
index = None
chunks = []
current_session_id = str(uuid.uuid4())

def init_model():
    global model
    if model is None:
        try:
            print("Initializing SentenceTransformer model (offline mode)...")
            # The local_files_only flag causes sentence-transformers to skip checking huggingface entirely
            # and just load from the local cache directory it created the first time.
            model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            print("Model initialized successfully from local cache!")
        except Exception as e:
            print(f"Error loading SentenceTransformer model from cache: {e}")
            traceback.print_exc()
            try:
                 print("Fallback: Attempting to load without local_files_only...")
                 model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e2:
                 print(f"Fallback failed: {e2}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run initialization at startup
    initialize_db()
    init_model() # Load the model when the server starts!
    yield
    # Clean up (if needed) on shutdown

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text):
    return [c for c in text.split(". ") if c.strip()]

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    global index, chunks, model
    contents = await file.read()

    try:
        if model is None:
            raise Exception("Failed to initialize embedding model. Please check network connection or local cache.")

        if file.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(contents)
        else:
            text = contents.decode("utf-8", errors="ignore")

        new_chunks = chunk_text(text)
        if not new_chunks:
            return {"message": "No text found in document"}

        # Store the start index of these new chunks
        start_idx = len(chunks)
        chunks.extend(new_chunks)

        chunk_embeddings = model.encode(new_chunks)
        chunk_embeddings = np.array(chunk_embeddings).astype(np.float32)

        if index is None:
            index = faiss.IndexFlatL2(chunk_embeddings.shape[1])

        index.add(chunk_embeddings)

        return {"message": "File indexed successfully", "chunks": len(new_chunks)}
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_index():
    global index, chunks, current_session_id
    index = None
    chunks = []
    # Optionally reset session ID when documents are cleared so chat history starts fresh
    current_session_id = str(uuid.uuid4())
    return {"message": "Index reset successfully."}

class ChatRequest(BaseModel):
    query: str
    session_id: str = None

@app.post("/api/chat")
async def chat(request: ChatRequest):
    global index, chunks, current_session_id, model

    session_id = request.session_id or current_session_id

    if index is None or len(chunks) == 0:
        return {"text": "No documents indexed. Please upload a document first.", "sources": []}

    try:
        if model is None:
             raise Exception("Embedding model is not initialized.")

        query_embedding = model.encode([request.query]).astype(np.float32)
        k = min(3, len(chunks))
        distances, indices = index.search(query_embedding, k=k)

        relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        relevant_text = " ".join(relevant_chunks)

        # Retrieve history from Postgres
        history = ""
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT question, answer
                        FROM userchat_history
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT 3
                    """, (session_id,))
                    rows = cur.fetchall()
                    history_lines = [f"Q: {row[0]}\nA: {row[1]}" for row in reversed(rows)]
                    history = "\n".join(history_lines)
            finally:
                conn.close()

        prompt = f"{history}\nBased on the following text: {relevant_text}\nQuestion: {request.query}"

        client = ollama.Client()
        response = client.generate("llama3.2:latest", prompt)
        answer = response.get('response', "No response text found.")

        # Save to Postgres
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO userchat_history (session_id, question, answer)
                        VALUES (%s, %s, %s)
                    """, (session_id, request.query, answer))
                conn.commit()
            finally:
                conn.close()

        return {"text": answer, "sources": relevant_chunks[:2], "session_id": session_id}
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
