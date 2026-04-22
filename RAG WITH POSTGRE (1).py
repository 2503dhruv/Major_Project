import fitz
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import uuid  # For generating unique session IDs
import re  # Import the regex module
import psycopg2  # PostgreSQL library
from dotenv import load_dotenv

# Regex pattern for user ID
USER_ID_PATTERN = r'^user\d+$'  # Matches "user" followed by one or more digits (e.g., "user1", "user123")

load_dotenv()

USER_ID_PATTERN = r'^user\d+$'

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Connect to PostgreSQL database
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# Initialize database tables
def initialize_db():
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Create table for user history
        cur.execute("""
            CREATE TABLE IF NOT EXISTS userchat_history (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.commit()
    conn.close()

# Validate user ID
def is_valid_user_id(user_id):
    return re.match(USER_ID_PATTERN, user_id) is not None

# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Chunk the extracted text into sentences or paragraphs
def chunk_text(text):
    return text.split(". ")

# Step 3: Store and retrieve embeddings using FAISS
def store_embeddings(chunks, model):
    chunk_embeddings = model.encode(chunks)
    chunk_embeddings = np.array(chunk_embeddings).astype(np.float32)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    return index

def retrieve_relevant_chunk(query, index, chunks, model):
    query_embedding = model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, k=1)
    return chunks[indices[0][0]], distances[0][0]

# Step 4: Generate answer using Ollama model
def generate_answer(relevant_text, query, user_id, session_id):
    client = ollama.Client()
    model_name = "llama3.2:latest"
    temperature = 0.9
    
    # Retrieve the last 5 responses for the user
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT question, answer
            FROM userchat_history
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT 5
        """, (user_id,))
        history = "\n".join([f"Q: {row[0]}\nA: {row[1]}" for row in cur.fetchall()])
    conn.close()
    
    # Construct the prompt with history
    prompt = f"{history}\nBased on the following text: {relevant_text}\nQuestion: {query}"
    response = client.generate(model_name, prompt)
    return response.get('response', "No response text found.")

# RAG System with session management
def rag_system():
    initialize_db()  # Ensure database is ready
    
    pdf_path = input("Enter the path to the PDF file: ").strip()
    user_id = input("Enter your user ID: ").strip()
    
    # Validate user ID using regex
    if not is_valid_user_id(user_id):
        print("Invalid user ID. It must start with 'user' followed by digits (e.g., 'user1').")
        return
    
    # Assign a session ID for the user
    session_id = str(uuid.uuid4())
    print(f"Session ID for this session: {session_id}")
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        index = store_embeddings(chunks, model)
        
        while True:
            question = input("Enter your question (or 'exit' to quit): ").strip()
            if question.lower() == "exit":
                print("Exiting the system...")
                break
            
            relevant_chunk, _ = retrieve_relevant_chunk(question, index, chunks, model)
            answer = generate_answer(relevant_chunk, question, user_id, session_id)
            
            # Save interaction in the database
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO userchat_history (user_id, session_id, question, answer)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, session_id, question, answer))
            conn.commit()
            conn.close()
            
            print("\nGenerated Answer:")
            print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    rag_system()
