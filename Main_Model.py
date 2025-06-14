
# --- START OF REFACTORED app.py ---

import os
import json
import sqlite3
import numpy as np
import re
import asyncio
import logging
import traceback
from typing import Optional, List
import aiohttp

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# ==============================================================================
#  CONFIGURATION & INITIALIZATION
# ==============================================================================

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_API")

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
# Database path
DB_PATH = "knowledge_base.db"
# API key for the LLM service
API_KEY = os.getenv("API_KEY")

# --- RAG Tuning Parameters ---
# Lower threshold to retrieve more potential matches
SIMILARITY_THRESHOLD = 0.45
# Maximum number of initial results to consider from the database
MAX_RESULTS = 20
# Max chunks to use from a single source document to avoid over-concentration
MAX_CONTEXT_CHUNKS_PER_SOURCE = 5

# --- Pydantic Models for API data validation ---
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image string

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Virtual TA RAG API",
    description="API for querying a knowledge base of course materials and discussions.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ==============================================================================
#  DATABASE SETUP & UTILITIES
# ==============================================================================

def initialize_database():
    """
    Connects to the SQLite database, creates tables and indexes if they don't exist.
    This function is crucial for setting up the database schema correctly.
    """
    logger.info(f"Initializing database at: {DB_PATH}")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # --- Create discourse_chunks table ---
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS discourse_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                topic_id INTEGER,
                topic_title TEXT,
                post_number INTEGER,
                author TEXT,
                created_at TEXT,
                likes INTEGER,
                chunk_index INTEGER,
                content TEXT,
                url TEXT,
                embedding BLOB
            )
            ''')
           
            # --- Create markdown_chunks table ---
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS markdown_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_title TEXT,
                original_url TEXT,
                downloaded_at TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB
            )
            ''')

            # --- Create Indexes for Performance ---
            # These indexes are CRITICAL for fast lookups in a large database.
            logger.info("Creating database indexes if they don't exist...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_discourse_post_id ON discourse_chunks (post_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_discourse_chunk_index ON discourse_chunks (post_id, chunk_index);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_markdown_doc_title ON markdown_chunks (doc_title);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_markdown_chunk_index ON markdown_chunks (doc_title, chunk_index);")

            conn.commit()
            logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.critical(f"FATAL: Database initialization failed: {e}", exc_info=True)
        raise

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not connect to the database.")

# Initialize the database on startup
if not API_KEY:
    logger.critical("FATAL: API_KEY environment variable is not set. The application cannot start.")
else:
    initialize_database()

# ==============================================================================
#  CORE RAG LOGIC
# ==============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    try:
        # Ensure vectors are numpy arrays of type float
        vec1 = np.asarray(vec1, dtype=np.float32)
        vec2 = np.asarray(vec2, dtype=np.float32)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return float(dot_product / (norm_vec1 * norm_vec2))
    except (ValueError, TypeError) as e:
        logger.error(f"Error in cosine_similarity: {e}", exc_info=True)
        return 0.0

async def get_embedding_from_api(text: str, max_retries: int = 3) -> List[float]:
    """
    Gets text embedding from the API with a retry mechanism and longer timeout.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY is not configured on the server.")
    
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": text}
    
    # Increased timeout to handle potentially slow API responses
    request_timeout = aiohttp.ClientTimeout(total=90.0)

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=request_timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Successfully generated embedding for text (length: {len(text)}).")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:
                        logger.warning(f"Rate limit hit. Retrying in {5 * (attempt + 1)} seconds...")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        error_text = await response.text()
                        logger.error(f"API Error (Status {response.status}): {error_text}")
                        # Don't retry on definitive errors like 400 or 401
                        if response.status in [400, 401, 403]:
                            break
                        # For server errors, retry
                        await asyncio.sleep(3 * (attempt + 1))
        except asyncio.TimeoutError:
            logger.warning(f"Request timed out. Retrying... (Attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(5 * (attempt + 1))
        except Exception as e:
            logger.error(f"Exception during embedding request: {e}", exc_info=True)
            await asyncio.sleep(3 * (attempt + 1))
            
    raise HTTPException(status_code=503, detail="The embedding service is currently unavailable. Please try again later.")

async def process_multimodal_query(question: str, image_base64: Optional[str]) -> List[float]:
    """
    Generates a text description from an image if provided and combines it with the question
    to create a unified embedding.
    """
    if not image_base64:
        logger.info("No image provided. Generating embedding for text query only.")
        return await get_embedding_from_api(question)

    logger.info("Image provided. Analyzing with vision model to enhance query.")
    try:
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
        image_content = f"data:image/jpeg;base64,{image_base64}"
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Describe the key elements in this image relevant to the following question: '{question}'"},
                        {"type": "image_url", "image_url": {"url": image_content}}
                    ]
                }
            ]
        }
        
        request_timeout = aiohttp.ClientTimeout(total=120.0)
        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Image description received: '{image_description[:100]}...'")
                    combined_query = f"{question}\n\n[Context from image: {image_description}]"
                    return await get_embedding_from_api(combined_query)
                else:
                    error_text = await response.text()
                    logger.error(f"Vision API failed (Status {response.status}): {error_text}. Falling back to text-only query.")
                    return await get_embedding_from_api(question)
    except Exception as e:
        logger.error(f"Exception in multimodal processing: {e}. Falling back to text-only.", exc_info=True)
        return await get_embedding_from_api(question)


async def find_similar_content(query_embedding: List[float], conn: sqlite3.Connection) -> List[dict]:
    """
    Searches the database for content chunks with high cosine similarity to the query embedding.
    """
    logger.info("Searching for relevant content in the database...")
    cursor = conn.cursor()
    all_chunks = []
    
    # Fetch all chunks with embeddings from both tables
    cursor.execute("SELECT id, content, embedding, 'discourse' as source, topic_title as title, url FROM discourse_chunks WHERE embedding IS NOT NULL")
    all_chunks.extend(cursor.fetchall())
    cursor.execute("SELECT id, content, embedding, 'markdown' as source, doc_title as title, original_url as url FROM markdown_chunks WHERE embedding IS NOT NULL")
    all_chunks.extend(cursor.fetchall())
    
    logger.info(f"Comparing query against {len(all_chunks)} chunks from the knowledge base.")
    
    # Calculate similarity for each chunk
    results = []
    for chunk in all_chunks:
        try:
            chunk_embedding = json.loads(chunk["embedding"])
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            
            if similarity >= SIMILARITY_THRESHOLD:
                results.append({
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "similarity": similarity,
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "url": chunk["url"]
                })
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Skipping chunk ID {chunk['id']} from source '{chunk['source']}' due to invalid embedding data: {e}")

    # Sort by similarity and return the top results
    results.sort(key=lambda x: x["similarity"], reverse=True)
    logger.info(f"Found {len(results)} relevant chunks above threshold {SIMILARITY_THRESHOLD}.")
    
    return results[:MAX_RESULTS]


async def generate_llm_answer(question: str, context_chunks: List[dict], max_retries: int = 2) -> str:
    """
    Generates a final answer by feeding the question and retrieved context to an LLM.
    """
    if not context_chunks:
        return "I could not find any relevant information in the knowledge base to answer this question."

    logger.info(f"Generating answer using {len(context_chunks)} context chunks.")
    context_str = ""
    for i, chunk in enumerate(context_chunks):
        context_str += f"--- Context Snippet {i+1} ---\n"
        context_str += f"Source Type: {'Documentation' if chunk['source'] == 'markdown' else 'Discourse Forum'}\n"
        context_str += f"Document Title: {chunk['title']}\n"
        context_str += f"URL: {chunk['url']}\n"
        context_str += f"Content: {chunk['content']}\n\n"

    # A more robust and directive prompt
    prompt = f"""You are an expert Teaching Assistant for a 'Tools in Data Science' course.
Your task is to answer the user's question accurately and exclusively based on the provided context snippets.
Do not use any external knowledge. If the context does not contain the answer, state that clearly.

Follow these rules STRICTLY:
1.  **Answer First:** Provide a direct and comprehensive answer to the question.
2.  **Cite Sources:** After the answer, include a "Sources:" section. List the URLs of the documents you used.
3.  **Exact Format:** Your entire response MUST follow this structure:
    
    [Your comprehensive answer here]

    Sources:
    - [Full URL of Source 1]
    - [Full URL of Source 2]
    ...

HERE IS THE CONTEXT:
{context_str}

HERE IS THE USER'S QUESTION:
{question}

Your response:
"""

    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Lower temperature for more factual, less creative answers
        "n": 1,
    }
    request_timeout = aiohttp.ClientTimeout(total=120.0)

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=request_timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received response from LLM.")
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"LLM API Error (Status {response.status}): {error_text}. Retrying...")
                        await asyncio.sleep(5 * (attempt + 1))
        except asyncio.TimeoutError:
            logger.warning(f"LLM request timed out. Retrying... (Attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.error(f"Exception during LLM request: {e}", exc_info=True)
    
    raise HTTPException(status_code=503, detail="The language model service is currently unavailable.")


def parse_llm_response(llm_response_text: str, relevant_chunks: List[dict]) -> dict:
    """
    Parses the LLM's text response to extract the answer and source links.
    Includes a fallback mechanism if parsing fails.
    """
    try:
        # Split the response into answer and sources parts
        parts = re.split(r'\n\s*Sources:', llm_response_text, maxsplit=1, flags=re.IGNORECASE)
        answer = parts[0].strip()
        
        links = []
        unique_urls = set()

        if len(parts) > 1:
            sources_section = parts[1]
            # Regex to find URLs, robustly
            found_urls = re.findall(r'https?://[^\s\)]+', sources_section)
            for url in found_urls:
                if url not in unique_urls:
                    links.append(LinkInfo(url=url, text="Relevant source document"))
                    unique_urls.add(url)
        
        # Fallback: If parsing fails, use the URLs from the top context chunks
        if not links and relevant_chunks:
            logger.warning("Could not parse source URLs from LLM response. Using top context chunks as sources.")
            for chunk in relevant_chunks:
                if chunk['url'] not in unique_urls:
                    links.append(LinkInfo(url=chunk['url'], text=chunk['title']))
                    unique_urls.add(chunk['url'])
                if len(links) >= 3:  # Limit to 3 fallback links
                    break

        return {"answer": answer, "links": links}
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}. Returning raw response.", exc_info=True)
        # Return the raw response as the answer if parsing completely fails
        return {"answer": llm_response_text, "links": []}

# ==============================================================================
#  API ENDPOINTS
# ==============================================================================

@app.post("/api", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Main endpoint to handle user queries. It orchestrates the entire RAG pipeline.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server is not configured with an API key.")
    
    logger.info(f"Received query: '{request.question[:70]}...'")
    
    conn = None
    try:
        # 1. Get database connection
        conn = get_db_connection()
        
        # 2. Process query (text + optional image) and get embedding
        query_embedding = await process_multimodal_query(request.question, request.image)
        
        # 3. Find similar content in the database
        similar_chunks = await find_similar_content(query_embedding, conn)
        
        if not similar_chunks:
            logger.warning("No relevant content found for the query.")
            return QueryResponse(
                answer="I could not find any relevant information to answer this question.",
                links=[]
            )
        
        # 4. Generate a response using the LLM with the retrieved context
        llm_response_text = await generate_llm_answer(request.question, similar_chunks)
        
        # 5. Parse the LLM response to structure it for the API
        final_result = parse_llm_response(llm_response_text, similar_chunks)
        
        logger.info(f"Successfully processed query. Returning answer and {len(final_result['links'])} sources.")
        return QueryResponse(**final_result)

    except HTTPException as http_exc:
        # Re-raise FastAPI's HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the query process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        if conn:
            conn.close()

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status, DB connection, and data presence.
    """
    db_status = "disconnected"
    db_error = ""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_error = str(e)

    return {
        "status": "healthy" if db_status == "connected" and API_KEY else "unhealthy",
        "services": {
            "database": {
                "status": db_status,
                "markdown_chunks": markdown_count if db_status == "connected" else 0,
                "discourse_chunks": discourse_count if db_status == "connected" else 0,
                "error": db_error
            },
            "llm_api": {
                "api_key_set": bool(API_KEY)
            }
        }
    }

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: API_KEY environment variable is not set. Please create a .env file and add it.")
    else:
        print("Starting FastAPI server...")
        uvicorn.run("Main_Model:app", host="0.0.0.0", port=8000, reload=True)