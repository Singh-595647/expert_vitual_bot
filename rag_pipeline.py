import os
import json
import glob
import re
from typing import List, Dict, Tuple
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3

# Placeholder for AIPipe API integration
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
AIPIPE_EMBEDDING_ENDPOINT = "https://api.aipipe.io/v1/embeddings"
AIPIPE_COMPLETION_ENDPOINT = "https://api.aipipe.io/v1/completions"

# Local embedding model for vector DB creation
_local_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
DB_PATH = "vector_store.db"

# --- Data Loading ---
def load_markdown_files(md_dir: str) -> List[Dict]:
    data = []
    for md_file in glob.glob(os.path.join(md_dir, "*.md")):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        data.append({"type": "markdown", "file": md_file, "content": content})
    return data

def load_json_files(json_dir: str) -> List[Dict]:
    data = []
    for json_file in glob.glob(os.path.join(json_dir, "*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        data.append({"type": "json", "file": json_file, "content": content})
    return data

def chunk_markdown(text: str, max_tokens: int = 300) -> List[str]:
    # Simple chunking by paragraphs, can be improved
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current = ""
    for para in paragraphs:
        if len((current + para).split()) > max_tokens:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current:
        chunks.append(current.strip())
    return chunks

def extract_json_posts(json_content: dict) -> List[str]:
    # Assumes Discourse JSON with 'post_stream' and 'posts'
    posts = json_content.get('post_stream', {}).get('posts', [])
    return [post.get('cooked', '') for post in posts if post.get('cooked')]

def get_aipipe_embedding(text: str) -> List[float]:
    headers = {"Authorization": f"Bearer {AIPIPE_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": text, "model": "text-embedding-ada-002"}  # Adjust model as needed
    response = requests.post(AIPIPE_EMBEDDING_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# Vector DB helpers
SCHEMA = '''
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    file TEXT,
    chunk_index INTEGER,
    content TEXT,
    embedding TEXT
);
'''

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn

def save_chunks_with_embeddings(chunks):
    conn = get_db_connection()
    for idx, chunk in enumerate(chunks):
        conn.execute(
            "INSERT INTO chunks (source, file, chunk_index, content, embedding) VALUES (?, ?, ?, ?, ?)",
            (
                chunk.get("source", "unknown"),
                chunk.get("file", "unknown"),
                chunk.get("chunk_index", idx),
                chunk["text"],
                json.dumps(chunk["embedding"])
            )
        )
    conn.commit()
    conn.close()

def load_chunks_with_embeddings():
    conn = get_db_connection()
    rows = conn.execute("SELECT source, file, chunk_index, content, embedding FROM chunks").fetchall()
    conn.close()
    return [
        {
            "source": row[0],
            "file": row[1],
            "chunk_index": row[2],
            "text": row[3],
            "embedding": json.loads(row[4])
        }
        for row in rows
    ]

def is_db_populated():
    conn = get_db_connection()
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    return count > 0

def build_corpus_embeddings(md_data, json_data):
    corpus = []
    for item in md_data:
        for i, chunk in enumerate(chunk_markdown(item["content"])):
            emb = _local_model.encode(chunk).tolist()
            corpus.append({"text": chunk, "embedding": emb, "source": item["file"], "chunk_index": i})
    for item in json_data:
        for i, post in enumerate(extract_json_posts(item["content"])):
            emb = _local_model.encode(post).tolist()
            corpus.append({"text": post, "embedding": emb, "source": item["file"], "chunk_index": i})
    return corpus

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve_relevant_chunks(query: str, corpus: List[Dict], top_k: int = 5) -> List[Dict]:
    query_emb = get_aipipe_embedding(query)
    scored = [
        {**item, "score": cosine_similarity(query_emb, item["embedding"])}
        for item in corpus
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

def generate_answer_with_aipipe(query: str, context_chunks: List[str]) -> str:
    headers = {"Authorization": f"Bearer {AIPIPE_API_KEY}", "Content-Type": "application/json"}
    context = "\n\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(AIPIPE_COMPLETION_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def load_chunks_from_db(db_path: str, table: str) -> list:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    columns = [desc[0] for desc in conn.execute(f'SELECT * FROM {table}').description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]

# --- Main Pipeline (to be expanded) ---
def main():
    # Load your improved DB
    discourse_chunks = load_chunks_from_db("improved_knowledge_base.db", "discourse_chunks")
    markdown_chunks = load_chunks_from_db("improved_knowledge_base.db", "markdown_chunks")
    corpus = []
    for row in discourse_chunks:
        corpus.append({
            "text": row["content"],
            "embedding": json.loads(row["embedding"]),
            "source": row["url"] or row["topic_title"],
            "meta": row
        })
    for row in markdown_chunks:
        corpus.append({
            "text": row["content"],
            "embedding": json.loads(row["embedding"]),
            "source": row["original_url"] or row["doc_title"],
            "meta": row
        })
    print(f"Loaded {len(corpus)} chunks from improved_knowledge_base.db.")
    while True:
        query = input("Ask a question (or 'exit'): ")
        if query.lower() == 'exit':
            break
        query_emb = get_aipipe_embedding(query)
        scored = [
            {**item, "score": cosine_similarity(query_emb, item["embedding"])}
            for item in corpus
        ]
        top_chunks = sorted(scored, key=lambda x: x["score"], reverse=True)[:10]
        print("\n--- Retrieved Context ---")
        for i, c in enumerate(top_chunks):
            print(f"[{i+1}] {c['text'][:200]}\n---")
        print("------------------------\n")
        answer = generate_answer_with_aipipe(query, [c["text"] for c in top_chunks])
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
