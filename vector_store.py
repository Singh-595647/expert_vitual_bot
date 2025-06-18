import sqlite3
import os
import json
from typing import List, Dict

DB_PATH = "vector_store.db"

# Create tables if not exist
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

def save_chunks_with_embeddings(chunks: List[Dict]):
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

def load_chunks_with_embeddings() -> List[Dict]:
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

def is_db_populated() -> bool:
    conn = get_db_connection()
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    return count > 0
