import os
import json
import glob
import re
import sqlite3
from typing import List, Dict
from chunker import overlapping_chunks
from data_loader import load_markdown_files, load_json_files, extract_json_posts
from embedder import get_aipipe_embedding, get_aipipe_embedding_batch
import logging

# ---
# Embeddings are stored as JSON arrays for transparency and portability.
# To use a more advanced embedding model (e.g., BAAI/bge-base-en-v1.5), set the environment variable:
#   export EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
# Or pass model_name to get_aipipe_embedding(chunk, model_name="BAAI/bge-base-en-v1.5")
# You can also add more metadata fields (e.g., section, tags) in the schema and insert logic below.
# ---

DB_PATH = "improved_knowledge_base.db"

SCHEMA = '''
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
    embedding TEXT
);
CREATE TABLE IF NOT EXISTS markdown_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_title TEXT,
    original_url TEXT,
    downloaded_at TEXT,
    chunk_index INTEGER,
    content TEXT,
    embedding TEXT
);
'''

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn

def build_and_store_discourse_chunks(json_dir: str):
    conn = get_db_connection()
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            topic_id = data.get("id")
            topic_title = data.get("title", "")
            posts = data.get("post_stream", {}).get("posts", [])
            for post in posts:
                post_id = post.get("id")
                post_number = post.get("post_number")
                author = post.get("username", "")
                created_at = post.get("created_at", "")
                likes = post.get("like_count", 0)
                url = post.get("url", "")
                content = post.get("cooked", "")
                print(f"{jf} post {post_id}: content length = {len(content.split())} tokens")
                chunks = overlapping_chunks(content, max_tokens=350, overlap=100)
                print(f"{jf} post {post_id}: {len(chunks)} chunks")
                if not chunks:
                    continue
                # Batch embed all chunks for this post
                try:
                    embeddings = get_aipipe_embedding_batch(chunks)
                except Exception as e:
                    logging.error(f"Batch embedding error in {jf} post {post_id}: {e}")
                    embeddings = [get_aipipe_embedding(chunk) for chunk in chunks]
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    try:
                        conn.execute(
                            "INSERT INTO discourse_chunks (post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (post_id, topic_id, topic_title, post_number, author, created_at, likes, idx, chunk, url, json.dumps(emb))
                        )
                    except Exception as e:
                        logging.error(f"DB error in {jf} post {post_id}: {e}")
            logging.info(f"Processed {jf}")
        except Exception as e:
            logging.error(f"Failed to process {jf}: {e}")
    conn.commit()
    conn.close()

def build_and_store_markdown_chunks(md_dir: str):
    conn = get_db_connection()
    md_files = glob.glob(os.path.join(md_dir, "*.md"))
    for mf in md_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                content = f.read()
            doc_title = os.path.splitext(os.path.basename(mf))[0]
            original_url = ""
            downloaded_at = ""
            # Try to extract metadata from markdown frontmatter if present
            if content.startswith("---"):
                meta_match = re.match(r"---\n(.*?)---", content, re.DOTALL)
                if meta_match:
                    meta = meta_match.group(1)
                    for line in meta.splitlines():
                        if line.startswith("title:"):
                            doc_title = line.split(":",1)[1].strip().strip('"')
                        if line.startswith("original_url:"):
                            original_url = line.split(":",1)[1].strip().strip('"')
                        if line.startswith("downloaded_at:"):
                            downloaded_at = line.split(":",1)[1].strip().strip('"')
            # Remove frontmatter for chunking
            content_body = re.sub(r"---.*?---", "", content, flags=re.DOTALL).strip()
            print(f"{mf}: content length = {len(content_body.split())} tokens")
            chunks = overlapping_chunks(content_body, max_tokens=350, overlap=100)
            print(f"{mf}: {len(chunks)} chunks")
            if not chunks:
                continue
            # Batch embed all chunks for this markdown file
            try:
                embeddings = get_aipipe_embedding_batch(chunks)
            except Exception as e:
                logging.error(f"Batch embedding error in {mf}: {e}")
                embeddings = [get_aipipe_embedding(chunk) for chunk in chunks]
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                try:
                    conn.execute(
                        "INSERT INTO markdown_chunks (doc_title, original_url, downloaded_at, chunk_index, content, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                        (doc_title, original_url, downloaded_at, idx, chunk, json.dumps(emb))
                    )
                except Exception as e:
                    logging.error(f"DB error in {mf} chunk {idx}: {e}")
            logging.info(f"Processed {mf}")
        except Exception as e:
            logging.error(f"Failed to process {mf}: {e}")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Building improved knowledge base...")
    build_and_store_discourse_chunks("discourse_json")
    build_and_store_markdown_chunks("tds_pages_md")
    print("Done! Database saved as improved_knowledge_base.db")
