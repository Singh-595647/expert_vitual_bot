import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json
from data_loader import load_markdown_files, load_json_files, extract_json_posts
from chunker import overlapping_chunks
from embedder import get_aipipe_embedding
from retriever import retrieve_top_k
import requests
from vector_store import is_db_populated, load_chunks_with_embeddings, save_chunks_with_embeddings, get_db_connection
from dotenv import load_dotenv
import numpy as np

AIPIPE_COMPLETION_ENDPOINT = "https://aipipe.org/openai/v1/chat/completions"

load_dotenv()
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

app = FastAPI()

corpus = None

def get_corpus():
    global corpus
    if corpus is not None:
        return corpus
    if is_db_populated():
        corpus = load_chunks_with_embeddings()
    else:
        md_data = load_markdown_files("tds_pages_md")
        json_data = load_json_files("discourse_json")
        temp_corpus = []
        for item in md_data:
            for i, chunk in enumerate(overlapping_chunks(item["content"], max_tokens=300, overlap=50)):
                temp_corpus.append({"text": chunk, "source": "markdown", "file": item["file"], "chunk_index": i})
        for item in json_data:
            for post in extract_json_posts(item["content"]):
                for i, chunk in enumerate(overlapping_chunks(post, max_tokens=300, overlap=50)):
                    temp_corpus.append({"text": chunk, "source": "discourse", "file": item["file"], "chunk_index": i})
        for c in temp_corpus:
            c["embedding"] = get_aipipe_embedding(c["text"])
        save_chunks_with_embeddings(temp_corpus)
        corpus = temp_corpus
    return corpus

@app.on_event("startup")
def startup_event():
    # Optionally warm up the corpus in the background
    import threading
    threading.Thread(target=get_corpus, daemon=True).start()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def clean_llm_output(raw_output):
    """Clean LLM output when JSON parsing fails"""
    # Remove markdown code blocks
    lines = raw_output.strip().split('\n')
    cleaned_lines = []
    in_code_block = False
    
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def is_valid_url(url):
    return url and url.startswith("http")

def format_chunk_with_url(chunk):
    url = chunk.get("source")
    text = chunk["text"]
    # Make the URL extremely prominent for the LLM
    if url and url.startswith("http"):
        return f"[IMPORTANT SOURCE URL: {url}]\n{text}\n[END OF CHUNK]"
    return text + "\n[END OF CHUNK]"

@app.post("/api/", response_model=QueryResponse)
def rag_query(req: QueryRequest):
    query_emb = get_aipipe_embedding(req.question)
    conn = get_db_connection()
    
    # Query from chunks table
    scored = []
    row_count = 0
    
    cursor = conn.execute("SELECT source, file, chunk_index, content, embedding FROM chunks")
    for row in cursor:
        row_count += 1
        try:
            chunk_emb = json.loads(row[4])
            score = cosine_similarity(query_emb, chunk_emb)
            scored.append({
                "source": row[0] or "unknown",
                "file": row[1] or "unknown",
                "chunk_index": row[2],
                "text": row[3],
                "embedding": chunk_emb,
                "score": score
            })
            if row_count <= 3:
                print(f"Row {row_count}: content[:100]={row[3][:100]!r}, embedding_dim={len(chunk_emb)}")
        except Exception as e:
            print(f"Error parsing row {row_count}: {e}, content[:100]={row[3][:100]!r}")
    
    conn.close()
    print(f"Total rows processed: {row_count}, scored: {len(scored)}")
    
    top_k = 10
    top_chunks = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
    print("\n--- Top Retrieved Chunks ---")
    for i, c in enumerate(top_chunks):
        print(f"[{i+1}] Score: {c['score']:.4f} | Text: {c['text'][:200]}")
    print("---------------------------\n")
    # Use a moderate chunk size and overlap for best balance
    # (You should also update this in your DB build script and rebuild the DB)
    top_k = 10  # Increase number of retrieved chunks for more context
    filtered_chunks = [c for c in top_chunks if is_valid_url(c.get("source"))]
    context = "\n\n".join([format_chunk_with_url(c) for c in filtered_chunks])
    print("\n--- Context sent to LLM ---\n" + context + "\n--------------------------\n")
    prompt = f"""You are a helpful assistant that ONLY answers questions based on the provided context. Return your answer as a JSON object with two fields: 'answer' (string) and 'links' (array of objects with 'url' and 'text').

CRITICAL INSTRUCTIONS:
- ONLY answer if the context contains relevant information that directly answers the question
- If the context does not contain information to answer the question, respond with: "I don't have enough information in the provided context to answer this question."
- Do NOT make up, guess, or hallucinate any information
- ONLY use URLs that appear in the provided context
- Do NOT invent or create any URLs

Context:
{context}

Question: {req.question}

Format:
{{
  "answer": "<your answer here or 'I don't have enough information in the provided context to answer this question.'>",
  "links": [
    {{
      "url": "<url from context>",
      "text": "<brief quote or description>"
    }}
  ]
}}"""
    headers = {"Authorization": f"Bearer {AIPIPE_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "gpt-4o-mini", "messages": [
        {"role": "system", "content": "You are a helpful assistant that provides accurate answers based ONLY on the provided context. If the context does not contain relevant information to answer the question, you must say 'I don't have enough information in the provided context to answer this question.' Do NOT make up or hallucinate any information."},
        {"role": "user", "content": prompt}
    ], "temperature": 0.1}
    response = requests.post(AIPIPE_COMPLETION_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    raw_answer = response.json()["choices"][0]["message"]["content"].strip()
    # Try to parse as JSON, fallback to previous cleaning if needed
    import json as _json
    try:
        parsed = _json.loads(raw_answer)
        answer = parsed.get("answer", "")
        links = [Link(**l) for l in parsed.get("links", []) if "url" in l and "text" in l]
    except Exception as e:
        print(f"Warning: LLM did not return valid JSON. Error: {e}\nRaw answer: {raw_answer}")
        answer = clean_llm_output(raw_answer) # type: ignore
        links = []
    return QueryResponse(answer=answer, links=links)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("rag_api:app", host="0.0.0.0", port=port)