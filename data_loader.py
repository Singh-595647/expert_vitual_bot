import os
import glob
import json
from typing import List, Dict

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

def extract_json_posts(json_content: dict) -> List[str]:
    posts = json_content.get('post_stream', {}).get('posts', [])
    return [post.get('cooked', '') for post in posts if post.get('cooked')]
