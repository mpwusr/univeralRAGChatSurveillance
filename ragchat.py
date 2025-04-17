import os
import logging
import requests
from dotenv import load_dotenv
import time
from dataclasses import dataclass
from openai import OpenAI
import cohere
from requests import Response
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from logging.handlers import RotatingFileHandler
import argparse
import inspect
from pinecone import Pinecone, ServerlessSpec
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageTk
import torch
import tkinter as tk
from typing import List, Dict, Tuple

# Logging setup
handler = RotatingFileHandler("chatbot.log", maxBytes=10 * 1024 * 1024, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from environment variables")
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "surveillance-my_images"
DIMENSION = 512
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
index = pc.Index(INDEX_NAME)

clip_processor = None
clip_model = None

def load_clip():
    global clip_processor, clip_model
    if clip_processor is None or clip_model is None:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    return clip_processor, clip_model

@dataclass
class Config:
    grok_api_key: str
    openai_api_key: str
    co_api_key: str
    grok_api_url: str = "https://api.x.ai/v1/chat/completions"
    default_service: str = "grok"
    default_model: str = "grok-2"

    def __post_init__(self):
        self.co_client = cohere.Client(self.co_api_key)
        self.openai_client = OpenAI(api_key=self.openai_api_key)

    def grok_headers(self):
        return {"Authorization": f"Bearer {self.grok_api_key}", "Content-Type": "application/json"}

def load_config():
    grok_key = os.getenv("XAI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    co_key = os.getenv("CO_API_KEY")
    if not all([grok_key, openai_key, co_key]):
        raise ValueError("Missing one or more required API keys in environment")
    return Config(grok_api_key=grok_key, openai_api_key=openai_key, co_api_key=co_key)

def build_prompt(base_role: str, prompt: str, conversation_history: List[Dict[str, str]] = None, extra_instructions: str = "") -> str:
    base_prompt = f"You are a {base_role}. {extra_instructions}"
    history = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history]) if conversation_history else ""
    return f"{history}\n{base_prompt}\n\nUser's question: {prompt}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_grok_response(prompt, model, use_deep_search=False, conversation_history=None, grok_url=None, grok_headers=None, extra_instructions=""):
    full_prompt = build_prompt("physical security consultant", prompt, conversation_history, extra_instructions)
    payload = {"model": model, "messages": [{"role": "user", "content": full_prompt}], "max_tokens": 300}
    resp: Response = requests.post(grok_url, headers=grok_headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def embed_text(text: str) -> torch.Tensor:
    processor, model = load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs.cpu().numpy()[0]

def retrieve_matches(query_embedding: torch.Tensor, top_k: int = 5):
    return index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)["matches"]

def summarize_segments(matches):
    summaries = []
    for match in matches:
        meta = match.get("metadata", {})
        label = meta.get("label", "unknown")
        friendly = "friendly" if meta.get("friendly") else "unfriendly"
        score = round(meta.get("score", 0.0), 2)
        summaries.append(f"- {label} (score: {score}, {friendly})")
    return "\n".join(summaries)

def display_gui(query: str, reply: str, matches):
    root = tk.Tk()
    root.title("Security Analysis")
    tk.Label(root, text=f"Query: {query}", wraplength=600, font=("Arial", 12, "bold")).pack(pady=5)
    tk.Label(root, text=f"Response: {reply}", wraplength=600).pack(pady=5)

    for match in matches:
        path = match.get("metadata", {}).get("path")
        if path and os.path.exists(path):
            try:
                image = Image.open(path).convert("RGB")
                image = image.resize((200, 200))
                photo = ImageTk.PhotoImage(image)
                label_text = match.get("metadata", {}).get("label", "unknown")
                friendly = "Friendly" if match.get("metadata", {}).get("friendly") else "Unfriendly"
                summary = f"{label_text} - {friendly}"

                img_label = tk.Label(root, image=photo)
                img_label.image = photo  # keep a reference
                img_label.pack(pady=5)
                tk.Label(root, text=summary).pack()
            except Exception as e:
                tk.Label(root, text=f"Failed to load image: {path}").pack()

    tk.Label(root, text="Was this helpful? (Y/N)").pack(pady=10)
    root.mainloop()

def main():
    config = load_config()
    SERVICE = "grok"
    MODEL = config.default_model
    conversation_history = []

    print("Surveillance RAGChat is ready. Type 'exit' to quit.")
    while True:
        query = input("Ask a security question: ")
        if query.lower() in ["exit", "quit"]:
            break
        embedding = embed_text(query)
        matches = retrieve_matches(embedding)
        summary = summarize_segments(matches)
        extra = f"Use the following scene elements:\n\n{summary}\n\n"
        try:
            response = get_grok_response(
                prompt=query,
                model=MODEL,
                use_deep_search=False,
                conversation_history=conversation_history,
                grok_url=config.grok_api_url,
                grok_headers=config.grok_headers(),
                extra_instructions=extra
            )
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": response})
            display_gui(query, response, matches)
        except Exception as e:
            logger.error(f"Error: {e}")
            print("An error occurred. Please try again.")

if __name__ == "__main__":
    main()