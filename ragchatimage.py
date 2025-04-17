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
import yaml

# Setup logging with detailed metrics
handler = RotatingFileHandler("chatbot.log", maxBytes=10 * 1024 * 1024, backupCount=5)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[handler])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from environment variables")

# Load configuration from YAML (optional, fallback to defaults)
try:
    with open("config.yaml", "r") as f:
        config_yaml = yaml.safe_load(f)
except FileNotFoundError:
    config_yaml = {}

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = config_yaml.get("index_name", "surveillance-my_images")
DIMENSION = config_yaml.get("dimension", 512)  # Matches CLIP ViT-B/16 output

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
    logger.info(f"Created and activated Pinecone index '{INDEX_NAME}'")
index = pc.Index(INDEX_NAME)

# Lazy-load CLIP model and processor
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
    missing_keys = [k for k, v in [("XAI_API_KEY", grok_key), ("OPENAI_API_KEY", openai_key), ("CO_API_KEY", co_key)] if not v]
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    return Config(grok_api_key=grok_key, openai_api_key=openai_key, co_api_key=co_key)

def print_help():
    print(
        "I can help assess physical security using surveillance my_images. Try asking about suspicious activities, "
        "alarms, or trends. Commands: 'help' (this message), 'exit' (quit), 'switch to <service>' (change service), "
        "'set model <model>' (change model), 'feedback <Y/N>' (rate last response)."
    )

def build_prompt(base_role: str, prompt: str, conversation_history: List[Dict[str, str]] = None, extra_instructions: str = "") -> str:
    base_prompt = f"You are a {base_role}. {extra_instructions}"
    if conversation_history:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        return f"{history_text}\n{base_prompt}\n\nUser's question: {prompt}"
    return f"{base_prompt}\n\nUser's question: {prompt}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_grok_response(prompt, model, use_deep_search=False, conversation_history=None, grok_url=None, grok_headers=None, extra_instructions=""):
    start_time = time.time()
    deep_search_text = "Use DeepSearch to analyze recent X posts and provide insights. " if use_deep_search else ""
    full_prompt = build_prompt("physical security consultant", prompt, conversation_history, extra_instructions + deep_search_text)
    payload = {"model": model, "messages": [{"role": "user", "content": full_prompt}], "max_tokens": 300}
    logger.info("Sending payload to Grok: %s", payload)
    try:
        resp_grok: Response = requests.post(grok_url, headers=grok_headers, json=payload, timeout=10)
        resp_grok.raise_for_status()
        data = resp_grok.json()
        logger.info("Grok API response: %s", data)
        logger.info("Response time: %.2f seconds", time.time() - start_time)
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as err:
        logger.error("Grok request failed: %s", str(err))
        raise

def get_openai_response(prompt, model="gpt-4o", conversation_history=None, openai_client=None, extra_instructions=""):
    if openai_client is None:
        raise ValueError("OpenAI client must be provided")
    full_prompt = build_prompt("physical security consultant", prompt, conversation_history, extra_instructions)
    messages = [{"role": "user", "content": full_prompt}] if not conversation_history else \
               [{"role": msg["role"], "content": msg["content"]} for msg in conversation_history] + \
               [{"role": "user", "content": full_prompt}]
    try:
        resp_openai = openai_client.chat.completions.create(model=model, messages=messages, max_tokens=300)
        return resp_openai.choices[0].message.content
    except Exception as e:
        logger.error("OpenAI API error: %s", str(e))
        raise

def get_cohere_response(prompt, model="command-r", conversation_history=None, co_client=None, extra_instructions=""):
    if co_client is None:
        raise ValueError("Cohere client must be provided")
    base_prompt = build_prompt("physical security consultant", "", conversation_history, extra_instructions)
    chat_history = [{"role": "User" if msg["role"] == "user" else "Chatbot", "message": msg["content"]}
                    for msg in conversation_history] if conversation_history else []
    try:
        resp_co = co_client.chat(message=prompt, preamble=base_prompt, chat_history=chat_history, model=model, max_tokens=300, temperature=0.7)
        return resp_co.text
    except Exception as e:
        logger.error("Cohere API error: %s", str(e))
        raise

def trim_conversation_history(history: List[Dict[str, str]], max_messages: int = 10) -> List[Dict[str, str]]:
    return history[-max_messages:] if len(history) > max_messages else history

def fetch_x_trends(query: str) -> str:
    logger.info("Fetching X trends for: %s", query)
    # Placeholder for real API integration (e.g., Twitter API)
    return "Recent X posts suggest a rise in smart lock vulnerabilities (placeholder)."

SERVICE_HANDLERS = {
    "grok": get_grok_response,
    "openai": get_openai_response,
    "cohere": get_cohere_response
}

# RAG-specific functions
def embed_text(text: str) -> torch.Tensor:
    processor, model = load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    logger.info("Text embedding generated for: %s", text[:50])
    return outputs.cpu().numpy()[0]

def upsert_data(image_paths: List[str], captions: List[str], timestamps: List[str], batch_size: int = 100) -> None:
    processor, model = load_clip()
    vectors = []
    for img_path, caption, timestamp in zip(image_paths, captions, timestamps):
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                embedding = model.get_image_features(**inputs).cpu().numpy()[0]
            vectors.append({
                "id": str(hash(caption + timestamp)),  # Unique ID with timestamp
                "values": embedding.tolist(),
                "metadata": {"caption": caption, "timestamp": timestamp, "path": img_path}
            })
            if len(vectors) >= batch_size:
                index.upsert(vectors=vectors)
                logger.info("Upserted batch of %d vectors", len(vectors))
                vectors = []
        except Exception as e:
            logger.error("Error processing image %s: %s", img_path, str(e))
    if vectors:
        index.upsert(vectors=vectors)
        logger.info("Upserted final batch of %d vectors", len(vectors))

def retrieve_images(query_embedding: torch.Tensor, top_k: int = 5) -> List[Dict[str, str]]:
    try:
        results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
        return [{"id": match["id"], "path": match["metadata"]["path"]} for match in results["matches"]]
    except Exception as e:
        logger.error("Error retrieving my_images from Pinecone: %s", str(e))
        return []

def get_captions_from_pinecone(image_ids: List[str]) -> List[str]:
    try:
        captions = []
        fetch_result = index.fetch(ids=image_ids)
        for id in image_ids:
            vector = fetch_result["vectors"].get(id)
            captions.append(vector["metadata"]["caption"] if vector and "metadata" in vector else "No caption available")
        return captions
    except Exception as e:
        logger.error("Error fetching captions from Pinecone: %s", str(e))
        return ["Error fetching caption"] * len(image_ids)

def get_response(prompt, service, model, deep_search, conversation_history, config, extra_instructions=""):
    handler = SERVICE_HANDLERS.get(service)
    if not handler:
        raise ValueError(f"Unknown service: {service}")
    if deep_search:
        prompt += f"\nAdditional context: {fetch_x_trends(prompt)}"
    args = {
        "prompt": prompt,
        "model": model,
        "use_deep_search": deep_search,
        "conversation_history": conversation_history,
        "grok_url": config.grok_api_url if service == "grok" else None,
        "grok_headers": config.grok_headers() if service == "grok" else None,
        "openai_client": config.openai_client if service == "openai" else None,
        "co_client": config.co_client if service == "cohere" else None,
        "extra_instructions": extra_instructions
    }
    sig = inspect.signature(handler)
    filtered_args = {k: v for k, v in args.items() if k in sig.parameters}
    return handler(**filtered_args)

def validate_input(user_input: str) -> Tuple[bool, str]:
    if not user_input.strip():
        return False, "Input cannot be empty. Please provide some details."
    if len(user_input) > 500:
        return False, "Input is too long. Please keep it under 500 characters."
    if re.search(r'[<>{}]', user_input):
        return False, "Input contains invalid characters (e.g., <, >, {})."
    return True, ""

def display_response_with_images(user_input: str, reply: str, image_data: List[Dict[str, str]]) -> None:
    root = tk.Tk()
    root.title(f"Security Chatbot - {SERVICE.capitalize()} Response")
    tk.Label(root, text=f"Query: {user_input}", wraplength=400).pack(pady=5)
    tk.Label(root, text=f"{SERVICE.capitalize()} says: {reply}", wraplength=400).pack(pady=5)
    for data in image_data:
        try:
            img = Image.open(data["path"])
            img = img.resize((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            tk.Label(root, image=photo).pack(pady=5)
            tk.Label(root).image = photo  # Prevent garbage collection
        except Exception as e:
            tk.Label(root, text=f"Error loading image {data['path']}: {str(e)}").pack(pady=5)
    tk.Label(root, text="Was this helpful? (Y/N in terminal)").pack(pady=5)
    root.mainloop()

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Chatbot for Physical Security with Surveillance Images")
    parser.add_argument("--service", default=config_yaml.get("default_service", "grok"), choices=["grok", "openai", "cohere"], help="AI service to use")
    parser.add_argument("--model", default=None, help="Model to use (overrides default)")
    return parser.parse_args()

if __name__ == "__main__":
    config = load_config()
    conversation_history = []
    args = parse_args()
    SERVICE = args.service
    DEFAULT_MODELS = {"grok": "grok-2", "openai": "gpt-4o", "cohere": "command-r"}
    MODEL = args.model or DEFAULT_MODELS.get(SERVICE)
    last_response_helpful = None

    # Populate Pinecone with sample data (enhanced with timestamps)
    sample_images = ["path/to/img1.jpg", "path/to/img2.jpg"]  # Replace with real paths
    sample_captions = ["Person at gate", "Car in driveway"]
    sample_timestamps = ["2025-03-19 10:00:00", "2025-03-19 12:00:00"]
    upsert_data(sample_images, sample_captions, sample_timestamps)

    print(f"Starting with {SERVICE.capitalize()} (model: {MODEL})")
    print("This chatbot uses surveillance my_images to assist with physical security queries.")
    while True:
        user_input = input(
            f"[{SERVICE.capitalize()}:{MODEL}] How can I assist you today? (Type 'exit', 'help', 'switch to <service>', 'set model <model>', or 'feedback <Y/N>'): ")
        is_valid, error_msg = validate_input(user_input)
        if not is_valid:
            print(error_msg)
            continue

        if user_input.lower() == "help":
            print_help()
        elif user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input.lower().startswith("switch to "):
            new_service = user_input.lower().replace("switch to ", "").strip()
            if new_service in SERVICE_HANDLERS:
                SERVICE = new_service
                MODEL = DEFAULT_MODELS.get(SERVICE)
                print(f"Switched to {SERVICE.capitalize()} (model: {MODEL})")
            else:
                print(f"Service {new_service} not recognized.")
        elif user_input.lower().startswith("set model "):
            new_model = user_input.lower().replace("set model ", "").strip()
            MODEL = new_model
            print(f"Model set to {MODEL} for {SERVICE.capitalize()}")
        elif user_input.lower().startswith("feedback "):
            feedback = user_input.lower().replace("feedback ", "").strip()
            if feedback in ["y", "n"]:
                last_response_helpful = feedback == "y"
                logger.info("User feedback recorded: %s", "Helpful" if last_response_helpful else "Not helpful")
                print("Feedback recorded. Please ask another question.")
            else:
                print("Please provide feedback as 'Y' or 'N'.")
        else:
            try:
                start_time = time.time()
                query_embedding = embed_text(user_input)
                image_data = retrieve_images(query_embedding)
                image_descriptions = get_captions_from_pinecone([data["id"] for data in image_data])
                extra_instructions = (
                    "You are provided with the following surveillance my_images and their descriptions to help answer the user's question:\n\n" +
                    "\n".join(f"- {desc}" for desc in image_descriptions) +
                    "\n\nPlease use this information to provide a detailed response to the user's question related to security scenarios, events, or alarms."
                )

                conversation_history.append({"role": "user", "content": user_input})
                conversation_history = trim_conversation_history(conversation_history)
                deep_search = "trend" in user_input.lower()
                reply = get_response(user_input, SERVICE, MODEL, deep_search, conversation_history, config, extra_instructions)
                logger.info("Response generated in %.2f seconds", time.time() - start_time)
                print(f"{SERVICE.capitalize()} says: {reply}")
                display_response_with_images(user_input, reply, image_data)
                conversation_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                logger.exception("Error during response retrieval: %s", e)
                print(f"Sorry, something went wrong: {str(e)}. Falling back to text-only response.")
                reply = "Unable to retrieve my_images due to an error. Please try again or refine your query."
                print(f"{SERVICE.capitalize()} says: {reply}")
                conversation_history.append({"role": "assistant", "content": reply})