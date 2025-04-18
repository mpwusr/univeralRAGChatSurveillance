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
from datetime import datetime

# Logging setup
handler = RotatingFileHandler("chatbot.log", maxBytes=10 * 1024 * 1024, backupCount=5)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[handler])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from environment variables")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "surveillance-my-images"
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

# Initialize CLIP
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


def summarize_recent_detections(top_k: int = 10):
    """Summarize unfriendly detections from the most recent preprocessing run."""
    try:
        # Query Pinecone for recent entries (using a neutral vector to fetch latest metadata)
        neutral_vector = [0.0] * DIMENSION
        matches = index.query(vector=neutral_vector, top_k=top_k, include_metadata=True)["matches"]

        if not matches:
            return "No recent preprocessing data found in Pinecone."

        # Sort by timestamp to get the most recent entries
        matches = sorted(
            matches,
            key=lambda x: datetime.strptime(x["metadata"].get("timestamp", "1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S"),
            reverse=True
        )

        # Group by image path to identify the latest preprocessing run
        recent_images = {}
        for match in matches:
            meta = match["metadata"]
            path = meta.get("path")
            if path not in recent_images:
                recent_images[path] = []
            recent_images[path].append(meta)

        # Summarize unfriendly detections
        unfriendly_detections = []
        for path, metas in recent_images.items():
            for meta in metas:
                if not meta.get("friendly", True):
                    model = meta.get("model", "unknown")
                    description = meta.get("description", "unknown")
                    unfriendly_detections.append(f"{model}: {description}")

        image_count = len(recent_images)
        if unfriendly_detections:
            return f"Latest preprocessing run ({image_count} images): {len(unfriendly_detections)} unfriendly detections:\n" + "\n".join(
                unfriendly_detections)
        else:
            return f"Latest preprocessing run ({image_count} images): No unfriendly detections found."
    except Exception as e:
        logger.error(f"Error summarizing recent detections: {e}")
        return "Error summarizing recent detections."


def build_prompt(base_role: str, prompt: str, conversation_history: List[Dict[str, str]] = None,
                 extra_instructions: str = "") -> str:
    base_prompt = f"You are a {base_role}. {extra_instructions}"
    history = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history]) if conversation_history else ""
    return f"{history}\n{base_prompt}\n\nUser's question: {prompt}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_grok_response(prompt, model, use_deep_search=False, conversation_history=None, grok_url=None, grok_headers=None,
                      extra_instructions=""):
    full_prompt = build_prompt("physical security consultant", prompt, conversation_history, extra_instructions)
    payload = {"model": model, "messages": [{"role": "user", "content": full_prompt}], "max_tokens": 300}
    resp: Response = requests.post(grok_url, headers=grok_headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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
        logger.error(f"OpenAI API error: {e}")
        raise


def get_cohere_response(prompt, model="command-r", conversation_history=None, co_client=None, extra_instructions=""):
    if co_client is None:
        raise ValueError("Cohere client must be provided")
    base_prompt = build_prompt("physical security consultant", "", conversation_history, extra_instructions)
    chat_history = [{"role": "User" if msg["role"] == "user" else "Chatbot", "message": msg["content"]}
                    for msg in conversation_history] if conversation_history else []
    try:
        resp_co = co_client.chat(message=prompt, preamble=base_prompt, chat_history=chat_history, model=model,
                                 max_tokens=300, temperature=0.7)
        return resp_co.text
    except Exception as e:
        logger.error(f"Cohere API error: {e}")
        raise


def trim_conversation_history(history: List[Dict[str, str]], max_messages: int = 10) -> List[Dict[str, str]]:
    return history[-max_messages:] if len(history) > max_messages else history


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
        model = meta.get("model", "unknown")
        description = meta.get("description", "unknown")
        caption = meta.get("caption", "none")
        friendly = "Friendly" if meta.get("friendly", True) else "Unfriendly"
        summaries.append(f"- {model}: {description} (Caption: {caption}, {friendly})")
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
                model = match.get("metadata", {}).get("model", "unknown")
                friendly = "Friendly" if match.get("metadata", {}).get("friendly", True) else "Unfriendly"
                summary = f"{model} - {friendly}"

                img_label = tk.Label(root, image=photo)
                img_label.image = photo
                img_label.pack(pady=5)
                tk.Label(root, text=summary).pack()
            except Exception as e:
                tk.Label(root, text=f"Failed to load image: {path}").pack()

    tk.Label(root, text="Was this helpful? (Y/N)").pack(pady=10)
    root.mainloop()


def validate_input(user_input: str) -> Tuple[bool, str]:
    if not user_input.strip():
        return False, "Input cannot be empty. Please provide some details."
    if len(user_input) > 500:
        return False, "Input is too long. Please keep it under 500 characters."
    if re.search(r'[<>{}]', user_input):
        return False, "Input contains invalid characters (e.g., <, >, {})."
    return True, ""


SERVICE_HANDLERS = {
    "grok": get_grok_response,
    "openai": get_openai_response,
    "cohere": get_cohere_response
}


def main():
    config = load_config()
    SERVICE = config.default_service
    MODEL = config.default_model
    conversation_history = []

    # Print summary of recent detections at startup
    print("\nSummary of Recent Preprocessing Run:")
    print(summarize_recent_detections(top_k=10))
    print("\n")

    print(f"Starting with {SERVICE.capitalize()} (model: {MODEL})")
    print("This chatbot uses surveillance images to assist with physical security queries.")
    while True:
        user_input = input(
            f"[{SERVICE.capitalize()}:{MODEL}] How can I assist you today? (Type 'exit', 'help', 'switch to <service>', 'set model <model>', or 'feedback <Y/N>'): ")
        is_valid, error_msg = validate_input(user_input)
        if not is_valid:
            print(error_msg)
            continue

        if user_input.lower() == "help":
            print(
                "Try asking about suspicious activities, alarms, or trends. Commands: 'help', 'exit', 'switch to <service>', 'set model <model>', 'feedback <Y/N>'.")
        elif user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input.lower().startswith("switch to "):
            new_service = user_input.lower().replace("switch to ", "").strip()
            if new_service in SERVICE_HANDLERS:
                SERVICE = new_service
                MODEL = {"grok": "grok-2", "openai": "gpt-4o", "cohere": "command-r"}.get(new_service)
                print(f"Switched to {SERVICE.capitalize()} (model: {MODEL})")
            else:
                print(f"Service {new_service} not recognized.")
        elif user_input.lower().startswith("set model "):
            MODEL = user_input.lower().replace("set model ", "").strip()
            print(f"Model set to {MODEL} for {SERVICE.capitalize()}")
        elif user_input.lower().startswith("feedback "):
            feedback = user_input.lower().replace("feedback ", "").strip()
            if feedback in ["y", "n"]:
                logger.info(f"User feedback recorded: {'Helpful' if feedback == 'y' else 'Not helpful'}")
                print("Feedback recorded. Please ask another question.")
            else:
                print("Please provide feedback as 'Y' or 'N'.")
        else:
            try:
                start_time = time.time()
                query_embedding = embed_text(user_input)
                matches = retrieve_matches(query_embedding)
                summary = summarize_segments(matches)
                extra = f"Use the following scene elements from YOLOv3, Mask R-CNN, and CLIP/BLIP:\n\n{summary}\n\n"
                response = SERVICE_HANDLERS[SERVICE](
                    prompt=user_input,
                    model=MODEL,
                    use_deep_search="trend" in user_input.lower(),
                    conversation_history=conversation_history,
                    grok_url=config.grok_api_url if SERVICE == "grok" else None,
                    grok_headers=config.grok_headers() if SERVICE == "grok" else None,
                    openai_client=config.openai_client if SERVICE == "openai" else None,
                    co_client=config.co_client if SERVICE == "cohere" else None,
                    extra_instructions=extra
                )
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
                conversation_history = trim_conversation_history(conversation_history)
                logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
                print(f"{SERVICE.capitalize()} says: {response}")
                display_gui(user_input, response, matches)
            except Exception as e:
                logger.error(f"Error: {e}")
                print("An error occurred. Please try again.")


if __name__ == "__main__":
    main()