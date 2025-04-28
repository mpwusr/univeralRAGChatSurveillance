import json
import logging
import os
import re
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Tuple

import cohere
import requests
import torch
from PIL import Image, ImageTk
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import CLIPProcessor, CLIPModel
from langchain.llms import HuggingFaceTextGenInference, LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Logging setup
handler = RotatingFileHandler("chatbot.log", maxBytes=10 * 1024 * 1024, backupCount=5)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[handler])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")  # For Hugging Face Inference API
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # For local Ollama server
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
    openai_api_key: str
    cohere_api_key: str
    xai_api_key: str
    hf_api_key: str
    ollama_host: str
    xai_api_url: str = "https://api.x.ai/v1/chat/completions"
    default_service: str = "grok"
    default_model: str = "grok-2"

    def __post_init__(self):
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.cohere_client = cohere.Client(self.cohere_api_key)
        self.hf_client = HuggingFaceTextGenInference(
            inference_server_url="https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
            api_key=self.hf_api_key
        ) if self.hf_api_key else None
        self.ollama_client = LlamaCpp(
            model_path=None,  # Set to local path if using local model, e.g., "/path/to/llama-2-7b.Q4_0.gguf"
            n_ctx=3900,
            max_tokens=256,
            temperature=0.1,
            model_kwargs={"n_gpu_layers": 1}  # Use GPU if available
        ) if os.path.exists(self.ollama_host) else None

    def xai_headers(self):
        return {"Authorization": f"Bearer {self.xai_api_key}", "Content-Type": "application/json"}

def load_config():
    if not all([OPENAI_API_KEY, COHERE_API_KEY, XAI_API_KEY]):
        raise ValueError("Missing one or more required API keys in environment")
    return Config(
        openai_api_key=OPENAI_API_KEY,
        cohere_api_key=COHERE_API_KEY,
        xai_api_key=XAI_API_KEY,
        hf_api_key=HF_API_KEY,
        ollama_host=OLLAMA_HOST
    )

# LLM Handler Class
class LLMHandler:
    def __init__(self, config: Config):
        self.config = config
        self.handlers = {
            "openai": self.get_openai_response,
            "cohere": self.get_cohere_response,
            "grok": self.get_grok_response,
            "huggingface": self.get_huggingface_response,
            "ollama": self.get_ollama_response
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_grok_response(self, prompt, model, conversation_history=None, extra_instructions=""):
        full_prompt = build_prompt("physical security consultant", prompt, conversation_history, extra_instructions)
        payload = {"model": model, "messages": [{"role": "user", "content": full_prompt}], "max_tokens": 300}
        resp = requests.post(self.config.xai_api_url, headers=self.config.xai_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def get_openai_response(self, prompt, model="gpt-4o", conversation_history=None, extra_instructions=""):
        full_prompt = build_prompt("physical security consultant", prompt, conversation_history, extra_instructions)
        messages = [{"role": "user", "content": full_prompt}] if not conversation_history else \
            [{"role": msg["role"], "content": msg["content"]} for msg in conversation_history] + \
            [{"role": "user", "content": full_prompt}]
        try:
            resp = self.config.openai_client.chat.completions.create(model=model, messages=messages, max_tokens=300)
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def get_cohere_response(self, prompt, model="command-r", conversation_history=None, extra_instructions=""):
        base_prompt = build_prompt("physical security consultant", "", conversation_history, extra_instructions)
        chat_history = [{"role": "User" if msg["role"] == "user" else "Chatbot", "message": msg["content"]}
                        for msg in conversation_history] if conversation_history else []
        try:
            resp = self.config.cohere_client.chat(
                message=prompt,
                preamble=base_prompt,
                chat_history=chat_history,
                model=model,
                max_tokens=300,
                temperature=0.7
            )
            return resp.text
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            raise

    def get_huggingface_response(self, prompt, model="meta-llama/Llama-2-7b-chat-hf", conversation_history=None,
                                 extra_instructions=""):
        if not self.config.hf_client:
            raise ValueError("Hugging Face client not initialized. Check HF_API_KEY.")
        full_prompt = build_prompt("physical security consultant", prompt, conversation_history, extra_instructions)
        try:
            response = self.config.hf_client.invoke(full_prompt)  # <--- FIXED
            return response
        except Exception as e:
            logger.error(f"Hugging Face API error: {e}")
            raise

    def get_ollama_response(self, prompt, model="llama2", conversation_history=None, extra_instructions=""):
        if not self.config.ollama_client:
            raise ValueError("Ollama client not initialized. Check OLLAMA_HOST or local model path.")
        template = PromptTemplate(
            input_variables=["history", "prompt"],
            template="{history}\nYou are a physical security consultant. {extra_instructions}\n\nUser's question: {prompt}"
        )
        chain = LLMChain(llm=self.config.ollama_client, prompt=template)
        history = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history]) if conversation_history else ""
        try:
            response = chain.run(history=history, prompt=prompt, extra_instructions=extra_instructions)
            return response
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    def get_response(self, service, prompt, model, conversation_history=None, extra_instructions=""):
        handler = self.handlers.get(service)
        if not handler:
            raise ValueError(f"Service {service} not supported")
        return handler(prompt, model, conversation_history, extra_instructions)

# Existing functions (unchanged)
def load_daily_summary_json(date_str=None):
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"pinecone_summary_{date_str}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load daily summary: {e}")
            return []
    else:
        logger.info(f"No daily summary found for {filename}")
        return []

def summarize_recent_detections():
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"pinecone_summary_{date_str}.json"
        if not os.path.exists(filename):
            return "No JSON summary found for today's preprocessing run."
        with open(filename, "r") as f:
            data = json.load(f)
        if not data:
            return "No detections recorded in today's summary."
        unfriendly_detections = [
            f"{entry.get('model', 'unknown')}: {entry.get('description', 'No description')}"
            for entry in data if not entry.get("friendly", True)
        ]
        image_paths = {entry.get("path") for entry in data}
        image_count = len(image_paths)
        if unfriendly_detections:
            return f"Today's preprocessing ({image_count} images): {len(unfriendly_detections)} unfriendly detections:\n" + \
                   "\n".join(unfriendly_detections[:10])
        else:
            return f"Today's preprocessing ({image_count} images): No unfriendly detections found."
    except Exception as e:
        logger.error(f"Failed to load or summarize JSON summary: {e}")
        return "Error loading today's preprocessing summary."

def build_prompt(base_role: str, prompt: str, conversation_history: List[Dict[str, str]] = None,
                 extra_instructions: str = "") -> str:
    base_prompt = f"You are a {base_role}. {extra_instructions}"
    history = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history]) if conversation_history else ""
    return f"{history}\n{base_prompt}\n\nUser's question: {prompt}"

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
                tk.Label(root, text=f"Failed to load image: {path} error {e}").pack()
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

def main():
    config = load_config()
    llm_handler = LLMHandler(config)
    llm_service= config.default_service
    llm_model = config.default_model
    conversation_history = []

    print("\nSummary of Recent Preprocessing Run:")
    print(summarize_recent_detections())
    print("\n")

    # 🧠 List available LLMs
    print("\n🧠 Available LLM Services:")
    print(" - grok (model: grok-2)")
    print(" - openai (model: gpt-4o)")
    print(" - cohere (model: command-r)")
    print(" - huggingface (model: meta-llama/Llama-2-7b-chat-hf)")
    print(" - ollama (model: llama2)")
    print("-" * 60)

    print(f"Starting with {llm_service.capitalize()} (model: {llm_model})")
    print("This chatbot uses surveillance images to assist with physical security queries.")
    while True:
        user_input = input(
            f"[{llm_service.capitalize()}:{llm_model}] How can I assist you today? (Type 'exit', 'help', 'switch to <service>', 'set model <model>', or 'feedback <Y/N>'): ")
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
            if new_service in llm_handler.handlers:
                llm_service = new_service
                llm_model = {
                    "grok": "grok-2",
                    "openai": "gpt-4o",
                    "cohere": "command-r",
                    "huggingface": "meta-llama/Llama-2-7b-chat-hf",
                    "ollama": "llama2"
                }.get(new_service)
                print(f"Switched to {llm_service.capitalize()} (model: {llm_model})")
            else:
                print(f"Service {new_service} not recognized.")
        elif user_input.lower().startswith("set model "):
            llm_model = user_input.lower().replace("set model ", "").strip()
            print(f"Model set to {llm_model} for {llm_service.capitalize()}")
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
                response = llm_handler.get_response(
                    service=llm_service,
                    prompt=user_input,
                    model=llm_model,
                    conversation_history=conversation_history,
                    extra_instructions=extra
                )
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
                conversation_history = trim_conversation_history(conversation_history)
                logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
                print(f"{llm_service.capitalize()} says: {response}")
                display_gui(user_input, response, matches)
            except Exception as e:
                logger.error(f"Error: {e}")
                print("An error occurred. Please try again.")

if __name__ == "__main__":
    main()