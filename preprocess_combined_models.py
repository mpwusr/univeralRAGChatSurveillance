import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from pinecone import Pinecone
from transformers import CLIPProcessor, CLIPModel

# Load environment variables
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("Missing ROBOFLOW_API_KEY in .env")
if not pinecone_key:
    raise ValueError("Missing PINECONE_API_KEY in .env")

# Constants
#MODEL_ID = "wpndts_3js_psvl/2"
MODEL_ID = "ragchat-nonfriendly/3"
INDEX_NAME = "surveillance-my-images"
DIMENSION = 512
CONFIDENCE_THRESHOLD = 0.25
SUSPICIOUS_LABELS = {"gun", "pistol", "rifle", "revolver", "handgun", "knife", "weapon"}

# Initialize clients
rf_client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index(INDEX_NAME)

# Initialize CLIP
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# Helpers
def generate_vector_id(path: str, label: str, timestamp: str):
    return hashlib.md5(f"{path}-{label}-{timestamp}".encode()).hexdigest()

def generate_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().flatten().astype(np.float32)
        if np.all(embedding == 0) or embedding.shape[0] != DIMENSION:
            return None
        return embedding
    except Exception as e:
        print(f"Error generating embedding for {image_path}: {e}")
        return None

def already_upserted(image_path, label, timestamp):
    vector_id = generate_vector_id(str(image_path), label, timestamp)
    try:
        res = index.fetch(ids=[vector_id])
        return vector_id in res.vectors
    except Exception:
        return False

# Main detection function
def run_roboflow_detection(input_dir: str, output_dir: str, overwrite: bool = False, draw_only: bool = False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(input_dir.rglob("*.[jp][pn]g"))
    print(f"📂 Found {len(image_paths)} images in {input_dir}")

    summary_buffer = []

    for img_path in image_paths:
        timestamp = datetime.fromtimestamp(img_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"🔍 Processing {img_path}")
        result = rf_client.infer(str(img_path), model_id=MODEL_ID)
        img = cv2.imread(str(img_path))

        for prediction in result.get("predictions", []):
            label = prediction["class"]
            conf = float(prediction["confidence"])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x = int(prediction['x'] - prediction['width'] / 2)
            y = int(prediction['y'] - prediction['height'] / 2)
            w = int(prediction['width'])
            h = int(prediction['height'])
            vector_id = generate_vector_id(str(img_path), label, timestamp)

            # Draw box and label
            color = (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not draw_only and (overwrite or not already_upserted(img_path, label, timestamp)):
                embedding = generate_image_embedding(img_path)
                if embedding is None:
                    print(f"Skipping upsert for {img_path.name}, embedding failed or all zeros.")
                    continue
                metadata = {
                    "model": "Roboflow-WPNDTS",
                    "description": f"{label} {conf:.2f} at ({x},{y},{w},{h})",
                    "path": str(img_path),
                    "timestamp": timestamp,
                    "friendly": label.lower() not in SUSPICIOUS_LABELS
                }
                try:
                    index.upsert([{"id": vector_id, "values": embedding.tolist(), "metadata": metadata}])
                    print(f"🧠 Upserted to Pinecone: {metadata['description']}")
                    summary_buffer.append({
                        "id": vector_id,
                        "label": label,
                        "confidence": conf,
                        "image": str(img_path),
                        "timestamp": timestamp,
                        "friendly": metadata["friendly"]
                    })
                except Exception as e:
                    print(f"Pinecone upsert failed: {e}")

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"Saved annotated image: {out_path}")

    # Final summary output
    if not draw_only and summary_buffer:
        date_tag = datetime.now().strftime("%Y-%m-%d")
        csv_path = output_dir / f"pinecone_upsert_summary_{date_tag}.csv"
        json_path = output_dir / f"pinecone_upsert_summary_{date_tag}.json"

        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = summary_buffer[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_buffer)

        with open(json_path, "w") as jsonfile:
            json.dump(summary_buffer, jsonfile, indent=2)

        print(f"📝 Summary saved: {csv_path} and {json_path}")

    print("Roboflow detection complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Roboflow gun detection on existing images.")
    parser.add_argument("--input", default="surveillance/segmented_images/yolo", help="Directory with images to process")
    parser.add_argument("--output", default="surveillance/cleared_images", help="Output directory for annotated images")
    parser.add_argument("--overwrite", action="store_true", help="Force reprocessing of previously upserted images")
    parser.add_argument("--draw-only", action="store_true", help="Only draw boxes, skip upsert to Pinecone")
    args = parser.parse_args()

    run_roboflow_detection(args.input, args.output, args.overwrite, args.draw_only)
