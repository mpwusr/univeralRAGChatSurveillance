import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import hashlib

# Initialize Pinecone and CLIP
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from environment variables")
pc = Pinecone(api_key=PINECONE_API_KEY)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "surveillance-my_images"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs.cpu().detach().numpy()[0]

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    return blip_processor.decode(outputs[0], skip_special_tokens=True)

image_dir = "/surveillance/my_images"  # Update this path
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith((".jpg", ".png"))]

for image_path in image_paths:
    embedding = generate_image_embedding(image_path)
    caption = generate_caption(image_path)
    vector_id = hashlib.md5(image_path.encode('utf-8')).hexdigest()
    index.upsert([(vector_id, embedding, {"path": image_path, "caption": caption})])
    print(f"Processed and stored: {image_path}")

print("Preprocessing complete.")