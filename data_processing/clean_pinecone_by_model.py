# clean_pinecone_by_model.py
import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "surveillance-my-images"
index = pc.Index(INDEX_NAME)

# Set which model you want to delete
MODEL_TO_DELETE = "YOLOv3"  # Change to "Mask R-CNN" or "CLIP-BLIP" as needed

# Fetch all vectors matching that model
print(f"Fetching vectors to delete where model = {MODEL_TO_DELETE}...")
query_filter = {"model": {"$eq": MODEL_TO_DELETE}}
result = index.query(vector=[0.0] * 512, top_k=10000, include_metadata=True, filter=query_filter)

ids_to_delete = [match['id'] for match in result.get('matches', [])]
print(f"Found {len(ids_to_delete)} vectors to delete.")

# Delete them
if ids_to_delete:
    index.delete(ids=ids_to_delete)
    print(f"✅ Deleted {len(ids_to_delete)} vectors where model = {MODEL_TO_DELETE}")
else:
    print("No vectors found matching filter.")
