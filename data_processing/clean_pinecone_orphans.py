# clean_pinecone_orphans.py
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

print(f"Scanning index {INDEX_NAME} for orphaned entries...")

# Query Pinecone for all entries
result = index.query(vector=[0.0] * 512, top_k=10000, include_metadata=True)

orphans = []
for match in result.get('matches', []):
    metadata = match.get("metadata", {})
    if not metadata.get("path") or not metadata.get("description"):
        orphans.append(match['id'])

print(f"Found {len(orphans)} orphaned vectors with missing metadata.")

# Delete them
if orphans:
    index.delete(ids=orphans)
    print(f"✅ Deleted {len(orphans)} orphaned vectors.")
else:
    print("No orphaned entries found.")
