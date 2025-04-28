import os
import csv
import json
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "surveillance-my-images"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def fetch_vectors_by_date(date_str):
    """Fetch all Pinecone vectors with metadata.timestamp starting with `date_str`."""
    print(f"📆 Generating summary for {date_str} ...")

    # Get index stats and all vector IDs
    stats = index.describe_index_stats()
    count = stats.get("total_vector_count", 0)
    if count == 0:
        print("⚠️ No vectors found in index.")
        return []

    print(f"📊 Total vector count: {count}")

    # WARNING: this assumes your app tracks all vector IDs or inserts manageable batch sizes.
    # We will use Pinecone's `query()` trick to pull top_k vectors and then filter manually.
    # But here, we’ll do it with dummy queries and skip pagination for now.
    all_matches = []
    next_token = None

    while True:
        query_result = index.query(
            vector=[0.0] * 512,  # dummy vector
            top_k=1000,
            include_metadata=True,
            filter={},  # no filter
            namespace="",
            next=next_token,
        )

        matches = query_result.get("matches", [])
        for match in matches:
            meta = match.get("metadata", {})
            ts = meta.get("timestamp", "")
            if ts.startswith(date_str):
                all_matches.append(meta)

        next_token = query_result.get("next")
        if not next_token:
            break

    return all_matches


def write_report(date_str, results):
    if not results:
        print("⚠️ No records found for the date.")
        return

    base = f"pinecone_summary_{date_str}"
    csv_file = f"{base}.csv"
    json_file = f"{base}.json"

    # Delete existing CSV and JSON files if they exist
    for file in [csv_file, json_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ Deleted existing file: {file}")

    # Gather all unique keys from metadata
    all_keys = set()
    for row in results:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Report written to {csv_file} and {json_file}")


if __name__ == "__main__":
    target_date = datetime.now().strftime("%Y-%m-%d")  # Or hardcode: '2025-04-21'
    records = fetch_vectors_by_date(target_date)
    write_report(target_date, records)