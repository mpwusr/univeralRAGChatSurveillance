import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load .env
env_path = Path("/.env")
print(f"🔍 Loading environment variables from: {env_path}")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise EnvironmentError("Missing ROBOFLOW_API_KEY in .env file.")

zip_path = Path("/autolabel_export.zip")
if not zip_path.exists():
    raise FileNotFoundError(f"ZIP file not found: {zip_path}")

print(f"📦 Found ZIP file: {zip_path.name} ({zip_path.stat().st_size / 1024**2:.2f} MB)")

# Roboflow API endpoint (correct!)
upload_url = f"https://api.roboflow.com/universalragchat/ragchat-surveillance/upload"

# Upload
print("Uploading ZIP to Roboflow API...")
with open(zip_path, 'rb') as f:
    files = {'file': f}
    data = {
        "api_key": api_key,
        "name": "auto_labeled_data",
        "split": "train"  # or val/test if needed
    }
    response = requests.post(upload_url, data=data, files=files)

if response.ok:
    print("Upload complete!")
    print("Server response:", response.json())
else:
    print("Upload failed.")
    print("Status:", response.status_code)
    print("Error:", response.text)
