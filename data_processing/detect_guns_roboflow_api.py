import os
import cv2
from pathlib import Path
from inference_sdk import InferenceHTTPClient

# Roboflow credentials
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file after code reset
env_path = Path("/.env")
load_dotenv(dotenv_path=env_path)

# Check if ROBOFLOW_API_KEY is available
api_key = os.getenv("ROBOFLOW_API_KEY")
api_key if api_key else "❌ API key not found."
model_id = "wpndts_3js_psvl/2"

# Initialize client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# Directories
input_dir = Path("../data/WPNDTS_3JS_PSVL/wpndts_3js_psvl-1/train/images")  # Change to 'val' if needed
output_dir = Path("surveillance/segmented_images/wpndts_3js_psv1-1/trained")
output_dir.mkdir(parents=True, exist_ok=True)

# Process images
image_paths = sorted(input_dir.glob("*.[jp][pn]g"))
print(f"📂 Found {len(image_paths)} images.")

for img_path in image_paths:
    print(f"🔍 Processing {img_path.name}")
    result = CLIENT.infer(str(img_path), model_id=model_id)

    # Load original image
    img = cv2.imread(str(img_path))
    height, width = img.shape[:2]

    # Draw predictions
    for prediction in result['predictions']:
        class_name = prediction['class']
        conf = float(prediction['confidence'])
        if conf < 0.25:
            continue  # Filter low confidence

        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        w = int(prediction['width'])
        h = int(prediction['height'])

        color = (0, 0, 255)
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save result
    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"✅ Saved to: {out_path}")

print("🎯 Roboflow inference and visualization complete.")
