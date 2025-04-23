import json
import pandas as pd

from run_inference import detections

# Export path
json_path = "output/predictions.json"
csv_path = "output/predictions.csv"

# Convert detections to dict list
detection_data = [
    {
        "label": det.class_name,
        "confidence": float(det.confidence),
        "x": float(det.xyxy[0]),
        "y": float(det.xyxy[1]),
        "x2": float(det.xyxy[2]),
        "y2": float(det.xyxy[3])
    }
    for det in detections
]

# Save to JSON
with open(json_path, "w") as f:
    json.dump(detection_data, f, indent=4)

# Save to CSV
df = pd.DataFrame(detection_data)
df.to_csv(csv_path, index=False)
