import os
import cv2
import torch
from pathlib import Path
import numpy as np
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.models.experimental import Ensemble

# Register YOLOModel for torch unpickling
try:
    import torch.serialization
    from yolov7.models.yolo import Model as YOLOModel
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals({"models.yolo.Model": YOLOModel})
except Exception as e:
    print(f"⚠️ Could not register YOLOModel for torch safe unpickling: {e}")

# Configuration
model_path = Path("../yolov7-tiny.pt")  # Update path if needed
image_dir = Path("../data/WPNDTS_3JS_PSVL/wpndts_3js_psvl-1/test")
output_dir = Path("../surveillance/segmented_images/")
gun_class_ids = [0]  # Use the correct index for 'gun' from dataset.yaml (0 if single class)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load model manually to avoid import issues
print("🔄 Loading model...")
model = Ensemble()
ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())
model = model[-1].to(device)

stride = int(model.stride.max())
names = model.names if hasattr(model, 'names') else [str(i) for i in range(model.model[-1].nc)]
output_dir.mkdir(parents=True, exist_ok=True)

# Load images
image_paths = sorted(image_dir.glob("*.[jp][pn]g"))
print(f"📂 Found {len(image_paths)} image(s) in {image_dir}")

for img_path in image_paths:
    print(f"🔍 Processing: {img_path.name}")
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        print(f"⚠️ Could not read {img_path.name}")
        continue

    img = letterbox(img0, 640, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    det = pred[0]
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in det:
            cls_id = int(cls.item())
            if cls_id in gun_class_ids:
                label = f"{names[cls_id]} {conf:.2f}"
                xyxy = [int(x.item()) for x in xyxy]
                cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                cv2.putText(img0, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), img0)
    print(f"✅ Saved: {out_path}")

print("🎉 Visualization complete.")
