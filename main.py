import os
import cv2
import torch
import torchvision
from yolov7.models.yolo import Model
from yolov7.utils.general import check_img_size
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'yolo_weights': 'runs/train/yolov7_surveillance/weights/best.pt',
    'yolo_cfg': 'cfg/training/yolov7-tiny.yaml',
    'maskrcnn_weights': 'runs/train/maskrcnn_epoch_10.pth',
    'img_size': 640,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'confidence_threshold': 0.5,
    'iou_threshold': 0.4,
    'pinecone_api_key': os.getenv('PINECONE_API_KEY', 'your-pinecone-api-key'),
    'pinecone_index_name': 'surveillance-detections',
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'classes': [
        'person_friendly', 'vehicle_authorized', 'gun', 'knife', 'rifle',
        'mask_unfriendly', 'vehicle_unknown'
    ],
    'host': '0.0.0.0',
    'port': 8000,
}


# Initialize Pinecone and embedding model
def init_pinecone():
    pc = Pinecone(api_key=CONFIG['pinecone_api_key'])
    index_name = CONFIG['pinecone_index_name']

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Matches MiniLM embedding size
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )

    index = pc.Index(index_name)
    embedder = SentenceTransformer(CONFIG['embedding_model'])
    return index, embedder


# Store detection in Pinecone
def store_detection(class_name: str, confidence: float, bbox: List[float], mask_path: str, frame_idx: int, index,
                    embedder):
    timestamp = datetime.now().isoformat()
    # Create text description for embedding
    text = f"{class_name} detected with confidence {confidence:.2f} at frame {frame_idx}"
    if mask_path:
        text += f" with mask saved at {mask_path}"
    vector = embedder.encode(text).tolist()

    # Metadata for Pinecone
    metadata = {
        'timestamp': timestamp,
        'class_name': class_name,
        'confidence': confidence,
        'bbox': ','.join(map(str, bbox)),
        'mask_path': mask_path,
        'video_frame': frame_idx,
        'text': text
    }

    # Upsert to Pinecone
    index.upsert(vectors=[{
        'id': f'det_{timestamp}_{frame_idx}',
        'values': vector,
        'metadata': metadata
    }])


# Query Pinecone for detections
def query_detections(query: str, index, embedder, top_k: int = 10) -> List[Dict]:
    query_vector = embedder.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    detections = [
        {
            'id': match['id'],
            'timestamp': match['metadata']['timestamp'],
            'class_name': match['metadata']['class_name'],
            'confidence': match['metadata']['confidence'],
            'bbox': match['metadata']['bbox'],
            'mask_path': match['metadata']['mask_path'],
            'video_frame': match['metadata']['video_frame'],
            'score': match['score']
        } for match in results['matches']
    ]
    return detections


# Load YOLO model
def load_yolo_model():
    device = torch.device(CONFIG['device'])
    model = Model(CONFIG['yolo_cfg'], ch=3, nc=len(CONFIG['classes'])).to(device)
    ckpt = torch.load(CONFIG['yolo_weights'], map_location=device)
    model.load_state_dict(ckpt if 'model' not in ckpt else ckpt['model'].float().state_dict())
    model.eval()
    return model


# Load Mask R-CNN model
def load_maskrcnn_model():
    device = torch.device(CONFIG['device'])
    model = maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, len(CONFIG['classes']) + 1  # +1 for background
    )
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        256, 256, len(CONFIG['classes']) + 1
    )
    model.load_state_dict(torch.load(CONFIG['maskrcnn_weights'], map_location=device))
    model.to(device)
    model.eval()
    return model


# Process video with YOLO (real-time detection)
def process_video_yolo(video_path: str, model, index, embedder, stride: int = 32):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    detections = []
    img_size = check_img_size(CONFIG['img_size'], s=stride)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Preprocess frame
        img = cv2.resize(frame, (img_size, img_size))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).to(CONFIG['device']).float() / 255.0
        img = img.unsqueeze(0)  # Add batch dimension

        # Inference
        with torch.no_grad():
            pred = model(img)[0]
            from yolov7.utils.general import non_max_suppression
            pred = non_max_suppression(
                pred, CONFIG['confidence_threshold'], CONFIG['iou_threshold'], classes=None, agnostic=False
            )

        # Process detections
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    class_name = CONFIG['classes'][int(cls)]
                    bbox = [float(x) for x in xyxy]
                    # Store detection in Pinecone
                    store_detection(class_name, float(conf), bbox, '', frame_idx, index, embedder)
                    detections.append({
                        'frame': frame_idx,
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': bbox
                    })
                    # Trigger alert for unfriendly markers
                    if class_name in ['gun', 'knife', 'rifle', 'mask_unfriendly', 'vehicle_unknown']:
                        logger.warning(f"Alert: {class_name} detected at frame {frame_idx}")

        # Run Mask R-CNN for unfriendly markers every 10 frames
        if frame_idx % 10 == 0 and any(
                d['class_name'] in ['gun', 'knife', 'rifle', 'mask_unfriendly'] for d in detections):
            maskrcnn_model = load_maskrcnn_model()
            process_frame_maskrcnn(frame, frame_idx, maskrcnn_model, index, embedder)

    cap.release()
    return detections


# Process frame with Mask R-CNN (detailed analysis)
def process_frame_maskrcnn(frame: np.ndarray, frame_idx: int, model, index, embedder):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torchvision.transforms.ToTensor()(img).to(CONFIG['device'])

    with torch.no_grad():
        predictions = model([img])

    for i, pred in enumerate(predictions):
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()

        for j in range(len(boxes)):
            if scores[j] >= CONFIG['confidence_threshold']:
                class_idx = labels[j] - 1  # Subtract 1 for background
                if class_idx < len(CONFIG['classes']):
                    class_name = CONFIG['classes'][class_idx]
                    bbox = boxes[j].tolist()
                    mask = masks[j, 0] > 0.5  # Threshold mask
                    mask_path = f"masks/frame_{frame_idx}_mask_{j}.png"
                    os.makedirs('masks', exist_ok=True)
                    cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
                    store_detection(class_name, scores[j], bbox, mask_path, frame_idx, index, embedder)
                    if class_name in ['gun', 'knife', 'rifle', 'mask_unfriendly', 'vehicle_unknown']:
                        logger.warning(f"Mask R-CNN Alert: {class_name} detected at frame {frame_idx}")


# FastAPI setup
app = FastAPI(title="UniveralRAGChatSurveillance")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    results: List[Dict]


@app.post("/detect", response_model=List[Dict])
async def detect_video(file: UploadFile = File(...)):
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    yolo_model = load_yolo_model()
    index, embedder = init_pinecone()
    detections = process_video_yolo(video_path, yolo_model, index, embedder)
    os.remove(video_path)
    return detections


@app.post("/query", response_model=QueryResponse)
async def query_detections(request: QueryRequest):
    index, embedder = init_pinecone()
    results = query_detections(request.query, index, embedder)
    return {"results": results}


# Chat interface (console-based)
async def chat_interface():
    index, embedder = init_pinecone()
    print("Starting chat interface. Type 'exit' to quit.")
    while True:
        query = input("Enter query (e.g., 'show guns detected'): ")
        if query.lower() == 'exit':
            break
        results = query_detections(query, index, embedder)
        if results:
            for res in results:
                print(
                    f"[{res['timestamp']}] {res['class_name']} (conf: {res['confidence']:.2f}, score: {res['score']:.2f}) at frame {res['video_frame']}")
        else:
            print("No matching detections found.")


# Main entry point
if __name__ == '__main__':
    # Run FastAPI server and chat interface
    asyncio.create_task(uvicorn.run(app, host=CONFIG['host'], port=CONFIG['port']))
    asyncio.run(chat_interface())