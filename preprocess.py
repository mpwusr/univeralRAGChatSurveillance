
import random
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from dotenv import load_dotenv
import hashlib
import logging
import os
import shutil

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from environment variables")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "surveillance-my-images"
DIMENSION = 512
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
index = pc.Index(INDEX_NAME)

# Initialize CLIP and BLIP
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize Mask R-CNN
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mask_rcnn_model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
mask_rcnn_model.eval()
mask_rcnn_model.to(device)

# Initialize YOLOv3
yolo_weights = "yolov3.weights"
yolo_config = "yolov3.cfg"
yolo_names = "coco.names"
for file_path in [yolo_weights, yolo_config, yolo_names]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required YOLOv3 file missing: {file_path}. Please download from https://pjreddie.com/darknet/yolo/.")
yolo_model = cv2.dnn.readNet(yolo_weights, yolo_config)
yolo_layer_names = yolo_model.getLayerNames()
yolo_output_layers = [yolo_layer_names[i - 1] for i in yolo_model.getUnconnectedOutLayers()]
with open(yolo_names, "r") as f:
    yolo_classes = [line.strip() for line in f.readlines()]

# COCO classes for Mask R-CNN
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def generate_image_embedding(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        # Resize to CLIP's expected size (224x224) to normalize input
        image = image.resize((224, 224))
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        # Convert to NumPy array and ensure correct shape
        embedding = outputs.cpu().numpy()
        logger.info(f"Image embedding type: {type(embedding)}, shape: {embedding.shape}")
        # Handle unexpected shapes
        if embedding.shape != (1, DIMENSION):
            logger.error(f"Unexpected embedding shape for {image_path}: {embedding.shape}")
            return None
        embedding = embedding[0]  # Extract single embedding
        if not isinstance(embedding, np.ndarray):
            logger.error(f"Invalid embedding type for {image_path}: {type(embedding)}")
            return None
        if embedding.shape != (DIMENSION,):
            logger.error(f"Invalid embedding shape for {image_path}: {embedding.shape}")
            return None
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for {image_path}: {str(e)}")
        return None


def empty_directory(dir_path):
    if not os.path.isdir(dir_path):
        print(f" Error: {dir_path} is not a valid directory.")
        return

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory and its contents
            print(f" Removed: {file_path}")
        except Exception as e:
            print(f" Failed to delete {file_path}. Reason: {e}")


def sanitize_embedding(embedding, dim=512):
    try:
        if hasattr(embedding, 'detach'):
            embedding = embedding.detach().cpu().numpy()
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        embedding = embedding.flatten().astype(np.float32)
        if embedding.shape != (dim,):
            logger.error(f"Sanitized embedding has wrong shape: {embedding.shape}")
            return None
        return embedding
    except Exception as e:
        logger.error(f"Embedding sanitization failed: {e}")
        return None

def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        inputs = blip_processor(images=image, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        return blip_processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error generating caption for {image_path}: {str(e)}")
        return "Error generating caption"

def embed_text(text):
    try:
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
        embedding = outputs.cpu().numpy()
        logger.info(f"Text embedding type: {type(embedding)}, shape: {embedding.shape}")
        if embedding.shape != (1, DIMENSION):
            logger.error(f"Unexpected text embedding shape: {embedding.shape}")
            return None
        embedding = embedding[0]
        if not isinstance(embedding, np.ndarray):
            logger.error(f"Invalid text embedding type: {type(embedding)}")
            return None
        if embedding.shape != (DIMENSION,):
            logger.error(f"Invalid text embedding shape: {embedding.shape}")
            return None
        return embedding
    except Exception as e:
        logger.error(f"Error embedding text: {str(e)}")
        return None

def segment_image_yolo(image_path, threshold=0.5, timestamp=None):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(blob)
    outs = yolo_model.forward(yolo_output_layers)

    boxes, pred_cls, scores = [], [], []
    for out in outs:
        for detection in out:
            score = float(detection[5])
            if score > threshold:
                class_id = int(np.argmax(detection[6:]))
                if 0 <= class_id < len(yolo_classes):
                    pred_cls.append(yolo_classes[class_id])
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, x + w, y + h])
                    scores.append(score)

    description = f"YOLOv3: Image contains {', '.join(pred_cls) if pred_cls else 'no objects'} at coordinates {[box for box in boxes]}. Time: {timestamp}."
    return description, boxes, pred_cls, scores

def segment_image_mask_rcnn(image_path, threshold=0.5, timestamp=None):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        prediction = mask_rcnn_model([img_tensor])

    masks = prediction[0]['masks'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    pred_cls, valid_masks, valid_boxes, valid_scores = [], [], [], []
    for i in range(len(scores)):
        if scores[i] > threshold:
            label_id = labels[i]
            if 0 <= label_id < len(COCO_CLASSES):
                pred_cls.append(COCO_CLASSES[label_id])
                valid_masks.append(masks[i])
                valid_boxes.append(boxes[i])
                valid_scores.append(scores[i])

    valid_masks = np.array(valid_masks) if valid_masks else np.empty((0, 1, img.height, img.width))
    valid_boxes = np.array(valid_boxes) if valid_boxes else np.empty((0, 4))
    description = f"Mask R-CNN: Image contains {', '.join(pred_cls) if pred_cls else 'no objects'} at coordinates {[box.tolist() for box in valid_boxes]}. Time: {timestamp}."

    return description, valid_boxes, pred_cls, valid_scores, valid_masks

def upsert_metadata(description, image_path, timestamp, model_name, is_image=False, caption=None):
    embedding = generate_image_embedding(image_path) if is_image else embed_text(description)

    if embedding is None:
        logger.error(f"Skipping upsert for {model_name} due to invalid embedding: {image_path}")
        return False

    # Normalize to NumPy float32 array of shape (512,)
    try:
        logger.info(f"Raw embedding type: {type(embedding)}, shape: {getattr(embedding, 'shape', 'N/A')}")
        if isinstance(embedding, list):
            logger.warning(f"Embedding is a list: {embedding[:5]}... (first 5 elements)")
            embedding = np.array(embedding, dtype=np.float32)
        elif isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy().astype(np.float32)
        elif isinstance(embedding, np.ndarray):
            embedding = embedding.astype(np.float32)
        else:
            raise ValueError(f"Unsupported embedding type: {type(embedding)}")

        if embedding.ndim > 1:
            embedding = embedding.flatten()

        if embedding.shape != (DIMENSION,):
            logger.error(f"Invalid embedding shape for {model_name} at {image_path}: {embedding.shape}")
            return False
    except Exception as e:
        logger.error(f"Embedding conversion error for {image_path}: {e}")
        return False

    # Unique vector ID
    vector_id = f"{model_name}-{hashlib.md5((description + timestamp).encode('utf-8')).hexdigest()}"
    metadata = {
        "description": description,
        "timestamp": timestamp,
        "path": image_path,
        "model": model_name,
        "friendly": not any(obj in description.lower() for obj in ['gun', 'knife', 'weapon'])
    }
    if caption:
        metadata["caption"] = caption

    try:
        logger.info(f"Final embedding type: {type(embedding)}, shape: {embedding.shape}")
        index.upsert([{
            "id": vector_id,
            "values": embedding.tolist(),
            "metadata": metadata
        }])
        print(f"Upserted {model_name} {'image' if is_image else 'metadata'} to Pinecone: {description}")
        return True
    except Exception as e:
        logger.error(f"Error upserting {model_name} metadata for {image_path}: {str(e)}")
        return False

def process_image(image_path, output_dir, threshold=0.5, save_output=True):
    yolo_dir = os.path.join(output_dir, "yolo")
    mask_rcnn_dir = os.path.join(output_dir, "mask_rcnn")
    if save_output:
        os.makedirs(yolo_dir, exist_ok=True)
        os.makedirs(mask_rcnn_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # CLIP/BLIP preprocessing
    caption = generate_caption(image_path)
    if not upsert_metadata(caption, image_path, timestamp, "CLIP-BLIP", is_image=True, caption=caption):
        print(f"Skipping further processing for {os.path.basename(image_path)} due to CLIP-BLIP embedding error")
        return False

    # YOLOv3 processing
    yolo_description, yolo_boxes, yolo_cls, yolo_scores = segment_image_yolo(image_path, threshold, timestamp)
    print(f"\nYOLOv3 Results for {os.path.basename(image_path)}:")
    print(yolo_description)
    upsert_metadata(yolo_description, image_path, timestamp, "YOLOv3")

    if save_output:
        for i, (box, cls) in enumerate(zip(yolo_boxes, yolo_cls)):
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(img_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(img_rgb, cls, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        yolo_output_path = os.path.join(yolo_dir, f"yolo_{os.path.basename(image_path)}")
        cv2.imwrite(yolo_output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved YOLOv3 output to: {yolo_output_path}")

    # Mask R-CNN processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_rcnn_description, mask_rcnn_boxes, mask_rcnn_cls, mask_rcnn_scores, mask_rcnn_masks = segment_image_mask_rcnn(
        image_path, threshold, timestamp)
    print(f"\nMask R-CNN Results for {os.path.basename(image_path)}:")
    print(mask_rcnn_description)
    upsert_metadata(mask_rcnn_description, image_path, timestamp, "Mask R-CNN")

    if save_output:
        for i, (mask, box, cls) in enumerate(zip(mask_rcnn_masks, mask_rcnn_boxes, mask_rcnn_cls)):
            mask = (mask[0] > 0.5).astype(np.uint8)
            color = [random.randint(0, 255) for _ in range(3)]
            for c in range(3):
                img_rgb[:, :, c] = np.where(mask == 1, img_rgb[:, :, c] * 0.5 + color[c] * 0.5, img_rgb[:, :, c])
            cv2.rectangle(img_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(img_rgb, cls, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        mask_rcnn_output_path = os.path.join(mask_rcnn_dir, f"mask_rcnn_{os.path.basename(image_path)}")
        cv2.imwrite(mask_rcnn_output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved Mask R-CNN output to: {mask_rcnn_output_path}")

    return True

def process_video_stream(input_source, output_dir, threshold=0.5, frame_interval=1.0, max_frames=100,
                         max_duration=60.0):
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {input_source}")

    frame_count = 0
    start_time = time.time()

    yolo_dir = os.path.join(output_dir, "yolo")
    mask_rcnn_dir = os.path.join(output_dir, "mask_rcnn")
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(mask_rcnn_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    last_processed = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Video stream ended or failed at frame {frame_count}")
            break

        current_time = time.time()
        if current_time - last_processed < frame_interval:
            continue

        frame_count += 1
        if frame_count > max_frames:
            print(f"Reached maximum frame limit: {max_frames}")
            break
        if current_time - start_time > max_duration:
            print(f"Reached maximum duration: {max_duration} seconds")
            break

        temp_path = os.path.join(tmp_dir, f"temp_frame_{frame_count}.jpg")
        cv2.imwrite(temp_path, frame)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # CLIP/BLIP preprocessing
        caption = generate_caption(temp_path)
        if not upsert_metadata(caption, temp_path, timestamp, "CLIP-BLIP", is_image=True, caption=caption):
            print(f"Skipping further processing for frame {frame_count} due to CLIP-BLIP embedding error")
            try:
                os.remove(temp_path)
                print(f"Removed temporary frame: {temp_path}")
            except Exception as e:
                print(f"Error removing temporary frame {temp_path}: {str(e)}")
            continue

        # YOLOv3
        yolo_description, yolo_boxes, yolo_cls, yolo_scores = segment_image_yolo(temp_path, threshold, timestamp)
        print(f"\nYOLOv3 Frame {frame_count}: {yolo_description}")
        upsert_metadata(yolo_description, temp_path, timestamp, "YOLOv3")

        for i, (box, cls) in enumerate(zip(yolo_boxes, yolo_cls)):
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(img_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(img_rgb, cls, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        yolo_output_path = os.path.join(yolo_dir, f"yolo_frame_{frame_count}.jpg")
        cv2.imwrite(yolo_output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved YOLOv3 frame to: {yolo_output_path}")

        # Mask R-CNN
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_rcnn_description, mask_rcnn_boxes, mask_rcnn_cls, mask_rcnn_scores, mask_rcnn_masks = segment_image_mask_rcnn(
            temp_path, threshold, timestamp)
        print(f"\nMask R-CNN Frame {frame_count}: {mask_rcnn_description}")
        upsert_metadata(mask_rcnn_description, temp_path, timestamp, "Mask R-CNN")

        for i, (mask, box, cls) in enumerate(zip(mask_rcnn_masks, mask_rcnn_boxes, mask_rcnn_cls)):
            mask = (mask[0] > 0.5).astype(np.uint8)
            color = [random.randint(0, 255) for _ in range(3)]
            for c in range(3):
                img_rgb[:, :, c] = np.where(mask == 1, img_rgb[:, :, c] * 0.5 + color[c] * 0.5, img_rgb[:, :, c])
            cv2.rectangle(img_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(img_rgb, cls, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        mask_rcnn_output_path = os.path.join(mask_rcnn_dir, f"mask_rcnn_frame_{frame_count}.jpg")
        cv2.imwrite(mask_rcnn_output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved Mask R-CNN frame to: {mask_rcnn_output_path}")

        last_processed = current_time
        try:
            os.remove(temp_path)
            print(f"Removed temporary frame: {temp_path}")
        except Exception as e:
            print(f"Error removing temporary frame {temp_path}: {str(e)}")

    print(f"Processed {frame_count} frames")
    cap.release()


def copy_first_n_files(src_dir, dst_dir, n=250):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"📁 Created destination directory: {dst_dir}")

    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    files.sort()  # Sort alphabetically; remove if order doesn't matter

    for i, filename in enumerate(files[:n]):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        shutil.copy2(src_file, dst_file)  # copy2 preserves metadata
        print(f"📄 Copied ({i+1}/{n}): {filename}")

def main():
    wpndts_image_dir = "/Users/michaelwilliams/PycharmProjects/RAGChat/data/WPNDTS_3JS_PSVL/wpndts_3js_psvl-1/test/images"
    yolo_image_dir = "/Users/michaelwilliams/PycharmProjects/RAGChat/dataset/images/train"
    created_images = "/Users/michaelwilliams/PycharmProjects/RAGChat/data_processing/images"
    image_dir ="/Users/michaelwilliams/PycharmProjects/RAGChat/surveillance/incoming_images"
    output_dir = "/Users/michaelwilliams/PycharmProjects/RAGChat/surveillance/segmented_images"
    input_source = -1  # -1 for static images, 0 for webcam, "rtsp://your_camera_ip/stream" for RTSP, or "/path/to/video.mp4"
    # Clean input directory completely
    if os.path.exists(image_dir):
        try:
            shutil.rmtree(image_dir)
            print(f"Cleaned output directory: {image_dir}")
        except Exception as e:
            print(f"Error cleaning output directory {image_dir}: {str(e)}")
    os.makedirs(image_dir, exist_ok=True)
    print(f"Created output directory: {image_dir}")

    # Clean output directory completely
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"Cleaned output directory: {output_dir}")
        except Exception as e:
            print(f"Error cleaning output directory {output_dir}: {str(e)}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    copy_first_n_files(wpndts_image_dir, image_dir, 100)
    copy_first_n_files(yolo_image_dir, image_dir, 100)
    copy_first_n_files(created_images, image_dir, 25)
    if input_source == -1:
        if not os.path.exists(image_dir):
            print(f"Image directory {image_dir} does not exist.")
            return

        # Filter out processed files (e.g., yolo_*, mask_rcnn_*)
        image_files = [f for f in os.listdir(image_dir) if
                       f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith(('yolo_', 'mask_rcnn_'))]
        print(f"Found {len(image_files)} images in {image_dir}: {image_files}")

        max_images = 250
        processed_count = 0
        for file_name in image_files:
            if processed_count >= max_images:
                print(f"Reached maximum image limit: {max_images}")
                break
            img_path = os.path.join(image_dir, file_name)
            print(f"\nProcessing image: {img_path}")

            try:
                if process_image(img_path, output_dir, threshold=0.75, save_output=True):
                    processed_count += 1
                else:
                    print(f"Skipping {file_name} due to processing error")
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue
        print(f"Processed {processed_count} images")

    else:
        print(f"\nProcessing video stream: {input_source}")
        try:
            process_video_stream(
                input_source,
                output_dir,
                threshold=0.75,
                frame_interval=1.0,
                max_frames=100,
                max_duration=60.0
            )
        except Exception as e:
            print(f"Error processing video stream: {str(e)}")

if __name__ == "__main__":
    main()