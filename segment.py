import os
import random
import shutil
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# Set device (GPU if available, else CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
model.to(device)

# COCO classes
COCO_INSTANCE_CATEGORY_NAMES = [
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


def get_prediction(img_path, threshold):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])

    masks = prediction[0]['masks'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    pred_cls = [COCO_INSTANCE_CATEGORY_NAMES[labels[i]] for i in range(len(scores)) if scores[i] > threshold]
    masks = masks[scores > threshold]
    boxes = boxes[scores > threshold]

    return masks, boxes, pred_cls


def instance_segmentation_api(img_path, output_dir=None, threshold=0.5, save_output=True):
    masks, boxes, pred_cls = get_prediction(img_path, threshold)

    print(f"\nSegmentation Details for {os.path.basename(img_path)}:")
    print(f"Detected objects: {len(pred_cls)}")
    for i, (cls, box) in enumerate(zip(pred_cls, boxes)):
        print(f"  [{i+1}] Class: {cls} | BBox: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i, (mask, box, cls) in enumerate(zip(masks, boxes, pred_cls)):
        mask = (mask[0] > 0.5).astype(np.uint8)
        color = [random.randint(0, 255) for _ in range(3)]

        for c in range(3):
            img[:, :, c] = np.where(mask == 1, img[:, :, c] * 0.5 + color[c] * 0.5, img[:, :, c])

        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img, cls, (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if save_output:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"output_{os.path.basename(img_path)}")
        else:
            output_path = os.path.join(os.path.dirname(img_path), f"output_{os.path.basename(img_path)}")

        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved output to: {output_path}")
    else:
        cv2.imshow("Segmentation", img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


def main():
    image_dir = "/Users/michaelwilliams/PycharmProjects/MaskRCNNAnalysis-main/my_images"
    output_dir = os.path.join(image_dir, "processed")

    # Ensure the directory exists
    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} not found.")
        return

    # Clean output directory
    if os.path.exists(output_dir):
        print(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get list of images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images.")

    for file_name in image_files:
        img_path = os.path.join(image_dir, file_name)

        # Skip already processed outputs if re-run mid-script
        if file_name.startswith("output_"):
            continue

        try:
            instance_segmentation_api(img_path, output_dir=output_dir, threshold=0.75, save_output=True)
        except Exception as e:
            print(f"Error with {file_name}: {str(e)}")


if __name__ == "__main__":
    main()
