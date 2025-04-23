import os
import json
from sklearn.model_selection import train_test_split
import shutil
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'dataset_path': '/Users/michaelwilliams/PycharmProjects/RAGChat/roboflow_dataset',
    'output_dir': '/Users/michaelwilliams/PycharmProjects/RAGChat/dataset',
    'classes': [
        'person_friendly', 'vehicle_authorized', 'gun', 'knife', 'rifle',
        'mask_unfriendly', 'vehicle_unknown'
    ],
    'train_split': 0.8,  # 80% train, 20% val
}

def load_coco_annotations(json_path):
    """Load COCO JSON annotations."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Annotation file not found: {json_path}")
        raise

def save_coco_annotations(data, output_path):
    """Save COCO JSON annotations."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved annotations to {output_path}")

def adapt_dataset(dataset_path):
    """Adapt Roboflow dataset to match required classes and structure."""
    # Load annotations
    train_json_path = os.path.join(dataset_path, 'train', '_annotations.coco.json')
    coco_data = load_coco_annotations(train_json_path)

    # Define category mapping
    category_map = {
        'handgun': 3,  # gun
        'pistol': 3,   # gun
        'rifle': 5,    # rifle
        'shotgun': 5   # rifle
    }
    new_categories = [
        {"id": i + 1, "name": name} for i, name in enumerate(CONFIG['classes'])
    ]

    # Update annotations
    new_annotations = []
    for ann in coco_data['annotations']:
        old_cat_id = ann['category_id']
        old_cat_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == old_cat_id)
        if old_cat_name in category_map:
            ann['category_id'] = category_map[old_cat_name]
            new_annotations.append(ann)

    # Split images into train/val
    images = coco_data['images']
    train_images, val_images = train_test_split(
        images, train_size=CONFIG['train_split'], random_state=42
    )
    train_image_ids = set(img['id'] for img in train_images)
    val_image_ids = set(img['id'] for img in val_images)

    train_annotations = [ann for ann in new_annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in new_annotations if ann['image_id'] in val_image_ids]

    # Create train and val JSONs
    train_coco = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': new_categories
    }
    val_coco = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': new_categories
    }

    # Save annotations
    save_coco_annotations(train_coco, os.path.join(CONFIG['output_dir'], 'annotations', 'train.json'))
    save_coco_annotations(val_coco, os.path.join(CONFIG['output_dir'], 'annotations', 'val.json'))

    # Copy images
    for split, image_list in [('train', train_images), ('val', val_images)]:
        split_dir = os.path.join(CONFIG['output_dir'], 'images', split)
        os.makedirs(split_dir, exist_ok=True)
        for img in image_list:
            src_path = os.path.join(dataset_path, 'train', img['file_name'])
            dst_path = os.path.join(split_dir, img['file_name'])
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                logger.warning(f"Image not found: {src_path}")

def convert_to_yolo_format(json_path, images_dir, output_dir):
    """Convert COCO annotations to YOLO format for train_yolo.py."""
    coco = load_coco_annotations(json_path)
    os.makedirs(output_dir, exist_ok=True)

    for img in coco['images']:
        img_path = os.path.join(images_dir, img['file_name'])
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except FileNotFoundError:
            logger.warning(f"Image not found for YOLO conversion: {img_path}")
            continue

        label_path = os.path.join(output_dir, img['file_name'].replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            for ann in [a for a in coco['annotations'] if a['image_id'] == img['id']]:
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                class_id = ann['category_id'] - 1  # 0-based for YOLO
                f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

    # Create dataset.yaml
    yaml_content = f"""
train: {os.path.join(CONFIG['output_dir'], 'images/train')}
val: {os.path.join(CONFIG['output_dir'], 'images/val')}
nc: {len(CONFIG['classes'])}
names: {CONFIG['classes']}
"""
    with open(os.path.join(CONFIG['output_dir'], 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)
    logger.info(f"Created dataset.yaml at {CONFIG['output_dir']}/dataset.yaml")

def main():
    # Verify dataset path
    dataset_path = CONFIG['dataset_path']
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Adapt dataset for Mask R-CNN
    adapt_dataset(dataset_path)

    # Convert to YOLO format
    convert_to_yolo_format(
        json_path=os.path.join(CONFIG['output_dir'], 'annotations', 'train.json'),
        images_dir=os.path.join(CONFIG['output_dir'], 'images', 'train'),
        output_dir=os.path.join(CONFIG['output_dir'], 'labels', 'train')
    )
    convert_to_yolo_format(
        json_path=os.path.join(CONFIG['output_dir'], 'annotations', 'val.json'),
        images_dir=os.path.join(CONFIG['output_dir'], 'images', 'val'),
        output_dir=os.path.join(CONFIG['output_dir'], 'labels', 'val')
    )

    logger.info(f"Dataset integrated at {CONFIG['output_dir']}. Run train_maskrcnn.py and train_yolo.py to train models.")

if __name__ == '__main__':
    main()
