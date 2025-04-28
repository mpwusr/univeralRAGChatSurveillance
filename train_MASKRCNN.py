import json
import logging
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Dataset for COCO-style annotations
class SurveillanceDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = self.coco['annotations']
        self.cat_id_to_label = {cat['id']: i + 1 for i, cat in enumerate(self.coco['categories'])}  # 1-based for Mask R-CNN

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = list(self.images.keys())[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Get annotations
        anns = [ann for ann in self.annotations if ann['image_id'] == img_id]
        boxes = []
        labels = []
        masks = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann['category_id']])
            # Convert segmentation to mask
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for seg in ann['segmentation']:
                coords = np.array(seg).reshape(-1, 2)
                from skimage.draw import polygon
                rr, cc = polygon(coords[:, 1], coords[:, 0], mask.shape)
                mask[rr, cc] = 1
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        image_id = torch.tensor([img_id])
        area = torch.tensor([ann['area'] for ann in anns])
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

# Configuration
cfg = {
    'data_root': 'dataset/images',
    'train_ann': 'dataset/annotations/train.json',
    'val_ann': 'dataset/annotations/val.json',
    'num_classes': 8,  # 7 classes + background
    'batch_size': 4,
    'epochs': 10,
    'lr': 0.005,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'weights': 'maskrcnn_resnet50_fpn_coco.pth',  # Pre-trained weights
    'log_dir': 'runs/train/maskrcnn',
}

def get_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.ToTensor()])

def main():
    # Create log directory
    os.makedirs(cfg['log_dir'], exist_ok=True)

    # Datasets
    try:
        train_dataset = SurveillanceDataset(
            root=os.path.join(cfg['data_root'], 'train'),
            annotation_file=cfg['train_ann'],
            transforms=get_transform()
        )
        val_dataset = SurveillanceDataset(
            root=os.path.join(cfg['data_root'], 'val'),
            annotation_file=cfg['val_ann'],
            transforms=get_transform()
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset files not found: {e}")
        return

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, cfg['num_classes']
    )
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        256, 256, cfg['num_classes']
    )
    model.to(cfg['device'])

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg['lr'],
        momentum=0.9,
        weight_decay=0.0005
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = [image.to(cfg['device']) for image in images]
            targets = [{k: v.to(cfg['device']) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        lr_scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(cfg['device']) for image in images]
                targets = [{k: v.to(cfg['device']) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()

        # Log and save model
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'Epoch {epoch+1}/{cfg["epochs"]}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        torch.save(model.state_dict(), f'{cfg["log_dir"]}/maskrcnn_epoch_{epoch+1}.pth')

    logger.info('Training complete.')

if __name__ == '__main__':
    main()