import os
import yaml
import argparse
import torch

from yolov7.models.yolo import Model
from yolov7.train import train
from yolov7.utils.datasets import LoadImagesAndLabels
from yolov7.utils.general import check_file, check_img_size
from yolov7.utils.torch_utils import select_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov7/cfg/training/yolov7-tiny.yaml', help='Path to model config .yaml')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='Path to dataset.yaml')
    parser.add_argument('--weights', type=str, default='', help='Pretrained weights path')
    parser.add_argument('--img', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='', help='Device ID (e.g., 0 or "cpu")')
    parser.add_argument('--name', type=str, default='exp', help='Training run name')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='Hyperparameter YAML')
    return parser.parse_args()

def main():
    opt = parse_args()

    # Resolve absolute paths
    cfg_path = os.path.abspath(opt.cfg)
    data_path = os.path.abspath(opt.data)
    hyp_path = os.path.abspath(opt.hyp)

    device = select_device(opt.device)
    img_size = check_img_size(opt.img, s=32)

    # Load dataset YAML
    with open(check_file(data_path), 'r') as f:
        data = yaml.safe_load(f)

    # ✅ Load hyperparameter YAML as dict
    with open(hyp_path, 'r') as f:
        hyp = yaml.safe_load(f)

    # Load model
    model = Model(cfg_path, ch=3, nc=data['nc']).to(device)
    if opt.weights:
        ckpt = torch.load(opt.weights, map_location=device)
        model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.937,
        nesterov=True,
        weight_decay=0.0005
    )

    # Load train/val data
    train_loader = LoadImagesAndLabels(
        path=data['train'],
        img_size=img_size,
        batch_size=opt.batch,
        augment=True,
        hyp=hyp,
        rect=False,
        cache_images=True,
        single_cls=False,
        stride=32,
        pad=0.0,
    )

    val_loader = LoadImagesAndLabels(
        path=data['val'],
        img_size=img_size,
        batch_size=opt.batch,
        augment=False,
        hyp=hyp,
        rect=True,
        cache_images=True,
        single_cls=False,
        stride=32,
        pad=0.5,
    )

    # Train
    train(
        opt=vars(opt),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=opt.epochs,
        optimizer=optimizer,
        log_dir=f"runs/train/{opt.name}",
    )

    # Save model
    os.makedirs(f"runs/train/{opt.name}/weights", exist_ok=True)
    torch.save(model.state_dict(), f"runs/train/{opt.name}/weights/best.pt")
    print(f"✅ Training complete. Model saved to runs/train/{opt.name}/weights/best.pt")

if __name__ == '__main__':
    main()
