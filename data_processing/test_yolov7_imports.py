import os
import sys
import torch
from torch.serialization import add_safe_globals
import yaml
import logging

# Add yolov7 to Python path
yolov7_path = '/'
#sys.path.append(yolov7_path)
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "yolov7"))
if yolo_path not in sys.path:
    sys.path.insert(0, yolo_path)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify yolov7 directory
if not os.path.exists(yolov7_path):
    logger.error(f"YOLOv7 directory not found at {yolov7_path}")
    sys.exit(1)

# Verify specific files
files_to_check = [
    os.path.join(yolov7_path, 'models', 'yolo.py'),
    os.path.join(yolov7_path, 'utils', 'datasets.py'),
    os.path.join(yolov7_path, 'utils', 'general.py'),
    os.path.join(yolov7_path, 'utils', 'google_utils.py'),
    os.path.join(yolov7_path, 'utils', 'torch_utils.py'),
    os.path.join(yolov7_path, 'train.py'),
    os.path.join(yolov7_path, 'utils', '__init__.py')
]
for file_path in files_to_check:
    if os.path.exists(file_path):
        logger.info(f"File found at {file_path}")
    else:
        logger.error(f"File not found at {file_path}")
        sys.exit(1)

# Try importing utils.google_utils
try:
    from yolov7.utils import google_utils
    logger.info("Imported yolov7.utils.google_utils successfully")
except ImportError as e:
    logger.error(f"Failed to import yolov7.utils.google_utils: {e}")
    sys.exit(1)

# Try importing utils.general
try:
    from yolov7.utils import general
    logger.info("Imported yolov7.utils.general successfully")
except ImportError as e:
    logger.error(f"Failed to import yolov7.utils.general: {e}")
    sys.exit(1)

# Try importing utils.datasets
try:
    from yolov7.utils import datasets
    logger.info("Imported yolov7.utils.datasets successfully")
except ImportError as e:
    logger.error(f"Failed to import yolov7.utils.datasets: {e}")
    sys.exit(1)

try:
    from yolov7.models.yolo import Model
    from yolov7.utils.datasets import LoadImagesAndLabels
    from yolov7.utils.general import check_file, check_img_size
    from yolov7.utils.torch_utils import select_device, TracedModel
    from yolov7.train import train
except ImportError as e:
    logger.error(f"Failed to import yolov7 modules: {e}")
    sys.exit(1)

# Configuration
cfg = {
    'data': '/Users/michaelwilliams/PycharmProjects/RAGChat/dataset/dataset.yaml',
    'weights': '/Users/michaelwilliams/PycharmProjects/RAGChat/yolov7-tiny.pt',
    'cfg': '/Users/michaelwilliams/PycharmProjects/RAGChat/yolov7/cfg/training/yolov7-tiny.yaml',
    'img_size': 640,
    'batch_size': 8,  # Reduced for macOS CPU/MPS
    'epochs': 50,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'name': 'yolov7_surveillance',
    'hyp': '/Users/michaelwilliams/PycharmProjects/RAGChat/yolov7/data/hyp.scratch.yaml',
    'workers': 4,
    'log_dir': '/Users/michaelwilliams/PycharmProjects/RAGChat/runs/train',
}

def main():
    # Set device
    device = select_device(cfg['device'])
    logger.info(f"Using device: {device}")

    # Load dataset configuration
    try:
        data_dict = check_file(cfg['data'])
        with open(data_dict, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        logger.error(f"Dataset YAML file not found: {e}")
        return

    # Check image size
    img_size = check_img_size(cfg['img_size'], s=32)  # YOLO requires multiples of 32

    # Load model
    try:
        model = Model(cfg['cfg'], ch=3, nc=data['nc']).to(device)
        add_safe_globals([Model])
        if cfg['weights']:
            ckpt = torch.load(cfg['weights'], map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    except FileNotFoundError as e:
        logger.error(f"Model weights or config file not found: {e}")
        return
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.937,
        nesterov=True,
        weight_decay=0.0005
    )

    # Data loaders
    try:
        train_loader = LoadImagesAndLabels(
            path=data['train'],
            img_size=img_size,
            batch_size=cfg['batch_size'],
            augment=True,
            hyp=cfg['hyp'],
            rect=False,
            cache_images=True,
            single_cls=False,
            stride=32,
            pad=0.0,
        )
        val_loader = LoadImagesAndLabels(
            path=data['val'],
            img_size=img_size,
            batch_size=cfg['batch_size'],
            augment=False,
            hyp=cfg['hyp'],
            rect=True,
            cache_images=True,
            single_cls=False,
            stride=32,
            pad=0.5,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Train
    os.makedirs(cfg['log_dir'], exist_ok=True)
    try:
        train(
            opt=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=cfg['epochs'],
            optimizer=optimizer,
            log_dir=f"{cfg['log_dir']}/{cfg['name']}",
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # Save final model
    model_path = f"{cfg['log_dir']}/{cfg['name']}/weights/best.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Training complete. Model saved to {model_path}")

if __name__ == '__main__':
    main()
