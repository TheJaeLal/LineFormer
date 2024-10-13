from mmcv import Config
from mmdet.apis import train_detector, init_detector
from mmdet.datasets import build_dataset

# Paths to your checkpoint and config file
CKPT = None # "iter_3000.pth"
CONFIG_PATH = "km_swin_t_config.py"

# Load the config file into a Config object
cfg = Config.fromfile(CONFIG_PATH)

# Initialize the model with the config and checkpoint
model = init_detector(cfg, CKPT, device='cuda')

# Build the dataset from the config
datasets = [build_dataset(cfg.data.train)]

# Pass the config object and model to train_detector
train_detector(model, datasets, cfg)

