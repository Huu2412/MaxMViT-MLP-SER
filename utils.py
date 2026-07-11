
import os
import yaml
import logging
import random
import numpy as np
import torch
import sys
from datetime import datetime

def load_config(config_path):
    """Load YAML config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config):
    """
    Setup logging to file and console.
    Structure: logs/{experiment_name}/{timestamp}.log
    """
    exp_name = config.get("experiment_name", "experiment")
    log_root = config.get("paths", {}).get("log_dir", "logs")
    
    # Create experiment directory
    exp_dir = os.path.join(log_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(exp_dir, f"{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging configured. Saving logs to: {log_file}")
    
    # Return path to save checkpoints nearby if needed
    ckpt_root = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    ckpt_dir = os.path.join(ckpt_root, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    return ckpt_dir

def seed_everything(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Seeded everything with seed: {seed}")


def freeze_backbone_layers(backbone, unfreeze_last_n=0):
    """
    Freeze all parameters of a backbone, optionally leaving the last N
    top-level child modules trainable.

    This is a heuristic: timm backbones (MaxViT, MViTv2) expose their stages
    as top-level children (stem, stages.0, stages.1, ..., norm/head). Freezing
    everything except the last few stages is a common transfer-learning trick
    to reduce overfitting on small datasets while still allowing the
    high-level features to adapt to the new domain.

    Args:
        backbone (nn.Module): e.g. model.maxvit or model.mvitv2
        unfreeze_last_n (int): number of trailing top-level child modules to
            keep trainable. 0 means the whole backbone is frozen.

    Returns:
        (int, int): (num_frozen_params, num_trainable_params)
    """
    # Freeze everything first
    for p in backbone.parameters():
        p.requires_grad = False

    if unfreeze_last_n > 0:
        # Only consider top-level children that actually own parameters.
        # timm classification backbones created with num_classes=0 typically
        # end in a parameter-less nn.Identity() (the removed head) and/or a
        # pooling layer -- naively taking the last N named_children() would
        # silently "unfreeze" nothing useful. Filter those out first, then
        # take the last N from what remains (the real stages/blocks/norm).
        children_with_params = [
            (name, module) for name, module in backbone.named_children()
            if any(True for _ in module.parameters())
        ]
        for name, module in children_with_params[-unfreeze_last_n:]:
            for p in module.parameters():
                p.requires_grad = True

    frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    return frozen, trainable


def compute_class_weights(dataset, num_classes):
    """
    Compute inverse-frequency class weights from a dataset's `.indices` list
    (used by ViSEC/RAVDESS/IEMOCAP loaders, where each entry is either
    (ds_idx, label) or (ds_idx, label, accent_label)).

    weight_i = total_samples / (num_classes * count_i)

    Returns a torch.FloatTensor of shape [num_classes], suitable for
    nn.CrossEntropyLoss(weight=...).
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for entry in dataset.indices:
        label = entry[1]
        if 0 <= label < num_classes:
            counts[label] += 1

    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for absent classes
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
