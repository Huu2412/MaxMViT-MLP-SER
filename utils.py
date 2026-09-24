
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
    with open(config_path, "r", encoding="utf-8") as f:
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


def compute_accent_weights(dataset, num_accent_classes):
    """
    Compute inverse-frequency accent/region class weights from a dataset's `.indices` list
    where each entry is (ds_idx, label, accent_label).

    weight_i = total_valid_samples / (num_accent_classes * count_i)

    Returns a torch.FloatTensor of shape [num_accent_classes], suitable for
    nn.CrossEntropyLoss(weight=..., ignore_index=-1).
    """
    counts = np.zeros(num_accent_classes, dtype=np.float64)
    for entry in dataset.indices:
        if len(entry) >= 3:
            acc = entry[2]
            if 0 <= acc < num_accent_classes:
                counts[acc] += 1

    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for absent classes
    total = counts.sum()
    weights = total / (num_accent_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_detailed_metrics(y_true, y_pred, class_names=None):
    """
    Tính toán chi tiết số lượng nhãn dự đoán đúng và bị dự đoán sai so với các nhãn khác.
    
    Args:
        y_true: List hoặc 1D array chứa nhãn thực tế (ground truth).
        y_pred: List hoặc 1D array chứa nhãn mô hình dự đoán.
        class_names: List tên các nhãn tương ứng theo thứ tự (ví dụ: ['happy', 'neutral', 'sad', 'angry']).
        
    Returns:
        dict gồm:
            - 'confusion_matrix': List 2D (K x K)
            - 'class_names': List tên các nhãn
            - 'per_class': Dict chi tiết từng nhãn (số lượng đúng, sai, nhầm sang nhãn nào)
            - 'summary_text': Chuỗi văn bản định dạng đẹp để in log và lưu báo cáo.
    """
    from sklearn.metrics import confusion_matrix
    
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    num_classes = len(class_names) if class_names else max(int(np.max(y_true)), int(np.max(y_pred))) + 1
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
        
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    per_class = {}
    lines = []
    lines.append("=" * 80)
    lines.append("DETAILED PREDICTION BREAKDOWN & CONFUSION MATRIX")
    lines.append("=" * 80)
    
    # 1. Bảng ma trận nhầm lẫn
    col_title = "Actual / Pred"
    tot_title = "Total"
    header = f"{col_title:<20}" + "".join([f"{name:>12}" for name in class_names]) + f"{tot_title:>10}"
    lines.append("Confusion Matrix:")
    lines.append(header)
    lines.append("-" * len(header))
    
    for i, name in enumerate(class_names):
        row_str = f"{name:<20}"
        for j in range(num_classes):
            row_str += f"{cm[i, j]:>12d}"
        row_str += f"{cm[i].sum():>10d}"
        lines.append(row_str)
    lines.append("-" * len(header))
    
    # 2. Chi tiết từng nhãn
    lines.append("\nPer-class accuracy and misclassification details:")
    for i, name in enumerate(class_names):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        correct_pct = (correct / total * 100.0) if total > 0 else 0.0
        misclassified = total - correct
        misc_pct = (misclassified / total * 100.0) if total > 0 else 0.0
        
        # Nhãn này bị đoán nhầm thành các nhãn khác:
        misclassified_to = {}
        for j in range(num_classes):
            if j != i and cm[i, j] > 0:
                other_name = class_names[j]
                misclassified_to[other_name] = int(cm[i, j])
                
        # Các nhãn khác bị đoán nhầm thành nhãn này:
        confused_from = {}
        for j in range(num_classes):
            if j != i and cm[j, i] > 0:
                other_name = class_names[j]
                confused_from[other_name] = int(cm[j, i])
                
        per_class[name] = {
            'total_samples': total,
            'correct_count': correct,
            'correct_rate_pct': round(correct_pct, 2),
            'misclassified_count': misclassified,
            'misclassified_rate_pct': round(misc_pct, 2),
            'misclassified_to': misclassified_to,
            'confused_from': confused_from,
        }
        
        lines.append(f"\n[*] Class '{name}' (Total: {total} samples):")
        lines.append(f"    [+] Correct predictions    : {correct:>4d} / {total} ({correct_pct:.2f}%)")
        lines.append(f"    [-] Misclassified (Errors) : {misclassified:>4d} / {total} ({misc_pct:.2f}%)")
        if misclassified_to:
            lines.append("        Misclassified as:")
            for other_name, count in sorted(misclassified_to.items(), key=lambda x: x[1], reverse=True):
                sub_pct = (count / total * 100.0) if total > 0 else 0.0
                lines.append(f"          -> '{other_name}': {count:>3d} samples ({sub_pct:.2f}%)")
        else:
            lines.append("        -> No misclassifications!")
            
        if confused_from:
            confused_total = sum(confused_from.values())
            lines.append(f"    [<-] Other classes confused as '{name}': {confused_total} samples:")
            for other_name, count in sorted(confused_from.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"          <- From '{other_name}': {count:>3d} samples")
                
    lines.append("\n" + "=" * 80)
    summary_text = "\n".join(lines)
    
    return {
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'per_class': per_class,
        'summary_text': summary_text,
    }


def load_checkpoint(ckpt_path, model, device='cpu'):
    """
    Hỗ trợ nạp checkpoint an toàn cho cả định dạng raw state_dict và structured dict.
    """
    checkpoint_data = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint_data, dict):
        if 'model_state_dict' in checkpoint_data:
            state_dict = checkpoint_data['model_state_dict']
        elif 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
        else:
            state_dict = checkpoint_data
    else:
        state_dict = checkpoint_data
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return checkpoint_data

