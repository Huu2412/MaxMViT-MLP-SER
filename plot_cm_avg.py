"""
Plot averaged confusion matrix from multiple checkpoints in the Results folder.
Completely self-contained - bypasses slow ViSECDataset and timm pretrained downloads.

Usage:
    python plot_cm_avg.py
    python plot_cm_avg.py --config configs/visec_optimized.yaml --results_dir Results --output avg_confusion_matrix
"""

import argparse
import torch
import numpy as np
import os
import sys
import glob
import random
import copy
import io
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import yaml

# Suppress warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal')
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Lightweight Dataset - loads HF dataset WITHOUT cast_column
# ============================================================
class LightweightViSECDataset(torch.utils.data.Dataset):
    """
    Minimal ViSEC dataset for inference only.
    Reads directly from pyarrow table.
    """
    def __init__(self, arrow_table, indices, target_size=(224, 224), sr=44100):
        self.table = arrow_table
        self.indices = indices
        self.target_size = target_size
        self.sr = sr
        self.target_classes = ['happy', 'neutral', 'sad', 'angry']
        self.n_fft = 4096
        self.hop_length = 256
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        import librosa
        import soundfile as sf

        ds_idx, label, accent_label = self.indices[idx]
        
        # Access pyarrow table directly
        audio_dict = self.table.column('path')[ds_idx].as_py()
        audio_bytes = audio_dict['bytes']
        
        y, orig_sr = sf.read(io.BytesIO(audio_bytes))

        if orig_sr != self.sr:
            y = y.astype(np.float32)
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
        else:
            y = y.astype(np.float32)

        if y.ndim > 1:
            y = np.mean(y, axis=0)

        if len(y) < self.n_fft:
            padding = self.n_fft - len(y) + 1
            y = np.pad(y, (0, padding), mode='constant')

        try:
            cqt = librosa.cqt(y, sr=self.sr)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            cqt_img = self._resize_normalize(cqt_db)
            mel_img = self._resize_normalize_3ch(mel_db)

            cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32)
            mel_tensor = torch.tensor(mel_img, dtype=torch.float32)

            return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(accent_label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing sample {ds_idx}: {e}")
            dummy = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return dummy, dummy, torch.tensor(label, dtype=torch.long), torch.tensor(accent_label, dtype=torch.long)

    def _resize_normalize(self, spec):
        import cv2
        spec_min, spec_max = spec.min(), spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        spec_3ch = np.stack([spec_resized] * 3, axis=0)
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
        return spec_3ch

    def _resize_normalize_3ch(self, spec):
        """Same as _resize_normalize but stacks same spec 3 times (matching ablation in visec.py)."""
        import cv2
        spec_min, spec_max = spec.min(), spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        spec_3ch = np.stack([spec_resized] * 3, axis=0)
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
        return spec_3ch


# ============================================================
# Model creation with pretrained=False
# ============================================================
def create_model(model_type, num_classes, model_cfg):
    """Create model with pretrained=False (we load our own checkpoint)."""
    import timm
    _original_create = timm.create_model
    def _create_no_pretrained(*args, **kwargs):
        kwargs['pretrained'] = False
        return _original_create(*args, **kwargs)
    timm.create_model = _create_no_pretrained

    hidden_size = model_cfg.get('hidden_size', 512)
    dropout_rate = model_cfg.get('dropout_rate', 0.2)
    num_accent_classes = model_cfg.get('num_accent_classes', 0)
    freeze_backbone = model_cfg.get('freeze_backbone', False)
    unfreeze_last_n_blocks = model_cfg.get('unfreeze_last_n_blocks', 0)

    if model_type == 'gmu':
        from model_gmu import MaxMViT_MLP_GMU
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        model = MaxMViT_MLP_GMU(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            fusion_hidden_dim=fusion_hidden_dim,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
    elif model_type == 'crossattn':
        from model_crossattn import MaxMViT_MLP_CrossAttn
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        num_heads = model_cfg.get('num_heads', 8)
        fusion_type = model_cfg.get('fusion_type', 'concat')
        model = MaxMViT_MLP_CrossAttn(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            fusion_hidden_dim=fusion_hidden_dim,
            num_heads=num_heads,
            fusion_type=fusion_type,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
    elif model_type == 'original':
        from model import MaxMViT_MLP
        model = MaxMViT_MLP(
            num_classes=num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone, unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    timm.create_model = _original_create
    return model


# ============================================================
# Dataset creation - uses HF cache, no cast_column
# ============================================================
def create_val_loader(config, custom_seed=None):
    """Load HF dataset fast using pyarrow directly to bypass HF hangs, build val indices, return DataLoader."""
    from datasets import load_dataset
    
    ds_cfg = config.get('dataset', {})
    args = ds_cfg.get('args', {})
    hf_id = args.get('hf_id', 'hustep-lab/ViSEC')
    batch_size = args.get('batch_size', 8)
    target_size = tuple(args.get('target_size', [224, 224]))
    seed = custom_seed if custom_seed is not None else config.get('training', {}).get('seed', 42)

    target_classes = ['happy', 'neutral', 'sad', 'angry']
    class_map = {c: i for i, c in enumerate(target_classes)}
    accent_map = {'north': 0, 'south': 1, 'mid': 2}

    print(f"Loading dataset from HF (bypassing cast_column to avoid hangs)...")
    try:
        ds = load_dataset(hf_id, split='train')
        table = ds.data  # MemoryMappedTable behaves like pyarrow table
        print(f"Loaded {len(table)} samples. Columns: {table.column_names}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {hf_id}. Error: {e}")

    # Build indices
    aux_cfg = config.get('auxiliary_task', {})
    load_accent = aux_cfg.get('enabled', False)
    has_accent = load_accent and 'accent' in table.column_names

    emotions = table.column('emotion').to_pylist()
    accents = table.column('accent').to_pylist() if has_accent else [None] * len(table)

    all_indices = []
    for idx, (emo, accent_str) in enumerate(zip(emotions, accents)):
        if emo in target_classes:
            accent_label = -1
            if has_accent and accent_str is not None:
                if isinstance(accent_str, (int, float)) or (isinstance(accent_str, str) and str(accent_str).isdigit()):
                    acc_int = int(accent_str)
                    if 0 <= acc_int < 3:
                        accent_label = acc_int
                else:
                    clean_accent = str(accent_str).strip().lower()
                    if clean_accent in accent_map:
                        accent_label = accent_map[clean_accent]
            all_indices.append((idx, class_map[emo], accent_label))

    print(f"Filtered {len(all_indices)} samples")

    # Stratified Split (same as get_visec_dataloaders)
    split_ratio = tuple(args.get('split_ratio', [0.8, 0.1, 0.1]))
    from data_loaders.visec import stratified_split_indices
    _, val_indices, test_indices = stratified_split_indices(
        all_indices, split_ratio=split_ratio, seed=seed, label_index=1
    )

    eval_indices = val_indices
    print(f"Val split: {len(eval_indices)} samples")

    val_ds = LightweightViSECDataset(table, eval_indices, target_size=target_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loader


def run_inference(model, val_loader, device):
    """Run inference and return predictions + labels."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            cqt, mel, label = batch[0], batch[1], batch[2]
            cqt, mel = cqt.to(device), mel.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(cqt, mel)

            outputs = out[0] if isinstance(out, tuple) else out
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.numpy())

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(val_loader)}")

    return np.array(all_preds), np.array(all_labels)


def main(config_path, results_dir, output_img):
    config = load_config(config_path)
    train_cfg = config['training']
    model_cfg = config['model']

    SEED = train_cfg.get('seed', 42)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    # Model config
    num_classes = model_cfg.get('num_classes', 4)
    model_type = model_cfg.get('type', 'crossattn')
    aux_cfg = config.get('auxiliary_task', {})
    if aux_cfg.get('enabled', False):
        model_cfg['num_accent_classes'] = aux_cfg.get('num_accent_classes', 0)

    # Find checkpoints
    checkpoint_files = sorted(glob.glob(os.path.join(results_dir, "*.zip")))
    if not checkpoint_files:
        checkpoint_files = sorted(
            glob.glob(os.path.join(results_dir, "*.pth")) +
            glob.glob(os.path.join(results_dir, "*.pt")))
    if not checkpoint_files:
        print(f"No checkpoint files found in {results_dir}")
        return

    print(f"\n{len(checkpoint_files)} checkpoints found:")
    for f in checkpoint_files:
        print(f"  - {os.path.basename(f)}")

    class_names = ['happy', 'neutral', 'sad', 'angry']
    print(f"Class names: {class_names}")

    # Specific seeds used for the checkpoints (in alphabetical order of files)
    ckpt_seeds = [45, 46, 44]

    # Run inference for each checkpoint
    all_cms = []
    all_cms_normalized = []

    for i, ckpt_path in enumerate(checkpoint_files):
        ckpt_seed = ckpt_seeds[i] if i < len(ckpt_seeds) else 42
        
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(checkpoint_files)}] {os.path.basename(ckpt_path)} (Seed: {ckpt_seed})")
        print(f"{'='*60}")

        # Load dataset dynamically with the correct validation split seed for this checkpoint
        val_loader = create_val_loader(config, custom_seed=ckpt_seed)

        model = create_model(model_type, num_classes, model_cfg)
        model.to(DEVICE)

        loaded = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = loaded.get('model_state_dict', loaded.get('state_dict', loaded)) if isinstance(loaded, dict) and ('model_state_dict' in loaded or 'state_dict' in loaded) else loaded
        model.load_state_dict(state_dict)

        preds, labels = run_inference(model, val_loader, DEVICE)

        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        all_cms.append(cm)
        all_cms_normalized.append(cm_norm)

        print(classification_report(labels, preds, target_names=class_names, digits=4))

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average
    avg_cm = np.mean(all_cms, axis=0)
    avg_cm_norm = np.mean(all_cms_normalized, axis=0)
    std_cm_norm = np.std(all_cms_normalized, axis=0)

    print(f"\n{'='*60}")
    print(f"AVERAGED RESULTS ({len(checkpoint_files)} checkpoints)")
    print(f"{'='*60}")

    # --- Plot 1: Averaged Normalized CM ---
    fig, ax = plt.subplots(figsize=(10, 8))
    annot = np.empty_like(avg_cm_norm, dtype=object)
    for r in range(avg_cm_norm.shape[0]):
        for c in range(avg_cm_norm.shape[1]):
            # Chuyển đổi sang phần trăm (x100) và lấy 2 chữ số thập phân
            annot[r, c] = f"{avg_cm_norm[r, c]*100:.2f}%\n\u00b1{std_cm_norm[r, c]*100:.2f}%"

    sns.heatmap(avg_cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax, annot_kws={"size": 16, "weight": "bold"})
    
    # Tăng kích thước chữ cho trục và tiêu đề
    ax.set_ylabel('True Label', fontsize=16, weight='bold')
    ax.set_xlabel('Predicted Label', fontsize=16, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(f'Averaged Normalized Confusion Matrix\n({len(checkpoint_files)} checkpoints, mean \u00b1 std)', fontsize=18, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_img + "_avg_normalized.png", dpi=150)
    print(f"Saved: {output_img}_avg_normalized.png")
    plt.close()

    # --- Plot 2: Averaged Counts CM (Rounded to Integer) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Làm tròn để hiển thị số nguyên theo yêu cầu
    avg_cm_int = np.round(avg_cm).astype(int)
    annot_c = np.empty_like(avg_cm_int, dtype=object)
    for r in range(avg_cm_int.shape[0]):
        for c in range(avg_cm_int.shape[1]):
            annot_c[r, c] = f"{avg_cm_int[r, c]}"

    sns.heatmap(avg_cm, annot=annot_c, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"size": 12})
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_title(f'Averaged Confusion Matrix (Counts)\n({len(checkpoint_files)} checkpoints)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_img + "_avg_counts.png", dpi=150)
    print(f"Saved: {output_img}_avg_counts.png")
    plt.close()

    # --- Plot 3: Individual CMs (Raw Counts) ---
    n = len(checkpoint_files)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7))
    if n == 1:
        axes = [axes]
    for i, (cm_raw, ckpt) in enumerate(zip(all_cms, checkpoint_files)):
        sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i], annot_kws={"size": 12})
        axes[i].set_ylabel('True Label', fontsize=11)
        axes[i].set_xlabel('Predicted Label', fontsize=11)
        name = os.path.basename(ckpt).replace('.zip', '')
        axes[i].set_title(name, fontsize=10)

    plt.suptitle('Individual Confusion Matrices (Raw Counts)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_img + "_individual.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_img}_individual.png")
    plt.close()

    print(f"\nDone! All plots saved with prefix: {output_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/visec_optimized.yaml")
    parser.add_argument("--results_dir", type=str, default="Results")
    parser.add_argument("--output", type=str, default="confusion_matrix")
    args = parser.parse_args()
    main(args.config, args.results_dir, args.output)
