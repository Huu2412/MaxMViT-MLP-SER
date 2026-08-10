"""
Benchmark script for rank2_f167.80_acc68.47_loss2.1539_epoch29.zip
Format: Standard PyTorch zip (torch.load works directly on the zip)
Model type: GMU (visec_optimized.yaml)
"""

import torch
import time
import os
import numpy as np
from sklearn.metrics import (
    classification_report, recall_score, f1_score,
    precision_score, accuracy_score, confusion_matrix
)

# ── paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_ZIP = r"checkpoints\visec_test_1epoch\rank2_f167.80_acc68.47_loss2.1539_epoch29.zip"
CONFIG_PATH    = r"configs\visec_optimized.yaml"
REPORT_PATH    = r"checkpoints\visec_test_1epoch\rank2_f167.80_acc68.47_loss2.1539_epoch29_report.txt"
# ──────────────────────────────────────────────────────────────────────────────

# 0. Disable pretrained download for timm
import timm as _timm
_orig_create = _timm.create_model
def _mock_create(*a, **kw):
    kw['pretrained'] = False
    return _orig_create(*a, **kw)
_timm.create_model = _mock_create

from utils import load_config
from data_loaders import get_dataloaders

print("=" * 60)
print("Benchmark: rank2_f167.80_acc68.47_loss2.1539_epoch29.zip")
print("=" * 60)

# 1. Config & data ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading config & dataset ...")
config    = load_config(CONFIG_PATH)
model_cfg = config.get('model', {})
train_cfg = config.get('training', {})

num_classes = model_cfg.get('num_classes', 4)
model_type  = model_cfg.get('type', 'gmu')
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Model type : {model_type}")
print(f"  Num classes: {num_classes}")
print(f"  Device     : {DEVICE}")

_, val_loader = get_dataloaders(config)
if val_loader is None:
    raise RuntimeError("val_loader could not be created – check dataset config.")

# 2. Build model ───────────────────────────────────────────────────────────────
print("\n[2/5] Building model ...")
from train import get_model_and_optimizer
model, _ = get_model_and_optimizer(
    model_type, num_classes,
    lr=train_cfg.get('lr', 0.0002),
    model_cfg=model_cfg
)
model.to(DEVICE)
model.eval()

# 3. Load weights directly from zip (torch.load supports PyTorch zip format) ──
print("\n[3/5] Loading checkpoint weights ...")
state_dict = torch.load(CHECKPOINT_ZIP, map_location=DEVICE, weights_only=False)
print(f"  Loaded type: {type(state_dict)}")

# Unwrap if nested
if isinstance(state_dict, dict):
    for key in ('model', 'state_dict', 'model_state_dict'):
        if key in state_dict:
            state_dict = state_dict[key]
            print(f"  Unwrapped from key '{key}'")
            break

# Check key compatibility
model_keys  = set(model.state_dict().keys())
ckpt_keys   = set(state_dict.keys())
missing     = model_keys - ckpt_keys
unexpected  = ckpt_keys - model_keys
print(f"  Checkpoint keys : {len(ckpt_keys)}")
print(f"  Model keys      : {len(model_keys)}")
print(f"  Missing keys    : {len(missing)}")
print(f"  Unexpected keys : {len(unexpected)}")
if missing:
    print(f"  Sample missing: {list(missing)[:5]}")

model.load_state_dict(state_dict, strict=False)
print("  Weights loaded successfully.")

# 4. Model size & FLOPs ────────────────────────────────────────────────────────
print("\n[4/5] Computing model size & FLOPs ...")
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params    : {total_params/1e6:.2f}M ({total_params:,})")
print(f"  Trainable params: {trainable_params/1e6:.2f}M ({trainable_params:,})")

flops_str = "N/A"
try:
    import thop
    cqt_d = torch.randn(1, 3, 224, 224).to(DEVICE)
    mel_d = torch.randn(1, 3, 224, 224).to(DEVICE)
    flops_raw, _ = thop.profile(model, inputs=(cqt_d, mel_d), verbose=False)
    flops_str = f"{flops_raw/1e9:.2f} GFLOPs ({flops_raw:,.0f})"
    print(f"  FLOPs           : {flops_str}")
except Exception as e:
    print(f"  FLOPs: could not calculate ({e})")

# 5. Inference on val set ──────────────────────────────────────────────────────
print("\n[5/5] Running inference on validation set ...")
all_preds, all_labels = [], []
total_time = 0.0

batch_size_cfg = val_loader.batch_size if hasattr(val_loader, 'batch_size') else 8

with torch.no_grad():
    # Warmup
    cqt_w = torch.randn(1, 3, 224, 224).to(DEVICE)
    mel_w = torch.randn(1, 3, 224, 224).to(DEVICE)
    for _ in range(3):
        _ = model(cqt_w, mel_w)

    for i, batch in enumerate(val_loader):
        if len(batch) == 4:
            cqt, mel, label, _ = batch
        else:
            cqt, mel, label = batch

        cqt, mel = cqt.to(DEVICE), mel.to(DEVICE)
        t0 = time.time()
        out = model(cqt, mel)
        total_time += time.time() - t0

        if isinstance(out, tuple):
            out = out[0]

        _, pred = out.max(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(label.numpy())

        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1}/{len(val_loader)} done ...")

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
n_samples  = len(all_labels)

inf_per_sample_ms = (total_time / n_samples) * 1000.0
inf_per_batch_ms  = inf_per_sample_ms * batch_size_cfg

# ── Metrics ───────────────────────────────────────────────────────────────────
acc          = 100.0 * accuracy_score(all_labels, all_preds)
ua           = 100.0 * recall_score(all_labels, all_preds, average='macro',    zero_division=0)
mf1          = 100.0 * f1_score(    all_labels, all_preds, average='macro',    zero_division=0)
weighted_f1  = 100.0 * f1_score(    all_labels, all_preds, average='weighted', zero_division=0)
macro_prec   = 100.0 * precision_score(all_labels, all_preds, average='macro', zero_division=0)
f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

class_names = getattr(val_loader.dataset, 'target_classes', None) \
              or [f"Class {i}" for i in range(num_classes)]

report = classification_report(
    all_labels, all_preds, target_names=class_names, digits=4, zero_division=0
)
cm = confusion_matrix(all_labels, all_preds)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"Checkpoint     : {CHECKPOINT_ZIP}")
print(f"Config         : {CONFIG_PATH}")
print(f"Total samples  : {n_samples}")
print()
print("── Classification Report ──────────────────────────────────")
print(report)
print("── Model Size & Efficiency ────────────────────────────────")
print(f"  Total Parameters    : {total_params/1e6:.2f}M ({total_params:,})")
print(f"  Trainable Parameters: {trainable_params/1e6:.2f}M ({trainable_params:,})")
print(f"  FLOPs (per sample)  : {flops_str}")
print(f"  Inference / sample  : {inf_per_sample_ms:.2f} ms")
print(f"  Inference / batch({batch_size_cfg}): {inf_per_batch_ms:.2f} ms")
print()
print("── Emotion Recognition Metrics ────────────────────────────")
print(f"  Overall Accuracy (WA)    : {acc:.2f}%")
print(f"  Unweighted Accuracy (UA) : {ua:.2f}%")
print(f"  Macro F1-score (mF1)     : {mf1:.2f}%")
print(f"  Weighted F1-score        : {weighted_f1:.2f}%")
print(f"  Macro Precision          : {macro_prec:.2f}%")
print()
print("── Per-class F1 ───────────────────────────────────────────")
for i, name in enumerate(class_names):
    if i < len(f1_per_class):
        print(f"  {name:>10}: {f1_per_class[i]*100:.2f}%")
print()
print("── Confusion Matrix ───────────────────────────────────────")
header = "         " + "  ".join(f"{n[:7]:>7}" for n in class_names)
print(header)
for i, row in enumerate(cm):
    print(f"  {class_names[i]:>7}  " + "  ".join(f"{v:>7}" for v in row))

# ── Save report ───────────────────────────────────────────────────────────────
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Checkpoint: {CHECKPOINT_ZIP}\n")
    f.write(f"Config: {CONFIG_PATH}\n")
    f.write(f"Total samples: {n_samples}\n")
    f.write("=" * 60 + "\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n" + "=" * 60 + "\n")
    f.write("BENCHMARKS & METRICS:\n")
    f.write(f"Total Parameters    : {total_params/1e6:.2f}M ({total_params:,})\n")
    f.write(f"Trainable Parameters: {trainable_params/1e6:.2f}M ({trainable_params:,})\n")
    f.write(f"FLOPs (per sample)  : {flops_str}\n")
    f.write(f"Inference per sample: {inf_per_sample_ms:.2f} ms\n")
    f.write(f"Inference per batch({batch_size_cfg}): {inf_per_batch_ms:.2f} ms\n")
    f.write("-" * 60 + "\n")
    f.write(f"Overall Accuracy (WA)    : {acc:.2f}%\n")
    f.write(f"Unweighted Accuracy (UA) : {ua:.2f}%\n")
    f.write(f"Macro F1-score (mF1)     : {mf1:.2f}%\n")
    f.write(f"Weighted F1-score        : {weighted_f1:.2f}%\n")
    f.write(f"Macro Precision          : {macro_prec:.2f}%\n")
    f.write("F1-score per class:\n")
    for i, name in enumerate(class_names):
        if i < len(f1_per_class):
            f.write(f"  - {name}: {f1_per_class[i]*100:.2f}%\n")
    f.write("\nConfusion Matrix:\n")
    f.write("         " + "  ".join(f"{n[:7]:>7}" for n in class_names) + "\n")
    for i, row in enumerate(cm):
        f.write(f"  {class_names[i]:>7}  " + "  ".join(f"{v:>7}" for v in row) + "\n")

print(f"\nReport saved to: {REPORT_PATH}")
