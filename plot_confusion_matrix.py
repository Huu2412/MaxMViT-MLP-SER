#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, classification_report, recall_score, f1_score

# Patch timm.create_model to disable downloading pretrained models during evaluation,
# saving memory and avoiding Windows paging file exhaustion errors.
import timm
original_create_model = timm.create_model

def mock_create_model(*args, **kwargs):
    if 'pretrained' in kwargs:
        kwargs['pretrained'] = False
    return original_create_model(*args, **kwargs)

timm.create_model = mock_create_model

from utils import load_config, seed_everything
from data_loaders import get_dataloaders
from train import get_model_and_optimizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and plot its confusion matrix.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output files (default: checkpoint directory)")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on (cpu or cuda)")
    
    args = parser.parse_args()
    
    # 1. Load config
    config = load_config(args.config)
    train_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    
    # Set seed for reproducibility
    seed_everything(train_cfg.get('seed', 42))
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
    print(f"Using device: {device}")
    
    # 2. Initialize Model first (when memory is at its cleanest)
    num_classes = model_cfg.get('num_classes', 4)
    model_type = model_cfg.get('type', 'crossattn')
    
    print(f"Initializing model type '{model_type}' with {num_classes} classes...")
    model, _ = get_model_and_optimizer(model_type, num_classes, lr=0.0001, model_cfg=model_cfg)
    
    # Load Checkpoint incrementally to minimize memory footprint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device, mmap=True)
    
    print("Loading weights incrementally into model...")
    with torch.no_grad():
        model_state = model.state_dict()
        for name in list(state_dict.keys()):
            if name in model_state:
                model_state[name].copy_(state_dict[name])
                del state_dict[name]
            else:
                print(f"Warning: unexpected key {name} in checkpoint")
                
    model.to(device)
    model.eval()
    
    # Delete temporary state_dict and collect garbage immediately to free memory
    del state_dict
    import gc
    gc.collect()
    
    # 3. Setup Dataloaders
    print("Loading datasets...")
    # Use multiple workers for faster evaluation
    if 'dataset' in config and 'args' in config['dataset']:
        config['dataset']['args']['num_workers'] = 0
    _, val_loader = get_dataloaders(config)
    
    if val_loader is None:
        print("Error: Could not load validation dataloader.")
        return
        
    print(f"Validation dataset size: {len(val_loader.dataset)} samples")
    
    # 4. Evaluation Loop
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    cache_path = os.path.join(out_dir, f"{checkpoint_name}_preds.npz")
    
    if os.path.exists(cache_path):
        print(f"Loading predictions from cache: {cache_path}")
        cache = np.load(cache_path)
        all_preds = cache['preds']
        all_labels = cache['labels']
    else:
        print("Running evaluation...")
        all_preds = []
        all_labels = []
        
        num_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                cqt, mel, label = batch[0], batch[1], batch[2]
                print(f"Processing batch {batch_idx + 1}/{num_batches}...")
                cqt, mel = cqt.to(device), mel.to(device)
                outputs = model(cqt, mel)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.numpy())
                
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        np.savez(cache_path, preds=all_preds, labels=all_labels)
        print(f"Saved predictions cache to: {cache_path}")
        
    # Calculate accuracy
    acc = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"Evaluation Accuracy: {acc:.2f}%")
    
    # 5. Extract class names
    class_names = None
    if hasattr(val_loader, 'dataset'):
        class_names = getattr(val_loader.dataset, 'target_classes', None)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    print(f"Class mapping: {class_names}")
    
    # Determine output paths
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    cm_path = os.path.join(out_dir, f"{checkpoint_name}_confusion_matrix.png")
    report_path = os.path.join(out_dir, f"{checkpoint_name}_report.txt")
    
    # 6. Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Calculate additional benchmarks
    print("Calculating benchmarks (params, FLOPs, inference time)...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # FLOPs using thop
    flops_str = "N/A"
    try:
        import thop
        # Using shape [1, 3, 224, 224] as MaxViT interpolates to 224x224
        cqt_dummy = torch.randn(1, 3, 224, 224).to(device)
        mel_dummy = torch.randn(1, 3, 224, 224).to(device)
        flops_raw, _ = thop.profile(model, inputs=(cqt_dummy, mel_dummy), verbose=False)
        flops_str = f"{flops_raw / 1e9:.2f} GFLOPs ({flops_raw:,})"
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")
        
    # Inference time measurement
    print("Measuring inference time...")
    model.eval()
    num_batches_to_time = min(20, len(val_loader))
    total_time = 0.0
    num_samples_timed = 0
    with torch.no_grad():
        # Warmup
        cqt_dummy = torch.randn(1, 3, 224, 224).to(device)
        mel_dummy = torch.randn(1, 3, 224, 224).to(device)
        for _ in range(5):
            _ = model(cqt_dummy, mel_dummy)
            
        for idx, batch in enumerate(val_loader):
            if idx >= num_batches_to_time:
                break
            cqt, mel = batch[0].to(device), batch[1].to(device)
            start_t = time.time()
            _ = model(cqt, mel)
            total_time += time.time() - start_t
            num_samples_timed += cqt.size(0)
            
    inf_time_sample = (total_time / num_samples_timed) * 1000.0 if num_samples_timed > 0 else 0.0
    batch_size = val_loader.batch_size if hasattr(val_loader, 'batch_size') else 8
    inf_time_batch = inf_time_sample * batch_size
    
    # Calculate additional metrics
    ua = 100.0 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
    mf1 = 100.0 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = 100.0 * f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Validation Accuracy: {acc:.2f}%\n")
        f.write("="*60 + "\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
        f.write("BENCHMARKS & ADDED METRICS:\n")
        f.write(f"Total Parameters: {total_params / 1e6:.2f}M ({total_params:,})\n")
        f.write(f"Trainable Parameters: {trainable_params / 1e6:.2f}M ({trainable_params:,})\n")
        f.write(f"Total FLOPs (per sample): {flops_str}\n")
        f.write(f"Inference Time per batch (size {batch_size}): {inf_time_batch:.2f} ms\n")
        f.write(f"Inference Time per sample: {inf_time_sample:.2f} ms\n")
        f.write("-"*60 + "\n")
        f.write(f"Unweighted Accuracy (UA/UWA): {ua:.2f}%\n")
        f.write(f"Macro F1-score (mF1): {mf1:.2f}%\n")
        f.write(f"Weighted F1-score (F1): {weighted_f1:.2f}%\n")
        f.write("F1-score per class:\n")
        
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        for i, name in enumerate(class_names):
            if i < len(f1_per_class):
                f.write(f"  - {name}: {f1_per_class[i]*100.0:.2f}%\n")
                
    print(f"Saved classification report to: {report_path}")

    
    # 7. Plot and save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalized confusion matrix for cell colors
    # handle division by zero just in case a class has no samples in the split
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    
    # Plotting code using custom aesthetics
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # We will show only absolute count in annotation
    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            annot_labels[i, j] = f"{count}"
            
    # Beautiful heatmap
    sns.heatmap(
        cm_norm, 
        annot=annot_labels, 
        fmt="", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True,
        square=True,
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    plt.title(f"Confusion matrix of ViSec ViT-GMU\nAccuracy: {acc:.2f}%", fontsize=14, pad=20)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

if __name__ == "__main__":
    main()
