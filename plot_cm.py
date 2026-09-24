import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

# Suppress librosa n_fft warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal')

from utils import load_config, seed_everything
from data_loaders import get_dataloaders
from train import get_model_and_optimizer

def main(config_path, checkpoint_path, output_img):
    config = load_config(config_path)
    train_cfg = config['training']
    model_cfg = config['model']
    
    SEED = train_cfg.get('seed', 42)
    seed_everything(SEED)
    
    DEVICE = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    
    # Force num_workers to 0 to prevent multiprocessing hangs on Windows
    if 'args' in config.get('dataset', {}):
        config['dataset']['args']['num_workers'] = 0

    # Get DataLoaders
    loaders = get_dataloaders(config)
    if not loaders or loaders[0] is None:
        print("Failed to load data.")
        return
    val_loader = loaders[2] if len(loaders) > 2 else loaders[1]

    # Model
    num_classes = model_cfg.get('num_classes', 4)
    model_type = model_cfg.get('type', 'crossattn')
    
    aux_cfg = config.get('auxiliary_task', {})
    aux_enabled = aux_cfg.get('enabled', False)
    num_accent_classes = aux_cfg.get('num_accent_classes', 0) if aux_enabled else 0
    if num_accent_classes > 0:
        model_cfg['num_accent_classes'] = num_accent_classes
    
    print(f"Initializing Model ({model_type}) with {num_classes} classes...")
    model, _ = get_model_and_optimizer(model_type, num_classes, 0.0002, model_cfg)
    model.to(DEVICE)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if len(batch) == 4:
                cqt, mel, label, _ = batch
            else:
                cqt, mel, label = batch
                
            cqt, mel = cqt.to(DEVICE), mel.to(DEVICE)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                model_output = model(cqt, mel)
                
            if isinstance(model_output, tuple):
                outputs, _ = model_output
            else:
                outputs = model_output
                
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(val_loader)} batches")
                
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Class names
    class_names = getattr(val_loader.dataset, 'target_classes', None)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    print(f"Class names: {class_names}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    
    # Also create a non-normalized one with raw counts
    plt.savefig(output_img + "_normalized.png")
    print(f"Saved normalized confusion matrix to {output_img}_normalized.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Counts)')
    plt.savefig(output_img + "_counts.png")
    print(f"Saved confusion matrix counts to {output_img}_counts.png")
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/visec_optimized.yaml")
    parser.add_argument("--checkpoint", type=str, default="Results/rank1_f172.46_acc72.92_loss2.5503_epoch50.zip")
    parser.add_argument("--output", type=str, default="confusion_matrix")
    args = parser.parse_args()
    
    main(args.config, args.checkpoint, args.output)
