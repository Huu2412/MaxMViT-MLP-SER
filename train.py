
import argparse
import torch
import torch.nn as nn
import time
import os
import logging
import warnings
import numpy as np
from sklearn.metrics import classification_report, recall_score, f1_score


# Suppress librosa n_fft warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal')

from utils import load_config, setup_logging, seed_everything
from data_loaders import get_dataloaders

# Model imports
from model import MaxMViT_MLP, get_optimizer
from model_gmu import MaxMViT_MLP_GMU, get_optimizer_gmu
from model_crossattn import MaxMViT_MLP_CrossAttn, get_optimizer_crossattn

def get_model_and_optimizer(model_type, num_classes, lr, model_cfg):
    """
    Factory function to get model and optimizer based on model_type.
    
    Args:
        model_type: 'original', 'gmu', or 'crossattn'
        num_classes: Number of emotion classes
        lr: Learning rate
        model_cfg: Model configuration dict
        
    Returns:
        model: The model instance
        optimizers: List of optimizers
    """
    hidden_size = model_cfg.get('hidden_size', 512)
    dropout_rate = model_cfg.get('dropout_rate', 0.2)
    
    if model_type == 'original':
        logging.info("Using Original Model (Simple Concatenation Fusion)")
        model = MaxMViT_MLP(num_classes=num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate)
        optimizers = get_optimizer(model, lr=lr)
        
    elif model_type == 'gmu':
        logging.info("Using GMU Model (Gated Multimodal Unit Fusion)")
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MaxMViT_MLP_GMU(
            num_classes=num_classes, 
            hidden_size=hidden_size, 
            dropout_rate=dropout_rate,
            fusion_hidden_dim=fusion_hidden_dim,
            num_accent_classes=num_accent_classes
        )
        optimizers = get_optimizer_gmu(model, lr=lr)
        
    elif model_type == 'crossattn':
        logging.info("Using Cross-Attention Model (Bidirectional Cross-Attention Fusion)")
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        num_heads = model_cfg.get('num_heads', 8)
        fusion_type = model_cfg.get('fusion_type', 'concat')
        model = MaxMViT_MLP_CrossAttn(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            fusion_hidden_dim=fusion_hidden_dim,
            num_heads=num_heads,
            fusion_type=fusion_type
        )
        optimizers = get_optimizer_crossattn(model, lr=lr)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: original, gmu, crossattn")
        
    return model, optimizers

def train(config_path):
    # 1. Load Config & Setup
    config = load_config(config_path)
    ckpt_dir = setup_logging(config)
    
    # 2. Extract Configs
    train_cfg = config['training']
    model_cfg = config['model']
    
    SEED = train_cfg.get('seed', 42)
    seed_everything(SEED)
    
    DEVICE = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    EPOCHS = train_cfg.get('epochs', 50)
    LR = train_cfg.get('lr', 0.0002)
    PATIENCE = train_cfg.get('patience', 5)
    
    # 3. Data
    train_loader, val_loader = get_dataloaders(config)
    if not train_loader:
        logging.error("Failed to load data.")
        return

    # 4. Model - Select based on config
    num_classes = model_cfg.get('num_classes', 4)
    model_type = model_cfg.get('type', 'crossattn')  # Default to crossattn
    
    # Auxiliary task config (Region Recognition)
    aux_cfg = config.get('auxiliary_task', {})
    aux_enabled = aux_cfg.get('enabled', False)
    aux_alpha = aux_cfg.get('alpha', 0.3)
    num_accent_classes = aux_cfg.get('num_accent_classes', 0) if aux_enabled else 0
    
    # Pass num_accent_classes to model config
    if num_accent_classes > 0:
        model_cfg['num_accent_classes'] = num_accent_classes
    
    logging.info(f"Initializing Model with {num_classes} classes...")
    if aux_enabled:
        logging.info(f"Auxiliary Task: Region Recognition ({num_accent_classes} accent classes, alpha={aux_alpha})")
    
    model, optimizers = get_model_and_optimizer(model_type, num_classes, LR, model_cfg)
    model.to(DEVICE)

    
    sched_cfg = train_cfg.get('scheduler', {})
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', 
        factor=sched_cfg.get('factor', 0.1), 
        patience=sched_cfg.get('patience', 2), 
        min_lr=float(sched_cfg.get('min_lr', 1e-6))
    ) for opt in optimizers]
    
    criterion = nn.CrossEntropyLoss()
    
    # Auxiliary task loss (Weighted CrossEntropy for accent)
    criterion_accent = None
    if aux_enabled and num_accent_classes > 0:
        accent_weights_list = aux_cfg.get('accent_weights', None)
        if accent_weights_list:
            accent_weights = torch.tensor(accent_weights_list, dtype=torch.float).to(DEVICE)
            criterion_accent = nn.CrossEntropyLoss(weight=accent_weights, ignore_index=-1)
        else:
            criterion_accent = nn.CrossEntropyLoss(ignore_index=-1)
        logging.info(f"Accent loss: Weighted CrossEntropy (weights={accent_weights_list})")
    
    # 6. Training Loop
    logging.info("Starting Training...")
    best_val_acc = 0.0
    patience_counter = 0
    top_k_checkpoints = [] # {'acc': float, 'epoch': int, 'path': str}
    TOP_K = 3
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_loss_emo = 0
        total_loss_acc = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle both 3-element and 4-element batches
            if len(batch) == 4:
                cqt, mel, label, accent_label = batch
                accent_label = accent_label.to(DEVICE)
            else:
                cqt, mel, label = batch
                accent_label = None
            cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
            
            for opt in optimizers: opt.zero_grad()
            
            # Forward pass
            model_output = model(cqt, mel)
            if isinstance(model_output, tuple):
                outputs, accent_logits = model_output
            else:
                outputs = model_output
                accent_logits = None
            
            # Primary loss (Emotion)
            loss_emo = criterion(outputs, label)
            loss = loss_emo
            
            # Auxiliary loss (Accent) 
            loss_acc_value = 0.0
            if criterion_accent is not None and accent_logits is not None and accent_label is not None:
                # Only compute accent loss for samples with valid accent labels
                valid_mask = accent_label != -1
                if valid_mask.any():
                    loss_acc = criterion_accent(accent_logits[valid_mask], accent_label[valid_mask])
                    loss = loss_emo + aux_alpha * loss_acc
                    loss_acc_value = loss_acc.item()
            
            loss.backward()
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            for opt in optimizers: opt.step()
            
            total_loss += loss.item()
            total_loss_emo += loss_emo.item()
            total_loss_acc += loss_acc_value
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            # Log every 20 batches (optional)
            if batch_idx % 20 == 0:
                 logging.debug(f"Batch {batch_idx}: Loss {loss.item():.4f}")

        # Epoch Metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        val_loss = 0
        val_correct = 0
        val_total = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 4:
                        cqt, mel, label, _ = batch  # Ignore accent label in validation
                    else:
                        cqt, mel, label = batch
                    cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
                    
                    model_output = model(cqt, mel)
                    if isinstance(model_output, tuple):
                        outputs, _ = model_output
                    else:
                        outputs = model_output
                    
                    loss = criterion(outputs, label)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += label.size(0)
                    val_correct += predicted.eq(label).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
        else:
            val_loss = train_loss
            val_acc = train_acc

        # Step Scheduler
        for sch in schedulers: sch.step(val_loss)

        # Logging
        epoch_time = time.time() - start_time
        if criterion_accent is not None:
            avg_loss_emo = total_loss_emo / len(train_loader)
            avg_loss_acc = total_loss_acc / len(train_loader)
            logging.info(f"Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} L_emo:{avg_loss_emo:.4f} L_acc:{avg_loss_acc:.4f} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}%] | Time: {epoch_time:.1f}s")
        else:
            logging.info(f"Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}%] | Time: {epoch_time:.1f}s")
        
        # Checkpointing Strategy (Top-K)
        filename = f"epoch_{epoch+1}.pth"
        save_path = os.path.join(ckpt_dir, filename)
        torch.save(model.state_dict(), save_path)
        
        # Maintain Top-K list (sorted by accuracy descending - highest first)
        top_k_checkpoints.append({'acc': val_acc, 'loss': val_loss, 'epoch': epoch+1, 'path': save_path})
        top_k_checkpoints.sort(key=lambda x: x['acc'], reverse=True)
        
        # Cleanup - remove worst accuracy checkpoints
        while len(top_k_checkpoints) > TOP_K:
            to_remove = top_k_checkpoints.pop() # Worst one (lowest accuracy)
            if os.path.exists(to_remove['path']):
                os.remove(to_remove['path'])
                logging.info(f"Removed checkpoint: {os.path.basename(to_remove['path'])} (Acc: {to_remove['acc']:.2f}%)")
                
        # Early Stopping based on validation accuracy
        if top_k_checkpoints[0]['epoch'] == epoch + 1:
            patience_counter = 0 # New Best
            logging.info(f"New Best Model! Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logging.info("Early stopping triggered.")
                break

    # Final Rename
    logging.info("Renaming Top Checkpoints...")
    rank1_filename = None
    for i, ckpt in enumerate(top_k_checkpoints):
        rank = i + 1
        new_name = f"rank{rank}_acc{ckpt['acc']:.2f}_loss{ckpt['loss']:.4f}_epoch{ckpt['epoch']}.pth"
        new_path = os.path.join(ckpt_dir, new_name)
        if os.path.exists(ckpt['path']):
            os.rename(ckpt['path'], new_path)
            logging.info(f"Saved Rank {rank}: {new_name}")
            if rank == 1:
                rank1_filename = new_name
                
    if rank1_filename:
        # Load best model and run benchmarks
        rank1_path = os.path.join(ckpt_dir, rank1_filename)
        logging.info(f"Running full evaluation on best checkpoint: {rank1_filename}")
        
        # Load weights into model
        state_dict = torch.load(rank1_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate FLOPs
        flops_str = "N/A"
        try:
            import thop
            cqt_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            mel_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            flops_raw, _ = thop.profile(model, inputs=(cqt_dummy, mel_dummy), verbose=False)
            flops_str = f"{flops_raw / 1e9:.2f} GFLOPs ({flops_raw:,})"
        except Exception as e:
            logging.warning(f"Could not calculate FLOPs: {e}")
            
        # Measure inference time and generate predictions
        model.eval()
        all_preds = []
        all_labels = []
        total_time = 0.0
        num_samples_timed = 0
        with torch.no_grad():
            # Warmup
            cqt_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            mel_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            for _ in range(5):
                _ = model(cqt_dummy, mel_dummy)
                
            for batch in val_loader:
                if len(batch) == 4:
                    cqt, mel, label, _ = batch
                else:
                    cqt, mel, label = batch
                cqt, mel = cqt.to(DEVICE), mel.to(DEVICE)
                start_t = time.time()
                model_output = model(cqt, mel)
                total_time += time.time() - start_t
                
                if isinstance(model_output, tuple):
                    outputs, _ = model_output
                else:
                    outputs = model_output
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.numpy())
                num_samples_timed += cqt.size(0)
                
        inf_time_sample = (total_time / num_samples_timed) * 1000.0 if num_samples_timed > 0 else 0.0
        batch_size = val_loader.batch_size if hasattr(val_loader, 'batch_size') else 8
        inf_time_batch = inf_time_sample * batch_size
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate UA/UWA, mF1, F1 (weighted)
        ua = 100.0 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
        mf1 = 100.0 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = 100.0 * f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Class names
        class_names = getattr(val_loader.dataset, 'target_classes', None)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]
            
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
        
        # Save evaluation report to txt file
        report_filename = rank1_filename.replace(".pth", "_report.txt")
        report_path = os.path.join(ckpt_dir, report_filename)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Checkpoint: {rank1_path}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Validation Accuracy: {top_k_checkpoints[0]['acc']:.2f}%\n")
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
                    
        logging.info(f"Saved evaluation benchmarks report to: {report_path}")
        logging.info(f"Benchmarks:\nUA/UWA: {ua:.2f}%, mF1: {mf1:.2f}%, F1: {weighted_f1:.2f}%, FLOPs: {flops_str}, Inf time: {inf_time_sample:.2f} ms/sample")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    train(args.config)
