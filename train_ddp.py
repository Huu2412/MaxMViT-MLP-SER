
"""
Multi-GPU Training Script using PyTorch DistributedDataParallel (DDP).

Usage on Kaggle (2x T4):
    torchrun --nproc_per_node=2 train_ddp.py --config configs/visec_cross_eval.yaml

This wraps the same logic as train.py but distributes across multiple GPUs.
"""

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import time
import os
import logging
import warnings
import numpy as np
import json
from sklearn.metrics import classification_report, recall_score, f1_score

# Suppress librosa n_fft warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal')

from utils import load_config, setup_logging, seed_everything, compute_class_weights, compute_accent_weights, compute_detailed_metrics, load_checkpoint
from data_loaders.factory import get_dataloaders_ddp

# Model imports
from model import MaxMViT_MLP, get_optimizer
from model_gmu import MaxMViT_MLP_GMU, get_optimizer_gmu
from model_crossattn import MaxMViT_MLP_CrossAttn, get_optimizer_crossattn


def is_main_process():
    """Check if this is the main (rank 0) process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    """Initialize DDP process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Destroy DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model_and_optimizer(model_type, num_classes, lr, model_cfg, backbone_lr=None, head_lr=None):
    """
    Factory function to get model and optimizer based on model_type.
    (Same as train.py)
    """
    hidden_size = model_cfg.get('hidden_size', 512)
    dropout_rate = model_cfg.get('dropout_rate', 0.2)
    freeze_backbone = model_cfg.get('freeze_backbone', False)
    unfreeze_last_n_blocks = model_cfg.get('unfreeze_last_n_blocks', 0)

    if freeze_backbone and is_main_process():
        logging.info(f"Backbone freezing ENABLED (unfreeze_last_n_blocks={unfreeze_last_n_blocks})")
    if (backbone_lr is not None or head_lr is not None) and is_main_process():
        logging.info(f"Discriminative LR: backbone_lr={backbone_lr}, head_lr={head_lr}")
    
    if model_type == 'original':
        if is_main_process(): logging.info("Using Original Model (Simple Concatenation Fusion)")
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MaxMViT_MLP(
            num_classes=num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone, unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
        optimizers = get_optimizer(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
        
    elif model_type == 'gmu':
        if is_main_process(): logging.info("Using GMU Model (Gated Multimodal Unit Fusion)")
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MaxMViT_MLP_GMU(
            num_classes=num_classes, 
            hidden_size=hidden_size, 
            dropout_rate=dropout_rate,
            fusion_hidden_dim=fusion_hidden_dim,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
        optimizers = get_optimizer_gmu(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
        
    elif model_type == 'crossattn':
        if is_main_process(): logging.info("Using Cross-Attention Model (Bidirectional Cross-Attention Fusion)")
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        num_heads = model_cfg.get('num_heads', 8)
        fusion_type = model_cfg.get('fusion_type', 'concat')
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
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
        optimizers = get_optimizer_crossattn(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
        
    elif model_type == 'maxvit_unimodal':
        if is_main_process(): logging.info("Using MaxViT Unimodal Model with Self-Attention (CQT only)")
        from model_unimodal import MaxViT_SelfAttn_MLP, get_optimizer_unimodal
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MaxViT_SelfAttn_MLP(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
        optimizers = get_optimizer_unimodal(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
        
    elif model_type == 'mvitv2_unimodal':
        if is_main_process(): logging.info("Using MViTv2 Unimodal Model with Self-Attention (Mel-STFT only)")
        from model_unimodal import MViTv2_SelfAttn_MLP, get_optimizer_unimodal
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MViTv2_SelfAttn_MLP(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
        optimizers = get_optimizer_unimodal(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: original, gmu, crossattn, maxvit_unimodal, mvitv2_unimodal")
        
    return model, optimizers


def train_ddp(config_path):
    # ── DDP Setup ──
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    # 1. Load Config & Setup (only rank 0 sets up logging/dirs)
    config = load_config(config_path)
    if is_main_process():
        ckpt_dir = setup_logging(config)
    else:
        # Suppress logging on non-main processes
        logging.basicConfig(level=logging.WARNING)
        ckpt_dir = os.path.join(
            config.get("paths", {}).get("checkpoint_dir", "checkpoints"),
            config.get("experiment_name", "experiment")
        )

    # Wait for rank 0 to create directories
    dist.barrier()

    # 2. Extract Configs
    train_cfg = config['training']
    model_cfg = config['model']
    
    SEED = train_cfg.get('seed', 42)
    seed_everything(SEED)
    
    EPOCHS = train_cfg.get('epochs', 50)
    LR = train_cfg.get('lr', 0.0002)
    PATIENCE = train_cfg.get('patience', 5)

    BACKBONE_LR = train_cfg.get('backbone_lr', None)
    HEAD_LR = train_cfg.get('head_lr', None)
    LABEL_SMOOTHING = train_cfg.get('label_smoothing', 0.0)
    USE_CLASS_WEIGHT = train_cfg.get('class_weight', False)
    USE_AMP = train_cfg.get('amp', False) and torch.cuda.is_available()
    GRAD_CLIP_NORM = train_cfg.get('grad_clip_norm', 1.0)
    CHECKPOINT_METRIC = train_cfg.get('checkpoint_metric', 'macro_f1')

    if is_main_process():
        logging.info(f"DDP Training: {world_size} GPUs")
        logging.info(f"AMP: {USE_AMP} | Label smoothing: {LABEL_SMOOTHING} | Class weight: {USE_CLASS_WEIGHT} "
                     f"| Grad clip norm: {GRAD_CLIP_NORM} | Checkpoint metric: {CHECKPOINT_METRIC}")
    
    # 3. Data (with DistributedSampler)
    train_loader, val_loader, train_sampler = get_dataloaders_ddp(config, rank, world_size)
    if not train_loader:
        if is_main_process():
            logging.error("Failed to load data.")
        cleanup_ddp()
        return

    if is_main_process():
        # Log augmentation configs
        ds_args = config.get('dataset', {}).get('args', {})
        logging.info("--- Data Augmentation Settings ---")
        waveform_cfg = ds_args.get('waveform_augment', None)
        spec_cfg = ds_args.get('spec_augment', None)
        
        if waveform_cfg:
            logging.info(f"  waveform_augment (OneOf): {waveform_cfg}")
            if spec_cfg:
                logging.info(f"  spec_augment: {spec_cfg}")
        else:
            for aug_name in ['spec_augment', 'pitch_shift', 'time_shift']:
                aug_cfg = ds_args.get(aug_name, None)
                if aug_cfg:
                    logging.info(f"  {aug_name}: {aug_cfg}")
                else:
                    logging.info(f"  {aug_name}: Disabled")
        logging.info("----------------------------------")

    # 4. Model
    num_classes = model_cfg.get('num_classes', 4)
    model_type = model_cfg.get('type', 'crossattn')
    
    aux_cfg = config.get('auxiliary_task', {})
    aux_enabled = aux_cfg.get('enabled', False)
    aux_alpha = aux_cfg.get('alpha', 0.3)
    num_accent_classes = aux_cfg.get('num_accent_classes', 0) if aux_enabled else 0
    
    if num_accent_classes > 0:
        model_cfg['num_accent_classes'] = num_accent_classes
    
    if is_main_process():
        logging.info(f"Initializing Model with {num_classes} classes...")
        if aux_enabled:
            logging.info(f"Auxiliary Task: Region Recognition ({num_accent_classes} accent classes, alpha={aux_alpha})")
    
    model, optimizers = get_model_and_optimizer(
        model_type, num_classes, LR, model_cfg,
        backbone_lr=BACKBONE_LR, head_lr=HEAD_LR
    )
    model.to(device)
    
    # ── Wrap model with DDP ──
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # AMP GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    sched_cfg = train_cfg.get('scheduler', {})
    scheduler_type = sched_cfg.get('type', 'plateau')
    if scheduler_type == 'cosine':
        warmup_epochs = sched_cfg.get('warmup_epochs', 0)
        schedulers = []
        for opt in optimizers:
            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_epochs)
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - warmup_epochs, eta_min=float(sched_cfg.get('min_lr', 1e-6)))
                schedulers.append(torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]))
            else:
                schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=float(sched_cfg.get('min_lr', 1e-6))))
        SCHEDULER_STEPS_ON_METRIC = False
        if is_main_process():
            logging.info(f"Using CosineAnnealingLR scheduler (T_max={EPOCHS}, Warmup={warmup_epochs})")
    else:
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', 
            factor=sched_cfg.get('factor', 0.1), 
            patience=sched_cfg.get('patience', 2), 
            min_lr=float(sched_cfg.get('min_lr', 1e-6))
        ) for opt in optimizers]
        SCHEDULER_STEPS_ON_METRIC = True
        if is_main_process():
            logging.info(f"Using ReduceLROnPlateau scheduler (factor={sched_cfg.get('factor', 0.1)}, "
                         f"patience={sched_cfg.get('patience', 2)})")

    # Class weighting
    class_weights = None
    if USE_CLASS_WEIGHT:
        class_weights = compute_class_weights(train_loader.dataset, num_classes).to(device)
        if is_main_process():
            logging.info(f"Emotion class weights (inverse-frequency): {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    # Auxiliary task loss
    criterion_accent = None
    if aux_enabled and num_accent_classes > 0:
        accent_weights_list = aux_cfg.get('accent_weights', None)
        auto_accent_weight = aux_cfg.get('auto_class_weights', False) or aux_cfg.get('accent_class_weight', False) or accent_weights_list == "auto"
        
        if auto_accent_weight:
            accent_weights = compute_accent_weights(train_loader.dataset, num_accent_classes).to(device)
            criterion_accent = nn.CrossEntropyLoss(weight=accent_weights, ignore_index=-1)
            if is_main_process():
                logging.info(f"Auto-computed Region/Accent class weights (inverse-frequency): {accent_weights.tolist()}")
        elif accent_weights_list and isinstance(accent_weights_list, (list, tuple)):
            accent_weights = torch.tensor(accent_weights_list, dtype=torch.float).to(device)
            criterion_accent = nn.CrossEntropyLoss(weight=accent_weights, ignore_index=-1)
            if is_main_process():
                logging.info(f"Using configured accent loss weights: {accent_weights_list}")
        else:
            criterion_accent = nn.CrossEntropyLoss(ignore_index=-1)
            if is_main_process():
                logging.info("Accent loss weights: unweighted CrossEntropyLoss")
        if is_main_process():
            logging.info(f"Accent auxiliary task configured: alpha={aux_alpha}")
    
    # Class names for detailed prediction metrics and reporting
    class_names = getattr(val_loader.dataset if val_loader else train_loader.dataset, 'target_classes', None)
    if class_names is None:
        class_names = config.get('dataset', {}).get('args', {}).get('target_classes', None)
    if class_names is None:
        class_names = ['happy', 'neutral', 'sad', 'angry'] if num_classes == 4 else [f"Class_{i}" for i in range(num_classes)]

    # 6. Training Loop
    if is_main_process():
        logging.info("Starting DDP Training...")
    patience_counter = 0
    top_k_checkpoints = []
    TOP_K = 3
    
    for epoch in range(EPOCHS):
        model.train()
        
        # ── CRITICAL: Set epoch on DistributedSampler for proper shuffling ──
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        total_loss = 0
        total_loss_emo = 0
        total_loss_acc = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 4:
                cqt, mel, label, accent_label = batch
                accent_label = accent_label.to(device)
                if batch_idx == 0 and epoch == 0 and is_main_process():
                    logging.info(f"DEBUG - Batch 0 accent_labels: {accent_label.unique().tolist()}")
            else:
                cqt, mel, label = batch
                accent_label = None
            cqt, mel, label = cqt.to(device), mel.to(device), label.to(device)
            
            for opt in optimizers: opt.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                model_output = model(cqt, mel)
                if isinstance(model_output, tuple):
                    outputs, accent_logits = model_output
                else:
                    outputs = model_output
                    accent_logits = None

                loss_emo = criterion(outputs, label)
                loss = loss_emo

                loss_acc_value = 0.0
                if criterion_accent is not None and accent_logits is not None and accent_label is not None:
                    valid_mask = accent_label != -1
                    if valid_mask.any():
                        loss_acc = criterion_accent(accent_logits[valid_mask], accent_label[valid_mask])
                        loss = loss_emo + aux_alpha * loss_acc
                        loss_acc_value = loss_acc.item()

            scaler.scale(loss).backward()

            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
            
            total_loss += loss.item()
            total_loss_emo += loss_emo.item()
            total_loss_acc += loss_acc_value
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            if batch_idx % 20 == 0 and is_main_process():
                logging.debug(f"Batch {batch_idx}: Loss {loss.item():.4f}")

        # Epoch Metrics (per-GPU, then aggregate)
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation (run on all GPUs, but only rank 0 computes final metrics)
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 4:
                        cqt, mel, label, _ = batch
                    else:
                        cqt, mel, label = batch
                    cqt, mel, label = cqt.to(device), mel.to(device), label.to(device)
                    
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
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
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(label.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            val_f1 = 100. * f1_score(val_labels, val_preds, average='macro', zero_division=0)
        else:
            val_loss = train_loss
            val_acc = train_acc
            val_f1 = train_acc

        # Step Scheduler
        for sch in schedulers:
            if SCHEDULER_STEPS_ON_METRIC:
                sch.step(val_loss)
            else:
                sch.step()

        # ── Only rank 0 handles logging, checkpointing, early stopping ──
        if is_main_process():
            epoch_time = time.time() - start_time
            if criterion_accent is not None:
                avg_loss_emo = total_loss_emo / len(train_loader)
                avg_loss_acc = total_loss_acc / len(train_loader)
                logging.info(f"Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} L_emo:{avg_loss_emo:.4f} L_acc:{avg_loss_acc:.4f} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}% mF1:{val_f1:.1f}%] | Time: {epoch_time:.1f}s")
            else:
                logging.info(f"Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}% mF1:{val_f1:.1f}%] | Time: {epoch_time:.1f}s")
            
            # Compute detailed metrics on validation predictions
            detailed_metrics = None
            if val_loader is not None and len(val_labels) > 0:
                detailed_metrics = compute_detailed_metrics(val_labels, val_preds, class_names=class_names)

            # Checkpointing — save the underlying model (without DDP wrapper)
            filename = f"epoch_{epoch+1}.pth"
            save_path = os.path.join(ckpt_dir, filename)
            checkpoint_payload = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'model_state_dict': model.module.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_loss': val_loss,
                'class_names': class_names,
                'confusion_matrix': detailed_metrics['confusion_matrix'] if detailed_metrics else [],
                'per_class_results': detailed_metrics['per_class'] if detailed_metrics else {},
                'prediction_breakdown': detailed_metrics['summary_text'] if detailed_metrics else "",
            }
            torch.save(checkpoint_payload, save_path)
            
            rank_key = 'f1' if CHECKPOINT_METRIC == 'macro_f1' else 'acc'
            top_k_checkpoints.append({'acc': val_acc, 'f1': val_f1, 'loss': val_loss, 'epoch': epoch+1, 'path': save_path})
            top_k_checkpoints.sort(key=lambda x: x[rank_key], reverse=True)
            
            while len(top_k_checkpoints) > TOP_K:
                to_remove = top_k_checkpoints.pop()
                if os.path.exists(to_remove['path']):
                    os.remove(to_remove['path'])
                    logging.info(f"Removed checkpoint: {os.path.basename(to_remove['path'])} "
                                 f"(Acc: {to_remove['acc']:.2f}%, mF1: {to_remove['f1']:.2f}%)")
                    
            if top_k_checkpoints[0]['epoch'] == epoch + 1:
                patience_counter = 0
                logging.info(f"New Best Model! Acc: {val_acc:.2f}% | mF1: {val_f1:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logging.info("Early stopping triggered.")
        
        # Broadcast early stop decision from rank 0 to all ranks
        early_stop = torch.tensor([0], device=device)
        if is_main_process() and patience_counter >= PATIENCE:
            early_stop = torch.tensor([1], device=device)
        dist.broadcast(early_stop, src=0)
        if early_stop.item() == 1:
            break

    # ── Final evaluation (rank 0 only) ──
    if is_main_process() and top_k_checkpoints:
        logging.info("Renaming Top Checkpoints...")
        rank1_filename = None
        for i, ckpt in enumerate(top_k_checkpoints):
            ckpt_rank = i + 1
            new_name = f"rank{ckpt_rank}_f1{ckpt['f1']:.2f}_acc{ckpt['acc']:.2f}_loss{ckpt['loss']:.4f}_epoch{ckpt['epoch']}.pth"
            new_path = os.path.join(ckpt_dir, new_name)
            if os.path.exists(ckpt['path']):
                os.rename(ckpt['path'], new_path)
                logging.info(f"Saved Rank {ckpt_rank}: {new_name}")
                if ckpt_rank == 1:
                    rank1_filename = new_name
                    
        if rank1_filename:
            rank1_path = os.path.join(ckpt_dir, rank1_filename)
            logging.info(f"Running full evaluation on best checkpoint: {rank1_filename}")
            
            # Load weights into the underlying model safely (without DDP wrapper)
            load_checkpoint(rank1_path, model.module, device=device)
            
            total_params = sum(p.numel() for p in model.module.parameters())
            trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
            
            flops_str = "N/A"
            try:
                import thop
                cqt_dummy = torch.randn(1, 3, 224, 224).to(device)
                mel_dummy = torch.randn(1, 3, 224, 224).to(device)
                flops_raw, _ = thop.profile(model.module, inputs=(cqt_dummy, mel_dummy), verbose=False)
                flops_str = f"{flops_raw / 1e9:.2f} GFLOPs ({flops_raw:,})"
            except Exception as e:
                logging.warning(f"Could not calculate FLOPs: {e}")
                
            model.eval()
            all_preds = []
            all_labels = []
            total_time = 0.0
            num_samples_timed = 0
            with torch.no_grad():
                cqt_dummy = torch.randn(1, 3, 224, 224).to(device)
                mel_dummy = torch.randn(1, 3, 224, 224).to(device)
                for _ in range(5):
                    _ = model(cqt_dummy, mel_dummy)
                    
                for batch in val_loader:
                    if len(batch) == 4:
                        cqt, mel, label, _ = batch
                    else:
                        cqt, mel, label = batch
                    cqt, mel = cqt.to(device), mel.to(device)
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
            
            ua = 100.0 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
            mf1 = 100.0 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
            weighted_f1 = 100.0 * f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Thống kê chi tiết số lượng nhãn dự đoán đúng và các nhãn bị dự đoán sai
            final_detailed = compute_detailed_metrics(all_labels, all_preds, class_names=class_names)
            logging.info("\n" + final_detailed['summary_text'])
            
            report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
            
            report_filename = rank1_filename.replace(".pth", "_report.txt")
            report_path = os.path.join(ckpt_dir, report_filename)
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Checkpoint: {rank1_path}\n")
                f.write(f"Config: {config_path}\n")
                f.write(f"Training: {world_size} GPUs (DDP)\n")
                f.write(f"Validation Accuracy: {top_k_checkpoints[0]['acc']:.2f}%\n")
                f.write(f"Validation Macro F1: {top_k_checkpoints[0]['f1']:.2f}%\n")
                f.write(f"Validation Loss: {top_k_checkpoints[0]['loss']:.4f}\n")
                f.write("=" * 80 + "\n")
                f.write(final_detailed['summary_text'] + "\n")
                f.write("=" * 80 + "\n")
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\n" + "=" * 80 + "\n")
                f.write("BENCHMARKS & ADDED METRICS:\n")
                f.write(f"Total Parameters: {total_params / 1e6:.2f}M ({total_params:,})\n")
                f.write(f"Trainable Parameters: {trainable_params / 1e6:.2f}M ({trainable_params:,})\n")
                f.write(f"Total FLOPs (per sample): {flops_str}\n")
                f.write(f"Inference Time per batch (size {batch_size}): {inf_time_batch:.2f} ms\n")
                f.write(f"Inference Time per sample: {inf_time_sample:.2f} ms\n")
                f.write("-" * 60 + "\n")
                f.write(f"Unweighted Accuracy (UA/UWA): {ua:.2f}%\n")
                f.write(f"Macro F1-score (mF1): {mf1:.2f}%\n")
                f.write(f"Weighted F1-score (F1): {weighted_f1:.2f}%\n")
                f.write("F1-score per class:\n")
                f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
                for i, name in enumerate(class_names):
                    if i < len(f1_per_class):
                        f.write(f"  - {name}: {f1_per_class[i]*100.0:.2f}%\n")
                        
            logging.info(f"Saved evaluation benchmarks report to: {report_path}")
            
            # Save structured JSON file with detailed predictions breakdown
            json_filename = rank1_filename.replace(".pth", "_predictions.json")
            json_path = os.path.join(ckpt_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    'checkpoint': rank1_filename,
                    'epoch': top_k_checkpoints[0]['epoch'],
                    'val_accuracy': top_k_checkpoints[0]['acc'],
                    'val_macro_f1': top_k_checkpoints[0]['f1'],
                    'val_loss': top_k_checkpoints[0]['loss'],
                    'class_names': class_names,
                    'confusion_matrix': final_detailed['confusion_matrix'],
                    'per_class_results': final_detailed['per_class'],
                }, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved detailed predictions JSON to: {json_path}")
            logging.info(f"Benchmarks:\nUA/UWA: {ua:.2f}%, mF1: {mf1:.2f}%, F1: {weighted_f1:.2f}%, FLOPs: {flops_str}, Inf time: {inf_time_sample:.2f} ms/sample")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    train_ddp(args.config)
