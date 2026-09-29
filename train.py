
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
import torch.nn as nn
import time
import logging
import warnings
import numpy as np
import json
from sklearn.metrics import classification_report, recall_score, f1_score


# Suppress librosa n_fft warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal')

from utils import load_config, setup_logging, seed_everything, compute_class_weights, compute_accent_weights, compute_detailed_metrics, load_checkpoint
from data_loaders import get_dataloaders

# Model imports
from model import MaxMViT_MLP, get_optimizer
from model_gmu import MaxMViT_MLP_GMU, get_optimizer_gmu
from model_crossattn import MaxMViT_MLP_CrossAttn, get_optimizer_crossattn
from model_disentangled import MaxMViT_MLP_Disentangled, get_optimizer_disentangled, DisentangledConstraintLoss

def get_model_and_optimizer(model_type, num_classes, lr, model_cfg, backbone_lr=None, head_lr=None):
    """
    Factory function to get model and optimizer based on model_type.
    
    Args:
        model_type: 'original', 'gmu', or 'crossattn'
        num_classes: Number of emotion classes
        lr: Learning rate (fallback if backbone_lr/head_lr not set)
        model_cfg: Model configuration dict
        backbone_lr: Discriminative LR for the pretrained MaxViT/MViTv2 backbones
            (low, since they're already pretrained). None = use `lr`.
        head_lr: Discriminative LR for the randomly-initialized fusion/MLP
            head (higher, since it needs to learn from scratch). None = use `lr`.
        
    Returns:
        model: The model instance
        optimizers: List of optimizers
    """
    hidden_size = model_cfg.get('hidden_size', 512)
    dropout_rate = model_cfg.get('dropout_rate', 0.2)
    freeze_backbone = model_cfg.get('freeze_backbone', False)
    unfreeze_last_n_blocks = model_cfg.get('unfreeze_last_n_blocks', 0)
    backbone_size = model_cfg.get('backbone_size', 'base')
    maxvit_variant = model_cfg.get('maxvit_variant', None)
    mvitv2_variant = model_cfg.get('mvitv2_variant', None)

    logging.info(f"Backbone configuration: scale='{backbone_size}' | MaxViT: {maxvit_variant or backbone_size} | MViTv2: {mvitv2_variant or backbone_size}")
    if freeze_backbone:
        logging.info(f"Backbone freezing ENABLED (unfreeze_last_n_blocks={unfreeze_last_n_blocks})")
    if backbone_lr is not None or head_lr is not None:
        logging.info(f"Discriminative LR: backbone_lr={backbone_lr}, head_lr={head_lr}")
    
    if model_type == 'original':
        logging.info("Using Original Model (Simple Concatenation Fusion)")
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MaxMViT_MLP(
            num_classes=num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone, unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=backbone_size, maxvit_variant=maxvit_variant, mvitv2_variant=mvitv2_variant
        )
        optimizers = get_optimizer(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
        
    elif model_type == 'gmu':
        logging.info("Using GMU Model (Gated Multimodal Unit Fusion)")
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MaxMViT_MLP_GMU(
            num_classes=num_classes, 
            hidden_size=hidden_size, 
            dropout_rate=dropout_rate,
            fusion_hidden_dim=fusion_hidden_dim,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=backbone_size,
            maxvit_variant=maxvit_variant,
            mvitv2_variant=mvitv2_variant
        )
        optimizers = get_optimizer_gmu(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
        
    elif model_type == 'crossattn':
        logging.info("Using Cross-Attention Model (Bidirectional Cross-Attention Fusion)")
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
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=backbone_size,
            maxvit_variant=maxvit_variant,
            mvitv2_variant=mvitv2_variant
        )
        optimizers = get_optimizer_crossattn(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
        
    elif model_type == 'maxvit_unimodal':
        logging.info("Using MaxViT Unimodal Model with Self-Attention (CQT only)")
        from model_unimodal import MaxViT_SelfAttn_MLP, get_optimizer_unimodal
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MaxViT_SelfAttn_MLP(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=backbone_size,
            maxvit_variant=maxvit_variant
        )
        optimizers = get_optimizer_unimodal(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)
        
    elif model_type == 'mvitv2_unimodal':
        logging.info("Using MViTv2 Unimodal Model with Self-Attention (Mel-STFT only)")
        from model_unimodal import MViTv2_SelfAttn_MLP, get_optimizer_unimodal
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MViTv2_SelfAttn_MLP(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=backbone_size,
            mvitv2_variant=mvitv2_variant
        )
        optimizers = get_optimizer_unimodal(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)

    elif model_type == 'disentangled':
        logging.info("Using Disentangled Model (Shared-Private Decomposition + 3-Way GMU Fusion)")
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        sub_dim = model_cfg.get('sub_dim', 256)
        num_accent_classes = model_cfg.get('num_accent_classes', 0)
        model = MaxMViT_MLP_Disentangled(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            sub_dim=sub_dim,
            fusion_dim=fusion_hidden_dim,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=backbone_size,
            maxvit_variant=maxvit_variant,
            mvitv2_variant=mvitv2_variant
        )
        optimizers = get_optimizer_disentangled(model, lr=lr, backbone_lr=backbone_lr, head_lr=head_lr)

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: original, gmu, crossattn, maxvit_unimodal, mvitv2_unimodal, disentangled")
        
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

    # --- New optimization knobs ---
    BACKBONE_LR = train_cfg.get('backbone_lr', None)   # discriminative LR for pretrained backbones
    HEAD_LR = train_cfg.get('head_lr', None)            # discriminative LR for new fusion/MLP layers
    LABEL_SMOOTHING = train_cfg.get('label_smoothing', 0.0)
    USE_CLASS_WEIGHT = train_cfg.get('class_weight', False)
    USE_AMP = train_cfg.get('amp', False) and torch.cuda.is_available()
    GRAD_CLIP_NORM = train_cfg.get('grad_clip_norm', 1.0)
    # 'macro_f1' (recommended, robust to class imbalance) or 'val_acc' (legacy behaviour)
    CHECKPOINT_METRIC = train_cfg.get('checkpoint_metric', 'macro_f1')
    logging.info(f"AMP: {USE_AMP} | Label smoothing: {LABEL_SMOOTHING} | Class weight: {USE_CLASS_WEIGHT} "
                 f"| Grad clip norm: {GRAD_CLIP_NORM} | Checkpoint metric: {CHECKPOINT_METRIC}")
    
    # 3. Data
    loaders = get_dataloaders(config)
    if not loaders or loaders[0] is None:
        logging.error("Failed to load data.")
        return
    train_loader = loaders[0]
    val_loader = loaders[1]
    test_loader = loaders[2] if len(loaders) > 2 else val_loader

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

    # Log accent distribution if applicable
    if train_loader.dataset and len(train_loader.dataset.indices) > 0:
        first_index = train_loader.dataset.indices[0]
        if len(first_index) == 3:
            accent_map = getattr(train_loader.dataset, 'accent_map', {'north': 0, 'south': 1, 'mid': 2})
            rev_accent_map = {v: k for k, v in accent_map.items()}
            
            # Train distribution
            train_accents = [idx_tuple[2] for idx_tuple in train_loader.dataset.indices]
            train_counts = {}
            for acc in train_accents:
                train_counts[acc] = train_counts.get(acc, 0) + 1
            train_dist_str = ", ".join([f"{rev_accent_map.get(k, f'class_{k}')}: {v} ({v/len(train_accents)*100:.2f}%)" for k, v in sorted(train_counts.items())])
            logging.info(f"Train accent distribution: {train_dist_str}")
            
            # Val distribution (only if val dataset also has accent labels, i.e. same format as train)
            if val_loader and hasattr(val_loader.dataset, 'indices') and len(val_loader.dataset.indices) > 0:
                first_val_index = val_loader.dataset.indices[0]
                if len(first_val_index) == 3:
                    val_accents = [idx_tuple[2] for idx_tuple in val_loader.dataset.indices]
                    val_counts = {}
                    for acc in val_accents:
                        val_counts[acc] = val_counts.get(acc, 0) + 1
                    val_dist_str = ", ".join([f"{rev_accent_map.get(k, f'class_{k}')}: {v} ({v/len(val_accents)*100:.2f}%)" for k, v in sorted(val_counts.items())])
                    logging.info(f"Val accent distribution: {val_dist_str}")
                else:
                    logging.info("Val set (cross-dataset) does not have accent labels — skipping accent distribution log.")

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
    
    model, optimizers = get_model_and_optimizer(
        model_type, num_classes, LR, model_cfg,
        backbone_lr=BACKBONE_LR, head_lr=HEAD_LR
    )
    model.to(DEVICE)

    # AMP GradScaler (no-op / disabled gracefully when USE_AMP=False)
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
        logging.info(f"Using CosineAnnealingLR scheduler (T_max={EPOCHS}, Warmup={warmup_epochs})")
    else:
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', 
            factor=sched_cfg.get('factor', 0.1), 
            patience=sched_cfg.get('patience', 2), 
            min_lr=float(sched_cfg.get('min_lr', 1e-6))
        ) for opt in optimizers]
        SCHEDULER_STEPS_ON_METRIC = True
        logging.info(f"Using ReduceLROnPlateau scheduler (factor={sched_cfg.get('factor', 0.1)}, "
                     f"patience={sched_cfg.get('patience', 2)})")

    # Optional class weighting for the (imbalanced) emotion classification task
    class_weights = None
    if USE_CLASS_WEIGHT:
        class_weights = compute_class_weights(train_loader.dataset, num_classes).to(DEVICE)
        logging.info(f"Emotion class weights (inverse-frequency): {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    # Auxiliary task loss (Weighted CrossEntropy for accent / Region Recognition)
    criterion_accent = None
    if aux_enabled and num_accent_classes > 0:
        accent_weights_list = aux_cfg.get('accent_weights', None)
        auto_accent_weight = aux_cfg.get('auto_class_weights', False) or aux_cfg.get('accent_class_weight', False) or accent_weights_list == "auto"
        
        if auto_accent_weight:
            accent_weights = compute_accent_weights(train_loader.dataset, num_accent_classes).to(DEVICE)
            criterion_accent = nn.CrossEntropyLoss(weight=accent_weights, ignore_index=-1)
            logging.info(f"Auto-computed Region/Accent class weights (inverse-frequency): {accent_weights.tolist()}")
        elif accent_weights_list and isinstance(accent_weights_list, (list, tuple)):
            accent_weights = torch.tensor(accent_weights_list, dtype=torch.float).to(DEVICE)
            criterion_accent = nn.CrossEntropyLoss(weight=accent_weights, ignore_index=-1)
            logging.info(f"Using configured accent loss weights: {accent_weights_list}")
        else:
            criterion_accent = nn.CrossEntropyLoss(ignore_index=-1)
            logging.info("Accent loss weights: unweighted CrossEntropyLoss")
        logging.info(f"Accent auxiliary task configured: alpha={aux_alpha}")

    # Disentanglement constraint loss (only for disentangled model)
    disentangle_cfg = config.get('disentangle', {})
    disentangle_enabled = disentangle_cfg.get('enable', False) and model_type == 'disentangled'
    criterion_disentangle = None
    if disentangle_enabled:
        dis_beta = disentangle_cfg.get('beta_recon', 0.1)
        dis_gamma = disentangle_cfg.get('gamma_sim', 0.05)
        dis_delta = disentangle_cfg.get('delta_orth', 0.05)
        dis_warmup = disentangle_cfg.get('warmup_epochs', 5)
        criterion_disentangle = DisentangledConstraintLoss(
            weight_recon=dis_beta, weight_sim=dis_gamma,
            weight_orth=dis_delta, warmup_epochs=dis_warmup
        )
        logging.info(f"Disentanglement constraints ENABLED: β_recon={dis_beta}, "
                     f"γ_sim={dis_gamma}, δ_orth={dis_delta}, warmup={dis_warmup} epochs")
    
    # Class names for detailed prediction metrics and reporting
    class_names = getattr(val_loader.dataset if val_loader else train_loader.dataset, 'target_classes', None)
    if class_names is None:
        class_names = config.get('dataset', {}).get('args', {}).get('target_classes', None)
    if class_names is None:
        class_names = ['happy', 'neutral', 'sad', 'angry'] if num_classes == 4 else [f"Class_{i}" for i in range(num_classes)]

    # 6. Training Loop
    logging.info("Starting Training...")
    patience_counter = 0
    top_k_checkpoints = [] # {'acc': float, 'f1': float, 'loss': float, 'epoch': int, 'path': str}
    TOP_K = 3
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_loss_emo = 0
        total_loss_acc = 0
        total_loss_dis = 0
        valid_batches_count = 0
        accent_batches_count = 0
        dis_batches_count = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 4:
                cqt, mel, label, accent_label = batch
                accent_label = accent_label.to(DEVICE)
                if batch_idx == 0 and epoch == 0:
                    logging.info(f"DEBUG - Batch 0 accent_labels: {accent_label.unique().tolist()}")
            else:
                cqt, mel, label = batch
                accent_label = None
            cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
            
            # Sanitize inputs to prevent NaNs/Infs entering the network
            cqt = torch.nan_to_num(cqt, nan=0.0, posinf=0.0, neginf=0.0)
            mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
            
            for opt in optimizers: opt.zero_grad()
            
            # Forward pass (autocast is a no-op when USE_AMP=False)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                model_output = model(cqt, mel)
                
                # Handle disentangled model (returns dict in training mode)
                if isinstance(model_output, dict):
                    outputs = model_output['emotion_logits']
                    accent_logits = model_output.get('accent_logits', None)
                elif isinstance(model_output, tuple):
                    outputs, accent_logits = model_output
                else:
                    outputs = model_output
                    accent_logits = None

                # Primary loss (Emotion)
                loss_emo = criterion(outputs, label)
                if not torch.isfinite(loss_emo):
                    logging.warning(f"Epoch {epoch+1} Batch {batch_idx}: Non-finite loss_emo ({loss_emo.item() if hasattr(loss_emo, 'item') else loss_emo}). Skipping batch.")
                    for opt in optimizers: opt.zero_grad()
                    continue
                loss = loss_emo

                # Auxiliary loss (Accent) - Strictly filter valid classes: 0 <= acc < num_accent_classes
                loss_acc_value = 0.0
                if criterion_accent is not None and accent_logits is not None and accent_label is not None:
                    valid_mask = (accent_label >= 0) & (accent_label < num_accent_classes)
                    if valid_mask.any():
                        loss_acc = criterion_accent(accent_logits[valid_mask], accent_label[valid_mask])
                        if torch.isfinite(loss_acc):
                            loss = loss_emo + aux_alpha * loss_acc
                            loss_acc_value = loss_acc.item()
                        else:
                            logging.warning(f"Epoch {epoch+1} Batch {batch_idx}: Non-finite loss_acc ({loss_acc.item()}). Ignoring accent loss for this batch.")

                # Disentanglement auxiliary losses (L_recon + L_sim + L_orth)
                loss_dis_value = 0.0
                if criterion_disentangle is not None and isinstance(model_output, dict):
                    dis_loss, dis_dict = criterion_disentangle(
                        z_cqt=model_output['z_cqt'], z_mel=model_output['z_mel'],
                        h_cqt_s=model_output['h_cqt_s'], h_cqt_p=model_output['h_cqt_p'],
                        z_cqt_recon=model_output['z_cqt_recon'],
                        h_mel_s=model_output['h_mel_s'], h_mel_p=model_output['h_mel_p'],
                        z_mel_recon=model_output['z_mel_recon'],
                        current_epoch=epoch
                    )
                    if torch.isfinite(dis_loss):
                        loss = loss + dis_loss
                        loss_dis_value = dis_loss.item()
                    else:
                        logging.warning(f"Epoch {epoch+1} Batch {batch_idx}: Non-finite disentangle loss. Ignoring.")

            if not torch.isfinite(loss):
                logging.warning(f"Epoch {epoch+1} Batch {batch_idx}: Non-finite combined loss. Skipping backward pass.")
                for opt in optimizers: opt.zero_grad()
                continue

            if USE_AMP:
                scaler.scale(loss).backward()
                for opt in optimizers:
                    scaler.unscale_(opt)
                
                # Check for inf/nan gradients across all parameters BEFORE clipping.
                # Calling clip_grad_norm_ when grads have inf produces NaNs (inf * 0.0 = NaN)!
                has_inf_or_nan = any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if not has_inf_or_nan:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                else:
                    logging.debug(f"Epoch {epoch+1} Batch {batch_idx}: Inf/NaN gradients detected in AMP. Skipping clip_grad_norm_ to allow scaler backoff.")
                
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                has_inf_or_nan = any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if not has_inf_or_nan:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                    for opt in optimizers:
                        opt.step()
                else:
                    logging.warning(f"Epoch {epoch+1} Batch {batch_idx}: Inf/NaN gradients detected without AMP. Skipping step.")
                    for opt in optimizers: opt.zero_grad()
            
            total_loss += loss.item()
            total_loss_emo += loss_emo.item()
            if loss_acc_value > 0.0:
                total_loss_acc += loss_acc_value
                accent_batches_count += 1
            if loss_dis_value > 0.0:
                total_loss_dis += loss_dis_value
                dis_batches_count += 1
            valid_batches_count += 1
            
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            # Log every 20 batches (optional)
            if batch_idx % 20 == 0:
                 logging.debug(f"Batch {batch_idx}: Loss {loss.item():.4f}")

        # Epoch Metrics
        train_loss = total_loss / max(1, valid_batches_count)
        train_acc = 100. * correct / max(1, total)
        
        # Validation
        val_loss = 0
        val_batches_count = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 4:
                        cqt, mel, label, _ = batch  # Ignore accent label in validation
                    else:
                        cqt, mel, label = batch
                    cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
                    cqt = torch.nan_to_num(cqt, nan=0.0, posinf=0.0, neginf=0.0)
                    mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        model_output = model(cqt, mel)
                        if isinstance(model_output, tuple):
                            outputs, _ = model_output
                        else:
                            outputs = model_output
                        loss = criterion(outputs, label)

                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        val_batches_count += 1
                    _, predicted = outputs.max(1)
                    val_total += label.size(0)
                    val_correct += predicted.eq(label).sum().item()
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(label.cpu().numpy())
            
            val_loss = val_loss / max(1, val_batches_count)
            val_acc = 100. * val_correct / max(1, val_total)
            val_f1 = 100. * f1_score(val_labels, val_preds, average='macro', zero_division=0)
        else:
            val_loss = train_loss
            val_acc = train_acc
            val_f1 = train_acc  # no separate val set: fall back to train acc as a proxy

        # Step Scheduler (plateau schedulers need the metric; cosine steps unconditionally)
        for sch in schedulers:
            if SCHEDULER_STEPS_ON_METRIC:
                sch.step(val_loss)
            else:
                sch.step()

        # Logging
        epoch_time = time.time() - start_time
        avg_loss_dis = (total_loss_dis / dis_batches_count) if dis_batches_count > 0 else 0.0
        if criterion_accent is not None:
            avg_loss_emo = total_loss_emo / max(1, valid_batches_count)
            avg_loss_acc = (total_loss_acc / accent_batches_count) if accent_batches_count > 0 else 0.0
            dis_str = f" L_dis:{avg_loss_dis:.4f}" if criterion_disentangle is not None else ""
            logging.info(f"Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} L_emo:{avg_loss_emo:.4f} L_acc:{avg_loss_acc:.4f}{dis_str} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}% mF1:{val_f1:.1f}%] | Time: {epoch_time:.1f}s")
        else:
            dis_str = f" L_dis:{avg_loss_dis:.4f}" if criterion_disentangle is not None else ""
            logging.info(f"Epoch {epoch+1:02d} | Train [L:{train_loss:.4f}{dis_str} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}% mF1:{val_f1:.1f}%] | Time: {epoch_time:.1f}s")
        
        # Compute detailed metrics on validation predictions
        detailed_metrics = None
        if val_loader is not None and len(val_labels) > 0:
            detailed_metrics = compute_detailed_metrics(val_labels, val_preds, class_names=class_names)

        # Checkpointing Strategy (Top-K)
        filename = f"epoch_{epoch+1}.pth"
        save_path = os.path.join(ckpt_dir, filename)
        checkpoint_payload = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_loss': val_loss,
            'class_names': class_names,
            'confusion_matrix': detailed_metrics['confusion_matrix'] if detailed_metrics else [],
            'per_class_results': detailed_metrics['per_class'] if detailed_metrics else {},
            'prediction_breakdown': detailed_metrics['summary_text'] if detailed_metrics else "",
        }
        torch.save(checkpoint_payload, save_path)
        
        # Maintain Top-K list (sorted by CHECKPOINT_METRIC descending - highest first).
        # Default is macro_f1: robust to class imbalance, unlike raw accuracy which can
        # be dominated by majority classes.
        rank_key = 'f1' if CHECKPOINT_METRIC == 'macro_f1' else 'acc'
        top_k_checkpoints.append({'acc': val_acc, 'f1': val_f1, 'loss': val_loss, 'epoch': epoch+1, 'path': save_path})
        top_k_checkpoints.sort(key=lambda x: x[rank_key], reverse=True)
        
        # Cleanup - remove worst checkpoints
        while len(top_k_checkpoints) > TOP_K:
            to_remove = top_k_checkpoints.pop() # Worst one (lowest ranked metric)
            if os.path.exists(to_remove['path']):
                os.remove(to_remove['path'])
                logging.info(f"Removed checkpoint: {os.path.basename(to_remove['path'])} "
                             f"(Acc: {to_remove['acc']:.2f}%, mF1: {to_remove['f1']:.2f}%)")
                
        # Early Stopping based on the same ranking metric
        if top_k_checkpoints[0]['epoch'] == epoch + 1:
            patience_counter = 0 # New Best
            logging.info(f"New Best Model! Acc: {val_acc:.2f}% | mF1: {val_f1:.2f}%")
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
        new_name = f"rank{rank}_f1{ckpt['f1']:.2f}_acc{ckpt['acc']:.2f}_loss{ckpt['loss']:.4f}_epoch{ckpt['epoch']}.pth"
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
        
        # Load weights into model safely
        load_checkpoint(rank1_path, model, device=DEVICE)
        
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
                
        eval_target_loader = test_loader if 'test_loader' in locals() and test_loader is not None else val_loader
        eval_split_name = "TEST Set (Hold-out)" if eval_target_loader is test_loader else "VALIDATION Set"
        logging.info(f"Running final evaluation on {eval_split_name} ({len(eval_target_loader.dataset)} samples)...")
        
        with torch.no_grad():
            # Warmup
            cqt_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            mel_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            for _ in range(5):
                _ = model(cqt_dummy, mel_dummy)
                
            for batch in eval_target_loader:
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
        batch_size = eval_target_loader.batch_size if hasattr(eval_target_loader, 'batch_size') else 8
        inf_time_batch = inf_time_sample * batch_size
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        test_acc = 100.0 * np.mean(all_preds == all_labels)
        ua = 100.0 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
        mf1 = 100.0 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = 100.0 * f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Thống kê chi tiết số lượng nhãn dự đoán đúng và các nhãn bị dự đoán sai
        final_detailed = compute_detailed_metrics(all_labels, all_preds, class_names=class_names)
        logging.info("\n" + final_detailed['summary_text'])
        
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
        
        # Save evaluation report to txt file
        report_filename = rank1_filename.replace(".pth", "_report.txt")
        report_path = os.path.join(ckpt_dir, report_filename)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Checkpoint: {rank1_path}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Split Mode: 80% Train, 10% Val, 10% Test\n")
            f.write(f"Evaluation Target: {eval_split_name} ({len(eval_target_loader.dataset)} samples)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Validation Accuracy (Best Epoch): {top_k_checkpoints[0]['acc']:.2f}%\n")
            f.write(f"Validation Macro F1 (Best Epoch): {top_k_checkpoints[0]['f1']:.2f}%\n")
            f.write(f"Validation Loss: {top_k_checkpoints[0]['loss']:.4f}\n")
            f.write("=" * 80 + "\n")
            f.write(f"FINAL INDEPENDENT TEST RESULTS:\n")
            f.write(f"  Test Accuracy (WA): {test_acc:.2f}%\n")
            f.write(f"  Test Unweighted Accuracy (UA): {ua:.2f}%\n")
            f.write(f"  Test Macro F1 (mF1): {mf1:.2f}%\n")
            f.write(f"  Test Weighted F1: {weighted_f1:.2f}%\n")
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
                'eval_split': eval_split_name,
                'val_accuracy': top_k_checkpoints[0]['acc'],
                'val_macro_f1': top_k_checkpoints[0]['f1'],
                'val_loss': top_k_checkpoints[0]['loss'],
                'test_accuracy': float(test_acc),
                'test_macro_f1': float(mf1),
                'test_unweighted_accuracy': float(ua),
                'test_weighted_f1': float(weighted_f1),
                'class_names': class_names,
                'confusion_matrix': final_detailed['confusion_matrix'],
                'per_class_results': final_detailed['per_class'],
            }, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved detailed predictions JSON to: {json_path}")
        logging.info(f"Final Test Results -> Acc: {test_acc:.2f}%, UA: {ua:.2f}%, mF1: {mf1:.2f}%, F1: {weighted_f1:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    train(args.config)
