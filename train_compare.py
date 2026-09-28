"""
Train and Compare All Model Variants

This script trains all 3 fusion methods:
1. Original (Simple Concatenation)
2. GMU (Gated Multimodal Unit)
3. CrossAttn (Bidirectional Cross-Attention)

And generates a comparison table at the end.
"""

import argparse
import torch
import torch.nn as nn
import time
import os
import logging
import warnings
import json
from datetime import datetime
from tabulate import tabulate
import copy
from sklearn.metrics import recall_score, f1_score


# Suppress librosa n_fft warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal')

from utils import load_config, setup_logging, seed_everything
from data_loaders import get_dataloaders

# Model imports
from model import MaxMViT_MLP, get_optimizer
from model_gmu import MaxMViT_MLP_GMU, get_optimizer_gmu
from model_crossattn import MaxMViT_MLP_CrossAttn, get_optimizer_crossattn


def get_model_and_optimizer(model_type, num_classes, lr, model_cfg, backbone_size=None):
    """Factory function to get model and optimizer based on model_type."""
    hidden_size = model_cfg.get('hidden_size', 512)
    dropout_rate = model_cfg.get('dropout_rate', 0.2)
    num_accent_classes = model_cfg.get('num_accent_classes', 0)
    freeze_backbone = model_cfg.get('freeze_backbone', False)
    unfreeze_last_n_blocks = model_cfg.get('unfreeze_last_n_blocks', 0)
    
    current_backbone_size = backbone_size or model_cfg.get('backbone_size', 'base')
    maxvit_variant = model_cfg.get('maxvit_variant', None)
    mvitv2_variant = model_cfg.get('mvitv2_variant', None)
    
    if model_type == 'original':
        model = MaxMViT_MLP(
            num_classes=num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone, unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=current_backbone_size, maxvit_variant=maxvit_variant, mvitv2_variant=mvitv2_variant
        )
        optimizers = get_optimizer(model, lr=lr)
        
    elif model_type == 'gmu':
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        model = MaxMViT_MLP_GMU(
            num_classes=num_classes, 
            hidden_size=hidden_size, 
            dropout_rate=dropout_rate,
            fusion_hidden_dim=fusion_hidden_dim,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=current_backbone_size,
            maxvit_variant=maxvit_variant,
            mvitv2_variant=mvitv2_variant
        )
        optimizers = get_optimizer_gmu(model, lr=lr)
        
    elif model_type == 'crossattn':
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
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=current_backbone_size,
            maxvit_variant=maxvit_variant,
            mvitv2_variant=mvitv2_variant
        )
        optimizers = get_optimizer_crossattn(model, lr=lr)
        
    elif model_type == 'maxvit_unimodal':
        from model_unimodal import MaxViT_SelfAttn_MLP, get_optimizer_unimodal
        model = MaxViT_SelfAttn_MLP(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=current_backbone_size,
            maxvit_variant=maxvit_variant
        )
        optimizers = get_optimizer_unimodal(model, lr=lr)
        
    elif model_type == 'mvitv2_unimodal':
        from model_unimodal import MViTv2_SelfAttn_MLP, get_optimizer_unimodal
        model = MViTv2_SelfAttn_MLP(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_accent_classes=num_accent_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            backbone_size=current_backbone_size,
            mvitv2_variant=mvitv2_variant
        )
        optimizers = get_optimizer_unimodal(model, lr=lr)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return model, optimizers


def train_single_model(model_type, config, train_loader, val_loader, logger, backbone_size=None, run_name=None):
    """
    Train a single model variant and return results.
    
    Returns:
        dict with keys: model_type, best_val_acc, best_val_loss, best_epoch, total_time, etc.
    """
    train_cfg = config['training']
    model_cfg = config['model']
    
    DEVICE = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    EPOCHS = train_cfg.get('epochs', 50)
    LR = train_cfg.get('lr', 0.0002)
    PATIENCE = train_cfg.get('patience', 10)
    
    current_backbone_size = backbone_size or model_cfg.get('backbone_size', 'base')
    display_name = run_name or f"{model_type.upper()}-{current_backbone_size.upper()}"

    # Get model
    num_classes = model_cfg.get('num_classes', 4)
    model, optimizers = get_model_and_optimizer(model_type, num_classes, LR, model_cfg, backbone_size=current_backbone_size)
    model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"Training: {display_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    
    # Schedulers
    sched_cfg = train_cfg.get('scheduler', {})
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max',  # Now monitoring accuracy (higher is better)
        factor=sched_cfg.get('factor', 0.1), 
        patience=sched_cfg.get('patience', 3), 
        min_lr=float(sched_cfg.get('min_lr', 1e-6))
    ) for opt in optimizers]
    
    criterion = nn.CrossEntropyLoss()
    
    # Training state
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            cqt, mel, label = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            
            for opt in optimizers: opt.zero_grad()
            
            model_out = model(cqt, mel)
            outputs = model_out[0] if isinstance(model_out, tuple) else model_out
            loss = criterion(outputs, label)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            for opt in optimizers: opt.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                cqt, mel, label = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
                model_out = model(cqt, mel)
                outputs = model_out[0] if isinstance(model_out, tuple) else model_out
                loss = criterion(outputs, label)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += label.size(0)
                val_correct += predicted.eq(label).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Step Scheduler (monitoring accuracy now)
        for sch in schedulers: sch.step(val_acc)

        epoch_time = time.time() - epoch_start
        logger.info(f"[{model_type}] Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}%] | Time: {epoch_time:.1f}s")
        
        # Early Stopping based on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"[{model_type}] ★ New Best! Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"[{model_type}] Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    
    # Reload best model state dict for benchmarking
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    logger.info(f"[{model_type}] Running final benchmarks on best model (Epoch {best_epoch})...")
    
    # Calculate FLOPs using thop
    flops_val = 0.0
    try:
        import thop
        cqt_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        mel_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        flops_raw, _ = thop.profile(model, inputs=(cqt_dummy, mel_dummy), verbose=False)
        flops_val = flops_raw / 1e9  # in GFLOPs
    except Exception as e:
        logger.warning(f"Could not calculate FLOPs: {e}")
        
    # Measure inference time and gather predictions
    model.eval()
    all_preds = []
    all_labels = []
    num_batches_to_time = min(20, len(val_loader))
    total_inf_time = 0.0
    num_samples_timed = 0
    with torch.no_grad():
        # Warmup
        cqt_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        mel_dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        for _ in range(5):
            _ = model(cqt_dummy, mel_dummy)
            
        for idx, batch in enumerate(val_loader):
            cqt, mel, label = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2]
            start_t = time.time()
            model_out = model(cqt, mel)
            outputs = model_out[0] if isinstance(model_out, tuple) else model_out
            elapsed = time.time() - start_t
            
            if idx < num_batches_to_time:
                total_inf_time += elapsed
                num_samples_timed += cqt.size(0)
                
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.numpy() if hasattr(label, 'numpy') else label)
            
    inf_time_sample = (total_inf_time / num_samples_timed) * 1000.0 if num_samples_timed > 0 else 0.0
    batch_size = val_loader.batch_size if hasattr(val_loader, 'batch_size') else 8
    inf_time_batch = inf_time_sample * batch_size
    
    # Calculate evaluation metrics
    val_uwa = 100.0 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
    val_mf1 = 100.0 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    val_f1_weighted = 100.0 * f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    result = {
        'model_type': model_type,
        'backbone_size': current_backbone_size,
        'run_name': display_name,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'total_time': total_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'val_uwa': val_uwa,
        'val_mf1': val_mf1,
        'val_f1_weighted': val_f1_weighted,
        'flops': flops_val,
        'inference_time_sample_ms': inf_time_sample,
        'inference_time_batch_ms': inf_time_batch
    }

    logger.info(f"[{display_name}] Finished! Best Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return result


def generate_comparison_table(results, logger):
    """Generate a markdown comparison table from results."""
    
    # Prepare table data
    headers = [
        "Model", "Backbone", "Fusion Type", "Best Val Acc (%)", "Val UWA (%)", 
        "Val mF1 (%)", "FLOPs (G)", "Inf Sample (ms)", "Params (M)"
    ]
    
    fusion_names = {
        'original': 'Concatenation',
        'gmu': 'Gated Multimodal Unit',
        'crossattn': 'Cross-Attention',
        'maxvit_unimodal': 'CQT Unimodal',
        'mvitv2_unimodal': 'Mel Unimodal'
    }
    
    table_data = []
    for r in results:
        table_data.append([
            r.get('run_name', r['model_type'].upper()),
            r.get('backbone_size', 'base').upper(),
            fusion_names.get(r['model_type'], r['model_type']),
            f"{r['best_val_acc']:.2f}",
            f"{r.get('val_uwa', 0.0):.2f}",
            f"{r.get('val_mf1', 0.0):.2f}",
            f"{r.get('flops', 0.0):.2f}",
            f"{r.get('inference_time_sample_ms', 0.0):.2f}",
            f"{r['total_params']/1e6:.2f}"
        ])
    
    # Sort by best accuracy (descending)
    table_data.sort(key=lambda x: float(x[3]), reverse=True)
    
    # Generate table
    table_str = tabulate(table_data, headers=headers, tablefmt="pipe")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(table_str)
    logger.info("")
    
    # Find winner
    winner = max(results, key=lambda x: x['best_val_acc'])
    logger.info(f"🏆 WINNER: {winner.get('run_name', winner['model_type'].upper())} with {winner['best_val_acc']:.2f}% accuracy")
    
    return table_str


def train_compare(config_path, ablation='fusion', backbone_size=None, model_type=None):
    """
    Main function to train and compare models.
    
    Args:
        config_path: Path to YAML config.
        ablation: 'fusion' (original vs gmu vs crossattn) or 'backbone' (tiny vs small vs base) or 'all'.
        backbone_size: Backbone scale override ('tiny', 'small', 'base').
        model_type: Specific model type to use for backbone ablation ('gmu', 'crossattn', 'original').
    """
    
    # Load config
    config = load_config(config_path)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config['dataset']['name']
    log_dir = config.get('paths', {}).get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"compare_{dataset_name}_{ablation}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"="*80)
    logger.info(f"MODEL COMPARISON EXPERIMENT (Ablation Mode: {ablation.upper()})")
    logger.info(f"="*80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {timestamp}")
    
    # Seed
    SEED = config['training'].get('seed', 42)
    seed_everything(SEED)
    logger.info(f"Random seed: {SEED}")
    
    # Load data ONCE (shared across all models)
    logger.info("Loading dataset...")
    loaders = get_dataloaders(config)
    if not loaders or loaders[0] is None:
        logger.error("Failed to load data.")
        return
    train_loader = loaders[0]
    val_loader = loaders[1]
    test_loader = loaders[2] if len(loaders) > 2 else val_loader
    # Pass accent config to model if enabled
    aux_cfg = config.get('auxiliary_task', {})
    if aux_cfg.get('enabled', False) and aux_cfg.get('task', '') == 'accent':
        config['model']['num_accent_classes'] = aux_cfg.get('num_accent_classes', 3)

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Determine experiment runs based on ablation mode
    if ablation == 'backbone':
        target_model = model_type or config['model'].get('type', 'gmu')
        scales = ['tiny', 'small', 'base']
        logger.info(f"Running BACKBONE ABLATION on model '{target_model.upper()}': {scales}")
        runs = [(target_model, s, f"{target_model.upper()}-{s.upper()}") for s in scales]
    elif ablation == 'all':
        fusion_types = ['original', 'gmu', 'crossattn']
        scales = ['tiny', 'small', 'base']
        runs = [(m, s, f"{m.upper()}-{s.upper()}") for m in fusion_types for s in scales]
    else:  # 'fusion'
        target_scale = backbone_size or config['model'].get('backbone_size', 'base')
        model_types = ['original', 'gmu', 'crossattn']
        logger.info(f"Running FUSION ABLATION with backbone scale '{target_scale.upper()}': {model_types}")
        runs = [(m, target_scale, f"{m.upper()}-{target_scale.upper()}") for m in model_types]

    results = []
    for m_type, b_size, r_name in runs:
        seed_everything(SEED)  # Reset seed for fair comparison
        result = train_single_model(m_type, config, train_loader, val_loader, logger, backbone_size=b_size, run_name=r_name)
        results.append(result)
    
    # Generate comparison table
    table_str = generate_comparison_table(results, logger)
    
    # Save results as JSON
    results_file = os.path.join(log_dir, f"compare_{dataset_name}_{ablation}_{timestamp}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Save markdown table
    table_file = os.path.join(log_dir, f"compare_{dataset_name}_{ablation}_{timestamp}_table.md")
    with open(table_file, 'w') as f:
        f.write(f"# Model Comparison Results (Ablation: {ablation.upper()})\n\n")
        f.write(f"**Dataset:** {dataset_name}\n\n")
        f.write(f"**Date:** {timestamp}\n\n")
        f.write(table_str)
        f.write("\n\n")
        
        # Add detailed results
        f.write("## Detailed Results\n\n")
        for r in results:
            run_title = r.get('run_name', f"{r['model_type'].upper()}-{r.get('backbone_size', 'base').upper()}")
            f.write(f"### {run_title}\n")
            f.write(f"- Backbone Scale: **{r.get('backbone_size', 'base').upper()}**\n")
            f.write(f"- Fusion Paradigm: **{r['model_type'].upper()}**\n")
            f.write(f"- Best Val Accuracy: **{r['best_val_acc']:.2f}%**\n")
            f.write(f"- Best Val Loss: {r['best_val_loss']:.4f}\n")
            f.write(f"- Best Epoch: {r['best_epoch']}\n")
            f.write(f"- Total Epochs: {r['total_epochs']}\n")
            f.write(f"- Training Time: {r['total_time']/60:.1f} minutes\n")
            f.write(f"- Unweighted Accuracy (UA/UWA): {r.get('val_uwa', 0.0):.2f}%\n")
            f.write(f"- Macro F1-score (mF1): {r.get('val_mf1', 0.0):.2f}%\n")
            f.write(f"- Weighted F1-score: {r.get('val_f1_weighted', 0.0):.2f}%\n")
            f.write(f"- FLOPs (per sample): {r.get('flops', 0.0):.2f} GFLOPs\n")
            f.write(f"- Inference Time per sample: {r.get('inference_time_sample_ms', 0.0):.2f} ms\n")
            f.write(f"- Inference Time per batch (size {config['dataset']['args']['batch_size']}): {r.get('inference_time_batch_ms', 0.0):.2f} ms\n")
            f.write(f"- Total Parameters: {r['total_params']:,} ({r['total_params']/1e6:.2f}M)\n")
            f.write(f"- Trainable Parameters: {r['trainable_params']:,}\n\n")
    
    logger.info(f"Markdown table saved to: {table_file}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare all model variants")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ablation", type=str, choices=['fusion', 'backbone', 'all'], default='fusion',
                        help="Ablation type: 'fusion' (compare original vs gmu vs crossattn), 'backbone' (compare tiny vs small vs base), or 'all'")
    parser.add_argument("--backbone_size", type=str, choices=['tiny', 'small', 'base'], default=None,
                        help="Override backbone size for fusion ablation (e.g. tiny, small, base)")
    parser.add_argument("--model_type", type=str, choices=['original', 'gmu', 'crossattn', 'maxvit_unimodal', 'mvitv2_unimodal'], default=None,
                        help="Specific fusion architecture to use when doing backbone ablation (default from config, typically gmu)")
    args = parser.parse_args()
    
    train_compare(args.config, ablation=args.ablation, backbone_size=args.backbone_size, model_type=args.model_type)
