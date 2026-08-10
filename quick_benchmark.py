import torch
import time
import timm

original_create_model = timm.create_model
def mock_create_model(*args, **kwargs):
    if 'pretrained' in kwargs:
        kwargs['pretrained'] = False
    return original_create_model(*args, **kwargs)
timm.create_model = mock_create_model

from train import get_model_and_optimizer
from utils import load_config
import os

# Set up paths and config
checkpoint_path = r"checkpoints\visec_test_1epoch\rank1_f166.84_acc67.14_loss3.1497_epoch39.pth"
config_path = r"configs\visec_optimized.yaml"
report_path = r"checkpoints\visec_test_1epoch\rank1_f166.84_acc67.14_loss3.1497_epoch39_report.txt"

print("Loading config...")
config = load_config(config_path)
model_cfg = config.get('model', {})
num_classes = model_cfg.get('num_classes', 4)
model_type = 'crossattn'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Initializing model...")
model, _ = get_model_and_optimizer(model_type, num_classes, lr=0.0001, model_cfg=model_cfg)
model.to(device)
model.eval()

print("Calculating parameters...")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Calculating FLOPs...")
flops_str = "N/A"
try:
    import thop
    cqt_dummy = torch.randn(1, 3, 224, 224).to(device)
    mel_dummy = torch.randn(1, 3, 224, 224).to(device)
    flops_raw, _ = thop.profile(model, inputs=(cqt_dummy, mel_dummy), verbose=False)
    flops_str = f"{flops_raw / 1e9:.2f} GFLOPs ({flops_raw:,})"
except Exception as e:
    print(f"Could not calculate FLOPs: {e}")

print("Measuring inference time...")
total_time = 0.0
num_samples_timed = 20
batch_size = 8

with torch.no_grad():
    # Warmup
    cqt_dummy = torch.randn(batch_size, 3, 224, 224).to(device)
    mel_dummy = torch.randn(batch_size, 3, 224, 224).to(device)
    for _ in range(5):
        _ = model(cqt_dummy, mel_dummy)
        
    for _ in range(num_samples_timed):
        cqt_dummy = torch.randn(batch_size, 3, 224, 224).to(device)
        mel_dummy = torch.randn(batch_size, 3, 224, 224).to(device)
        start_t = time.time()
        _ = model(cqt_dummy, mel_dummy)
        total_time += time.time() - start_t
        
inf_time_batch = (total_time / num_samples_timed) * 1000.0
inf_time_sample = inf_time_batch / batch_size

print("Writing report...")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Checkpoint: {checkpoint_path}\n")
    f.write(f"Config: {config_path}\n")
    f.write(f"Validation Accuracy: 67.14%\n")
    f.write("="*60 + "\n")
    f.write("BENCHMARKS & ADDED METRICS:\n")
    f.write(f"Total Parameters: {total_params / 1e6:.2f}M ({total_params:,})\n")
    f.write(f"Trainable Parameters: {trainable_params / 1e6:.2f}M ({trainable_params:,})\n")
    f.write(f"Total FLOPs (per sample): {flops_str}\n")
    f.write(f"Inference Time per batch (size {batch_size}): {inf_time_batch:.2f} ms\n")
    f.write(f"Inference Time per sample: {inf_time_sample:.2f} ms\n")
    f.write("-"*60 + "\n")
    f.write(f"Macro F1-score (mF1): 66.84%\n")
    f.write(f"Loss: 3.1497\n")
    
print(f"Report saved to {report_path}")
