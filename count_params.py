import torch
from model_unimodal import MaxViT_SelfAttn_MLP, MViTv2_SelfAttn_MLP

m1 = MaxViT_SelfAttn_MLP(num_classes=4)
p1 = sum(p.numel() for p in m1.parameters())
print(f"MaxViT branch (CQT): {p1/1e6:.2f}M parameters ({p1})")

m2 = MViTv2_SelfAttn_MLP(num_classes=4)
p2 = sum(p.numel() for p in m2.parameters())
print(f"MViTv2 branch (Mel): {p2/1e6:.2f}M parameters ({p2})")
