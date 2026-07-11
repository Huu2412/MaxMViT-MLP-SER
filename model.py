import torch
import torch.nn as nn
import timm
import torchaudio
import librosa
import numpy as np
import cv2
import os

from utils import freeze_backbone_layers

class MaxMViT_MLP(nn.Module):
    def __init__(self, num_classes=7, hidden_size=512, dropout_rate=0.2,
                 freeze_backbone=False, unfreeze_last_n_blocks=0):
        """
        MaxMViT and MViTv2 Fusion Network with Multilayer Perceptron (MaxMViT-MLP).
        
        Args:
            num_classes (int): Number of emotion classes (e.g., 7 for Emo-DB).
            hidden_size (int): Number of hidden nodes in MLP (default 512).
            dropout_rate (float): Dropout rate (default 0.2).
            freeze_backbone (bool): If True, freeze MaxViT/MViTv2 backbones
                (except the last `unfreeze_last_n_blocks` stages) to reduce
                overfitting on small datasets like ViSEC.
            unfreeze_last_n_blocks (int): Number of trailing backbone stages
                to keep trainable when freeze_backbone=True.
        """
        super(MaxMViT_MLP, self).__init__()
        
        # --- Path 1: CQT + MaxViT ---
        # Using 'maxvit_rmlp_base_rw_224' or similar. 
        # Paper mentions MaxViT.        # Paper likely uses base/large. Switching to base as per feedback.
        # MaxViT Base
        self.maxvit = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)
        # MViTv2 Base
        self.mvitv2 = timm.create_model('mvitv2_base', pretrained=True, num_classes=0)

        # Print config to verify window sizes if possible, or just the model name
        print(f"Initialized MaxViT: {self.maxvit.default_cfg['architecture']}")

        # Optionally freeze backbones (transfer-learning regularization)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            f1, t1 = freeze_backbone_layers(self.maxvit, unfreeze_last_n_blocks)
            f2, t2 = freeze_backbone_layers(self.mvitv2, unfreeze_last_n_blocks)
            print(f"Froze MaxViT backbone: {f1/1e6:.1f}M frozen / {t1/1e6:.1f}M trainable "
                  f"(last {unfreeze_last_n_blocks} blocks unfrozen)")
            print(f"Froze MViTv2 backbone: {f2/1e6:.1f}M frozen / {t2/1e6:.1f}M trainable "
                  f"(last {unfreeze_last_n_blocks} blocks unfrozen)")
        
        # Calculate feature dimension (fixed at 768 for both base backbones)
        maxvit_dim = 768
        mvitv2_dim = 768
        fusion_dim = maxvit_dim + mvitv2_dim
        
        # --- MLP Head ---
        # Dense layer -> Batch Norm -> Dropout -> Classification
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(), # Paper implies activation before classification? 
                       # "Dense layer... followed by classification layer... softmax"
                       # Usually Dense -> Activation -> BN -> Dropout -> FC.
                       # Paper text: "dense layer, batch normalization layer, dropout layer, and a classification layer."
                       # "two dense neural network layers activated by the ReLU function" (Ref to Vu et al. [13], not this work?)
                       # Section III.D.1: "Dense layer... applies linear transformation... BN... Dropout... Classification layer computes probabilities... softmax"
                       # Usually Linear implies just linear. But networks need non-linearity.
                       # I will add ReLU for safety as "Dense Layer" typically implies a hidden layer with activation.
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, cqt, mel):
        """
        Forward pass.
        
        Args:
           cqt (torch.Tensor): CQT Spectrogram [Batch, 1, 244, 244] -> will repeat to 3 channels
           mel (torch.Tensor): Mel-STFT Spectrogram [Batch, 1, 244, 244]
        """
        # Expand 1 channel to 3 channels for backbone compatibility
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
            
        # Resize to 224x224 if model expects it (timm models usually strict or better at native res)
        # Paper says 244x244. 
        # User requested 244x244.
        
        # MaxViT usually requires input divisible by 32 (224 is, 244 is NOT).
        # 244 / 32 = 7.625.
        # If we pass 244, MaxViT might error or perform poorly due to window padding.
        # However, to satisfy the requirement, we pass it through.
        # If strict compatibility is needed, we could interpolate to 224 here if we encounter errors.
        
        # Fix: MaxViT architecture restricts input size to be divisible by window size (7).
        # 244 is NOT divisible by 7. This causes a crash.
        # To support the user's request for 244 input (from config), we MUST interpolate to 224 
        # before the backbone to fit the fixed architecture constraints.
        if cqt.shape[-1] != 224:
             cqt = torch.nn.functional.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
             mel = torch.nn.functional.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)

        # Path 1
        feat_maxvit = self.maxvit(cqt) # [B, Dim1]
        
        # Path 2
        feat_mvitv2 = self.mvitv2(mel) # [B, Dim2]
        
        # Fusion
        fused = torch.cat((feat_maxvit, feat_mvitv2), dim=1)
        
        # MLP
        logits = self.mlp(fused)
        
        return logits

def get_optimizer(model, lr=0.02, backbone_lr=None, head_lr=None):
    """
    Returns the optimizers, with support for discriminative learning rates:
    - MaxViT backbone: Adam @ backbone_lr (low, since it's pretrained)
    - MViTv2 backbone: RAdam @ backbone_lr
    - MLP head (randomly initialized): Adam @ head_lr (higher)

    If backbone_lr/head_lr are not provided, both fall back to `lr`
    (reproduces the original paper-style behaviour of one shared LR).

    Only parameters with requires_grad=True are included, so this plays
    nicely with freeze_backbone=True.
    """
    backbone_lr = lr if backbone_lr is None else backbone_lr
    head_lr = lr if head_lr is None else head_lr

    # Split parameters, only keep trainable ones
    maxvit_params = [p for p in model.maxvit.parameters() if p.requires_grad]
    mvitv2_params = [p for p in model.mvitv2.parameters() if p.requires_grad]
    mlp_params = [p for p in model.mlp.parameters() if p.requires_grad]

    optimizers = []

    # Optimizer 1: MaxViT backbone + MLP head, different LR groups -> Adam
    param_groups = []
    if maxvit_params:
        param_groups.append({'params': maxvit_params, 'lr': backbone_lr})
    if mlp_params:
        param_groups.append({'params': mlp_params, 'lr': head_lr})
    if param_groups:
        optimizers.append(torch.optim.Adam(param_groups))

    # Optimizer 2: MViTv2 backbone -> RAdam
    if mvitv2_params:
        optimizers.append(torch.optim.RAdam(mvitv2_params, lr=backbone_lr))

    return optimizers
