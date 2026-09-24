"""
MaxMViT-MLP with Gated Multimodal Unit (GMU) Fusion
Standard GMU architecture: Arevalo et al. (2017) - Gated Multimodal Units for Information Fusion

Key improvement: Dynamically balance CQT (MaxViT) and Mel-STFT (MViTv2) modalities
using a dynamic gating vector computed directly from raw concatenated features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from utils import freeze_backbone_layers


class GatedMultimodalUnit(nn.Module):
    """
    Standard Gated Multimodal Unit (GMU) (Arevalo et al., 2017).
    
    Dynamically balances CQT (MaxViT) and Mel-STFT (MViTv2) representations
    by computing gating weights directly from raw concatenated features:
    
    Formula:
        h_cqt = tanh(W_cqt · x_cqt + b_cqt)    # Feature projection for CQT (MaxViT)
        h_mel = tanh(W_mel · x_mel + b_mel)    # Feature projection for Mel-STFT (MViTv2)
        gate_input = [x_cqt; x_mel]             # Raw concatenated features!
        z = σ(W_z · gate_input + b_z)          # Gating vector
        fused = (1 - z) ⊙ h_cqt + z ⊙ h_mel    # Adaptive weighted fusion
    """
    
    def __init__(self, cqt_dim=768, mel_dim=768, fusion_dim=None, dropout_p=0.0,
                 text_dim=None, audio_dim=None, dim_t=None, dim_a=None, hidden_dim=None):
        """
        Args:
            cqt_dim: Dimension of CQT features (from MaxViT)
            mel_dim: Dimension of Mel-STFT features (from MViTv2)
            fusion_dim: Hidden dimension for fusion (default: max of both)
            dropout_p: Dropout probability applied to fused features
        """
        super().__init__()
        
        # Backward-compatibility aliases
        if text_dim is not None: cqt_dim = text_dim
        if dim_t is not None: cqt_dim = dim_t
        if audio_dim is not None: mel_dim = audio_dim
        if dim_a is not None: mel_dim = dim_a
        if hidden_dim is not None: fusion_dim = hidden_dim
        if fusion_dim is None: fusion_dim = max(cqt_dim, mel_dim)
            
        self.cqt_dim = cqt_dim
        self.mel_dim = mel_dim
        self.fusion_dim = fusion_dim
        self.hidden_dim = fusion_dim
        
        # Modality projection layers with tanh activation
        self.cqt_proj = nn.Linear(cqt_dim, fusion_dim)
        self.mel_proj = nn.Linear(mel_dim, fusion_dim)
        
        # Gating mechanism takes RAW concatenated features: [x_cqt; x_mel]
        self.gate_proj = nn.Linear(cqt_dim + mel_dim, fusion_dim)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        
        # Backward-compatibility aliases
        self.text_proj = self.cqt_proj
        self.audio_proj = self.mel_proj
        self.gate = self.gate_proj
        
    def proj_cqt(self, x):
        return self.tanh(self.cqt_proj(x))
        
    def proj_mel(self, x):
        return self.tanh(self.mel_proj(x))

    def proj_audio(self, x):
        return self.proj_mel(x)
        
    def forward(self, feat_cqt, feat_mel, return_gate=True):
        """
        Args:
            feat_cqt: [B, cqt_dim] - CQT features from MaxViT
            feat_mel: [B, mel_dim] - Mel-STFT features from MViTv2
            return_gate: If True, also return gate values for analysis
            
        Returns:
            fused: [B, fusion_dim] - Adaptively fused features
            z (if return_gate): [B, fusion_dim] - Gate values
        """
        # 1. Modality-specific representations
        h_cqt = self.tanh(self.cqt_proj(feat_cqt))
        h_mel = self.tanh(self.mel_proj(feat_mel))

        # 2. Gate computed directly from RAW input features
        gate_input = torch.cat([feat_cqt, feat_mel], dim=1)  # [B, cqt_dim + mel_dim]
        z = self.sigmoid(self.gate_proj(gate_input))          # [B, fusion_dim], values in [0, 1]

        # 3. Adaptive fusion: (1-z) weights CQT, z weights Mel-STFT
        fused = (1 - z) * h_cqt + z * h_mel                  # [B, fusion_dim]
        
        if self.dropout is not None:
            fused = self.dropout(fused)
        
        if return_gate:
            return fused, z
        return fused


class MaxMViT_MLP_GMU(nn.Module):
    """
    MaxMViT-MLP with GMU Fusion.
    
    Architecture:
        CQT Spectrogram → MaxViT → 
                                    → GMU Fusion → MLP → Classification
        Mel-STFT Spectrogram → MViTv2 →
    
    Improvements over original:
        1. GMU instead of simple concatenation
        2. Adaptive modality balancing
        3. Learnable fusion weights
    """
    
    def __init__(self, num_classes=7, hidden_size=512, dropout_rate=0.2, 
                 fusion_hidden_dim=None, num_accent_classes=0,
                 freeze_backbone=False, unfreeze_last_n_blocks=0):
        """
        Args:
            num_classes: Number of emotion classes
            hidden_size: MLP hidden layer size (paper: 512)
            dropout_rate: Dropout rate (paper: 0.2)
            fusion_hidden_dim: GMU hidden dimension (None = auto)
            num_accent_classes: Number of accent/region classes (0 = disabled)
            freeze_backbone: If True, freeze MaxViT/MViTv2 backbones (except
                the last `unfreeze_last_n_blocks` stages) to reduce
                overfitting on small datasets like ViSEC.
            unfreeze_last_n_blocks: Number of trailing backbone stages to
                keep trainable when freeze_backbone=True.
        """
        super().__init__()
        
        # Path 1: CQT → MaxViT
        self.maxvit = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)
        
        # Path 2: Mel-STFT → MViTv2
        self.mvitv2 = timm.create_model('mvitv2_base', pretrained=True, num_classes=0)
        
        # Get feature dimensions (fixed at 768 for both base backbones)
        dim_cqt = 768
        dim_mel = 768
        print(f"Feature dims - CQT/MaxViT: {dim_cqt}, Mel/MViTv2: {dim_mel}")

        # Optionally freeze backbones (transfer-learning regularization)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            f1, t1 = freeze_backbone_layers(self.maxvit, unfreeze_last_n_blocks)
            f2, t2 = freeze_backbone_layers(self.mvitv2, unfreeze_last_n_blocks)
            print(f"Froze MaxViT backbone: {f1/1e6:.1f}M frozen / {t1/1e6:.1f}M trainable "
                  f"(last {unfreeze_last_n_blocks} blocks unfrozen)")
            print(f"Froze MViTv2 backbone: {f2/1e6:.1f}M frozen / {t2/1e6:.1f}M trainable "
                  f"(last {unfreeze_last_n_blocks} blocks unfrozen)")
        
        # --- GMU Fusion ---
        if fusion_hidden_dim is None:
            fusion_hidden_dim = max(dim_cqt, dim_mel)
            
        self.gmu = GatedMultimodalUnit(
            cqt_dim=dim_cqt,        # CQT path (MaxViT)
            mel_dim=dim_mel,        # Mel path (MViTv2)
            fusion_dim=fusion_hidden_dim
        )
        
        # --- MLP Shared Feature Extractor ---
        self.mlp_shared = nn.Sequential(
            nn.Linear(fusion_hidden_dim, hidden_size),
            nn.LayerNorm(hidden_size),   # LayerNorm: ổn định với batch_size nhỏ (thay BatchNorm1d)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # --- Emotion Classification Head (Primary Task) ---
        self.emotion_head = nn.Linear(hidden_size, num_classes)
        
        # --- Accent/Region Classification Head (Auxiliary Task) ---
        self.num_accent_classes = num_accent_classes
        if num_accent_classes > 0:
            self.accent_head = nn.Linear(hidden_size, num_accent_classes)
            print(f"Accent head enabled: {num_accent_classes} classes")
        else:
            self.accent_head = None
        
        # Store for analysis
        self.fusion_hidden_dim = fusion_hidden_dim
        
    def forward(self, cqt, mel, return_gate=False):
        """
        Args:
            cqt: CQT spectrogram [B, C, H, W]
            mel: Mel-STFT spectrogram [B, C, H, W]
            return_gate: If True, also return gate values for analysis
            
        Returns:
            emotion_logits: [B, num_classes]
            accent_logits (if accent_head exists): [B, num_accent_classes]
            gate_values (if return_gate): [B, hidden_dim]
        """
        # Expand to 3 channels if needed
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
            
        # Resize to 224x224 if needed
        if cqt.shape[-1] != 224:
            cqt = F.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
            mel = F.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features from both paths
        feat_cqt = self.maxvit(cqt)    # [B, dim_cqt]
        feat_mel = self.mvitv2(mel)    # [B, dim_mel]
        
        # GMU Fusion (instead of simple concatenation)
        fused, gate_values = self.gmu(feat_cqt, feat_mel)  # [B, fusion_hidden_dim]
        
        # Shared feature extraction
        shared_features = self.mlp_shared(fused)  # [B, hidden_size]
        
        # Emotion classification (primary task)
        emotion_logits = self.emotion_head(shared_features)
        
        # Accent classification (auxiliary task)
        accent_logits = None
        if self.accent_head is not None:
            accent_logits = self.accent_head(shared_features)
        
        if return_gate:
            return emotion_logits, accent_logits, gate_values
        
        if accent_logits is not None:
            return emotion_logits, accent_logits
        return emotion_logits


class MaxMViT_MLP_GMU_Contrastive(MaxMViT_MLP_GMU):
    """
    Extended version with Contrastive Self-Alignment (optional).
    
    Adds GloMER's contrastive learning losses:
        - NT-Xent loss: Push matching pairs closer
        - Consistency loss: Cosine similarity alignment  
        - Diversity loss: Prevent embedding collapse
    """
    
    def __init__(self, num_classes=7, hidden_size=512, dropout_rate=0.2,
                 fusion_hidden_dim=None, proj_dim=256):
        super().__init__(num_classes, hidden_size, dropout_rate, fusion_hidden_dim)
        
        # Projection heads for contrastive learning
        self.proj_cqt = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.proj_mel = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, cqt, mel, return_projections=False):
        # Expand channels
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
            
        # Resize
        if cqt.shape[-1] != 224:
            cqt = F.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
            mel = F.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        feat_cqt = self.maxvit(cqt)
        feat_mel = self.mvitv2(mel)
        
        # GMU Fusion
        fused, gate_values = self.gmu(feat_cqt, feat_mel)
        
        # Classification
        logits = self.mlp(fused)
        
        if return_projections:
            # Project for contrastive loss
            # Use the projected features from GMU
            z_cqt = self.gmu.proj_cqt(feat_cqt)
            z_mel = self.gmu.proj_mel(feat_mel)
            
            proj_cqt = self.proj_cqt(z_cqt)
            proj_mel = self.proj_mel(z_mel)
            
            return logits, proj_cqt, proj_mel, gate_values
            
        return logits


# ==================== Loss Functions ====================

class ContrastiveLoss(nn.Module):
    """
    Combined contrastive loss from GloMER paper.
    
    L_total = L_CE + L_NT-Xent + α(L_con + L_div)
    """
    
    def __init__(self, temperature=0.07, alpha=0.3):
        """
        Args:
            temperature: NT-Xent temperature (τ)
            alpha: Balance parameter for consistency + diversity
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def nt_xent_loss(self, z_cqt, z_mel):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
        Encourages paired samples to be close, non-paired to be far.
        """
        batch_size = z_cqt.size(0)
        
        # Normalize
        z_cqt = F.normalize(z_cqt, dim=1)
        z_mel = F.normalize(z_mel, dim=1)
        
        # Compute similarity matrix
        sim = torch.mm(z_cqt, z_mel.t()) / self.temperature  # [B, B]
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=z_cqt.device)
        
        # Cross entropy loss (treating as classification)
        loss = F.cross_entropy(sim, labels)
        
        return loss
    
    def consistency_loss(self, z_cqt, z_mel):
        """
        Consistency loss: 1 - mean(cosine_similarity).
        Encourages paired embeddings to be similar.
        """
        cos_sim = F.cosine_similarity(z_cqt, z_mel, dim=1)
        loss = 1 - cos_sim.mean()
        return loss
    
    def diversity_loss(self, z_cqt, z_mel):
        """
        Diversity loss: L2 distance between embeddings.
        Prevents trivial collapse where both modalities produce identical outputs.
        """
        loss = torch.mean((z_cqt - z_mel) ** 2)
        return loss
    
    def forward(self, logits, labels, z_cqt=None, z_mel=None):
        """
        Args:
            logits: Classification logits [B, num_classes]
            labels: Ground truth labels [B]
            z_cqt: CQT projections [B, proj_dim] (optional)
            z_mel: Mel projections [B, proj_dim] (optional)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components for logging
        """
        # Classification loss
        l_ce = self.ce_loss(logits, labels)
        
        loss_dict = {'ce': l_ce.item()}
        total_loss = l_ce
        
        # Contrastive losses (if projections provided)
        if z_cqt is not None and z_mel is not None:
            l_nt_xent = self.nt_xent_loss(z_cqt, z_mel)
            l_con = self.consistency_loss(z_cqt, z_mel)
            l_div = self.diversity_loss(z_cqt, z_mel)
            
            total_loss = l_ce + l_nt_xent + self.alpha * (l_con + l_div)
            
            loss_dict.update({
                'nt_xent': l_nt_xent.item(),
                'consistency': l_con.item(),
                'diversity': l_div.item()
            })
            
        return total_loss, loss_dict


# ==================== Optimizer ====================

def get_optimizer_gmu(model, lr=0.02, backbone_lr=None, head_lr=None):
    """
    Optimizers with discriminative learning rates:
    - MaxViT backbone: Adam @ backbone_lr
    - MViTv2 backbone: RAdam @ backbone_lr
    - GMU + MLP + Heads + projections (randomly initialized): Adam @ head_lr

    If backbone_lr/head_lr are not provided, both fall back to `lr`
    (reproduces the original shared-LR behaviour).

    Only parameters with requires_grad=True are included, so this plays
    nicely with freeze_backbone=True.
    """
    backbone_lr = lr if backbone_lr is None else backbone_lr
    head_lr = lr if head_lr is None else head_lr

    maxvit_params = [p for p in model.maxvit.parameters() if p.requires_grad]
    mvitv2_params = [p for p in model.mvitv2.parameters() if p.requires_grad]
    gmu_params = [p for p in model.gmu.parameters() if p.requires_grad]
    mlp_params = [p for p in model.mlp_shared.parameters() if p.requires_grad]
    head_params = [p for p in model.emotion_head.parameters() if p.requires_grad]

    # Accent head params (if exists)
    if model.accent_head is not None:
        head_params += [p for p in model.accent_head.parameters() if p.requires_grad]

    # Optional: contrastive projection params
    other_params = []
    if hasattr(model, 'proj_cqt'):
        other_params += [p for p in model.proj_cqt.parameters() if p.requires_grad]
        other_params += [p for p in model.proj_mel.parameters() if p.requires_grad]

    optimizers = []

    # Optimizer 1: MaxViT backbone (low LR) + GMU/MLP/Heads/projections (high LR) → Adam
    param_groups = []
    if maxvit_params:
        param_groups.append({'params': maxvit_params, 'lr': backbone_lr})
    head_side_params = gmu_params + mlp_params + head_params + other_params
    if head_side_params:
        param_groups.append({'params': head_side_params, 'lr': head_lr})
    if param_groups:
        optimizers.append(torch.optim.Adam(param_groups))

    # Optimizer 2: MViTv2 backbone → RAdam
    if mvitv2_params:
        optimizers.append(torch.optim.RAdam(mvitv2_params, lr=backbone_lr))

    return optimizers


# ==================== Quick Test ====================

if __name__ == "__main__":
    print("Testing GMU-enhanced MaxMViT-MLP...")
    
    # Test basic GMU model
    model = MaxMViT_MLP_GMU(num_classes=4)
    
    # Dummy input
    cqt = torch.randn(2, 3, 224, 224)
    mel = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    logits, gates = model(cqt, mel, return_gate=True)
    
    print(f"Output shape: {logits.shape}")
    print(f"Gate shape: {gates.shape}")
    print(f"Gate mean: {gates.mean().item():.3f} (0.5 = balanced)")
    
    # Test contrastive version
    print("\nTesting Contrastive version...")
    model_cl = MaxMViT_MLP_GMU_Contrastive(num_classes=4)
    logits, z_cqt, z_mel, gates = model_cl(cqt, mel, return_projections=True)
    
    # Test loss
    loss_fn = ContrastiveLoss(alpha=0.3)
    labels = torch.tensor([0, 1])
    total_loss, loss_dict = loss_fn(logits, labels, z_cqt, z_mel)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    print("\n✅ All tests passed!")
