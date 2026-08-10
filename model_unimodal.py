import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from utils import freeze_backbone_layers


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: [B, N, C]
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MaxViT_SelfAttn_MLP(nn.Module):
    """
    Unimodal baseline: CQT → MaxViT → Self-Attention → MLP → Classification.
    Supports optional Accent/Region Recognition auxiliary head (RR Loss).
    """
    def __init__(self, num_classes=4, hidden_size=512, dropout_rate=0.3, num_heads=8,
                 num_accent_classes=0,
                 freeze_backbone=False, unfreeze_last_n_blocks=0):
        super().__init__()
        # 1. Backbone: MaxViT base
        self.backbone = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)
        self.feature_dim = 768  # fixed for maxvit_base

        # Optionally freeze backbone
        if freeze_backbone:
            f, t = freeze_backbone_layers(self.backbone, unfreeze_last_n_blocks)
            print(f"Froze MaxViT backbone: {f/1e6:.1f}M frozen / {t/1e6:.1f}M trainable")

        # 2. Self Attention block
        self.self_attn = SelfAttentionBlock(dim=self.feature_dim, num_heads=num_heads, dropout=dropout_rate)
        
        # 3. Shared MLP trunk
        self.mlp_shared = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 4. Emotion classification head
        self.emotion_head = nn.Linear(hidden_size, num_classes)

        # 5. Accent/Region Recognition auxiliary head
        self.num_accent_classes = num_accent_classes
        if num_accent_classes > 0:
            self.accent_head = nn.Linear(hidden_size, num_accent_classes)
        else:
            self.accent_head = None

    def forward(self, cqt, mel=None):
        """
        Args:
            cqt: [B, C, H, W] - CQT spectrogram (mel is ignored for unimodal)
        Returns:
            logits or (logits, accent_logits)
        """
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if cqt.shape[-1] != 224:
            cqt = F.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)

        # Extract feature maps: [B, 768, 7, 7]
        features = self.backbone.forward_features(cqt)
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).transpose(1, 2)  # [B, 49, 768]

        # Self-attention
        attended = self.self_attn(features)         # [B, 49, 768]
        pooled = attended.mean(dim=1)               # [B, 768]

        # Shared MLP
        shared = self.mlp_shared(pooled)            # [B, hidden_size]
        logits = self.emotion_head(shared)

        if self.accent_head is not None:
            accent_logits = self.accent_head(shared)
            return logits, accent_logits

        return logits


class MViTv2_SelfAttn_MLP(nn.Module):
    """
    Unimodal baseline: Mel-STFT → MViTv2 → Self-Attention → MLP → Classification.
    Supports optional Accent/Region Recognition auxiliary head (RR Loss).
    """
    def __init__(self, num_classes=4, hidden_size=512, dropout_rate=0.3, num_heads=8,
                 num_accent_classes=0,
                 freeze_backbone=False, unfreeze_last_n_blocks=0):
        super().__init__()
        # 1. Backbone: MViTv2 small (outputs sequence features directly)
        self.backbone = timm.create_model('mvitv2_small', pretrained=True, num_classes=0)
        self.feature_dim = 768  # fixed for mvitv2_small

        # Optionally freeze backbone
        if freeze_backbone:
            f, t = freeze_backbone_layers(self.backbone, unfreeze_last_n_blocks)
            print(f"Froze MViTv2 backbone: {f/1e6:.1f}M frozen / {t/1e6:.1f}M trainable")

        # 2. Self Attention block
        self.self_attn = SelfAttentionBlock(dim=self.feature_dim, num_heads=num_heads, dropout=dropout_rate)
        
        # 3. Shared MLP trunk
        self.mlp_shared = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 4. Emotion classification head
        self.emotion_head = nn.Linear(hidden_size, num_classes)

        # 5. Accent/Region Recognition auxiliary head
        self.num_accent_classes = num_accent_classes
        if num_accent_classes > 0:
            self.accent_head = nn.Linear(hidden_size, num_accent_classes)
        else:
            self.accent_head = None

    def forward(self, cqt, mel):
        """
        Args:
            mel: [B, C, H, W] - Mel-STFT spectrogram (cqt is ignored for unimodal)
        Returns:
            logits or (logits, accent_logits)
        """
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
        if mel.shape[-1] != 224:
            mel = F.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)

        # MViTv2 returns sequence features: [B, N, 768]
        features = self.backbone.forward_features(mel)  # [B, 49, 768]

        # Self-attention
        attended = self.self_attn(features)             # [B, 49, 768]
        pooled = attended.mean(dim=1)                   # [B, 768]

        # Shared MLP
        shared = self.mlp_shared(pooled)                # [B, hidden_size]
        logits = self.emotion_head(shared)

        if self.accent_head is not None:
            accent_logits = self.accent_head(shared)
            return logits, accent_logits

        return logits


def get_optimizer_unimodal(model, lr=0.0002, backbone_lr=None, head_lr=None):
    """
    Discriminative learning rates:
    - Backbone (pretrained): backbone_lr
    - Self-attention + MLP head (randomly initialized): head_lr
    """
    backbone_lr = lr if backbone_lr is None else backbone_lr
    head_lr = lr if head_lr is None else head_lr
    
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = []
    if hasattr(model, 'self_attn'):
        head_params.extend([p for p in model.self_attn.parameters() if p.requires_grad])
    if hasattr(model, 'mlp_shared'):
        head_params.extend([p for p in model.mlp_shared.parameters() if p.requires_grad])
    if hasattr(model, 'emotion_head'):
        head_params.extend([p for p in model.emotion_head.parameters() if p.requires_grad])
    if getattr(model, 'accent_head', None) is not None:
        head_params.extend([p for p in model.accent_head.parameters() if p.requires_grad])
        
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr})
        
    optimizers = []
    if param_groups:
        optimizers.append(torch.optim.Adam(param_groups))
    return optimizers


if __name__ == '__main__':
    print('Testing Unimodal Self-Attention architectures with accent head...')
    dummy = torch.randn(2, 3, 224, 224)

    print('1. MaxViT Self-Attn (with accent)...')
    m1 = MaxViT_SelfAttn_MLP(num_classes=4, num_accent_classes=3)
    out1 = m1(dummy)
    print(f'   logits: {out1[0].shape}, accent_logits: {out1[1].shape}')

    print('2. MViTv2 Self-Attn (with accent)...')
    m2 = MViTv2_SelfAttn_MLP(num_classes=4, num_accent_classes=3)
    out2 = m2(dummy, dummy)
    print(f'   logits: {out2[0].shape}, accent_logits: {out2[1].shape}')

    print('3. MaxViT Self-Attn (no accent)...')
    m3 = MaxViT_SelfAttn_MLP(num_classes=4, num_accent_classes=0)
    out3 = m3(dummy)
    print(f'   logits: {out3.shape}')
    print('Done!')
