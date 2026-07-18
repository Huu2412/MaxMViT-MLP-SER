import torch
import torch.nn as nn
import timm

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
        # Self attention: query, key, value are all the same
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class MaxViT_SelfAttn_MLP(nn.Module):
    def __init__(self, num_classes=4, hidden_size=512, dropout_rate=0.3, num_heads=8):
        super().__init__()
        # 1. Backbone: Extract raw feature maps instead of pooled vector
        self.backbone = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features # 768
        
        # 2. Self Attention block
        self.self_attn = SelfAttentionBlock(dim=self.feature_dim, num_heads=num_heads, dropout=dropout_rate)
        
        # 3. Classifier
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, cqt, mel=None):
        # cqt shape: [B, 3, 224, 224] (CQT input)
        # mel is ignored for this unimodal branch
        
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if cqt.shape[-1] != 224:
            cqt = torch.nn.functional.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)

        # 1. Extract feature maps
        features = self.backbone.forward_features(cqt) # [B, 768, 7, 7]
        
        # 2. Reshape to sequence for Attention
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).transpose(1, 2) # [B, 49, 768]
        
        # 3. Apply Self-Attention
        attended_features = self.self_attn(features) # [B, 49, 768]
        
        # 4. Global Average Pooling (average over sequence length)
        pooled = attended_features.mean(dim=1) # [B, 768]
        
        # 5. Classify
        out = self.mlp(pooled)
        return out

class MViTv2_SelfAttn_MLP(nn.Module):
    def __init__(self, num_classes=4, hidden_size=512, dropout_rate=0.3, num_heads=8):
        super().__init__()
        # 1. Backbone
        self.backbone = timm.create_model('mvitv2_small', pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features # 768
        
        # 2. Self Attention block
        self.self_attn = SelfAttentionBlock(dim=self.feature_dim, num_heads=num_heads, dropout=dropout_rate)
        
        # 3. Classifier
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, cqt, mel):
        # mel shape: [B, 3, 224, 224] (Mel-STFT input)
        # cqt is ignored for this unimodal branch
        
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
        if mel.shape[-1] != 224:
            mel = torch.nn.functional.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)

        # 1. Extract sequence features
        features = self.backbone.forward_features(mel) # [B, 49, 768]
        
        # 2. Apply Self-Attention
        attended_features = self.self_attn(features) # [B, 49, 768]
        
        # 3. Global Average Pooling
        pooled = attended_features.mean(dim=1) # [B, 768]
        
        # 4. Classify
        out = self.mlp(pooled)
        return out

if __name__ == "__main__":
    print("Testing Uni-modal Self-Attention architectures...")
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("1. MaxViT Self-Attn...")
    model1 = MaxViT_SelfAttn_MLP()
    out1 = model1(dummy_input)
    print(f"Output shape: {out1.shape}")
    
    print("2. MViTv2 Self-Attn...")
    model2 = MViTv2_SelfAttn_MLP()
    out2 = model2(dummy_input)
    print(f"Output shape: {out2.shape}")
    print("Done!")

def get_optimizer_unimodal(model, lr=0.0002, backbone_lr=None, head_lr=None):
    backbone_lr = lr if backbone_lr is None else backbone_lr
    head_lr = lr if head_lr is None else head_lr
    
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = []
    if hasattr(model, "self_attn"):
        head_params.extend([p for p in model.self_attn.parameters() if p.requires_grad])
    if hasattr(model, "mlp"):
        head_params.extend([p for p in model.mlp.parameters() if p.requires_grad])
        
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
        
    optimizers = []
    if param_groups:
        optimizers.append(torch.optim.Adam(param_groups))
    return optimizers
