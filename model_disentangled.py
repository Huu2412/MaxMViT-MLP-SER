"""
MaxMViT-MLP with Disentangled Representation Learning & Self-Reconstruction.

Based on the theoretical framework from docs/DISENTANGLED_REPRESENTATION.md,
adapting principles from MISA (Hazarika et al., ACM MM 2020) and Barlow Twins
to the Dual-Stream Speech Emotion Recognition architecture (CQT + Mel-STFT).

Key design decisions vs. MISA:
  1. Linear output (no Sigmoid) on projectors — preserves geometric orthogonality in R^d.
  2. Supervised similarity (SupCon / emotion-anchored cosine) — prevents speaker leakage.
  3. 3-Way Disentangled GMU fusion — lightweight alternative to MISA's 6-way Transformer.
  4. Multi-task accent head on h_cqt_private — exploits CQT's low-freq F0 resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from utils import freeze_backbone_layers


# ==================== Disentangled Components ====================

class ModalityDisentangler(nn.Module):
    """
    Phân rã đặc trưng backbone z ∈ R^D thành:
      - h_shared  ∈ R^d  (ngữ nghĩa cảm xúc chung giữa 2 phương thức)
      - h_private ∈ R^d  (đặc trưng âm học riêng biệt)
    
    Kèm Decoder tự tái tạo: [h_s || h_p] → ẑ ≈ z.
    
    Design notes (from DISENTANGLED_REPRESENTATION.md §3.1):
      - Lớp cuối cùng của projector là Linear thuần túy (không ReLU/Sigmoid)
        để vector sống trong R^d đầy đủ, cho phép trực giao hình học thực sự.
      - MISA gốc dùng Sigmoid → ép [0,1] → disjoint support thay vì trực giao.
    """
    
    def __init__(self, in_dim=768, sub_dim=256):
        """
        Args:
            in_dim: Dimension of backbone feature z (e.g., 768 for base, 512 for tiny).
            sub_dim: Dimension of shared/private subspaces (d = D/2 or 256).
        """
        super().__init__()
        self.in_dim = in_dim
        self.sub_dim = sub_dim
        
        # Shared Projector: z → h_shared
        # Architecture: Linear → LayerNorm → GELU → Linear (no activation at output)
        self.shared_proj = nn.Sequential(
            nn.Linear(in_dim, sub_dim),
            nn.LayerNorm(sub_dim),
            nn.GELU(),
            nn.Linear(sub_dim, sub_dim),  # Pure linear output — critical for orthogonality
        )
        
        # Private Projector: z → h_private
        self.private_proj = nn.Sequential(
            nn.Linear(in_dim, sub_dim),
            nn.LayerNorm(sub_dim),
            nn.GELU(),
            nn.Linear(sub_dim, sub_dim),  # Pure linear output
        )
        
        # Self-Reconstruction Decoder: [h_s || h_p] → ẑ ≈ z
        self.decoder = nn.Sequential(
            nn.Linear(sub_dim * 2, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )
    
    def forward(self, z):
        """
        Args:
            z: [B, D] — backbone feature vector.
            
        Returns:
            h_s:     [B, d] — shared (emotion-semantic) component
            h_p:     [B, d] — private (modality-specific) component
            z_recon: [B, D] — reconstructed backbone feature
        """
        h_s = self.shared_proj(z)
        h_p = self.private_proj(z)
        h_comb = torch.cat([h_s, h_p], dim=-1)   # [B, 2d]
        z_recon = self.decoder(h_comb)              # [B, D]
        return h_s, h_p, z_recon


# ==================== Loss Functions ====================

def batch_orthogonality_loss(H_s, H_p, eps=1e-8):
    """
    Batch-level cross-covariance orthogonality loss (MISA / Barlow Twins style).
    
    Triệt tiêu tương quan tuyến tính giữa MỌI cặp chiều đặc trưng (j, k) 
    của phần chung H_s và phần riêng H_p trên toàn bộ phân phối batch.
    
    So sánh với MISA DiffLoss gốc:
      - MISA: L2-normalize theo chiều batch (dim=1) → cross-covariance qua sample
      - Ở đây: L2-normalize theo chiều feature (dim=0) → cross-covariance qua feature
      Cả hai đều triệt tiêu Frobenius norm, nhưng normalize theo dim=0 
      tương thích hơn với Barlow Twins và ổn định hơn khi batch nhỏ.
    
    Args:
        H_s: [B, d] — Shared representations for a mini-batch.
        H_p: [B, d] — Private representations for a mini-batch.
        
    Returns:
        Scalar loss ∈ [0, ∞), thấp → ít tương quan tuyến tính giữa H_s và H_p.
    """
    # 1. Zero-center theo batch dimension
    H_s_cent = H_s - H_s.mean(dim=0, keepdim=True)
    H_p_cent = H_p - H_p.mean(dim=0, keepdim=True)
    
    # 2. L2-normalize mỗi cột đặc trưng (dim=0, theo batch axis)
    H_s_norm = F.normalize(H_s_cent, p=2, dim=0, eps=eps)
    H_p_norm = F.normalize(H_p_cent, p=2, dim=0, eps=eps)
    
    # 3. Ma trận tương quan chéo [d, d]: C_jk = Σ_i ĥ_s_ij * ĥ_p_ik
    correlation_matrix = torch.matmul(H_s_norm.t(), H_p_norm)
    
    # 4. Frobenius norm bình phương, trung bình hóa
    loss_orth = torch.mean(correlation_matrix ** 2)
    return loss_orth


class DisentangledConstraintLoss(nn.Module):
    """
    Tính toán 3 hàm mất mát phụ trợ cho Disentanglement, có hỗ trợ warm-up schedule.
    
    L_disentangle = β * L_recon + γ * L_sim + δ * L_orth
    
    Trong đó:
      - L_recon: MSE self-reconstruction loss (bảo toàn thông tin trong h_p)
      - L_sim:   Cosine distance giữa h_cqt_shared và h_mel_shared (ép cùng hướng cảm xúc)
      - L_orth:  Batch-level Frobenius norm (triệt tiêu tương quan tuyến tính h_s ↔ h_p)
    
    Warm-up: Trong warmup_epochs đầu, trọng số tăng tuyến tính từ 0 → giá trị đích.
    """
    
    def __init__(self, weight_recon=0.1, weight_sim=0.05, weight_orth=0.05, warmup_epochs=5):
        super().__init__()
        self.w_recon = weight_recon
        self.w_sim = weight_sim
        self.w_orth = weight_orth
        self.warmup_epochs = warmup_epochs
    
    def get_warmup_scale(self, current_epoch):
        """Tính hệ số warm-up tuyến tính: 0 → 1 trong warmup_epochs.
        current_epoch: 0-indexed (0 cho Epoch 1, 1 cho Epoch 2, ...)
        Epoch 1 bắt đầu ở mức 1/warmup_epochs, đạt 1.0 ở Epoch warmup_epochs.
        """
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, max(0.0, (current_epoch + 1) / self.warmup_epochs))
    
    def forward(self, z_cqt, z_mel, h_cqt_s, h_cqt_p, z_cqt_recon,
                h_mel_s, h_mel_p, z_mel_recon, current_epoch=0):
        """
        Args:
            z_cqt, z_mel:             [B, D] — backbone features (targets for reconstruction)
            h_cqt_s, h_mel_s:         [B, d] — shared components
            h_cqt_p, h_mel_p:         [B, d] — private components
            z_cqt_recon, z_mel_recon: [B, D] — reconstructed features
            current_epoch:            int    — for warm-up schedule
            
        Returns:
            total_aux_loss: Scalar — weighted sum of all auxiliary losses.
            loss_dict:      dict   — individual loss values for logging.
        """
        warmup = self.get_warmup_scale(current_epoch)
        
        # 1. Reconstruction Loss (MSE): ẑ ≈ z
        loss_recon = F.mse_loss(z_cqt_recon, z_cqt.detach()) + F.mse_loss(z_mel_recon, z_mel.detach())
        
        # 2. Similarity Loss: Cosine distance giữa 2 nhánh Shared
        cos_sim = F.cosine_similarity(h_cqt_s, h_mel_s, dim=-1)
        loss_sim = torch.mean(1.0 - cos_sim)
        
        # 3. Batch-level Orthogonality Loss (CQT + Mel)
        orth_cqt = batch_orthogonality_loss(h_cqt_s, h_cqt_p)
        orth_mel = batch_orthogonality_loss(h_mel_s, h_mel_p)
        loss_orth = orth_cqt + orth_mel
        
        # Weighted sum with warm-up scaling
        total_aux_loss = warmup * (
            self.w_recon * loss_recon +
            self.w_sim * loss_sim +
            self.w_orth * loss_orth
        )
        
        return total_aux_loss, {
            'loss_recon': loss_recon.item(),
            'loss_sim': loss_sim.item(),
            'loss_orth': loss_orth.item(),
            'warmup_scale': warmup,
        }


# ==================== Disentangled GMU (3-Way) ====================

class DisentangledGMU(nn.Module):
    """
    3-Way Disentangled Gated Multimodal Unit.
    
    Thay vì dung hợp 2 luồng thô (z_cqt, z_mel) như GMU chuẩn,
    module này dung hợp 3 luồng đã phân rã:
      1. h_shared  = avg(h_cqt_s, h_mel_s)  — ngữ nghĩa cảm xúc chung
      2. h_cqt_p                              — đặc trưng pitch/thanh điệu CQT
      3. h_mel_p                              — đặc trưng formant/timbre Mel
    
    Gating: 3 gate vectors g_1, g_2, g_3 ∈ [0,1]^d từ σ(W_g · [h_s; h_cqt_p; h_mel_p])
    Fused = g_1 ⊙ h_s + g_2 ⊙ h_cqt_p + g_3 ⊙ h_mel_p  (normalized via softmax)
    """
    
    def __init__(self, sub_dim=256, fusion_dim=None, dropout_p=0.0):
        """
        Args:
            sub_dim:    Dimension of each disentangled subspace (d).
            fusion_dim: Output dimension of fused representation (default: sub_dim).
            dropout_p:  Dropout applied after fusion.
        """
        super().__init__()
        fusion_dim = fusion_dim or sub_dim
        self.sub_dim = sub_dim
        self.fusion_dim = fusion_dim
        
        # Modality projections (tanh activated, following GMU convention)
        self.proj_shared = nn.Linear(sub_dim, fusion_dim)
        self.proj_cqt_p = nn.Linear(sub_dim, fusion_dim)
        self.proj_mel_p = nn.Linear(sub_dim, fusion_dim)
        
        # 3-way gating: input is concatenation of all 3 streams
        self.gate_proj = nn.Linear(sub_dim * 3, fusion_dim * 3)
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
    
    def forward(self, h_shared, h_cqt_p, h_mel_p, return_gate=False):
        """
        Args:
            h_shared: [B, d] — averaged shared component
            h_cqt_p:  [B, d] — CQT private component
            h_mel_p:  [B, d] — Mel private component
            
        Returns:
            fused:  [B, fusion_dim]
            gates:  [B, 3, fusion_dim] (if return_gate=True)
        """
        # 1. Project each stream
        p_s = self.tanh(self.proj_shared(h_shared))     # [B, fusion_dim]
        p_cqt = self.tanh(self.proj_cqt_p(h_cqt_p))    # [B, fusion_dim]
        p_mel = self.tanh(self.proj_mel_p(h_mel_p))     # [B, fusion_dim]
        
        # 2. Compute 3 gate vectors from raw concatenated inputs
        gate_input = torch.cat([h_shared, h_cqt_p, h_mel_p], dim=-1)  # [B, 3d]
        gate_raw = self.gate_proj(gate_input)                           # [B, 3*fusion_dim]
        
        # Reshape to [B, 3, fusion_dim] and apply softmax across the 3 streams
        B = gate_raw.size(0)
        gate_3way = gate_raw.view(B, 3, self.fusion_dim)  # [B, 3, fusion_dim]
        gate_weights = torch.softmax(gate_3way, dim=1)     # [B, 3, fusion_dim], sums to 1 along dim=1
        
        g_s = gate_weights[:, 0, :]     # [B, fusion_dim]
        g_cqt = gate_weights[:, 1, :]   # [B, fusion_dim]
        g_mel = gate_weights[:, 2, :]   # [B, fusion_dim]
        
        # 3. Weighted fusion
        fused = g_s * p_s + g_cqt * p_cqt + g_mel * p_mel  # [B, fusion_dim]
        fused = self.dropout(fused)
        
        if return_gate:
            return fused, gate_weights
        return fused


# ==================== Backbone Presets ====================

BACKBONE_PRESETS = {
    'tiny': {
        'maxvit': 'maxvit_tiny_tf_224',
        'mvitv2': 'mvitv2_tiny',
    },
    'small': {
        'maxvit': 'maxvit_small_tf_224',
        'mvitv2': 'mvitv2_small',
    },
    'base': {
        'maxvit': 'maxvit_base_tf_224',
        'mvitv2': 'mvitv2_base',
    },
}


# ==================== Full Model ====================

class MaxMViT_MLP_Disentangled(nn.Module):
    """
    MaxMViT-MLP with Disentangled Representation Learning & Self-Reconstruction.
    
    Architecture:
        CQT Spectrogram → MaxViT → z_cqt → CQT Disentangler → (h_cqt_s, h_cqt_p, ẑ_cqt)
        Mel Spectrogram → MViTv2 → z_mel → Mel Disentangler → (h_mel_s, h_mel_p, ẑ_mel)
                                                    ↓
                    h_shared = avg(h_cqt_s, h_mel_s)
                                                    ↓
                    DisentangledGMU(h_shared, h_cqt_p, h_mel_p)
                                                    ↓
                              Shared MLP Trunk
                                                    ↓
                    ┌───────────────────────────────────┐
                    Emotion Head              Accent Head (from h_cqt_p)
    
    Training output (model.training=True):
        Returns a dict with all components needed for loss computation.
    
    Inference output (model.training=False):
        Returns (emotion_logits,) or (emotion_logits, accent_logits).
    """
    
    def __init__(self, num_classes=7, hidden_size=512, dropout_rate=0.2,
                 sub_dim=256, fusion_dim=None, num_accent_classes=0,
                 freeze_backbone=False, unfreeze_last_n_blocks=0,
                 backbone_size='base', maxvit_variant=None, mvitv2_variant=None):
        """
        Args:
            num_classes:          Number of emotion classes.
            hidden_size:          MLP hidden layer size.
            dropout_rate:         Dropout rate.
            sub_dim:              Dimension of shared/private subspaces.
            fusion_dim:           DisentangledGMU output dim (default: sub_dim).
            num_accent_classes:   Number of accent classes (0 = no accent head).
            freeze_backbone:      Freeze backbone except last N blocks.
            unfreeze_last_n_blocks: Number of trailing blocks to keep trainable.
            backbone_size:        'tiny', 'small', or 'base'.
            maxvit_variant:       Custom timm model name for MaxViT.
            mvitv2_variant:       Custom timm model name for MViTv2.
        """
        super().__init__()
        
        # Resolve backbone variants
        preset = BACKBONE_PRESETS.get(
            backbone_size.lower() if isinstance(backbone_size, str) else 'base',
            BACKBONE_PRESETS['base']
        )
        maxvit_name = maxvit_variant or preset['maxvit']
        mvitv2_name = mvitv2_variant or preset['mvitv2']
        
        # Path 1: CQT → MaxViT
        self.maxvit = timm.create_model(maxvit_name, pretrained=True, num_classes=0)
        
        # Path 2: Mel-STFT → MViTv2
        self.mvitv2 = timm.create_model(mvitv2_name, pretrained=True, num_classes=0)
        
        # Get feature dimensions dynamically
        dim_cqt = getattr(self.maxvit, 'num_features', 768)
        dim_mel = getattr(self.mvitv2, 'num_features', 768)
        print(f"[Disentangled] Feature dims - CQT/MaxViT ({maxvit_name}): {dim_cqt}, "
              f"Mel/MViTv2 ({mvitv2_name}): {dim_mel}")
        
        # Store backbone dims for optimizer
        self.dim_cqt = dim_cqt
        self.dim_mel = dim_mel
        
        # Optionally freeze backbones
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            f1, t1 = freeze_backbone_layers(self.maxvit, unfreeze_last_n_blocks)
            f2, t2 = freeze_backbone_layers(self.mvitv2, unfreeze_last_n_blocks)
            print(f"Froze MaxViT backbone: {f1/1e6:.1f}M frozen / {t1/1e6:.1f}M trainable "
                  f"(last {unfreeze_last_n_blocks} blocks unfrozen)")
            print(f"Froze MViTv2 backbone: {f2/1e6:.1f}M frozen / {t2/1e6:.1f}M trainable "
                  f"(last {unfreeze_last_n_blocks} blocks unfrozen)")
        
        # --- Disentanglers ---
        self.cqt_disentangler = ModalityDisentangler(in_dim=dim_cqt, sub_dim=sub_dim)
        self.mel_disentangler = ModalityDisentangler(in_dim=dim_mel, sub_dim=sub_dim)
        print(f"[Disentangled] Subspace dim: {sub_dim} (CQT: {dim_cqt}→{sub_dim}, Mel: {dim_mel}→{sub_dim})")
        
        # --- 3-Way Disentangled GMU ---
        fusion_dim = fusion_dim or sub_dim
        self.disentangled_gmu = DisentangledGMU(
            sub_dim=sub_dim,
            fusion_dim=fusion_dim,
            dropout_p=dropout_rate
        )
        
        # --- Shared MLP Feature Trunk ---
        self.mlp_shared = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # --- Emotion Classification Head (Primary Task) ---
        self.emotion_head = nn.Linear(hidden_size, num_classes)
        
        # --- Accent Classification Head (Auxiliary Task, from h_cqt_private) ---
        self.num_accent_classes = num_accent_classes
        if num_accent_classes > 0:
            # Dedicated accent MLP: h_cqt_private → accent logits
            # CQT's low-freq F0 resolution captures regional tonal patterns
            self.accent_mlp = nn.Sequential(
                nn.Linear(sub_dim, sub_dim // 2),
                nn.LayerNorm(sub_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            self.accent_head = nn.Linear(sub_dim // 2, num_accent_classes)
            print(f"[Disentangled] Accent head enabled: {num_accent_classes} classes "
                  f"(from h_cqt_private, sub_dim={sub_dim})")
        else:
            self.accent_mlp = None
            self.accent_head = None
        
        # Store dims for optimizer
        self.fusion_dim = fusion_dim
        self.sub_dim = sub_dim
    
    def forward(self, cqt, mel, return_gate=False):
        """
        Args:
            cqt: CQT spectrogram [B, C, H, W]
            mel: Mel-STFT spectrogram [B, C, H, W]
            return_gate: If True, include gate values in output.
            
        Returns (training mode):
            dict with keys:
              - 'emotion_logits':  [B, num_classes]
              - 'accent_logits':   [B, num_accent_classes] or None
              - 'z_cqt', 'z_mel': [B, D] — backbone features (for L_recon targets)
              - 'h_cqt_s', 'h_mel_s': [B, d] — shared components (for L_sim)
              - 'h_cqt_p', 'h_mel_p': [B, d] — private components (for L_orth)
              - 'z_cqt_recon', 'z_mel_recon': [B, D] — reconstructed features
              - 'gate_weights': [B, 3, fusion_dim] (if return_gate)
              
        Returns (eval mode):
            emotion_logits or (emotion_logits, accent_logits)
        """
        # Expand to 3 channels if needed
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
        
        # Resize to 224x224 if needed (MaxViT requires input divisible by window size)
        if cqt.shape[-1] != 224:
            cqt = F.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
            mel = F.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)
        
        # === Backbone Feature Extraction ===
        z_cqt = self.maxvit(cqt)    # [B, dim_cqt]
        z_mel = self.mvitv2(mel)     # [B, dim_mel]
        
        # === Disentanglement ===
        h_cqt_s, h_cqt_p, z_cqt_recon = self.cqt_disentangler(z_cqt)
        h_mel_s, h_mel_p, z_mel_recon = self.mel_disentangler(z_mel)
        
        # === Shared representation (average of both shared components) ===
        h_shared = 0.5 * (h_cqt_s + h_mel_s)   # [B, d]
        
        # === 3-Way Disentangled GMU Fusion ===
        if return_gate:
            fused, gate_weights = self.disentangled_gmu(
                h_shared, h_cqt_p, h_mel_p, return_gate=True
            )
        else:
            fused = self.disentangled_gmu(h_shared, h_cqt_p, h_mel_p)
            gate_weights = None
        
        # === MLP Trunk ===
        shared_features = self.mlp_shared(fused)   # [B, hidden_size]
        
        # === Classification Heads ===
        emotion_logits = self.emotion_head(shared_features)   # [B, num_classes]
        
        # Accent head: uses h_cqt_private (not shared features)
        accent_logits = None
        if self.accent_head is not None:
            accent_features = self.accent_mlp(h_cqt_p)       # [B, sub_dim//2]
            accent_logits = self.accent_head(accent_features)  # [B, num_accent_classes]
        
        # === Output ===
        if self.training:
            # During training, return all components for loss computation
            output = {
                'emotion_logits': emotion_logits,
                'accent_logits': accent_logits,
                'z_cqt': z_cqt,
                'z_mel': z_mel,
                'h_cqt_s': h_cqt_s,
                'h_mel_s': h_mel_s,
                'h_cqt_p': h_cqt_p,
                'h_mel_p': h_mel_p,
                'z_cqt_recon': z_cqt_recon,
                'z_mel_recon': z_mel_recon,
            }
            if gate_weights is not None:
                output['gate_weights'] = gate_weights
            return output
        else:
            # During eval, return standard format compatible with existing train.py
            if accent_logits is not None:
                return emotion_logits, accent_logits
            return emotion_logits


# ==================== Optimizer ====================

def get_optimizer_disentangled(model, lr=0.02, backbone_lr=None, head_lr=None):
    """
    Optimizers with discriminative learning rates for the disentangled model.
    
    Follows the same convention as get_optimizer_gmu in model_gmu.py:
      - MaxViT backbone: Adam @ backbone_lr
      - MViTv2 backbone: RAdam @ backbone_lr
      - Disentanglers + GMU + MLP + Heads: Adam @ head_lr
    
    Only parameters with requires_grad=True are included.
    """
    backbone_lr = lr if backbone_lr is None else backbone_lr
    head_lr = lr if head_lr is None else head_lr
    
    maxvit_params = [p for p in model.maxvit.parameters() if p.requires_grad]
    mvitv2_params = [p for p in model.mvitv2.parameters() if p.requires_grad]
    
    # New head-side params: disentanglers + GMU + MLP + classification heads
    head_side_params = []
    head_side_params += [p for p in model.cqt_disentangler.parameters() if p.requires_grad]
    head_side_params += [p for p in model.mel_disentangler.parameters() if p.requires_grad]
    head_side_params += [p for p in model.disentangled_gmu.parameters() if p.requires_grad]
    head_side_params += [p for p in model.mlp_shared.parameters() if p.requires_grad]
    head_side_params += [p for p in model.emotion_head.parameters() if p.requires_grad]
    
    if model.accent_head is not None:
        head_side_params += [p for p in model.accent_mlp.parameters() if p.requires_grad]
        head_side_params += [p for p in model.accent_head.parameters() if p.requires_grad]
    
    optimizers = []
    
    # Optimizer 1: MaxViT backbone (low LR) + head-side (high LR) → Adam
    param_groups = []
    if maxvit_params:
        param_groups.append({'params': maxvit_params, 'lr': backbone_lr})
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
    print("=" * 60)
    print("Testing Disentangled MaxMViT-MLP...")
    print("=" * 60)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with tiny backbone for speed
    model = MaxMViT_MLP_Disentangled(
        num_classes=4,
        hidden_size=512,
        dropout_rate=0.2,
        sub_dim=256,
        num_accent_classes=3,
        backbone_size='tiny'
    ).to(DEVICE)
    
    # Dummy input
    cqt = torch.randn(2, 3, 224, 224).to(DEVICE)
    mel = torch.randn(2, 3, 224, 224).to(DEVICE)
    
    # --- Test Training Mode ---
    model.train()
    output = model(cqt, mel, return_gate=True)
    
    print("\n--- Training Mode Output ---")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
    
    # Test DisentangledConstraintLoss
    dis_loss_fn = DisentangledConstraintLoss(
        weight_recon=0.1, weight_sim=0.05, weight_orth=0.05, warmup_epochs=5
    )
    
    aux_loss, loss_dict = dis_loss_fn(
        z_cqt=output['z_cqt'], z_mel=output['z_mel'],
        h_cqt_s=output['h_cqt_s'], h_cqt_p=output['h_cqt_p'],
        z_cqt_recon=output['z_cqt_recon'],
        h_mel_s=output['h_mel_s'], h_mel_p=output['h_mel_p'],
        z_mel_recon=output['z_mel_recon'],
        current_epoch=0  # epoch 0 → warmup_scale = 0.0
    )
    print(f"\nDisentangled Aux Loss (epoch 0, warmup=0): {aux_loss.item():.6f}")
    print(f"  Loss dict: {loss_dict}")
    
    aux_loss_ep10, loss_dict_10 = dis_loss_fn(
        z_cqt=output['z_cqt'], z_mel=output['z_mel'],
        h_cqt_s=output['h_cqt_s'], h_cqt_p=output['h_cqt_p'],
        z_cqt_recon=output['z_cqt_recon'],
        h_mel_s=output['h_mel_s'], h_mel_p=output['h_mel_p'],
        z_mel_recon=output['z_mel_recon'],
        current_epoch=10  # epoch 10 → warmup_scale = 1.0
    )
    print(f"Disentangled Aux Loss (epoch 10, warmup=1): {aux_loss_ep10.item():.6f}")
    print(f"  Loss dict: {loss_dict_10}")
    
    # --- Test Eval Mode ---
    model.eval()
    with torch.no_grad():
        eval_output = model(cqt, mel)
    
    print("\n--- Eval Mode Output ---")
    if isinstance(eval_output, tuple):
        emo_logits, acc_logits = eval_output
        print(f"  emotion_logits: {emo_logits.shape}")
        print(f"  accent_logits: {acc_logits.shape}")
    else:
        print(f"  emotion_logits: {eval_output.shape}")
    
    # Test optimizer
    optimizers = get_optimizer_disentangled(model, lr=0.0002, backbone_lr=0.0001, head_lr=0.0005)
    print(f"\nOptimizers: {len(optimizers)} optimizer(s)")
    for i, opt in enumerate(optimizers):
        total = sum(p.numel() for group in opt.param_groups for p in group['params'])
        print(f"  Optimizer {i}: {opt.__class__.__name__} ({total/1e6:.2f}M params)")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params/1e6:.2f}M")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")
    
    print("\n✅ All Disentangled model tests passed!")
