import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder, Wav2Vec2FeatureProjection
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from typing import Optional, Union, Dict, Any


class AttentionLayer(nn.Module):
    """
    Cross-Attention / Self-Attention Transformer block with Feed-Forward network.
    """
    def __init__(self, d_model: int, nhead: int = 16, dropout: float = 0.1):
        super(AttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        # src, tar: (Batch, Time, Channels) -> MultiheadAttention expects (Time, Batch, Channels)
        src = src.transpose(0, 1)
        tar = tar.transpose(0, 1)
        
        src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # Transpose back: (Time, Batch, Channels) -> (Batch, Time, Channels)
        src = src.transpose(0, 1)
        return src


class Wav2Vec2CrossAttentionPitchForSER(Wav2Vec2PreTrainedModel):
    """
    ViSEC Pitch-Fusion Model (ICASSP 2024):
    Fuses Wav2Vec 2.0 acoustic representations with Kaldi/interpolated Pitch features
    via bi-directional cross-attention and self-attention.
    """
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # CNN Feature Encoder for Pitch sequence
        self.pitch_encoder = Wav2Vec2FeatureEncoder(config)
        self.pitch_projection = Wav2Vec2FeatureProjection(config)

        # Cross-Attention modules: Acoustic -> Pitch and Pitch -> Acoustic
        n_heads = getattr(config, 'num_attention_heads', 16)
        if config.hidden_size % n_heads != 0:
            n_heads = 8 if config.hidden_size % 8 == 0 else 4
            
        self.attn_a_p = AttentionLayer(config.hidden_size, nhead=n_heads, dropout=config.hidden_dropout)
        self.attn_p_a = AttentionLayer(config.hidden_size, nhead=n_heads, dropout=config.hidden_dropout)
        self.self_attn = AttentionLayer(config.hidden_size * 2, nhead=n_heads, dropout=config.hidden_dropout)
        
        proj_size = getattr(config, 'classifier_proj_size', 256)
        self.projector = nn.Linear(config.hidden_size * 2, proj_size)
        self.classifier = nn.Linear(proj_size, config.num_labels)
        
        self.post_init()
        
    def freeze_feature_extractor(self):
        """Freezes feature encoder of wav2vec2 as recommended in fine-tuning."""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def _extract_tensor(self, inp: Any) -> torch.Tensor:
        """Safely extracts input_values tensor regardless of BatchFeature or dict format."""
        if hasattr(inp, 'input_values'):
            return inp.input_values
        elif isinstance(inp, dict) and 'input_values' in inp:
            return inp['input_values']
        elif isinstance(inp, torch.Tensor):
            return inp
        raise ValueError(f"Cannot extract input tensor from type {type(inp)}")

    def forward(
        self,
        audio_input: Any = None,
        pitch_input: Any = None,
        label: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        
        # Handle label alias (HuggingFace Trainer uses 'label' or 'labels')
        if label is None and labels is not None:
            label = labels
            
        # Support fallback when audio_input or pitch_input are in kwargs
        if audio_input is None and 'input_values' in kwargs:
            audio_input = kwargs['input_values']
            
        audio_tensor = self._extract_tensor(audio_input)
        pitch_tensor = self._extract_tensor(pitch_input)
        
        # 1. Acoustic branch through Wav2Vec 2.0 Transformer
        acoustic = self.wav2vec2(audio_tensor, attention_mask=None)[0]  # (Batch, Time, Hidden)
        
        # 2. Pitch branch through CNN feature encoder + projection
        pitch = self.pitch_encoder(pitch_tensor)  # (Batch, Channels, Time)
        pitch = pitch.transpose(1, 2)             # (Batch, Time, Channels)
        pitch, _ = self.pitch_projection(pitch)   # (Batch, Time, Hidden)
        
        # Align temporal length if slight difference due to rounding/stride
        min_time = min(acoustic.size(1), pitch.size(1))
        acoustic = acoustic[:, :min_time, :]
        pitch = pitch[:, :min_time, :]
        
        # 3. Bi-directional Cross-Attention
        a_p_attn = self.attn_a_p(acoustic, pitch)
        p_a_attn = self.attn_p_a(pitch, acoustic)
        
        # 4. Fusion + Self-Attention
        fused = torch.cat((a_p_attn, p_a_attn), dim=-1)  # (Batch, Time, Hidden * 2)
        fused = self.self_attn(fused, fused)
        
        # 5. Temporal Pooling & Classification Head
        pooled = self.projector(fused)
        pooled = pooled.mean(dim=1)  # Global Average Pooling over time
        logits = self.classifier(pooled)
        
        # 6. Loss computation
        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), label.view(-1))
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
