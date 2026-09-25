import torch
import numpy as np
import torchaudio

_KALDI_AVAILABLE = False
_KALDI_FN = None

# Check if Kaldi pitch is available in torchaudio
try:
    from torchaudio.functional import compute_kaldi_pitch as _k_fn
    _KALDI_FN = _k_fn
    _KALDI_AVAILABLE = True
except Exception:
    try:
        from torchaudio.compliance.kaldi import compute_kaldi_pitch as _k_fn
        _KALDI_FN = _k_fn
        _KALDI_AVAILABLE = True
    except Exception:
        _KALDI_AVAILABLE = False


def extract_audio_and_pitch(speech_tensor: torch.Tensor, sr: int, target_sr: int = 16000, max_duration: float = 8.0):
    """
    Extracts 16kHz normalized mono audio and interpolated pitch sequence.
    
    Compatible across all versions of torchaudio:
    - Uses Kaldi pitch if available (torchaudio <= 2.0).
    - Uses torchaudio.functional.detect_pitch_frequency or librosa.yin as fallback (torchaudio >= 2.1).
    - Caps audio duration to max_duration (default 8.0s) to prevent CUDA OOM on long outliers.
    
    Returns:
        speech_array (np.ndarray): 1D float32 waveform at target_sr.
        pitch_input (np.ndarray): 1D float32 interpolated pitch matching waveform length.
    """
    if not isinstance(speech_tensor, torch.Tensor):
        speech_tensor = torch.tensor(speech_tensor, dtype=torch.float32)
        
    # Ensure 2D tensor (channels, time)
    if speech_tensor.ndim == 1:
        speech_tensor = speech_tensor.unsqueeze(0)
    elif speech_tensor.ndim > 2:
        speech_tensor = speech_tensor.squeeze()
        if speech_tensor.ndim == 1:
            speech_tensor = speech_tensor.unsqueeze(0)
            
    # Convert stereo to mono if necessary
    if speech_tensor.shape[0] > 1:
        speech_tensor = torch.mean(speech_tensor, dim=0, keepdim=True)
        
    # Resample to target_sr (16000 Hz for Wav2Vec2)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        speech_tensor = resampler(speech_tensor)
        sr = target_sr

    # Cap max duration to prevent CUDA OOM on extreme outlier long audios
    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * target_sr)
        if speech_tensor.shape[1] > max_samples:
            speech_tensor = speech_tensor[:, :max_samples]

    waveform_length = speech_tensor.shape[1]
    
    # ── Pitch Extraction ──
    pitch_raw = None
    if _KALDI_AVAILABLE and _KALDI_FN is not None:
        try:
            pitch_feature = _KALDI_FN(speech_tensor, sr)
            # Kaldi returns [batch, time, 2] -> index [..., 0][0] is NCCF / Pitch
            pitch_raw = pitch_feature[..., 0][0].cpu().numpy()
        except Exception:
            pitch_raw = None
            
    if pitch_raw is None:
        # Fallback 1: torchaudio.functional.detect_pitch_frequency
        try:
            pitch_tensor = torchaudio.functional.detect_pitch_frequency(speech_tensor, sr)
            pitch_raw = pitch_tensor[0].cpu().numpy()
        except Exception:
            pitch_raw = None

    if pitch_raw is None:
        # Fallback 2: librosa.yin
        try:
            import librosa
            y = speech_tensor[0].cpu().numpy()
            pitch_raw = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            pitch_raw = np.nan_to_num(pitch_raw, nan=0.0)
        except Exception:
            # Ultimate safety fallback: zero array
            pitch_raw = np.zeros(waveform_length // 160 + 1, dtype=np.float32)

    pitch_length = len(pitch_raw)
    if pitch_length == 0:
        pitch_raw = np.zeros(1, dtype=np.float32)
        pitch_length = 1

    # Linear interpolation to match exact audio waveform length
    pitch_input = np.interp(
        np.linspace(0, waveform_length, waveform_length),
        np.linspace(0, pitch_length, pitch_length),
        pitch_raw
    )
    
    speech_array = speech_tensor[0].cpu().numpy().astype(np.float32)
    pitch_input = pitch_input.astype(np.float32)
    
    return speech_array, pitch_input
