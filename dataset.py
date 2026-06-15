import os
import torch
import numpy as np
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader

class SERDataset(Dataset):
    def __init__(self, audio_paths, labels, sr=44100, target_size=(244, 244)): # Paper sr=44.1kHz, size=244x244
        """
        Dataset class for Speech Emotion Recognition.
        
        Args:
            audio_paths (list): List of paths to audio files.
            labels (list): List of integer labels.
            sr (int): Sampling rate.
            target_size (tuple): Target spectrogram size (H, W).
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.sr = sr
        self.target_size = target_size
        
        # Mel-STFT parameters (Paper Section III.C)
        self.n_fft = 4096
        self.hop_length = 256
        
        # CQT parameters (Default/Standard)
        # Paper mentions "logarithmic frequency binning".
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load audio using soundfile
        import soundfile as sf
        y, orig_sr = sf.read(path)
        
        # Resample if needed
        if orig_sr != self.sr:
             y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
             sr = self.sr
        else:
             sr = orig_sr
             
        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1) # Soundfile returns (samples, channels)
        
        # --- Generate CQT ---
        # "Constant-Q resolution... higher resolution at lower frequencies"
        cqt = librosa.cqt(y, sr=sr)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        # --- Generate Mel-STFT ---
        # "Frame length 4096 samples and a hop size of 256 samples"
        # "Logarithm of the energy values"
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Calculate Mel Delta and Delta-Delta robustly
        t = mel_db.shape[1]
        if t >= 3:
            width = min(9, t)
            if width % 2 == 0:
                width = max(3, width - 1)
            mel_delta = librosa.feature.delta(mel_db, order=1, width=width)
            mel_delta2 = librosa.feature.delta(mel_db, order=2, width=width)
        else:
            mel_delta = np.zeros_like(mel_db)
            mel_delta2 = np.zeros_like(mel_db)

        # Calculate CQT Delta and Delta-Delta robustly
        t_cqt = cqt_db.shape[1]
        if t_cqt >= 3:
            width_cqt = min(9, t_cqt)
            if width_cqt % 2 == 0:
                width_cqt = max(3, width_cqt - 1)
            cqt_delta = librosa.feature.delta(cqt_db, order=1, width=width_cqt)
            cqt_delta2 = librosa.feature.delta(cqt_db, order=2, width=width_cqt)
        else:
            cqt_delta = np.zeros_like(cqt_db)
            cqt_delta2 = np.zeros_like(cqt_db)
        
        # --- Resize & Normalize ---
        cqt_img = self._resize_normalize([cqt_db, cqt_delta, cqt_delta2])
        mel_img = self._resize_normalize([mel_db, mel_delta, mel_delta2])
        
        # To Tensor [3, H, W]
        cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32)
        mel_tensor = torch.tensor(mel_img, dtype=torch.float32)
        
        return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long)
        
    def _resize_normalize(self, spec):
        # Normalize to 0-255 or 0-1. Vision models usually like 0-1 or standard normalization.
        # Paper doesn't specify normalization, but implicitly required for images.
        # Let's normalize globally to 0-1 per image.
        if isinstance(spec, (list, tuple)):
            channels = []
            for s in spec:
                s_min = s.min()
                s_max = s.max()
                s_norm = (s - s_min) / (s_max - s_min + 1e-8)
                s_resized = cv2.resize(s_norm, (self.target_size[1], self.target_size[0]))
                channels.append(s_resized)
            return np.stack(channels, axis=0)
        else:
            spec_min = spec.min()
            spec_max = spec.max()
            spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
            spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
            return spec_resized

def get_dataloader(paths, labels, batch_size=32, shuffle=True):
    dataset = SERDataset(paths, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
