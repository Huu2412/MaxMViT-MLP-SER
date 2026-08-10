import os
import torch
import numpy as np
import librosa
import cv2
import random
from torch.utils.data import Dataset, DataLoader

class SERDataset(Dataset):
    def __init__(self, audio_paths, labels, sr=44100, target_size=(244, 244), augment=False, spec_augment_cfg=None, pitch_shift_cfg=None, time_shift_cfg=None): # Paper sr=44.1kHz, size=244x244
        """
        Dataset class for Speech Emotion Recognition.
        
        Args:
            audio_paths (list): List of paths to audio files.
            labels (list): List of integer labels.
            sr (int): Sampling rate.
            target_size (tuple): Target spectrogram size (H, W).
            augment (bool): Whether to apply data augmentation.
            spec_augment_cfg (dict): Configuration for SpecAugment.
            pitch_shift_cfg (dict): Configuration for Pitch Shift.
            time_shift_cfg (dict): Configuration for Time Shift.
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.sr = sr
        self.target_size = target_size
        self.augment = augment
        
        # SpecAugment parameters
        _cfg = spec_augment_cfg or {}
        self.freq_mask_param = _cfg.get('freq_mask_param', 27)
        self.time_mask_param = _cfg.get('time_mask_param', 100)
        self.num_freq_masks = _cfg.get('num_freq_masks', 1)
        self.num_time_masks = _cfg.get('num_time_masks', 1)
        self.spec_augment_prob = _cfg.get('prob', 0.5)

        # Waveform augmentation configs
        _pitch_cfg = pitch_shift_cfg or {}
        self.pitch_shift_prob = _pitch_cfg.get('prob', 0.0)
        self.pitch_shift_range = _pitch_cfg.get('n_steps_range', [-2.0, 2.0])
        
        _time_cfg = time_shift_cfg or {}
        self.time_shift_prob = _time_cfg.get('prob', 0.0)
        if 'range' in _time_cfg:
            self.time_shift_range = _time_cfg['range']
            self.use_time_stretch = True
        elif 'limit' in _time_cfg:
            self.time_shift_limit = _time_cfg['limit']
            self.use_time_stretch = False
        else:
            self.time_shift_range = [0.9, 1.1]
            self.use_time_stretch = True
        
        # Mel-STFT parameters (Paper Section III.C)
        self.n_fft = 4096
        self.hop_length = 256
        
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
             y = y.astype(np.float32)
             y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
             sr = self.sr
        else:
             y = y.astype(np.float32)
             sr = orig_sr
             
        # Ensure mono
        if len(y.shape) > 1:
             y = np.mean(y, axis=1) # Soundfile returns (samples, channels)
        
        # Waveform Augmentations (Pitch Shift and Time Shift/Stretch)
        if self.augment:
            # Time shift/stretch
            if self.time_shift_prob > 0 and random.random() < self.time_shift_prob:
                if self.use_time_stretch:
                    rate = random.uniform(self.time_shift_range[0], self.time_shift_range[1])
                    y = librosa.effects.time_stretch(y, rate=rate)
                else:
                    shift_amt = int(random.uniform(-self.time_shift_limit, self.time_shift_limit) * len(y))
                    y = np.roll(y, shift_amt)
            
            # Pitch shift
            if self.pitch_shift_prob > 0 and random.random() < self.pitch_shift_prob:
                n_steps = random.uniform(self.pitch_shift_range[0], self.pitch_shift_range[1])
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

        # Fix: Pad short audio to prevent Librosa warnings
        if len(y) < self.n_fft:
            padding = self.n_fft - len(y) + 1
            y = np.pad(y, (0, padding), mode='constant')

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

        # Apply SpecAugment before resize (training only) based on probability
        if self.augment and random.random() < self.spec_augment_prob:
            cqt_db = self._spec_augment(cqt_db)
            mel_db = self._spec_augment(mel_db)
            mel_delta = self._spec_augment(mel_delta)
            mel_delta2 = self._spec_augment(mel_delta2)

        # --- Resize & Normalize ---
        cqt_img = self._resize_normalize(cqt_db)
        mel_img = self._resize_normalize([mel_db, mel_delta, mel_delta2])
        
        # To Tensor [3, H, W]
        cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32)
        mel_tensor = torch.tensor(mel_img, dtype=torch.float32)
        
        return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long)
        
    def _spec_augment(self, spec):
        """
        Apply SpecAugment: frequency masking + time masking.
        Masks are filled with the mean value of the spectrogram.
        """
        spec = spec.copy()
        num_freq, num_time = spec.shape
        fill_value = spec.mean()
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, num_freq - 1))
            f0 = random.randint(0, num_freq - f)
            spec[f0:f0 + f, :] = fill_value
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, num_time - 1))
            t0 = random.randint(0, num_time - t)
            spec[:, t0:t0 + t] = fill_value
        
        return spec

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
            return np.stack([spec_resized]*3, axis=0)

def get_dataloader(paths, labels, batch_size=32, shuffle=True, augment=False, spec_augment_cfg=None, pitch_shift_cfg=None, time_shift_cfg=None):
    dataset = SERDataset(paths, labels, augment=augment, spec_augment_cfg=spec_augment_cfg, pitch_shift_cfg=pitch_shift_cfg, time_shift_cfg=time_shift_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
