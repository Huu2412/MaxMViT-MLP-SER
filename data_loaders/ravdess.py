
import torch
import numpy as np
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import logging
import random
import copy

class RAVDESSHFDataset(Dataset):
    def __init__(self, hf_id="TwinkStart/RAVDESS", split="ravdess_emo", 
                 target_classes=['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'], 
                 sr=44100, target_size=(224, 224), augment=False, spec_augment_cfg=None, pitch_shift_cfg=None, time_shift_cfg=None):
        """
        Dataset class for RAVDESS from Hugging Face.
        """
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
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
        
        # Normalization params (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        print(f"Loading {hf_id} [{split}]...")
        # Load dataset
        self.ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        # Audio params
        self.n_fft = 4096
        self.hop_length = 256
        
        self.filter_data()

    def filter_data(self):
        self.indices = []
        # RAVDESS usually has: neutral, calm, happy, sad, angry, fearful, disgust, surprised
        # Mapping to 8 classes
        emo_map = {
            'neutral': 'neutral',
            'calm': 'calm',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fear',
            'disgust': 'disgust',
            'surprised': 'surprise'
        }
        
        for idx, item in enumerate(self.ds):
            raw_emo = item.get('emotion') # e.g., "angry"
            
            # Simple normalization just in case
            if isinstance(raw_emo, str):
                raw_emo = raw_emo.lower().strip()
                
            short_emo = emo_map.get(raw_emo)
            
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
                
        print(f"Filtered {len(self.indices)} samples from {len(self.ds)} total.")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label = self.indices[idx]
        item = self.ds[ds_idx]
        
        # Audio processing
        audio_bytes = item['audio']['bytes']
        
        # Decode
        import soundfile as sf
        import io
        y, orig_sr = sf.read(io.BytesIO(audio_bytes))
        
        # Resample
        if orig_sr != self.sr:
            y = y.astype(np.float32)
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
        else:
            y = y.astype(np.float32)
            
        # Ensure mono
        if y.ndim > 1:
            y = np.mean(y, axis=0) 
            
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
                y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

        # Fix: Pad short audio to prevent Librosa warnings
        if len(y) < self.n_fft:
            padding = self.n_fft - len(y) + 1
            y = np.pad(y, (0, padding), mode='constant')
            
        # --- Preprocessing ---
        try:
            # CQT
            cqt = librosa.cqt(y, sr=self.sr)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
            # Mel-STFT
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
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

            # Resize & Normalize
            cqt_img = self._resize_normalize(cqt_db)
            mel_img = self._resize_normalize([mel_db, mel_delta, mel_delta2])
            
            cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32)
            mel_tensor = torch.tensor(mel_img, dtype=torch.float32)
            
            return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing sample {ds_idx}: {e}")
            dummy_img = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return dummy_img, dummy_img, torch.tensor(label, dtype=torch.long)

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
        if isinstance(spec, (list, tuple)):
            channels = []
            for s in spec:
                s_min = s.min()
                s_max = s.max()
                s_norm = (s - s_min) / (s_max - s_min + 1e-8)
                s_resized = cv2.resize(s_norm, (self.target_size[1], self.target_size[0]))
                channels.append(s_resized)
            spec_3ch = np.stack(channels, axis=0)
        else:
            spec_min = spec.min()
            spec_max = spec.max()
            spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
            spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
            spec_3ch = np.stack([spec_resized]*3, axis=0)
            
        # ImageNet Norm
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
            
        return spec_3ch

def get_ravdess_dataloaders(hf_id="TwinkStart/RAVDESS", batch_size=16, num_workers=4, spec_augment_cfg=None, pitch_shift_cfg=None, time_shift_cfg=None, seed=42):
    try:
        full_ds = RAVDESSHFDataset(
            hf_id, 
            split="ravdess_emo", 
            spec_augment_cfg=spec_augment_cfg,
            pitch_shift_cfg=pitch_shift_cfg,
            time_shift_cfg=time_shift_cfg
        )
        
        # Manual Split 80/20
        full_indices = full_ds.indices
        total = len(full_indices)
        val_len = int(total * 0.2)
        train_len = total - val_len
        
        # Ensure reproducibility
        random.seed(seed) 
        random.shuffle(full_indices)
        
        train_indices = full_indices[:train_len]
        val_indices = full_indices[train_len:]
        
        train_ds = copy.deepcopy(full_ds)
        train_ds.indices = train_indices
        train_ds.augment = True  # Enable augmentations for training
        
        val_ds = copy.deepcopy(full_ds)
        val_ds.indices = val_indices
        val_ds.augment = False  # No augment for validation
        
        print(f"Split RAVDESS: Train {len(train_ds)}, Val {len(val_ds)}")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader
    except Exception as e:
        print(f"RAVDESS load error: {e}")
        return None, None
