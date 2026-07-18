import torch
import numpy as np
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import io
import soundfile as sf
import random
import copy
import os

class ViSECDataset(Dataset):
    def __init__(self, hf_id="hustep-lab/ViSEC", split="train", target_classes=['happy', 'neutral', 'sad', 'angry'], sr=44100, target_size=(244, 244), augment=False, spec_augment_cfg=None, pitch_shift_cfg=None, time_shift_cfg=None, load_accent=False, csv_path=None, waveform_augment_cfg=None):
        """
        Dataset class for ViSEC from Hugging Face or Local Preprocessed WAVs.
        
        Args:
            hf_id (str): Hugging Face dataset ID.
            split (str): Dataset split (dummy parameter for API compatibility).
            target_classes (list): List of emotions to classify.
            sr (int): Sampling rate.
            target_size (tuple): Spec image size.
            augment (bool): Whether to apply SpecAugment (for training only).
            spec_augment_cfg (dict): SpecAugment config.
            pitch_shift_cfg (dict): Pitch shift config.
            time_shift_cfg (dict): Time shift config.
            csv_path (str): Path to local preprocessed CSV metadata.
            waveform_augment_cfg (dict): OneOf waveform augmentation config.
        """
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        self.augment = augment
        
        # Accent/Region Recognition
        self.load_accent = load_accent
        self.accent_map = {'north': 0, 'south': 1, 'mid': 2}
        self.num_accent_classes = len(self.accent_map)
        
        # SpecAugment parameters
        _cfg = spec_augment_cfg or {}
        self.freq_mask_param = _cfg.get('freq_mask_param', 27)
        self.time_mask_param = _cfg.get('time_mask_param', 100)
        self.num_freq_masks = _cfg.get('num_freq_masks', 1)
        self.num_time_masks = _cfg.get('num_time_masks', 1)
        self.spec_augment_prob = _cfg.get('prob', 0.5)  # Default: 50% probability
 
        # Waveform augmentation configs (old style fallback)
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
            
        # OneOf Waveform Augmentation (new style)
        self.waveform_augment_cfg = waveform_augment_cfg
        if waveform_augment_cfg is not None:
            self.use_oneof_augment = True
            self.waveform_augment_prob = waveform_augment_cfg.get('prob', 0.6)
            
            _pitch = waveform_augment_cfg.get('pitch_shift', {})
            _noise = waveform_augment_cfg.get('noise_injection', {})
            _time = waveform_augment_cfg.get('time_shift', {})
            
            # Normalize internal weights
            self.pitch_weight = _pitch.get('weight', 0.4)
            self.noise_weight = _noise.get('weight', 0.4)
            self.time_weight = _time.get('weight', 0.2)
            
            total_w = self.pitch_weight + self.noise_weight + self.time_weight
            if total_w > 0:
                self.pitch_weight /= total_w
                self.noise_weight /= total_w
                self.time_weight /= total_w
            else:
                self.pitch_weight, self.noise_weight, self.time_weight = 0.4, 0.4, 0.2
                
            self.pitch_shift_range = _pitch.get('n_steps_range', [-1.0, 1.0])
            self.noise_factor_range = _noise.get('noise_factor_range', [0.001, 0.015])
            self.time_shift_range = _time.get('range', [0.9, 1.1])
        else:
            self.use_oneof_augment = False
        
        # Normalization params (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Audio params
        self.n_fft = 4096
        self.hop_length = 256
        
        self.indices = []
        self.csv_path = csv_path
        
        if csv_path is not None and os.path.exists(csv_path):
            import pandas as pd
            print(f"Loading local dataset metadata from {csv_path}...")
            self.is_local = True
            self.df = pd.read_csv(csv_path)
            for idx, row in self.df.iterrows():
                emo = row['emotion']
                accent_label = int(row['accent'])
                self.indices.append((idx, self.class_map[emo], accent_label))
        else:
            self.is_local = False
            from datasets import Audio, load_dataset
            
            local_parquet = "local_data/train.parquet"
            if os.path.exists(local_parquet):
                print(f"Loading bypass dataset from local parquet: {local_parquet}...")
                self.ds = load_dataset("parquet", data_files=local_parquet, split="train").cast_column("path", Audio(decode=False))
            else:
                print(f"Loading {hf_id} from Hugging Face Hub...")
                self.ds = load_dataset(hf_id, split="train").cast_column("path", Audio(decode=False))
            
            emotions = self.ds['emotion']
            has_accent = self.load_accent and 'accent' in self.ds.column_names
            accents = self.ds['accent'] if has_accent else [None] * len(self.ds)
            
            for idx, (emo, accent_str) in enumerate(zip(emotions, accents)):
                if emo in self.target_classes:
                    # Get accent label if available
                    accent_label = -1  # Default: unknown/missing
                    if has_accent and accent_str and accent_str in self.accent_map:
                        accent_label = self.accent_map[accent_str]
                    self.indices.append((idx, self.class_map[emo], accent_label))
                
        print(f"Filtered {len(self.indices)} samples.")
        if self.load_accent:
            accent_counts = {}
            for _, _, acc in self.indices:
                accent_counts[acc] = accent_counts.get(acc, 0) + 1
            print(f"Accent distribution: {accent_counts}")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label, accent_label = self.indices[idx]
        
        if getattr(self, 'is_local', False):
            # Load from preprocessed WAV file path
            file_path = self.df.iloc[ds_idx]['file_path']
            y, orig_sr = sf.read(file_path)
            y = y.astype(np.float32)
        else:
            # Fallback to Hugging Face
            item = self.ds[ds_idx]
            audio_bytes = item['path']['bytes']
            
            y, orig_sr = sf.read(io.BytesIO(audio_bytes))
            
            if orig_sr != self.sr:
                y = y.astype(np.float32)
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
            else:
                y = y.astype(np.float32)
                
            if y.ndim > 1:
                y = np.mean(y, axis=0)
                
            # Waveform Augmentations
            if self.augment:
                if getattr(self, 'use_oneof_augment', False):
                    # Cụm 1: OneOf Waveform Augmentation
                    if random.random() < self.waveform_augment_prob:
                        r = random.random()
                        if r < self.pitch_weight:
                            # Pitch shift
                            n_steps = random.uniform(self.pitch_shift_range[0], self.pitch_shift_range[1])
                            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
                        elif r < self.pitch_weight + self.noise_weight:
                            # Noise injection
                            noise_factor = random.uniform(self.noise_factor_range[0], self.noise_factor_range[1])
                            noise = np.random.normal(0, 1, len(y))
                            y = y + noise_factor * noise
                        else:
                            # Time shift/stretch
                            rate = random.uniform(self.time_shift_range[0], self.time_shift_range[1])
                            y = librosa.effects.time_stretch(y, rate=rate)
                else:
                    # Old style fallback (Independent)
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

        if len(y) < self.n_fft:
            padding = self.n_fft - len(y) + 1
            y = np.pad(y, (0, padding), mode='constant')
            
        try:
            cqt = librosa.cqt(y, sr=self.sr)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
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
            
            cqt_img = self._resize_normalize(cqt_db)
            
            # ABLATION: Thay vì dùng delta và delta-delta, ta copy mel_db thành 3 channels
            # mel_img = self._resize_normalize([mel_db, mel_delta, mel_delta2])
            mel_img = self._resize_normalize([mel_db, mel_db, mel_db])
            
            cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32)
            mel_tensor = torch.tensor(mel_img, dtype=torch.float32)
            
            if self.load_accent:
                return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(accent_label, dtype=torch.long)
            return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing audio sample {ds_idx}: {e}")
            dummy_img = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            if self.load_accent:
                return dummy_img, dummy_img, torch.tensor(label, dtype=torch.long), torch.tensor(accent_label, dtype=torch.long)
            return dummy_img, dummy_img, torch.tensor(label, dtype=torch.long)
            
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
            
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
            
        return spec_3ch

    def _spec_augment(self, spec):
        """
        Apply SpecAugment: frequency masking + time masking.
        Masks are filled with the mean value of the spectrogram.
        
        Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method
        for Automatic Speech Recognition", Interspeech 2019.
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

def get_visec_dataloaders(hf_id="hustep-lab/ViSEC", batch_size=16, num_workers=4, spec_augment_cfg=None, pitch_shift_cfg=None, time_shift_cfg=None, seed=42, load_accent=False, waveform_augment_cfg=None):
    try:
        local_train_csv = os.path.join("visec_dataset", "train.csv")
        local_val_csv = os.path.join("visec_dataset", "val.csv")
        
        if os.path.exists(local_train_csv) and os.path.exists(local_val_csv):
            print(f"Loading local preprocessed datasets from {local_train_csv} and {local_val_csv}...")
            train_ds = ViSECDataset(
                hf_id=hf_id,
                spec_augment_cfg=spec_augment_cfg,
                pitch_shift_cfg=pitch_shift_cfg,
                time_shift_cfg=time_shift_cfg,
                load_accent=load_accent,
                csv_path=local_train_csv,
                augment=True,
                waveform_augment_cfg=waveform_augment_cfg
            )
            val_ds = ViSECDataset(
                hf_id=hf_id,
                spec_augment_cfg=spec_augment_cfg,
                pitch_shift_cfg=pitch_shift_cfg,
                time_shift_cfg=time_shift_cfg,
                load_accent=load_accent,
                csv_path=local_val_csv,
                augment=False,
                waveform_augment_cfg=waveform_augment_cfg
            )
            print(f"Split complete. Train: {len(train_ds)}, Val: {len(val_ds)}")
        else:
            dataset = ViSECDataset(
                hf_id=hf_id, 
                spec_augment_cfg=spec_augment_cfg, 
                pitch_shift_cfg=pitch_shift_cfg, 
                time_shift_cfg=time_shift_cfg,
                load_accent=load_accent,
                waveform_augment_cfg=waveform_augment_cfg
            )
            
            # Split indices
            full_indices = dataset.indices
            total_len = len(full_indices)
            val_len = int(total_len * 0.2)
            train_len = total_len - val_len
            
            # Set seed to ensure reproducible train/val splits
            rng = random.Random(seed)
            rng.shuffle(full_indices)
            
            train_indices = full_indices[:train_len]
            val_indices = full_indices[train_len:]
            
            train_ds = copy.copy(dataset)
            train_ds.indices = train_indices
            train_ds.augment = True  # Enable SpecAugment and Waveform augmentations for training
            
            val_ds = copy.copy(dataset)
            val_ds.indices = val_indices
            val_ds.augment = False  # No augment for validation
            
            print(f"Split complete. Train: {len(train_ds)}, Val: {len(val_ds)}")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader
    except Exception as e:
        print(f"Dataset load error: {e}")
        return None, None
