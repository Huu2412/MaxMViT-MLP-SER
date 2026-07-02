import torch
import numpy as np
import librosa
import cv2
import random
import copy
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class IEMOCAPHFDataset(Dataset):
    def __init__(self, hf_id="AbstractTTS/IEMOCAP", split="train", target_classes=['neu', 'hap', 'ang', 'sad'], sr=44100, target_size=(244, 244), augment=False, spec_augment_cfg=None, pitch_shift_cfg=None, time_shift_cfg=None):
        """
        Dataset class for IEMOCAP from Hugging Face.
        
        Args:
            hf_id (str): Hugging Face dataset ID.
            split (str): Dataset split ('train', 'validation', etc.).
            target_classes (list): List of emotions to classify.
            sr (int): Sampling rate.
            target_size (tuple): Spec image size.
            augment (bool): Whether to apply data augmentation.
            spec_augment_cfg (dict): SpecAugment config.
            pitch_shift_cfg (dict): Pitch shift config.
            time_shift_cfg (dict): Time shift config.
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
        
        print(f"Loading {hf_id} [{split}]...")
        # Load dataset
        # Disable auto-decoding to avoid torchcodec issues
        from datasets import Audio
        self.ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        # Normalization params (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Audio params
        self.n_fft = 4096
        self.hop_length = 256
        
        # Mapping from dataset emotion strings to our codes
        # IEMOCAP usually: neutral, happy, angry, sad, frustrated, excited, fear, surprise, disgust, other, xxx
        # 'exc' -> 'hap' is standard.
        self.filter_data()

    def filter_data(self):
        self.indices = []
        for idx, item in enumerate(self.ds):
            emo = item.get('major_emotion')
            
            # Map full names to abbreviations if needed or just use first 3 chars if consistent
            # Sample showed 'neutral'. Let's assume standard full names or use mapping
            # Standard IEMOCAP mapping often used:
            # ang: angry
            # hap: happy, excited
            # sad: sad
            # neu: neutral
            
            # if emo == 'excited':
            #     emo = 'happy'
                
            # Convert to 3-char code for consistency with my other code if needed, 
            # Or just map the string 'happy' to what my target_classes expects.
            # My target_classes in train script are ['neu', 'hap', 'ang', 'sad'].
            # So I should map 'neutral' -> 'neu', 'happy' -> 'hap', 'angry' -> 'ang', 'sad' -> 'sad'.
            emo_map = {
                'neutral': 'neu',
                'happy': 'hap',
                # 'excited': 'hap',
                'angry': 'ang',
                'sad': 'sad'
            }
            
            short_emo = emo_map.get(emo)
            
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
                
        print(f"Filtered {len(self.indices)} samples from {len(self.ds)} total.")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label = self.indices[idx]
        item = self.ds[ds_idx]
        
        # Audio processing
        # item['audio']['array'] is numpy array
        # item['audio']['sampling_rate'] is original SR
        # Audio processing
        # item['audio']['bytes'] contains raw audio bytes when decode=False
        audio_bytes = item['audio']['bytes']
        
        # Decode with soundfile
        import soundfile as sf
        import io
        y, orig_sr = sf.read(io.BytesIO(audio_bytes))
        
        # Resample if needed
        # Resample if needed
        if orig_sr != self.sr:
            # Resample needs float32
            y = y.astype(np.float32)
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
        else:
            y = y.astype(np.float32)
            
        # Ensure y is 1D
        if y.ndim > 1:
            y = np.mean(y, axis=0) # Convert stereo to mono

        # Determine whether to apply data augmentation (shared probability p = spec_augment_prob)
        apply_aug = self.augment and (random.random() < self.spec_augment_prob)

        # Waveform Augmentations (Pitch Shift and Time Shift/Stretch)
        if apply_aug:
            # Time shift/stretch
            if self.time_shift_prob > 0:
                if self.use_time_stretch:
                    rate = random.uniform(self.time_shift_range[0], self.time_shift_range[1])
                    y = librosa.effects.time_stretch(y, rate=rate)
                else:
                    shift_amt = int(random.uniform(-self.time_shift_limit, self.time_shift_limit) * len(y))
                    y = np.roll(y, shift_amt)
            
            # Pitch shift
            if self.pitch_shift_prob > 0:
                n_steps = random.uniform(self.pitch_shift_range[0], self.pitch_shift_range[1])
                y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

        # Fix: Pad short audio to prevent Librosa warnings
        if len(y) < self.n_fft:
            padding = self.n_fft - len(y) + 1
            y = np.pad(y, (0, padding), mode='constant')
            
        # --- Preprocessing same as SERDataset ---
        
        # CQT
        try:
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
            
            # Apply SpecAugment before resize (training only) based on the shared decision
            if apply_aug:
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
            # In case of empty audio or error, return correct shapes with zeros
            print(f"Error processing audio sample {ds_idx}: {e}")
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
            spec_3ch = np.stack([spec_resized]*3, axis=0) # [3, H, W]
        
        # Normalize with ImageNet mean/std
        # spec_resized is in [0, 1].
        # We need to perform (x - mean) / std for each channel
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
            
        return spec_3ch # Returns [3, H, W] numpy array

def get_hf_dataloaders(hf_id, batch_size=32, num_workers=4, spec_augment_cfg=None, pitch_shift_cfg=None, time_shift_cfg=None, seed=42):
    # Hugging Face datasets usually have 'train', 'validation', 'test' splits or just 'train'.
    # AbstractTTS/IEMOCAP structure: checking...
    # If standard splits exist:
    try:
        train_ds = IEMOCAPHFDataset(
            hf_id, 
            split="train",
            spec_augment_cfg=spec_augment_cfg,
            pitch_shift_cfg=pitch_shift_cfg,
            time_shift_cfg=time_shift_cfg
        )
        # Try to load validation/test if exists, else split train
        try:
            val_ds = IEMOCAPHFDataset(
                hf_id, 
                split="validation",
                spec_augment_cfg=spec_augment_cfg,
                pitch_shift_cfg=pitch_shift_cfg,
                time_shift_cfg=time_shift_cfg
            )
        except:
             # If no validation split, manually split train using datasets library feature
             print("No validation split found. Automatically splitting train set (80/20)...")
             # We need to access the underlying HF dataset object to split it
             # BUT IEMOCAPHFDataset wraps it and applies filtering in __init__.
             # Splitting the filtered indices is cleaner.
             
             # Let's split indices of train_ds
             full_indices = train_ds.indices
             total_len = len(full_indices)
             val_len = int(total_len * 0.2)
             train_len = total_len - val_len
             
             # Random shuffle with seed
             random.seed(seed)
             random.shuffle(full_indices)
             
             train_indices = full_indices[:train_len]
             val_indices = full_indices[train_len:]
             
             # Assign back to train_ds
             train_ds.indices = train_indices
             
             # Create val_ds as a copy but with val indices
             val_ds = copy.deepcopy(train_ds) 
             val_ds.indices = val_indices
             print(f"Split complete. Train: {len(train_ds)}, Val: {len(val_ds)}")
             
        # Set augment flag appropriately
        train_ds.augment = True
        if val_ds:
            val_ds.augment = False
            
        test_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_ds else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        
        return train_loader, test_loader
    except Exception as e:
        print(f"Dataset load error: {e}")
        return None, None
