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

class ViSECDataset(Dataset):
    def __init__(self, hf_id="hustep-lab/ViSEC", split="train", target_classes=['happy', 'neutral', 'sad', 'angry'], sr=44100, target_size=(244, 244)):
        """
        Dataset class for ViSEC from Hugging Face.
        
        Args:
            hf_id (str): Hugging Face dataset ID.
            split (str): Dataset split (dummy parameter for API compatibility).
            target_classes (list): List of emotions to classify.
            sr (int): Sampling rate.
            target_size (tuple): Spec image size.
        """
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        
        print(f"Loading {hf_id}...")
        # Disable auto-decoding to avoid torchcodec issues
        from datasets import Audio
        self.ds = load_dataset(hf_id, split="train").cast_column("path", Audio(decode=False))
        
        # Normalization params (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Audio params
        self.n_fft = 4096
        self.hop_length = 256
        
        self.indices = []
        for idx, item in enumerate(self.ds):
            emo = item.get('emotion')
            if emo in self.target_classes:
                self.indices.append((idx, self.class_map[emo]))
                
        print(f"Filtered {len(self.indices)} samples from {len(self.ds)} total.")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label = self.indices[idx]
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
            
        if len(y) < self.n_fft:
            padding = self.n_fft - len(y) + 1
            y = np.pad(y, (0, padding), mode='constant')
            
        try:
            cqt = librosa.cqt(y, sr=self.sr)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            cqt_img = self._resize_normalize(cqt_db)
            mel_img = self._resize_normalize(mel_db)
            
            cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32)
            mel_tensor = torch.tensor(mel_img, dtype=torch.float32)
            
            return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing audio sample {ds_idx}: {e}")
            dummy_img = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return dummy_img, dummy_img, torch.tensor(label, dtype=torch.long)

    def _resize_normalize(self, spec):
        spec_min = spec.min()
        spec_max = spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        
        spec_3ch = np.stack([spec_resized]*3, axis=0)
        
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
            
        return spec_3ch

def get_visec_dataloaders(hf_id="hustep-lab/ViSEC", batch_size=16, num_workers=4):
    try:
        dataset = ViSECDataset(hf_id)
        
        # Split indices
        full_indices = dataset.indices
        total_len = len(full_indices)
        val_len = int(total_len * 0.2)
        train_len = total_len - val_len
        
        # Set seed to ensure reproducible train/val splits
        rng = random.Random(42)
        rng.shuffle(full_indices)
        
        train_indices = full_indices[:train_len]
        val_indices = full_indices[train_len:]
        
        train_ds = copy.deepcopy(dataset)
        train_ds.indices = train_indices
        
        val_ds = copy.deepcopy(dataset)
        val_ds.indices = val_indices
        
        print(f"Split complete. Train: {len(train_ds)}, Val: {len(val_ds)}")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader
    except Exception as e:
        print(f"Dataset load error: {e}")
        return None, None
