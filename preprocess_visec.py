import os
import io
import random
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio
from tqdm import tqdm

def preprocess_visec(hf_id="hustep-lab/ViSEC", output_dir="visec_dataset", seed=42, target_classes=None):
    if target_classes is None:
        target_classes = ['happy', 'neutral', 'sad', 'angry']
        
    accent_map = {'north': 0, 'south': 1, 'mid': 2}
    
    # Create directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Loading {hf_id} from Hugging Face...")
    ds = load_dataset(hf_id, split="train").cast_column("path", Audio(decode=False))
    
    # Filter by target classes
    filtered_samples = []
    for idx, item in enumerate(ds):
        emo = item.get('emotion')
        if emo in target_classes:
            accent_str = item.get('accent', None)
            accent_label = accent_map.get(accent_str, -1)
            filtered_samples.append({
                'hf_idx': idx,
                'emotion': emo,
                'accent': accent_label,
                'bytes': item['path']['bytes']
            })
            
    print(f"Filtered {len(filtered_samples)} samples from {len(ds)} total.")
    
    # Train / Val Split
    random.seed(seed)
    random.shuffle(filtered_samples)
    val_len = int(len(filtered_samples) * 0.2)
    train_samples = filtered_samples[val_len:]
    val_samples = filtered_samples[:val_len]
    
    print(f"Split size: Train = {len(train_samples)}, Val = {len(val_samples)}")
    
    # Process Train Samples (Expand 3x)
    train_records = []
    print("Processing & Augmenting Train Set (Original, Pitch-shifted, Time-shifted)...")
    for idx, sample in enumerate(tqdm(train_samples)):
        try:
            # Read audio bytes
            y, sr = sf.read(io.BytesIO(sample['bytes']))
            if y.ndim > 1:
                y = np.mean(y, axis=0)
            
            # Resample to 44100Hz if not already
            target_sr = 44100
            if sr != target_sr:
                y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # Save Original
            orig_name = f"train_{idx}_orig.wav"
            orig_path = os.path.join(train_dir, orig_name)
            sf.write(orig_path, y, sr)
            train_records.append({
                'file_path': orig_path,
                'emotion': sample['emotion'],
                'accent': sample['accent']
            })
            
            # Save Pitch Shifted (+1.0 semitone)
            y_pitch = librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=1.0)
            pitch_name = f"train_{idx}_pitch.wav"
            pitch_path = os.path.join(train_dir, pitch_name)
            sf.write(pitch_path, y_pitch, sr)
            train_records.append({
                'file_path': pitch_path,
                'emotion': sample['emotion'],
                'accent': sample['accent']
            })
            
            # Save Time Shifted/Stretched (0.9 speed)
            y_time = librosa.effects.time_stretch(y.astype(np.float32), rate=0.9)
            time_name = f"train_{idx}_time.wav"
            time_path = os.path.join(train_dir, time_name)
            sf.write(time_path, y_time, sr)
            train_records.append({
                'file_path': time_path,
                'emotion': sample['emotion'],
                'accent': sample['accent']
            })
            
        except Exception as e:
            print(f"Error processing train sample hf_idx {sample['hf_idx']}: {e}")
            
    # Process Val Samples (1x, No augmentation)
    val_records = []
    print("Processing Validation Set (Original only)...")
    for idx, sample in enumerate(tqdm(val_samples)):
        try:
            # Read audio bytes
            y, sr = sf.read(io.BytesIO(sample['bytes']))
            if y.ndim > 1:
                y = np.mean(y, axis=0)
                
            target_sr = 44100
            if sr != target_sr:
                y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                
            val_name = f"val_{idx}.wav"
            val_path = os.path.join(val_dir, val_name)
            sf.write(val_path, y, sr)
            val_records.append({
                'file_path': val_path,
                'emotion': sample['emotion'],
                'accent': sample['accent']
            })
            
        except Exception as e:
            print(f"Error processing val sample hf_idx {sample['hf_idx']}: {e}")
            
    # Save CSV metadata
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    
    print("\nPreprocessing complete!")
    print(f"Train dataset size: {len(train_df)} (Expanded from {len(train_samples)})")
    print(f"Val dataset size: {len(val_df)}")
    print(f"All files saved in: {output_dir}/")

if __name__ == "__main__":
    preprocess_visec()
