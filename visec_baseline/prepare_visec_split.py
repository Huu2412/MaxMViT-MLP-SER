"""
prepare_visec_split.py — Export HuggingFace ViSEC to local WAVs and CSV splits
(train.csv, valid.csv, test.csv) matching MaxMViT-MLP-SER (80/10/10, seed 42).
"""

import os
import io
import argparse
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Audio
from sklearn.model_selection import train_test_split

TARGET_CLASSES = ['happy', 'neutral', 'sad', 'angry']
CLASS_MAP = {c: i for i, c in enumerate(TARGET_CLASSES)}
ACCENT_MAP = {'north': 0, 'south': 1, 'mid': 2}


def export_visec_split(
    output_dir: str = "visec_dataset",
    hf_id: str = "hustep-lab/ViSEC",
    seed: int = 42,
    target_sr: int = 16000
):
    wav_dir = os.path.join(output_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    
    print(f"Loading '{hf_id}' from Hugging Face...")
    ds = load_dataset(hf_id, split="train").cast_column("path", Audio(decode=False))
    print(f"Total samples in dataset: {len(ds)}")
    
    records = []
    print("Filtering and saving WAV audio files...")
    for idx in tqdm(range(len(ds)), desc="Exporting audio"):
        item = ds[idx]
        emotion = str(item.get("emotion", "")).strip().lower()
        if emotion not in CLASS_MAP:
            continue
            
        emotion_id = CLASS_MAP[emotion]
        accent_str = item.get("accent", None)
        accent_id = -1
        if accent_str is not None:
            clean_acc = str(accent_str).strip().lower()
            if clean_acc in ACCENT_MAP:
                accent_id = ACCENT_MAP[clean_acc]
            elif clean_acc.isdigit():
                accent_id = int(clean_acc)
                
        speaker_id = item.get("speaker_id", "")
        gender = item.get("gender", "")
        
        # Audio decode & save as 16kHz WAV
        audio_data = item["path"]
        if isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"]:
            y, sr = sf.read(io.BytesIO(audio_data["bytes"]))
        elif isinstance(audio_data, str) and os.path.exists(audio_data):
            y, sr = sf.read(audio_data)
        else:
            continue
            
        # Resample if needed
        if sr != target_sr:
            import librosa
            y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            
        if y.ndim > 1:
            y = y.mean(axis=1)
            
        wav_filename = f"visec_{idx:05d}_{emotion}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)
        sf.write(wav_path, y, samplerate=target_sr, subtype='PCM_16')
        
        records.append({
            "idx": idx,
            "path": wav_path,
            "emotion": emotion,
            "emotion_id": emotion_id,
            "accent": accent_id,
            "speaker_id": speaker_id,
            "gender": gender
        })
        
    df = pd.DataFrame(records)
    print(f"Filtered {len(df)} samples across 4 target emotions.")
    
    # 80/10/10 Stratified Split
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["emotion_id"])
    val_df, test_df = train_test_split(eval_df, test_size=0.5, random_state=seed, stratify=eval_df["emotion_id"])
    
    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "valid.csv")
    test_csv = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print("\n" + "="*50)
    print("[SUCCESS] Dataset export & split completed!")
    print("="*50)
    print(f"  Train: {len(train_df)} samples -> {train_csv}")
    print(f"  Val:   {len(val_df)} samples   -> {val_csv}")
    print(f"  Test:  {len(test_df)} samples  -> {test_csv}")
    print(f"  WAVs saved to: {wav_dir}/")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ViSEC split to local CSVs")
    parser.add_argument("--output_dir", type=str, default="visec_dataset")
    parser.add_argument("--hf_id", type=str, default="hustep-lab/ViSEC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_sr", type=int, default=16000)
    args = parser.parse_args()
    
    export_visec_split(args.output_dir, args.hf_id, args.seed, args.target_sr)
