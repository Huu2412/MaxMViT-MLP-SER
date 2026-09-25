import os
import io
import torch
import numpy as np
import soundfile as sf
import torchaudio
from datasets import Dataset as HFDataset, load_dataset, Audio
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional, Any
from .pitch_utils import extract_audio_and_pitch

TARGET_CLASSES = ['happy', 'neutral', 'sad', 'angry']
CLASS_MAP = {c: i for i, c in enumerate(TARGET_CLASSES)}


def stratified_split(indices_and_labels: List[Tuple[int, int]], split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42):
    """
    Stratified split matching MaxMViT-MLP-SER:
    80% Train, 10% Val, 10% Test with fixed seed.
    """
    indices = [x[0] for x in indices_and_labels]
    labels = [x[1] for x in indices_and_labels]
    
    val_ratio = split_ratio[1]
    test_ratio = split_ratio[2]
    eval_ratio = val_ratio + test_ratio
    
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=eval_ratio,
        random_state=seed,
        stratify=labels
    )
    
    eval_labels = [labels[i] for i in eval_idx]
    relative_test_ratio = test_ratio / eval_ratio
    
    val_idx, test_idx = train_test_split(
        eval_idx,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=eval_labels
    )
    
    return train_idx, val_idx, test_idx


def _process_audio_item(audio_data: Any, default_sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decodes audio from various formats (bytes, file path, dict, array)
    and extracts 16kHz audio array and pitch array.
    """
    speech_tensor = None
    orig_sr = default_sr
    
    if isinstance(audio_data, dict):
        if 'bytes' in audio_data and audio_data['bytes'] is not None:
            data, orig_sr = sf.read(io.BytesIO(audio_data['bytes']))
            speech_tensor = torch.tensor(data, dtype=torch.float32)
        elif 'path' in audio_data and audio_data['path'] is not None and os.path.exists(audio_data['path']):
            speech_tensor, orig_sr = torchaudio.load(audio_data['path'])
        elif 'array' in audio_data:
            speech_tensor = torch.tensor(audio_data['array'], dtype=torch.float32)
            orig_sr = audio_data.get('sampling_rate', default_sr)
    elif isinstance(audio_data, str) and os.path.exists(audio_data):
        speech_tensor, orig_sr = torchaudio.load(audio_data)
    elif isinstance(audio_data, (np.ndarray, torch.Tensor)):
        speech_tensor = torch.tensor(audio_data, dtype=torch.float32)
        orig_sr = default_sr
    else:
        raise ValueError(f"Unsupported audio data format: {type(audio_data)}")
        
    return extract_audio_and_pitch(speech_tensor, orig_sr, target_sr=16000)


def load_visec_datasets(
    hf_id: str = "hustep-lab/ViSEC",
    csv_dir: Optional[str] = None,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    num_proc: int = 4,
    cache_dir: Optional[str] = None
):
    """
    Loads ViSEC datasets (Train, Val, Test) for the Pitch-Fusion model.
    Supports either local CSVs (train.csv, valid.csv, test.csv) or Hugging Face hub.
    """
    # ── Option 1: Load from local CSV directory if available ──
    if csv_dir is not None and os.path.exists(os.path.join(csv_dir, "train.csv")):
        print(f"[Dataset] Loading local CSVs from {csv_dir}...")
        train_csv = os.path.join(csv_dir, "train.csv")
        val_csv = os.path.join(csv_dir, "valid.csv") if os.path.exists(os.path.join(csv_dir, "valid.csv")) else os.path.join(csv_dir, "val.csv")
        test_csv = os.path.join(csv_dir, "test.csv")
        
        raw_train = load_dataset("csv", data_files=train_csv, split="train")
        raw_val = load_dataset("csv", data_files=val_csv, split="train")
        raw_test = load_dataset("csv", data_files=test_csv, split="train") if os.path.exists(test_csv) else raw_val
        
        def _map_csv_row(batch):
            audio, pitch = _process_audio_item(batch["path"])
            label = batch["emotion_id"] if "emotion_id" in batch else CLASS_MAP.get(str(batch["emotion"]).lower().strip(), 0)
            return {"audio_input": audio, "pitch_input": pitch, "label": label}
            
        print("[Dataset] Extracting audio and pitch features for local CSVs...")
        train_ds = raw_train.map(_map_csv_row, num_proc=num_proc)
        val_ds = raw_val.map(_map_csv_row, num_proc=num_proc)
        test_ds = raw_test.map(_map_csv_row, num_proc=num_proc)
        return train_ds, val_ds, test_ds

    # ── Option 2: Load from Hugging Face Hub (hustep-lab/ViSEC) ──
    print(f"[Dataset] Loading '{hf_id}' from Hugging Face Hub...")
    raw_ds = load_dataset(hf_id, split="train", cache_dir=cache_dir).cast_column("path", Audio(decode=False))
    
    # Filter 4 emotion classes and collect indices
    filtered_items = []
    for idx in range(len(raw_ds)):
        emo = str(raw_ds[idx]["emotion"]).lower().strip()
        if emo in CLASS_MAP:
            filtered_items.append((idx, CLASS_MAP[emo]))
            
    print(f"[Dataset] Filtered {len(filtered_items)} samples for 4 emotions: {TARGET_CLASSES}")
    
    # Perform reproducible stratified split
    train_indices, val_indices, test_indices = stratified_split(filtered_items, split_ratio=split_ratio, seed=seed)
    print(f"[Dataset] Split sizes (seed {seed}): Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Build split sub-datasets
    train_raw = raw_ds.select([it for it in train_indices])
    val_raw = raw_ds.select([it for it in val_indices])
    test_raw = raw_ds.select([it for it in test_indices])
    
    def _map_hf_row(batch):
        audio, pitch = _process_audio_item(batch["path"])
        emo_str = str(batch["emotion"]).lower().strip()
        label = CLASS_MAP.get(emo_str, 0)
        return {"audio_input": audio, "pitch_input": pitch, "label": label}
        
    print(f"[Dataset] Extracting features (audio + pitch) with {num_proc} processes...")
    train_ds = train_raw.map(_map_hf_row, num_proc=num_proc)
    val_ds = val_raw.map(_map_hf_row, num_proc=num_proc)
    test_ds = test_raw.map(_map_hf_row, num_proc=num_proc)
    
    return train_ds, val_ds, test_ds


class ViSECPitchDataCollator:
    """
    Collator to pad audio and pitch tensors to batch maximum length.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_inputs = self.processor.pad(
            [{"input_values": example["audio_input"]} for example in examples],
            return_tensors="pt",
            padding=True
        )
        pitch_inputs = self.processor.pad(
            [{"input_values": example["pitch_input"]} for example in examples],
            return_tensors="pt",
            padding=True
        )
        labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
        
        return {
            "audio_input": audio_inputs,
            "pitch_input": pitch_inputs,
            "label": labels
        }
