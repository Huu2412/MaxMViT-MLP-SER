"""
preprocess_features.py — Trích xuất spectrogram CQT / Mel-STFT từ ViSEC
và lưu thành file .pkl để tăng tốc huấn luyện.

Thay vì tính CQT + Mel mỗi epoch (rất chậm do librosa chạy trên CPU),
script này tính một lần duy nhất, lưu kết quả spectrogram thô vào đĩa.
Khi huấn luyện, Dataset chỉ cần đọc .pkl → SpecAugment → resize/normalize → trả về tensor.

Tính năng:
- Multiprocessing tăng tốc tối đa (sử dụng đa nhân CPU).
- Resumable: Tự động bỏ qua các mẫu đã xử lý nếu bị dừng giữa chừng.
- Tùy chọn float16 giúp tiết kiệm 50% dung lượng ổ cứng (~1.8 GB thay vì 3.7 GB).
- Tự động phân chia train/val split indices theo seed cố định.

Usage:
    python preprocess_features.py [--output_dir visec_features] [--workers 6] [--float16]
"""

import os
import sys
import pickle
import argparse
import time
import io
import multiprocessing as mp
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


def extract_features(y, sr, n_fft=4096, hop_length=256, use_float16=True):
    """
    Trích xuất CQT và Mel-STFT spectrogram từ waveform.
    """
    # Pad nếu audio quá ngắn
    if len(y) < n_fft:
        padding = n_fft - len(y) + 1
        y = np.pad(y, (0, padding), mode='constant')

    # CQT spectrogram
    cqt = librosa.cqt(y, sr=sr)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    # Mel-STFT spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Delta và Delta-delta
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

    dtype = np.float16 if use_float16 else np.float32
    return {
        'cqt_db': cqt_db.astype(dtype),
        'mel_db': mel_db.astype(dtype),
        'mel_delta': mel_delta.astype(dtype),
        'mel_delta2': mel_delta2.astype(dtype),
    }


def _worker_process_sample(task):
    """Worker function chạy trong tiến trình con."""
    (feature_idx, original_idx, audio_data, emotion, emotion_id, accent_label,
     speaker_id, gender, output_dir, sr, n_fft, hop_length, use_float16) = task

    pkl_path = os.path.join(output_dir, f"{feature_idx}.pkl")
    meta = {
        'feature_idx': feature_idx,
        'original_ds_idx': original_idx,
        'emotion': emotion,
        'emotion_id': emotion_id,
        'accent': accent_label,
        'speaker_id': speaker_id,
        'gender': gender,
    }

    # Bỏ qua nếu file đã tồn tại và hợp lệ (hỗ trợ resume)
    if os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 1000:
        meta['status'] = 'skipped'
        return meta

    try:
        if isinstance(audio_data, dict):
            if 'bytes' in audio_data and audio_data['bytes'] is not None:
                y, orig_sr = sf.read(io.BytesIO(audio_data['bytes']))
                y = y.astype(np.float32)
            elif 'array' in audio_data:
                y = np.array(audio_data['array'], dtype=np.float32)
                orig_sr = audio_data['sampling_rate']
            else:
                return {'status': 'error', 'error': 'Unknown dict structure', 'ds_idx': original_idx}
        else:
            y, orig_sr = sf.read(audio_data)
            y = y.astype(np.float32)

        # Resample nếu cần
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

        # Đảm bảo mono
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        # Trích xuất đặc trưng
        features = extract_features(y, sr=sr, n_fft=n_fft, hop_length=hop_length, use_float16=use_float16)

        # Lưu atomic để tránh file hỏng khi bị ngắt giữa chừng
        tmp_pkl_path = pkl_path + ".tmp"
        with open(tmp_pkl_path, 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_pkl_path, pkl_path)

        meta['status'] = 'ok'
        return meta

    except Exception as e:
        return {'status': 'error', 'error': str(e), 'ds_idx': original_idx}


def main():
    parser = argparse.ArgumentParser(description='Pre-extract ViSEC spectrogram features')
    parser.add_argument('--output_dir', type=str, default='visec_features',
                        help='Directory to save cached features')
    parser.add_argument('--hf_id', type=str, default='hustep-lab/ViSEC',
                        help='HuggingFace dataset ID')
    parser.add_argument('--sr', type=int, default=44100,
                        help='Target sampling rate')
    parser.add_argument('--n_fft', type=int, default=4096,
                        help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='Hop length for Mel-STFT')
    parser.add_argument('--target_classes', nargs='+',
                        default=['happy', 'neutral', 'sad', 'angry'],
                        help='Target emotion classes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split')
    parser.add_argument('--workers', type=int, default=max(1, min(6, (os.cpu_count() or 4) - 1)),
                        help='Number of parallel worker processes')
    parser.add_argument('--float16', action='store_true', default=True,
                        help='Save features as float16 to save 50%% disk space (default: True)')
    parser.add_argument('--no-float16', dest='float16', action='store_false',
                        help='Save features as standard float32')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process (for quick testing)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load ViSEC dataset ───
    print(f"Loading ViSEC dataset from {args.hf_id}...")
    from datasets import load_dataset, Audio
    # decode=False avoids requiring torchcodec and gives raw bytes directly
    ds = load_dataset(args.hf_id, split='train').cast_column("path", Audio(decode=False))
    print(f"Total samples in ViSEC: {len(ds)}")

    # Class mapping
    class_map = {c: i for i, c in enumerate(args.target_classes)}
    accent_map = {'north': 0, 'south': 1, 'mid': 2}

    # Chuẩn bị danh sách các task
    tasks = []
    num_skipped = 0
    feature_idx = 0

    print("Filtering samples and preparing worker tasks...")
    for ds_idx in range(len(ds)):
        sample = ds[ds_idx]
        emotion = sample['emotion']

        if emotion not in class_map:
            num_skipped += 1
            continue

        emotion_id = class_map[emotion]

        accent_str = sample.get('accent', None)
        accent_label = -1
        if accent_str is not None:
            if isinstance(accent_str, (int, float)) or (isinstance(accent_str, str) and accent_str.isdigit()):
                acc_int = int(accent_str)
                if 0 <= acc_int < len(accent_map):
                    accent_label = acc_int
            else:
                clean_accent = str(accent_str).strip().lower()
                if clean_accent in accent_map:
                    accent_label = accent_map[clean_accent]

        audio_data = sample['path']
        speaker_id = sample.get('speaker_id', '')
        gender = sample.get('gender', '')

        tasks.append((
            feature_idx,
            ds_idx,
            audio_data,
            emotion,
            emotion_id,
            accent_label,
            speaker_id,
            gender,
            args.output_dir,
            args.sr,
            args.n_fft,
            args.hop_length,
            args.float16
        ))
        feature_idx += 1

    if args.limit:
        tasks = tasks[:args.limit]
        print(f"Limiting to first {len(tasks)} samples for testing.")

    print(f"Target samples to process: {len(tasks)} (Skipped {num_skipped} non-target emotions)")
    print(f"Parallel workers: {args.workers} (Float16: {args.float16})")

    # ─── Multiprocessing Execution ───
    metadata_dict = {}
    num_errors = 0
    num_cached = 0
    start_time = time.time()

    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            for res in tqdm(pool.imap(_worker_process_sample, tasks, chunksize=8), total=len(tasks), desc="Extracting features"):
                if res.get('status') == 'error':
                    num_errors += 1
                    print(f"\n[ERROR] Sample {res.get('ds_idx')}: {res.get('error')}")
                else:
                    if res.get('status') == 'skipped':
                        num_cached += 1
                    f_idx = res['feature_idx']
                    metadata_dict[f_idx] = {
                        'feature_idx': f_idx,
                        'original_ds_idx': res['original_ds_idx'],
                        'emotion': res['emotion'],
                        'emotion_id': res['emotion_id'],
                        'accent': res['accent'],
                        'speaker_id': res['speaker_id'],
                        'gender': res['gender'],
                    }
    else:
        for task in tqdm(tasks, desc="Extracting features"):
            res = _worker_process_sample(task)
            if res.get('status') == 'error':
                num_errors += 1
            else:
                if res.get('status') == 'skipped':
                    num_cached += 1
                f_idx = res['feature_idx']
                metadata_dict[f_idx] = {
                    'feature_idx': f_idx,
                    'original_ds_idx': res['original_ds_idx'],
                    'emotion': res['emotion'],
                    'emotion_id': res['emotion_id'],
                    'accent': res['accent'],
                    'speaker_id': res['speaker_id'],
                    'gender': res['gender'],
                }

    # Sắp xếp metadata theo đúng thứ tự feature_idx
    metadata = [metadata_dict[i] for i in range(len(metadata_dict)) if i in metadata_dict]
    elapsed = time.time() - start_time

    # ─── Save metadata ───
    metadata_path = os.path.join(args.output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ─── Save train/val split indices ───
    import random
    rng = random.Random(args.seed)
    all_indices = list(range(len(metadata)))
    rng.shuffle(all_indices)

    val_len = int(len(all_indices) * 0.2)
    train_indices = sorted(all_indices[val_len:])
    val_indices = sorted(all_indices[:val_len])

    split_path = os.path.join(args.output_dir, 'split_indices.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump({
            'train': train_indices,
            'val': val_indices,
            'seed': args.seed,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ─── Summary ───
    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir)
        if f.endswith('.pkl')
    )

    print(f"\n{'='*60}")
    print("[OK] Feature extraction complete!")
    print(f"{'='*60}")
    print(f"  Total samples extracted: {len(metadata)}")
    print(f"  Already cached (skipped): {num_cached}")
    print(f"  Errors:                  {num_errors}")
    print(f"  Time elapsed:            {elapsed:.1f}s ({elapsed/max(len(metadata),1):.2f}s/sample)")
    print(f"  Output directory:        {args.output_dir}/")
    print(f"  Total cache size:        {total_size/1024/1024:.1f} MB")
    print(f"  Train/Val split (seed={args.seed}): {len(train_indices)} / {len(val_indices)}")

    from collections import Counter
    emo_counts = Counter(m['emotion'] for m in metadata)
    print("\n  Emotion distribution:")
    for emo, count in emo_counts.most_common():
        print(f"    {emo}: {count}")

    acc_counts = Counter(m['accent'] for m in metadata)
    acc_names = {0: 'north', 1: 'south', 2: 'mid', -1: 'unknown'}
    print("\n  Accent distribution:")
    for acc, count in acc_counts.most_common():
        print(f"    {acc_names.get(acc, acc)}: {count}")

    print(f"\nĐể sử dụng cache khi huấn luyện, cập nhật configs/visec_optimized.yaml:")
    print("  dataset:")
    print("    args:")
    print(f"      cache_dir: \"{args.output_dir}\"")


if __name__ == '__main__':
    main()
