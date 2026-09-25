"""
run_visec_baseline.py — Entrypoint to run ViSEC baselines with your split data.

Examples:
    # 1. Export 80/10/10 split to local WAV and CSVs:
    python run_visec_baseline.py --mode export --output_dir visec_dataset

    # 2. Train Pitch-Fusion Model (ICASSP 2024):
    python run_visec_baseline.py --mode train_pitch --batch_size 4 --epochs 30

    # 3. Train Baseline Wav2Vec 2.0 (No Pitch):
    python run_visec_baseline.py --mode train_no_joint --batch_size 4 --epochs 30
"""

import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ViSEC Baselines")
    parser.add_argument("--mode", type=str, default="train_pitch", choices=["train_pitch", "train_no_joint", "export"],
                        help="Mode: train_pitch | train_no_joint | export")
    
    # Forward remaining arguments
    args, unknown_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown_args
    
    if args.mode == "export":
        from visec_baseline.prepare_visec_split import export_visec_split
        import argparse as ap
        p = ap.ArgumentParser()
        p.add_argument("--output_dir", type=str, default="visec_dataset")
        p.add_argument("--hf_id", type=str, default="hustep-lab/ViSEC")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--target_sr", type=int, default=16000)
        p_args = p.parse_args(unknown_args)
        export_visec_split(p_args.output_dir, p_args.hf_id, p_args.seed, p_args.target_sr)
        
    elif args.mode == "train_pitch":
        from visec_baseline.train_interpolated_pitch import main
        main()
        
    elif args.mode == "train_no_joint":
        from visec_baseline.train_no_joint import main
        main()
