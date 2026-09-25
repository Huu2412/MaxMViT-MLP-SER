"""
train_interpolated_pitch.py — Training & Evaluation Script for ViSEC Pitch-Fusion Model
Supports both local CSVs (visec_dataset/) and direct Hugging Face Hub (hustep-lab/ViSEC).
"""

import os
import sys
import json
import argparse
import warnings
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import Trainer, TrainingArguments, Wav2Vec2Processor

# Local module imports
from .ser_pitch_model import Wav2Vec2CrossAttentionPitchForSER
from .dataset import load_visec_datasets, ViSECPitchDataCollator, TARGET_CLASSES

warnings.filterwarnings('ignore')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_acc = balanced_accuracy_score(labels, preds)
    acc = accuracy_score(labels, preds)
    
    return {
        'macro_f1': macro_f1,
        'weighted_acc': weighted_acc,
        'accuracy': acc
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViSEC Pitch-Fusion Model")
    parser.add_argument("--hf_id", type=str, default="hustep-lab/ViSEC", help="Hugging Face Dataset ID")
    parser.add_argument("--csv_dir", type=str, default=None, help="Directory containing train.csv, valid.csv, test.csv")
    parser.add_argument("--pretrained_model", type=str, default="nguyenvulebinh/wav2vec2-base-vi", help="Pretrained Wav2Vec2 model")
    parser.add_argument("--processor_name", type=str, default="facebook/wav2vec2-base-100h", help="Processor / Feature Extractor name")
    parser.add_argument("--output_dir", type=str, default="checkpoints/visec_pitch_baseline", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.5e-5, help="Learning rate")
    parser.add_argument("--num_proc", type=int, default=4, help="Preprocessing worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split and training")
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(), help="Enable FP16 mixed precision")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(f"  ViSEC Pitch-Fusion Model Training (ICASSP 2024 Baseline)")
    print(f"  Device:       {device.upper()} (FP16: {args.fp16})")
    print(f"  Pretrained:   {args.pretrained_model}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Batch size:   {args.batch_size} (Grad Accum: {args.grad_accum} -> Effective BS: {args.batch_size * args.grad_accum})")
    print("=" * 65)

    # 1. Load Datasets
    train_ds, val_ds, test_ds = load_visec_datasets(
        hf_id=args.hf_id,
        csv_dir=args.csv_dir,
        split_ratio=(0.8, 0.1, 0.1),
        seed=args.seed,
        num_proc=args.num_proc
    )
    
    # 2. Processor & Data Collator
    print(f"[Init] Loading processor '{args.processor_name}'...")
    processor = Wav2Vec2Processor.from_pretrained(args.processor_name)
    data_collator = ViSECPitchDataCollator(processor=processor)

    # 3. Model Initialization
    print(f"[Init] Loading model from '{args.pretrained_model}'...")
    model = Wav2Vec2CrossAttentionPitchForSER.from_pretrained(
        args.pretrained_model,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.1,
        final_dropout=0.1,
        mask_time_prob=0.05,
        layerdrop=0.1,
        num_labels=len(TARGET_CLASSES),
        classifier_proj_size=256
    )
    model.freeze_feature_extractor()
    print("[Init] Feature extractor frozen successfully.")

    # 4. Training Arguments with backward/forward compatibility
    eval_arg_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    
    training_kwargs = {
        "output_dir": args.output_dir,
        "group_by_length": False,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        eval_arg_name: "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "num_train_epochs": args.epochs,
        "dataloader_num_workers": min(args.num_proc, 4),
        "learning_rate": args.lr,
        "warmup_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": args.fp16,
        "report_to": "none",
        "seed": args.seed
    }
    training_args = TrainingArguments(**training_kwargs)

    # 5. Trainer Initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    # 6. Train Model
    print("\n[Training] Starting Pitch-Fusion training...")
    train_result = trainer.train()
    print("[Training] Training completed!")
    
    # Save best model
    best_model_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_path)
    print(f"[Model] Saved best model to: {best_model_path}")

    # 7. Final Evaluation on Independent Test Set
    print("\n" + "=" * 65)
    print("  Evaluating Best Model on Independent Test Set...")
    print("=" * 65)
    test_predictions = trainer.predict(test_ds)
    test_logits = test_predictions.predictions
    test_labels = test_predictions.label_ids
    test_preds = np.argmax(test_logits, axis=-1)

    macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    weighted_acc = balanced_accuracy_score(test_labels, test_preds)
    acc = accuracy_score(test_labels, test_preds)

    print(f"\n[Test Metrics]")
    print(f"  Macro-F1 Score:    {macro_f1 * 100:.2f}%")
    print(f"  Balanced Accuracy: {weighted_acc * 100:.2f}%")
    print(f"  Overall Accuracy:  {acc * 100:.2f}%")

    # Classification Report
    report = classification_report(test_labels, test_preds, target_names=TARGET_CLASSES, digits=4)
    print("\n[Detailed Classification Report]")
    print(report)

    # Save metrics to JSON & TXT
    results = {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(weighted_acc),
        "accuracy": float(acc),
        "classification_report": report
    }
    with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # 8. Confusion Matrix Visualization
    cm = confusion_matrix(test_labels, test_preds, labels=range(len(TARGET_CLASSES)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.title(f"ViSEC Pitch-Fusion Baseline (Test Macro-F1: {macro_f1*100:.2f}%)")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Results] Confusion Matrix saved to: {cm_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
