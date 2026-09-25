"""
train_no_joint.py — Training & Evaluation for Baseline Wav2Vec 2.0 (without Pitch Fusion)
"""

import os
import json
import argparse
import warnings
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from .dataset import load_visec_datasets, TARGET_CLASSES

warnings.filterwarnings('ignore')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'weighted_acc': balanced_accuracy_score(labels, preds),
        'accuracy': accuracy_score(labels, preds)
    }


class DataCollatorWav2Vec2:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, examples):
        input_features = [{"input_values": example["audio_input"]} for example in examples]
        batch = self.feature_extractor.pad(input_features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([example["label"] for example in examples], dtype=torch.long)
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline Wav2Vec 2.0 (No Pitch)")
    parser.add_argument("--hf_id", type=str, default="hustep-lab/ViSEC", help="Hugging Face Dataset ID")
    parser.add_argument("--csv_dir", type=str, default=None, help="Directory containing train.csv, valid.csv, test.csv")
    parser.add_argument("--pretrained_model", type=str, default="nguyenvulebinh/wav2vec2-base-vi")
    parser.add_argument("--output_dir", type=str, default="checkpoints/visec_no_joint_baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(f"  ViSEC Baseline Wav2Vec 2.0 Training (No Pitch)")
    print(f"  Device:       {device.upper()} (FP16: {args.fp16})")
    print(f"  Pretrained:   {args.pretrained_model}")
    print(f"  Output Dir:   {args.output_dir}")
    print("=" * 65)

    train_ds, val_ds, test_ds = load_visec_datasets(
        hf_id=args.hf_id,
        csv_dir=args.csv_dir,
        split_ratio=(0.8, 0.1, 0.1),
        seed=args.seed,
        num_proc=args.num_proc
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        padding_side='right', do_normalize=True, return_attention_mask=False
    )
    data_collator = DataCollatorWav2Vec2(feature_extractor=feature_extractor)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
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

    eval_arg_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        **{eval_arg_name: "epoch"},
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=args.epochs,
        dataloader_num_workers=min(args.num_proc, 4),
        learning_rate=args.lr,
        warmup_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    print("\n[Training] Starting Wav2Vec 2.0 training...")
    trainer.train()

    best_model_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_path)
    print(f"[Model] Saved best model to: {best_model_path}")

    # Evaluate on Test Set
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

    report = classification_report(test_labels, test_preds, target_names=TARGET_CLASSES, digits=4)
    print("\n[Detailed Classification Report]")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=range(len(TARGET_CLASSES)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.title(f"ViSEC Wav2Vec 2.0 Baseline (Test Macro-F1: {macro_f1*100:.2f}%)")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Results] Confusion Matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
