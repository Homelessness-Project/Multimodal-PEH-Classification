import argparse
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import os
from tqdm import tqdm
from utils import (
    COMMENT_TYPES,
    CRITIQUE_CATEGORIES,
    RESPONSE_CATEGORIES,
    PERCEPTION_TYPES
)

ALL_LABELS = COMMENT_TYPES + CRITIQUE_CATEGORIES + RESPONSE_CATEGORIES + PERCEPTION_TYPES + ["racist"]


def load_data(text_file, label_file):
    # Load Reddit comments (text)
    df_text = pd.read_csv(text_file)
    # Load soft labels (multi-label targets)
    df_labels = pd.read_csv(label_file)
    # Align lengths
    assert len(df_text) == len(df_labels), "Text and label files must have the same number of rows."
    # Use 'Comment' as the text field
    texts = df_text["Comment"].astype(str).tolist()
    # Use all columns in label file as labels
    labels = df_labels[ALL_LABELS].values.astype(np.float32)
    return texts, labels


def create_dataset(texts, labels):
    # For HuggingFace Datasets, labels should be a list of dicts or arrays
    return Dataset.from_dict({
        "text": texts,
        "labels": labels.tolist()
    })


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    metrics = {}
    for i, name in enumerate(ALL_LABELS):
        metrics[f"f1_{name}"] = f1_score(labels[:, i], preds[:, i], zero_division=0)
        metrics[f"acc_{name}"] = accuracy_score(labels[:, i], preds[:, i])
        metrics[f"prec_{name}"] = precision_score(labels[:, i], preds[:, i], zero_division=0)
        metrics[f"rec_{name}"] = recall_score(labels[:, i], preds[:, i], zero_division=0)
    metrics["mean_f1"] = np.mean([metrics[f"f1_{name}"] for name in ALL_LABELS])
    metrics["mean_acc"] = np.mean([metrics[f"acc_{name}"] for name in ALL_LABELS])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Finetune BERT for multi-label Reddit comment classification.")
    parser.add_argument('--text_file', type=str, default='output/gold_subset_reddit_comments_by_city_deidentified.csv', help='CSV file with Reddit comments')
    parser.add_argument('--label_file', type=str, default='output/annotation/reddit_soft_labels.csv', help='CSV file with soft labels')
    parser.add_argument('--output_dir', type=str, default='finetuned_bert', help='Directory to save model and tokenizer')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--max_length', type=int, default=128, help='Max token length')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    print("\n=== Loading data ===")
    texts, labels = load_data(args.text_file, args.label_file)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    train_dataset = create_dataset(train_texts, train_labels)
    val_dataset = create_dataset(val_texts, val_labels)

    print("\n=== Loading tokenizer and model ===")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(ALL_LABELS),
        problem_type="multi_label_classification"
    )

    print("\n=== Tokenizing datasets ===")
    def token_map(examples):
        return tokenize_function(examples, tokenizer)
    train_dataset = train_dataset.map(token_map, batched=True)
    val_dataset = val_dataset.map(token_map, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    print("\n=== Setting up Trainer ===")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",  # Fixed deprecation warning
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="mean_f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n=== Training ===")
    trainer.train()

    print("\n=== Saving model and tokenizer ===")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main() 