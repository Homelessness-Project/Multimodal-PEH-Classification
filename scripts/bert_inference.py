import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from utils import (
    COMMENT_TYPES,
    CRITIQUE_CATEGORIES,
    RESPONSE_CATEGORIES,
    PERCEPTION_TYPES
)

ALL_LABELS = COMMENT_TYPES + CRITIQUE_CATEGORIES + RESPONSE_CATEGORIES + PERCEPTION_TYPES + ["racist"]


def load_model_and_tokenizer(model_path):
    """Load the finetuned BERT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def predict_single_text(model, tokenizer, text, max_length=128):
    """Predict labels for a single text."""
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).int().squeeze().numpy()
    
    return predictions, probabilities.squeeze().numpy()


def create_output_row(comment, city, predictions, probabilities, raw_response=""):
    """Create a standardized output row with all fields and flags."""
    # Create binary predictions for each category
    pred_dict = dict(zip(ALL_LABELS, predictions))
    prob_dict = dict(zip(ALL_LABELS, probabilities))
    
    # Create text outputs for each category type
    comment_types = [label for label in ALL_LABELS if label in COMMENT_TYPES and pred_dict[label] == 1]
    critique_categories = [label for label in ALL_LABELS if label in CRITIQUE_CATEGORIES and pred_dict[label] == 1]
    response_categories = [label for label in ALL_LABELS if label in RESPONSE_CATEGORIES and pred_dict[label] == 1]
    perception_types = [label for label in ALL_LABELS if label in PERCEPTION_TYPES and pred_dict[label] == 1]
    
    output_row = {
        "Comment": comment,
        "City": city,
        "Comment Type": ", ".join(comment_types) if comment_types else "none",
        "Critique Category": ", ".join(critique_categories) if critique_categories else "none",
        "Response Category": ", ".join(response_categories) if response_categories else "none",
        "Perception Type": ", ".join(perception_types) if perception_types else "none",
        "racist": "Yes" if pred_dict["racist"] == 1 else "No",
        "Reasoning": "BERT multi-label classification",
        "Raw Response": raw_response
    }
    
    # Add all flag columns
    for category, flags in [
        ("Comment", {label: pred_dict[label] for label in COMMENT_TYPES}),
        ("Critique", {label: pred_dict[label] for label in CRITIQUE_CATEGORIES}),
        ("Response", {label: pred_dict[label] for label in RESPONSE_CATEGORIES}),
        ("Perception", {label: pred_dict[label] for label in PERCEPTION_TYPES})
    ]:
        for flag, value in flags.items():
            output_row[f"{category}_{flag}"] = value
    
    # Add racist flag
    output_row["Racist_Flag"] = pred_dict["racist"]
    
    # Add probability scores
    for label in ALL_LABELS:
        output_row[f"prob_{label}"] = prob_dict[label]
    
    return output_row


def main():
    parser = argparse.ArgumentParser(description="Run inference with finetuned BERT model.")
    parser.add_argument('--model_path', type=str, default='finetuned_bert', help='Path to finetuned BERT model')
    parser.add_argument('--input_file', type=str, default='output/gold_subset_reddit_comments_by_city_deidentified.csv', help='Input CSV file with comments')
    parser.add_argument('--output_file', type=str, default='output/classified_comments_reddit_gold_subset_bert.csv', help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    args = parser.parse_args()

    print(f"\n=== Loading finetuned BERT model from {args.model_path} ===")
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"\n=== Loading data from {args.input_file} ===")
    try:
        df = pd.read_csv(args.input_file)
        print(f"Loaded {len(df)} comments")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\n=== Running inference ===")
    output_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing comments"):
        comment = row["Comment"]
        city = row["City"]
        
        try:
            predictions, probabilities = predict_single_text(model, tokenizer, comment)
            
            # Create raw response string with probabilities
            raw_response = "BERT Predictions:\n"
            for i, label in enumerate(ALL_LABELS):
                pred = "Yes" if predictions[i] == 1 else "No"
                prob = probabilities[i]
                raw_response += f"{label}: {pred} (prob: {prob:.3f})\n"
            
            output_row = create_output_row(
                comment=comment,
                city=city,
                predictions=predictions,
                probabilities=probabilities,
                raw_response=raw_response
            )
            output_data.append(output_row)
            
        except Exception as e:
            print(f"Error processing comment {idx}: {e}")
            continue

    print(f"\n=== Saving results to {args.output_file} ===")
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.output_file, index=False)
    print(f"Saved {len(output_data)} predictions")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for label in ALL_LABELS:
        positive_count = sum(1 for row in output_data if row[f"Comment_{label}" if label in COMMENT_TYPES else 
                           f"Critique_{label}" if label in CRITIQUE_CATEGORIES else
                           f"Response_{label}" if label in RESPONSE_CATEGORIES else
                           f"Perception_{label}" if label in PERCEPTION_TYPES else
                           f"Racist_Flag" if label == "racist" else f"Comment_{label}"] == 1)
        print(f"{label}: {positive_count}/{len(output_data)} ({positive_count/len(output_data)*100:.1f}%)")


if __name__ == "__main__":
    main() 