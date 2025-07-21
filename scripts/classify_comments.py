import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import Dataset
import os
from utils import (
    get_model_config, 
    create_classification_prompt, 
    extract_field,
    COMMENT_TYPES,
    CRITIQUE_CATEGORIES,
    RESPONSE_CATEGORIES,
    PERCEPTION_TYPES,
    create_output_row
)
import ast

def parse_bracketed_list(field):
    if not field or field.strip() in ["[]", "", "none", "n/a", "-", "no categories", "none applicable"]:
        return []
    field = field.strip()
    if field.startswith('[') and field.endswith(']'):
        field = field[1:-1]
    items = [v.strip() for v in field.split(',') if v.strip()]
    return items

def parse_bracketed_single_value(field):
    if not field or field.strip() in ["[]", "", "none", "n/a", "-", "no categories", "none applicable"]:
        return ""
    field = field.strip()
    if field.startswith('[') and field.endswith(']'):
        field = field[1:-1]
    items = [v.strip() for v in field.split(',') if v.strip()]
    return items[0] if items else ""

def process_raw_to_flags(raw_csv_path, flags_csv_path):
    df = pd.read_csv(raw_csv_path)
    output_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        comment = row['Comment']
        city = row['City']
        output = row['Raw Response']
        comment_text = ", ".join(parse_bracketed_list(extract_field(output, "Comment Type")))
        critique_text = ", ".join(parse_bracketed_list(extract_field(output, "Critique Category")))
        response_text = ", ".join(parse_bracketed_list(extract_field(output, "Response Category")))
        perception_text = ", ".join(parse_bracketed_list(extract_field(output, "Perception Type")))
        racist_text = extract_field(output, "racist")
        racist_value = parse_bracketed_single_value(racist_text).lower()
        racist_flag = 1 if racist_value in ["yes", "true", "1"] else 0
        reasoning = extract_field(output, "Reasoning")
        if not reasoning:
            reasoning = "No reasoning provided."
        output_row = create_output_row(
            comment=comment,
            city=city,
            comment_text=comment_text,
            critique_text=critique_text,
            response_text=response_text,
            perception_text=perception_text,
            racist_flag=racist_flag,
            reasoning=reasoning,
            raw_response=output
        )
        output_data.append(output_row)
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(flags_csv_path, index=False)
    print(f"Saved {len(output_data)} processed comments to {flags_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Classify Reddit comments using Llama or Qwen.")
    parser.add_argument('--model', type=str, default='qwen', choices=['llama', 'qwen', 'gemma3', 'phi4'], help='Model to use (llama, qwen, gemma3, or phi4)')
    parser.add_argument('--input', type=str, default=None, help='Input CSV file (for --process_raw_only, this should be the raw CSV)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (optional, for processed/flags CSV)')
    parser.add_argument('--few_shot', type=str, default=None, choices=['reddit', 'x', 'news_articles', 'meeting_minutes'], help='Append few-shot examples for the specified platform (if available)')
    parser.add_argument('--source', type=str, required=True, choices=['reddit', 'x', 'news_articles', 'meeting_minutes'], help='Specify the data source (required)')
    parser.add_argument('--dataset', type=str, required=True, choices=['all', 'gold_subset'], help='Specify which dataset to use (required)')
    parser.add_argument('--process_raw_only', action='store_true', help='Only process an existing raw CSV to produce the one-hot encoded CSV (no LLM inference)')
    args = parser.parse_args()

    # Compose output paths
    if args.output:
        flags_csv_path = args.output
    else:
        base_name = f"classified_comments_{args.source}_{args.dataset}_{args.model}"
        flags_csv_path = os.path.join('output', args.source, args.model, base_name + '_flags.csv')
    raw_csv_path = os.path.join('output', args.source, args.model, f"classified_comments_{args.source}_{args.dataset}_{args.model}_raw.csv")
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)

    if args.process_raw_only:
        if not args.input:
            print("Error: --input (raw CSV) is required when using --process_raw_only.")
            exit(1)
        process_raw_to_flags(args.input, flags_csv_path)
        return

    # LLM inference step
    model_config = get_model_config(args.model)
    model_id = model_config["model_id"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Set default input file based on source and dataset if not provided
    if not args.input:
        if args.source == 'reddit' and args.dataset == 'gold_subset':
            args.input = 'output/gold_subset_reddit_comments_by_city_deidentified.csv'
        elif args.source == 'reddit' and args.dataset == 'all':
            args.input = 'data/reddit/all_comments.csv'
        # Add similar logic for other sources as needed

    # Load data
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} comments to process")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    output_data = []
    BATCH_SIZE = 10
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    # Few-shot prompt selection
    few_shot_text = ''
    if args.few_shot:
        try:
            if args.few_shot == 'reddit':
                from utils import FEW_SHOT_REDDIT_PROMPT_TEXT
                few_shot_text = FEW_SHOT_REDDIT_PROMPT_TEXT
            elif args.few_shot == 'x':
                from utils import FEW_SHOT_X_PROMPT_TEXT
                few_shot_text = FEW_SHOT_X_PROMPT_TEXT
            elif args.few_shot == 'news_articles':
                from utils import FEW_SHOT_NEWS_ARTICLES_PROMPT_TEXT
                few_shot_text = FEW_SHOT_NEWS_ARTICLES_PROMPT_TEXT
            elif args.few_shot == 'meeting_minutes':
                from utils import FEW_SHOT_MEETING_MINUTES_PROMPT_TEXT
                few_shot_text = FEW_SHOT_MEETING_MINUTES_PROMPT_TEXT
        except ImportError:
            print(f"Few-shot prompt for {args.few_shot} not found in utils.py. Proceeding without few-shot examples.")
            few_shot_text = ''

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
        batch_data = {
            "Comment": batch_df["Comment"].tolist(),
            "City": batch_df["City"].tolist()
        }
        batch_dataset = Dataset.from_dict(batch_data)
        for item in tqdm(batch_dataset, total=len(batch_dataset)):
            comment = item["Comment"]
            city = item["City"]
            try:
                # Compose prompt with optional few-shot examples
                if few_shot_text:
                    prompt = create_classification_prompt(comment, few_shot_text)
                else:
                    prompt = create_classification_prompt(comment)
                output = pipe(
                    prompt,
                    max_new_tokens=model_config["max_new_tokens"],
                    do_sample=True,
                    temperature=model_config["temperature"],
                    top_p=model_config["top_p"],
                    repetition_penalty=model_config["repetition_penalty"],
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                analysis_start = output.find("Analysis:")
                if analysis_start != -1:
                    output = output[analysis_start + len("Analysis:"):].strip()
                # Save only raw output
                output_data.append({
                    "Comment": comment,
                    "City": city,
                    "Raw Response": output
                })
            except Exception as e:
                print(f"Error processing comment: {comment[:100]}...")
                print(f"Error: {e}")
                continue
    # Save raw CSV
    raw_df = pd.DataFrame(output_data)
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"Saved {len(output_data)} raw LLM outputs to {raw_csv_path}")
    # Process raw to flags CSV
    process_raw_to_flags(raw_csv_path, flags_csv_path)
    print(f"Done! Final output saved to {flags_csv_path}")

if __name__ == "__main__":
    main() 