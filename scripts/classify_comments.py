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

def main():
    parser = argparse.ArgumentParser(description="Classify Reddit comments using Llama or Qwen.")
    parser.add_argument('--model', type=str, default='qwen', choices=['llama', 'qwen', 'gemma3', 'phi4'], help='Model to use (llama, qwen, gemma3, or phi4)')
    parser.add_argument('--input', type=str, default=None, help='Input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (optional)')
    parser.add_argument('--few_shot', type=str, default=None, choices=['reddit', 'x', 'news_articles', 'meeting_minutes'], help='Append few-shot examples for the specified platform (if available)')
    parser.add_argument('--source', type=str, required=True, choices=['reddit', 'x', 'news_articles', 'meeting_minutes'], help='Specify the data source (required)')
    parser.add_argument('--dataset', type=str, required=True, choices=['all', 'gold_subset'], help='Specify which dataset to use (required)')
    args = parser.parse_args()

    model_config = get_model_config(args.model)
    model_id = model_config["model_id"]

    # Load model and tokenizer
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
            args.input = 'output/gold_subset_reddit_comments_by_city_deidentified.csv'  # gold subset for LLM runs
        elif args.source == 'reddit' and args.dataset == 'all':
            args.input = 'data/reddit/all_comments.csv'
        # Add similar logic for other sources as needed
        # e.g., elif args.source == 'x' and args.dataset == 'gold_subset': ...

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
                comment_text = extract_field(output, "Comment Type")
                critique_text = extract_field(output, "Critique Category")
                response_text = extract_field(output, "Response Category")
                perception_text = extract_field(output, "Perception Type")
                racist_text = extract_field(output, "racist")
                racist_flag = 0
                if racist_text:
                    racist_text = racist_text.lower().strip()
                    if racist_text in ["yes", "true", "1"]:
                        racist_flag = 1
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
            except Exception as e:
                print(f"Error processing comment: {comment[:100]}...")
                print(f"Error: {e}")
                continue
        if output_data:
            # Compose output filename
            out_path = args.output
            if not out_path:
                out_path = f"output/classified_comments_{args.source}_{args.dataset}_{args.model}"
                if args.few_shot:
                    out_path += "_fewshot"
                out_path += ".csv"
            output_df = pd.DataFrame(output_data)
            output_df.to_csv(out_path, index=False)
            print(f"Saved {len(output_data)} processed comments")
    print(f"\nDone! Final output saved to {out_path}")

if __name__ == "__main__":
    main() 