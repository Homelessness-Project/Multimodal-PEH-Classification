import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import Dataset
import os
import sys
import datetime
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
    parser = argparse.ArgumentParser(description="Classify content about homelessness using various LLMs with support for different content types.")
    parser.add_argument('--model', type=str, default='qwen', choices=['llama', 'qwen', 'gemma3', 'phi4'], help='Model to use (llama, qwen, gemma3, or phi4)')
    parser.add_argument('--input', type=str, default=None, help='Input CSV file (for --process_raw_only, this should be the raw CSV)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (optional, for processed/flags CSV)')
    parser.add_argument('--few_shot', type=str, default='none', choices=['reddit', 'x', 'news', 'meeting_minutes', 'none'], help='Append few-shot examples for the specified platform (if available). Use "none" for zero-shot (default).')
    parser.add_argument('--source', type=str, required=True, choices=['reddit', 'x', 'news', 'meeting_minutes'], help='Specify the data source (required)')
    parser.add_argument('--dataset', type=str, required=True, choices=['all', 'gold_subset'], help='Specify which dataset to use (required)')
    parser.add_argument('--process_raw_only', action='store_true', help='Only process an existing raw CSV to produce the one-hot encoded CSV (no LLM inference)')
    args = parser.parse_args()

    # Setup logging if running with nohup
    if not sys.stdout.isatty():  # Check if running in background (nohup)
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/{args.source}_{args.dataset}_{args.model}_{args.few_shot}_{timestamp}.log"
        
        # Redirect stdout and stderr to log file
        sys.stdout = open(log_filename, 'w')
        sys.stderr = sys.stdout
        
        print(f"Logging to: {log_filename}")
        print(f"Started at: {datetime.datetime.now()}")
        print(f"Command: {' '.join(sys.argv)}")
        print("-" * 80)

    # Compose output paths
    if args.output:
        flags_csv_path = args.output
    else:
        base_name = f"classified_comments_{args.source}_{args.dataset}_{args.model}_{args.few_shot}"
        flags_csv_path = os.path.join('output', args.source, args.model, base_name + '_flags.csv')
    raw_csv_path = os.path.join('output', args.source, args.model, f"classified_comments_{args.source}_{args.dataset}_{args.model}_{args.few_shot}_raw.csv")
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
            args.input = 'gold_standard/gold_subset_reddit_comments_by_city_deidentified.csv'
        elif args.source == 'reddit' and args.dataset == 'all':
            args.input = 'data/reddit/all_comments.csv'
        elif args.source == 'x' and args.dataset == 'gold_subset':
            args.input = 'output/gold_subset_x_posts_by_city_deidentified.csv'
        elif args.source == 'x' and args.dataset == 'all':
            args.input = 'data/x/all_posts.csv'
        elif args.source == 'news' and args.dataset == 'gold_subset':
            args.input = 'output/gold_subset_news_articles_by_city_deidentified.csv'
        elif args.source == 'news' and args.dataset == 'all':
            args.input = 'data/news/all_articles.csv'
        elif args.source == 'meeting_minutes' and args.dataset == 'gold_subset':
            args.input = 'output/gold_subset_meeting_minutes_by_city_deidentified.csv'
        elif args.source == 'meeting_minutes' and args.dataset == 'all':
            args.input = 'data/meeting_minutes/all_minutes.csv'
        else:
            print(f"Warning: No default input file found for source '{args.source}' and dataset '{args.dataset}'")
            print("Please specify --input file path")
            exit(1)

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

    # Validate content_type early
    try:
        # Test the content_type validation
        test_prompt = create_classification_prompt("test", content_type=args.source)
        print(f"Using content type: {args.source}")
        if args.few_shot == 'none':
            print("Using zero-shot classification (no examples)")
        else:
            print(f"Using few-shot examples for: {args.few_shot}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Check if we're resuming from a previous run
    if os.path.exists(raw_csv_path):
        existing_df = pd.read_csv(raw_csv_path)
        processed_count = len(existing_df)
        print(f"Found existing raw output with {processed_count} processed comments")
        if processed_count >= len(df):
            print("All comments already processed. Processing raw to flags...")
            process_raw_to_flags(raw_csv_path, flags_csv_path)
            print(f"Done! Final output saved to {flags_csv_path}")
            return
        else:
            print(f"Resuming from comment {processed_count}")
            output_data = existing_df.to_dict('records')
            start_batch = processed_count // BATCH_SIZE
    else:
        start_batch = 0

    for batch_idx in range(start_batch, total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
        batch_data = {
            "Comment": batch_df["Comment"].tolist(),
            "City": batch_df["City"].tolist()
        }
        batch_dataset = Dataset.from_dict(batch_data)
        batch_outputs = []
        
        for item in tqdm(batch_dataset, total=len(batch_dataset)):
            comment = item["Comment"]
            city = item["City"]
            try:
                # Compose prompt with content_type and few-shot examples
                prompt = create_classification_prompt(
                    comment, 
                    content_type=args.source, 
                    few_shot_text=args.few_shot
                )
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
                # Save to batch outputs
                batch_outputs.append({
                    "Comment": comment,
                    "City": city,
                    "Raw Response": output
                })
            except Exception as e:
                print(f"Error processing comment: {comment[:100]}...")
                print(f"Error: {e}")
                continue
        
        # Add batch outputs to main data
        output_data.extend(batch_outputs)
        
        # Save progress after each batch
        temp_df = pd.DataFrame(output_data)
        temp_df.to_csv(raw_csv_path, index=False)
        print(f"Saved batch {batch_idx + 1} progress: {len(output_data)}/{len(df)} comments processed")
        
        # Also save flags after each batch
        process_raw_to_flags(raw_csv_path, flags_csv_path)
        print(f"Updated flags CSV with {len(output_data)} processed comments")
    
    print(f"Completed! Final output saved to {flags_csv_path}")
    
    # Log completion if using nohup
    if not sys.stdout.isatty():
        print("-" * 80)
        print(f"Completed at: {datetime.datetime.now()}")
        print(f"Results saved to: {flags_csv_path}")
        sys.stdout.close()

if __name__ == "__main__":
    """
    Examples of how to use this script:
    
    # Basic usage - zero-shot classification (default)
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model qwen
    
    # Zero-shot classification (explicit)
    python scripts/classify_comments.py --source x --dataset gold_subset --model llama --few_shot none
    
    # Few-shot classification with automatic examples for the same content type
    python scripts/classify_comments.py --source news --dataset all --model qwen --few_shot news
    
    # Few-shot classification with examples from different content type
    python scripts/classify_comments.py --source meeting_minutes --dataset gold_subset --model gemma3 --few_shot reddit
    
    # Process existing raw CSV to generate flags CSV (no LLM inference)
    python scripts/classify_comments.py --source reddit --dataset gold_subset --process_raw_only --input output/reddit/qwen/classified_comments_reddit_gold_subset_qwen_raw.csv
    
    # Specify custom input and output files
    python scripts/classify_comments.py --source x --dataset all --model llama --input data/custom_x_posts.csv --output results/x_classified.csv
    
    # Different models available
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model llama    # Llama 3.2 3B
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model qwen     # Qwen 2.5 7B
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model gemma3   # Gemma 3 4B
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model phi4     # Phi-4 mini
    
    # Content types supported
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model qwen      # Reddit comments
    python scripts/classify_comments.py --source x --dataset gold_subset --model qwen          # X (Twitter) posts
    python scripts/classify_comments.py --source news --dataset gold_subset --model qwen       # News articles
    python scripts/classify_comments.py --source meeting_minutes --dataset gold_subset --model qwen  # Meeting minutes
    
    # Few-shot options
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model qwen --few_shot none        # Zero-shot
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model qwen --few_shot reddit      # Reddit examples
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model qwen --few_shot x           # X examples
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model qwen --few_shot news        # News examples
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model qwen --few_shot meeting_minutes  # Meeting minutes examples
    
    # Datasets available
    python scripts/classify_comments.py --source reddit --dataset gold_subset --model qwen     # Gold standard subset
    python scripts/classify_comments.py --source reddit --dataset all --model qwen             # All available data
    
    Note:
    - Default behavior is zero-shot classification (--few_shot none)
    - Content type is automatically determined from --source
    - Few-shot examples are automatically selected based on content type
    - Output files are automatically named based on source, dataset, and model
    - Use --process_raw_only to convert existing raw CSV to flags CSV without LLM inference
    """
    main() 