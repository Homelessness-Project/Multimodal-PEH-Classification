import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import Dataset
import os
from utils import (
    create_mitigation_prompt,
    extract_mitigation_results,
    get_model_config,
    create_mitigation_y_n_classification_prompt,
    create_recheck_prompt,
    create_classification_prompt,
    extract_field,
    extract_flags,
    COMMENT_TYPES,
    CRITIQUE_CATEGORIES,
    RESPONSE_CATEGORIES,
    PERCEPTION_TYPES
)

def main():
    parser = argparse.ArgumentParser(description="Mitigate and reclassify Reddit comments using Llama or Qwen.")
    parser.add_argument('--model', type=str, default='qwen', choices=['llama', 'qwen'], help='Model to use (llama or qwen)')
    parser.add_argument('--input', type=str, default=None, help='Input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (optional)')
    parser.add_argument('--source', type=str, required=True, choices=['reddit', 'x', 'news_articles', 'meeting_minutes'], help='Specify the data source (required)')
    parser.add_argument('--dataset', type=str, required=True, choices=['all', 'gold_subset'], help='Specify which dataset to use (required)')
    parser.add_argument('--few_shot', action='store_true', help='Use this flag if the data is a few-shot dataset')
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
                # Step 1: Detect bias
                detect_prompt = create_mitigation_y_n_classification_prompt(comment)
                bias_output = pipe(
                    detect_prompt,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=model_config["temperature"],
                    top_p=model_config["top_p"],
                    repetition_penalty=model_config["repetition_penalty"],
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                bias_result = "Yes" if "yes" in bias_output.lower() else "No"
                # Step 2: Mitigate if needed
                new_comment = comment
                reasoning = ""
                if bias_result == "Yes":
                    mitigate_prompt = create_mitigation_prompt(comment)
                    mitigation_output = pipe(
                        mitigate_prompt,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=model_config["temperature"],
                        top_p=model_config["top_p"],
                        repetition_penalty=model_config["repetition_penalty"],
                        pad_token_id=tokenizer.eos_token_id
                    )[0]['generated_text']
                    new_comment, reasoning = extract_mitigation_results(mitigation_output)
                # Step 3: Recheck mitigated comment
                recheck_prompt = create_recheck_prompt(new_comment)
                recheck_output = pipe(
                    recheck_prompt,
                    max_new_tokens=50,
                    temperature=model_config["temperature"],
                    top_p=model_config["top_p"],
                    repetition_penalty=model_config["repetition_penalty"],
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                is_still_biased = "Yes" if "yes" in recheck_output.lower() else "No"
                # Step 4: Classify the mitigated comment
                classify_prompt = create_classification_prompt(new_comment)
                classification_output = pipe(
                    classify_prompt,
                    max_new_tokens=model_config["max_new_tokens"],
                    do_sample=True,
                    temperature=model_config["temperature"],
                    top_p=model_config["top_p"],
                    repetition_penalty=model_config["repetition_penalty"],
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                analysis_start = classification_output.find("Analysis:")
                if analysis_start != -1:
                    classification_output = classification_output[analysis_start + len("Analysis:"):].strip()
                comment_type = extract_field(classification_output, "Comment Type")
                critique_category = extract_field(classification_output, "Critique Category")
                response_category = extract_field(classification_output, "Response Category")
                perception_type = extract_field(classification_output, "Perception Type")
                racist = extract_field(classification_output, "racist")
                classification_reasoning = extract_field(classification_output, "Reasoning")
                comment_flags = extract_flags(comment_type, COMMENT_TYPES)
                critique_flags = extract_flags(critique_category, CRITIQUE_CATEGORIES)
                response_flags = extract_flags(response_category, RESPONSE_CATEGORIES)
                perception_flags = extract_flags(perception_type, PERCEPTION_TYPES)
                racist_flag = 1 if racist and racist.lower() in ["yes", "true", "1"] else 0
                output_row = {
                    "Original Comment": comment,
                    "City": city,
                    "Is Biased (Original)?": bias_result,
                    "New Comment": new_comment,
                    "Mitigation Reasoning": reasoning,
                    "Is Biased (Mitigated)?": is_still_biased,
                    "Comment Type": comment_type,
                    "Critique Category": critique_category,
                    "Response Category": response_category,
                    "Perception Type": perception_type,
                    "racist": racist,
                    "Classification Reasoning": classification_reasoning,
                    "Raw Classification": classification_output
                }
                for category, flags in [
                    ("Comment", comment_flags),
                    ("Critique", critique_flags),
                    ("Response", response_flags),
                    ("Perception", perception_flags)
                ]:
                    for flag, value in flags.items():
                        output_row[f"{category}_{flag}"] = value
                output_row["Racist_Flag"] = racist_flag
                output_data.append(output_row)
            except Exception as e:
                print(f"Error processing comment: {comment[:100]}...")
                print(f"Error: {e}")
                continue
        if output_data:
            # Compose output filename
            out_path = args.output
            if not out_path:
                out_path = f"output/mitigated_comments_{args.source}_{args.dataset}_{args.model}"
                if args.few_shot:
                    out_path += "_fewshot"
                out_path += ".csv"
            output_df = pd.DataFrame(output_data)
            output_df.to_csv(out_path, index=False)
            print(f"Saved {len(output_data)} processed comments")
    print(f"\nDone! Final output saved to {out_path}")

if __name__ == "__main__":
    main() 