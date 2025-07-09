import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import Dataset
import os
from utils import (
    create_classification_prompt,
    create_mitigation_prompt,
    extract_mitigation_results,
    extract_classification_results,
    get_model_config,
    create_output_row
)

# Get model configuration
model_config = get_model_config("qwen")
model_id = model_config["model_id"]

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Set left padding for decoder-only architecture
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    # Set pad token for batching
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load classified data
try:
    df = pd.read_csv("output/classified_comments_qwen.csv")
    # Limit to first 20 comments for testing
    df = df.head(20)
    print(f"Loaded {len(df)} classified comments to process (testing mode)")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Output container
output_data = []

# Process in batches
BATCH_SIZE = 10  # Process 10 comments at a time
total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min((batch_idx + 1) * BATCH_SIZE, len(df))
    batch_df = df.iloc[start_idx:end_idx]
    
    print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
    
    # Convert batch to dataset for better efficiency
    batch_data = {
        "Comment": batch_df["Comment"].tolist(),
        "City": batch_df["City"].tolist(),
        "Classification": batch_df["Raw Response"].tolist()
    }
    batch_dataset = Dataset.from_dict(batch_data)
    
    # Process each row in the batch
    for item in tqdm(batch_dataset, total=len(batch_dataset)):
        comment = item["Comment"]
        city = item["City"]
        original_classification = item["Classification"]
        
        try:
            # Step 1: Mitigate based on classification
            mitigate_prompt = create_mitigation_prompt(comment, original_classification)
            mitigation_output = pipe(
                mitigate_prompt,
                max_new_tokens=model_config["max_new_tokens"],
                do_sample=True,
                temperature=model_config["temperature"],
                top_p=model_config["top_p"],
                repetition_penalty=model_config["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id
            )[0]['generated_text']
            
            # Extract new comment and reasoning
            new_comment, reasoning = extract_mitigation_results(mitigation_output)
            
            # Step 2: Reclassify the mitigated comment
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
            
            # Extract classification results
            new_classification = extract_classification_results(classification_output)
            
            # Create output row
            output_row = create_output_row(
                comment=comment,
                city=city,
                original_classification=original_classification,
                new_comment=new_comment,
                reasoning=reasoning,
                new_classification=new_classification
            )
            
            output_data.append(output_row)
            
        except Exception as e:
            print(f"Error processing comment: {comment[:100]}...")
            print(f"Error: {e}")
            continue
    
    # Save intermediate results after each batch
    if output_data:
        output_df = pd.DataFrame(output_data)
        output_df.to_csv("output/mitigated_and_reclassified_comments_qwen.csv", index=False)
        print(f"Saved {len(output_data)} processed comments")

print("\nDone! Final output saved to output/mitigated_and_reclassified_comments_qwen.csv") 