import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import Dataset
import os
import re
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

# Load data
try:
    df = pd.read_csv("output/sampled_reddit_comments_by_city_deidentified.csv")
    # Limit to first 20 comments for testing
    #df = df.head(20)
    print(f"Loaded {len(df)} comments to process (testing mode)")
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
        "City": batch_df["City"].tolist()
    }
    batch_dataset = Dataset.from_dict(batch_data)
    
    # Process each row in the batch
    for item in tqdm(batch_dataset, total=len(batch_dataset)):
        comment = item["Comment"]
        city = item["City"]
        
        try:
            # Create classification prompt
            prompt = create_classification_prompt(comment)
            
            # Generate classification
            output = pipe(
                prompt,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )[0]['generated_text']
            
            # Extract only the analysis part after "Analysis:"
            analysis_start = output.find("Analysis:")
            if analysis_start != -1:
                output = output[analysis_start + len("Analysis:"):].strip()
            
            # Extract each field from the response
            comment_text = extract_field(output, "Comment Type")
            critique_text = extract_field(output, "Critique Category")
            response_text = extract_field(output, "Response Category")
            perception_text = extract_field(output, "Perception Type")
            
            racist_text = extract_field(output, "racist")
            # More strict racist flag extraction
            racist_flag = 0
            if racist_text:
                racist_text = racist_text.lower().strip()
                if racist_text in ["yes", "true", "1"]:
                    racist_flag = 1
            
            reasoning = extract_field(output, "Reasoning")
            if not reasoning:
                reasoning = "No reasoning provided."
            
            # Create output row using utility function
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
    
    # Save intermediate results after each batch
    if output_data:
        output_df = pd.DataFrame(output_data)
        output_df.to_csv("output/classified_comments_qwen.csv", index=False)
        print(f"Saved {len(output_data)} processed comments")

print("\nDone! Final output saved to output/classified_comments_qwen.csv") 