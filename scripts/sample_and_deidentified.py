import pandas as pd
import re
from tqdm import tqdm
import spacy
import os
from utils import load_spacy_model, deidentify_text

def main():
    # Load spaCy model
    print("Loading spaCy model...")
    nlp = load_spacy_model()
    
    # Load the classified comments
    try:
        input_files = [
            "output/sampled_reddit_comments_by_city.csv"
        ]
        
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"Skipping {input_file} - file not found")
                continue
                
            print(f"\nProcessing {input_file}...")
            df = pd.read_csv(input_file)
            
            # Create output filename
            output_file = input_file.replace('.csv', '_identified_and_deidentified.csv')
            
            # Deidentify comments
            print("Deidentifying comments...")
            df['Deidentified_Comment'] = [deidentify_text(comment, nlp) for comment in tqdm(df['Comment'])]
            
            # Create new dataframe with only the three columns we want
            output_df = pd.DataFrame({
                'City': df['City'],
                'Comment': df['Comment'],
                'Deidentified_Comment': df['Deidentified_Comment']
            })
            
            # Save deidentified data
            output_df.to_csv(output_file, index=False)
            print(f"Saved deidentified data to {output_file}")
    except Exception as e:
        print(f"Error processing files: {e}")
        exit(1)

if __name__ == "__main__":
    main() 