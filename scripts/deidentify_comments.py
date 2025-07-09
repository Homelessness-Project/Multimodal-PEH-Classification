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
            output_file = input_file.replace('.csv', '_deidentified.csv')
            
            # Deidentify both Submission Title and Comment columns
            print("Deidentifying submission titles and comments...")
            df['Deidentified_Submission_Title'] = [deidentify_text(title, nlp) for title in tqdm(df['Submission Title'])]
            df['Deidentified_Comment'] = [deidentify_text(comment, nlp) for comment in tqdm(df['Comment'])]
            
            # Create new dataframe with original column order but deidentified text
            output_df = pd.DataFrame({
                'Submission Title': df['Deidentified_Submission_Title'],
                'Submission Score': df['Submission Score'],
                'Comment': df['Deidentified_Comment'],
                'Comment Score': df['Comment Score'],
                'City': df['City']
            })
            
            # Save deidentified data
            output_df.to_csv(output_file, index=False)
            print(f"Saved deidentified data to {output_file}")
    except Exception as e:
        print(f"Error processing files: {e}")
        exit(1)

if __name__ == "__main__":
    main() 