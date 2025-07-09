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
    
    # Get list of cities from the data directory
    data_dir = "data"
    cities = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for city in cities:
        print(f"\nProcessing {city}...")
        
        # Process both all_comments and filtered_comments
        for comment_type in ['all_comments', 'filtered_comments']:
            input_file = f"data/{city}/reddit/{comment_type}.csv"
            
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
            
            # Create new dataframe with specified columns
            output_df = pd.DataFrame({
                'Deidentified Submission Title': df['Deidentified_Submission_Title'],
                'Submission Score': df['Submission Score'],
                'Deidentified Comment': df['Deidentified_Comment'],
                'Comment Score': df['Comment Score']
            })
            
            # Save deidentified data
            output_df.to_csv(output_file, index=False)
            print(f"Saved deidentified data to {output_file}")

if __name__ == "__main__":
    main() 