import os
import pandas as pd
from utils import CITY_MAP

def sample_lexisnexis_news(base_data_dir='data', sample_size=50, output_file='gold_standard/sampled_lexisnexis_news.csv'):
    all_samples = []
    for city, city_dir in CITY_MAP.items():
        news_dir = os.path.join(base_data_dir, city_dir, 'newspaper')
        news_path = os.path.join(news_dir, f'{city_dir}_filtered_deidentified.csv')
        if not os.path.isfile(news_path):
            print(f"File not found: {news_path}")
            continue
        try:
            df = pd.read_csv(news_path)
            if df.empty:
                print(f"No rows in {news_path}.")
                continue
            sample = df.sample(n=min(sample_size, len(df)), random_state=42)
            sample['city'] = city
            all_samples.append(sample)
        except Exception as e:
            print(f"Error processing {news_path}: {e}")
    if all_samples:
        combined = pd.concat(all_samples, ignore_index=True)
        combined.to_csv(output_file, index=False)
        print(f"Combined sample saved to {output_file}")
    else:
        print("No samples collected.")

if __name__ == '__main__':
    sample_lexisnexis_news() 