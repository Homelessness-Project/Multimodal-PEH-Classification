import os
import glob
import pandas as pd
from utils import CITY_MAP

def sample_twitter_posts(base_data_dir='data', sample_size=50, output_file='gold_standard/sampled_twitter_posts.csv'):
    all_samples = []
    for city, city_dir in CITY_MAP.items():
        x_dir = os.path.join(base_data_dir, city_dir, 'x')
        posts_path = os.path.join(x_dir, 'posts_english_2015-2025_rt_deidentified.csv')
        if not os.path.isfile(posts_path):
            print(f"File not found: {posts_path}")
            continue
        try:
            df = pd.read_csv(posts_path)
            if 'is_retweet' not in df.columns:
                print(f"'is_retweet' column not found in {posts_path}. Skipping.")
                continue
            non_rt = df[df['is_retweet'] == False]
            if non_rt.empty:
                print(f"No non-retweet tweets in {posts_path}.")
                continue
            sample = non_rt.sample(n=min(sample_size, len(non_rt)), random_state=42)
            sample['city'] = city
            all_samples.append(sample)
        except Exception as e:
            print(f"Error processing {posts_path}: {e}")
    if all_samples:
        combined = pd.concat(all_samples, ignore_index=True)
        combined.to_csv(output_file, index=False)
        print(f"Combined sample saved to {output_file}")
    else:
        print("No samples collected.")

if __name__ == '__main__':
    sample_twitter_posts() 