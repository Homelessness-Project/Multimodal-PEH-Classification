import pandas as pd
import os

cities = [
    "atlanticcity", "baltimore", "elpaso",
    "kzoo", "portland", "sanfrancisco", "southbend"
]

base_path = "data"
sampled_frames = []

for city in cities:
    file_path = os.path.join(base_path, city, "reddit/filtered_comments.csv")
    df = pd.read_csv(file_path)

    # Optional: convert timestamp for future filtering/cleaning
    df["Comment Timestamp"] = pd.to_datetime(df["Comment Timestamp"], errors='coerce')

    # Drop rows with invalid timestamps
    df = df.dropna(subset=["Comment Timestamp"])

    # Remove [removed] and [deleted] comments
    df = df[~df["Comment"].isin(["[removed]", "[deleted]"])]

    # Random sample up to 50 comments
    if len(df) >= 50:
        city_sample = df.sample(n=50, random_state=42)
    else:
        print(f"Warning: {city} only has {len(df)} rows. Using all available.")
        city_sample = df.copy()

    city_sample["City"] = city
    sampled_frames.append(city_sample)

# Combine all city samples
combined_df = pd.concat(sampled_frames, ignore_index=True)

# Save to output CSV
combined_df.to_csv("output/sampled_reddit_comments_by_city.csv", index=False)
