#!/usr/bin/env python3
"""
Comprehensive Sentiment Analysis Script

Analyzes sentiment on homelessness-related comments from 4 data sources:
- Reddit comments
- X/Twitter posts  
- News articles
- Meeting minutes

Outputs sentiment as: 1 (positive), 0 (neutral), -1 (negative)

Usage:
    # Test with example comments
    python scripts/sentiment_analysis.py --test
    
    # Run on all datasets (small sample)
    python scripts/sentiment_analysis.py --max_samples 10
    
    # Run on all datasets (full analysis)
    python scripts/sentiment_analysis.py
"""

import argparse
import pandas as pd
from tqdm import tqdm
import os
import sys
import datetime
from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyze sentiment and return 1 (positive), 0 (neutral), -1 (negative).
    
    Uses homelessness-specific keywords first, then TextBlob polarity as fallback.
    """
    if not text or pd.isna(text):
        return 0
    
    text = str(text).strip()
    if not text:
        return 0
    
    # TextBlob sentiment analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Homelessness-specific keywords
    text_lower = text.lower()
    
    positive_keywords = [
        "help", "support", "assist", "aid", "care", "compassion", "empathy",
        "solution", "housing", "shelter", "program", "initiative", "positive",
        "improve", "better", "hope", "recovery", "rehabilitation"
    ]
    
    negative_keywords = [
        "problem", "issue", "crisis", "burden", "nuisance", "annoying",
        "dangerous", "threat", "criminal", "lazy", "drug addict", "alcoholic",
        "dirty", "filthy", "disgusting", "hate", "despise", "get rid of"
    ]
    
    positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    # Determine sentiment
    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return -1
    else:
        # Use TextBlob polarity as fallback
        if polarity > 0.1:
            return 1
        elif polarity < -0.1:
            return -1
        else:
            return 0

def test_sentiment_analysis():
    """Test sentiment analysis on example comments."""
    
    # Example comments about homelessness
    test_comments = [
        "We need to help the homeless people in our community. They deserve compassion and support.",
        "These homeless people are a nuisance and should be removed from our streets.",
        "The city council is working on solutions for homelessness through housing programs.",
        "I don't care about homeless people, they should just get jobs.",
        "The shelter program is making a positive difference in people's lives.",
        "Homeless people are dangerous criminals who threaten our safety.",
        "We should provide more resources and support for people experiencing homelessness.",
        "The homeless problem is getting worse and the government isn't doing enough.",
        "I support initiatives that help homeless individuals find housing and employment.",
        "These panhandlers are annoying and should be arrested.",
        "This is just a regular comment about the weather.",
        "The city has a new park opening next week."
    ]
    
    print("Testing Sentiment Analysis (1, 0, -1)")
    print("=" * 50)
    
    for i, comment in enumerate(test_comments, 1):
        sentiment = analyze_sentiment(comment)
        
        # Convert sentiment to text for display
        sentiment_text = {1: "Positive", 0: "Neutral", -1: "Negative"}[sentiment]
        
        print(f"{i:2d}. [{sentiment:2d}] {sentiment_text}: {comment}")
    
    print("\n" + "=" * 50)
    print("Legend:")
    print("  1 = Positive")
    print("  0 = Neutral") 
    print(" -1 = Negative")

def process_dataset(source, output_dir, max_samples=None):
    """Process a single dataset and save results."""
    
    # Data file mapping
    data_files = {
        "reddit": "complete_dataset/all_reddit_comments.csv",
        "x": "complete_dataset/all_twitter_posts.csv", 
        "news": "complete_dataset/all_newspaper_articles.csv",
        "meeting_minutes": "complete_dataset/all_meeting_minutes.csv"
    }
    
    # Column mapping for each dataset
    comment_columns = {
        "reddit": "Deidentified_Comment",
        "x": "Deidentified_text", 
        "news": "Deidentified_paragraph_text",
        "meeting_minutes": "Deidentified_paragraph"
    }
    
    data_file = data_files[source]
    comment_col = comment_columns[source]
    
    print(f"Loading {source} data from {data_file}...")
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} comments")
    
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
        print(f"Limited to {len(df)} samples for testing")
    
    # Process sentiment
    print(f"Analyzing sentiment for {len(df)} comments...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        comment = row[comment_col]
        city = row.get("city", "")
        
        sentiment = analyze_sentiment(comment)
        
        results.append({
            "comment": comment,
            "city": city,
            "sentiment": sentiment
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Create output directory structure
    source_output_dir = os.path.join(output_dir, source, "nlp")
    os.makedirs(source_output_dir, exist_ok=True)
    
    # Save results without timestamp
    output_filename = "sentiment.csv"
    output_path = os.path.join(source_output_dir, output_filename)
    
    results_df.to_csv(output_path, index=False)
    print(f"Saved {source} results to {output_path}")
    
    # Print summary
    sentiment_counts = results_df['sentiment'].value_counts()
    total = len(results_df)
    
    print(f"\n{source.upper()} Sentiment Summary:")
    print(f"Positive (1): {sentiment_counts.get(1, 0)} ({sentiment_counts.get(1, 0)/total*100:.1f}%)")
    print(f"Neutral (0): {sentiment_counts.get(0, 0)} ({sentiment_counts.get(0, 0)/total*100:.1f}%)")
    print(f"Negative (-1): {sentiment_counts.get(-1, 0)} ({sentiment_counts.get(-1, 0)/total*100:.1f}%)")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Sentiment analysis on homelessness comments')
    parser.add_argument('--test', action='store_true', 
                       help='Run test with example comments')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples per dataset (for testing)')
    
    args = parser.parse_args()
    
    # Run test if requested
    if args.test:
        test_sentiment_analysis()
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all datasets
    sources = ['reddit', 'x', 'news', 'meeting_minutes']
    results = {}
    
    print("Starting sentiment analysis on all datasets...")
    print("=" * 50)
    
    for source in sources:
        print(f"\nProcessing {source}...")
        output_path = process_dataset(source, args.output_dir, args.max_samples)
        results[source] = output_path
    
    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS COMPLETED")
    print("=" * 50)
    print("Output files:")
    for source, path in results.items():
        print(f"  {source}: {path}")
    
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nUsage:")
    print("  python scripts/sentiment_analysis.py --test          # Test with examples")
    print("  python scripts/sentiment_analysis.py --max_samples 10 # Small sample")
    print("  python scripts/sentiment_analysis.py                 # Full analysis")

if __name__ == "__main__":
    main() 