#!/usr/bin/env python3
"""
Script to calculate overall annotator agreement by category.
Outputs a single agreement rate number for each category.
"""

import pandas as pd
import numpy as np

def calculate_annotator_agreement():
    """Calculate overall annotator agreement by category."""
    
    # Load the category overall agreement data
    try:
        df = pd.read_csv('output/annotation/category_overall_agreement.csv')
        print("📊 Overall Annotator Agreement by Category")
        print("=" * 60)
        print(f"{'Category':<25} {'Agreement Rate (%)':<15}")
        print("-" * 60)
        
        # Sort by agreement rate (descending)
        df_sorted = df.sort_values('Overall Agreement Rate (%)', ascending=False)
        
        # Calculate overall average agreement
        overall_avg = df_sorted['Overall Agreement Rate (%)'].mean()
        
        # Print each category with its agreement rate
        for _, row in df_sorted.iterrows():
            category = row['Category']
            agreement_rate = row['Overall Agreement Rate (%)']
            print(f"{category:<25} {agreement_rate:<15.2f}")
        
        print("-" * 60)
        print(f"{'OVERALL AVERAGE':<25} {overall_avg:<15.2f}")
        print("=" * 60)
        
        # Return the data for potential further use
        return df_sorted
        
    except FileNotFoundError:
        print("❌ Error: Could not find category_overall_agreement.csv")
        print("   Expected file: output/annotation/category_overall_agreement.csv")
        return None
    except Exception as e:
        print(f"❌ Error calculating agreement: {e}")
        return None

def main():
    """Main function."""
    print("🔬 Calculating Overall Annotator Agreement by Category")
    print("=" * 60)
    
    result = calculate_annotator_agreement()
    
    if result is not None:
        print(f"\n✅ Successfully calculated agreement rates for {len(result)} categories")
        print("📁 Data source: output/annotation/category_overall_agreement.csv")
    else:
        print("❌ Failed to calculate agreement rates")

if __name__ == "__main__":
    main() 