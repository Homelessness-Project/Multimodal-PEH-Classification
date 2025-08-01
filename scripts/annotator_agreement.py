import pandas as pd
import os

# File paths for all 4 data types
RAW_SCORES_FILES = {
    'reddit': 'annotation/reddit_raw_scores.csv',
    'news': 'annotation/news_raw_scores.csv', 
    'meeting_minutes': 'annotation/meeting_minutes_raw_scores.csv',
    'x': 'annotation/x_raw_scores.csv'
}

OUTPUT_DIR = 'output/annotation'
SOFT_LABELS_DIR = os.path.join(OUTPUT_DIR, 'soft_labels')
OVERALL_STATS_FILE = os.path.join(OUTPUT_DIR, 'agreement_stats.csv')
COLUMN_STATS_FILE = os.path.join(OUTPUT_DIR, 'column_agreement_stats.csv')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SOFT_LABELS_DIR, exist_ok=True)

def load_annotations(file_path):
    """Load annotations from CSV file, handling different separators"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    return df

def calculate_agreement(df, label_columns, data_type):
    """Calculate agreement statistics for a dataset with 3 annotators"""
    total_labels = 0
    full_agreement = 0
    partial_agreement = 0
    no_agreement = 0

    soft_labels = []
    per_column_stats = []

    # Convert raw scores to soft labels (0/1/2/3 → 0.0/0.33/0.67/1.0) for 3 annotators
    for _, row in df.iterrows():
        row_soft = {}
        for col in label_columns:
            val = row[col]
            row_soft[col] = val / 3.0  # Convert 0/1/2/3 → 0.0/0.33/0.67/1.0 for 3 annotators
        soft_labels.append(row_soft)

    # Calculate per-column statistics
    for col in label_columns:
        col_values = df[col]
        full = (col_values == 3).sum()  # All 3 annotators agree
        partial_high = (col_values == 2).sum()  # 2 out of 3 annotators agree
        partial_low = (col_values == 1).sum()  # 1 out of 3 annotators agree
        none = (col_values == 0).sum()  # No annotators agree
        total = full + partial_high + partial_low + none

        # Agreement calculation for 3 annotators
        pos_agree = full  # Full agreement (3 annotators)
        neg_agree = none  # No agreement (0 annotators)
        disagree = partial_high + partial_low  # Partial agreement (1 or 2 annotators)
        percent_agree = (pos_agree + neg_agree) / total if total > 0 else 0
        prevalence = (col_values > 0).sum() / len(col_values)

        per_column_stats.append({
            'Data_Type': data_type,
            'Label': col,
            'Full Agreement (3)': pos_agree,
            'Partial Agreement (1-2)': disagree,
            'No Agreement (0)': neg_agree,
            'Total': total,
            'Agreement Rate (%)': round(percent_agree * 100, 2),
            'Positive Prevalence (%)': round(prevalence * 100, 2)
        })

        full_agreement += full
        partial_agreement += partial_high + partial_low
        no_agreement += none
        total_labels += total

    overall_agreement = full_agreement / total_labels if total_labels > 0 else 0.0
    overall_stats = {
        'Data_Type': data_type,
        'Total Labels': total_labels,
        'Full Agreement (3)': full_agreement,
        'Partial Agreement (1-2)': partial_agreement,
        'No Agreement (0)': no_agreement,
        'Gold Standard Agreement Rate (%)': round(overall_agreement * 100, 2)
    }

    soft_df = pd.DataFrame(soft_labels)
    per_column_df = pd.DataFrame(per_column_stats)
    return soft_df, overall_stats, per_column_df

def calculate_overall_category_agreement(all_per_column_stats):
    """Calculate overall agreement rates by category across all data types"""
    if not all_per_column_stats:
        return None
    
    # Combine all per-column stats
    combined_df = pd.concat(all_per_column_stats, ignore_index=True)
    
    # Group by label and calculate overall statistics
    category_stats = []
    for label in combined_df['Label'].unique():
        label_data = combined_df[combined_df['Label'] == label]
        
        total_full = label_data['Full Agreement (3)'].sum()
        total_partial = label_data['Partial Agreement (1-2)'].sum()
        total_none = label_data['No Agreement (0)'].sum()
        total_labels = total_full + total_partial + total_none
        
        if total_labels > 0:
            agreement_rate = (total_full + total_none) / total_labels * 100
            prevalence = (total_full + total_partial) / total_labels * 100
            
            category_stats.append({
                'Category': label,
                'Total Labels': total_labels,
                'Full Agreement (3)': total_full,
                'Partial Agreement (1-2)': total_partial,
                'No Agreement (0)': total_none,
                'Overall Agreement Rate (%)': round(agreement_rate, 2),
                'Overall Prevalence (%)': round(prevalence, 2)
            })
    
    return pd.DataFrame(category_stats)

def save_statistics(stats_dict, file_path):
    """Save statistics to CSV file"""
    stats_df = pd.DataFrame([stats_dict])
    stats_df.to_csv(file_path, index=False)

def main():
    all_overall_stats = []
    all_per_column_stats = []
    
    # Define label columns (same for all data types)
    label_columns = [
        'ask a genuine question',
        'ask a rhetorical question', 
        'provide a fact or claim',
        'provide an observation',
        'express their opinion',
        'express others opinions',
        'money aid allocation', 
        'government critique',
        'societal critique', 
        'solutions/interventions', 
        'personal interaction',
        'media portrayal', 
        'not in my backyard', 
        'harmful generalization',
        'deserving/undeserving', 
        'racist'
    ]
    
    # Handle different column name variations
    label_column_variations = {
        'ask a rheorical question': 'ask a rhetorical question',  # typo in some files
        'Racist': 'racist'  # capitalization difference
    }

    # Process each data type
    for data_type, file_path in RAW_SCORES_FILES.items():
        print(f"Processing {data_type} annotations...")
        
        try:
            df = load_annotations(file_path)
            
            # Fix column name variations
            for old_name, new_name in label_column_variations.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Filter to only include the standard label columns
            available_columns = [col for col in label_columns if col in df.columns]
            
            if not available_columns:
                print(f"Warning: No standard label columns found in {data_type}")
                print(f"Available columns: {df.columns.tolist()}")
                continue
                
            print(f"Found {len(available_columns)} label columns for {data_type}")
            soft_df, overall_stats, per_column_df = calculate_agreement(df, available_columns, data_type)
            
            # Save soft labels for this data type
            soft_labels_file = os.path.join(SOFT_LABELS_DIR, f'{data_type}_soft_labels.csv')
            soft_df.to_csv(soft_labels_file, index=False)
            print(f"Soft labels saved to: {soft_labels_file}")
            
            all_overall_stats.append(overall_stats)
            all_per_column_stats.append(per_column_df)
            
        except Exception as e:
            print(f"Error processing {data_type}: {e}")
            continue

    # Combine and save overall statistics
    if all_overall_stats:
        overall_stats_df = pd.DataFrame(all_overall_stats)
        overall_stats_df.to_csv(OVERALL_STATS_FILE, index=False)
        print(f"Overall agreement stats saved to: {OVERALL_STATS_FILE}")
        
        # Print summary
        print("\n=== ANNOTATOR AGREEMENT SUMMARY BY DATA TYPE ===")
        for _, row in overall_stats_df.iterrows():
            print(f"{row['Data_Type'].upper()}:")
            print(f"  Total Labels: {row['Total Labels']}")
            print(f"  Full Agreement: {row['Full Agreement (3)']}")
            print(f"  Partial Agreement: {row['Partial Agreement (1-2)']}")
            print(f"  No Agreement: {row['No Agreement (0)']}")
            print(f"  Overall Agreement Rate: {row['Gold Standard Agreement Rate (%)']}%")
            print()

    # Calculate and display overall category agreement
    if all_per_column_stats:
        category_overall_df = calculate_overall_category_agreement(all_per_column_stats)
        if category_overall_df is not None:
            print("\n=== OVERALL AGREEMENT BY CATEGORY (ACROSS ALL DATA TYPES) ===")
            for _, row in category_overall_df.iterrows():
                print(f"{row['Category']}:")
                print(f"  Total Labels: {row['Total Labels']}")
                print(f"  Full Agreement: {row['Full Agreement (3)']}")
                print(f"  Partial Agreement: {row['Partial Agreement (1-2)']}")
                print(f"  No Agreement: {row['No Agreement (0)']}")
                print(f"  Overall Agreement Rate: {row['Overall Agreement Rate (%)']}%")
                print(f"  Overall Prevalence: {row['Overall Prevalence (%)']}%")
                print()
            
            # Save category overall stats
            category_overall_file = os.path.join(OUTPUT_DIR, 'category_overall_agreement.csv')
            category_overall_df.to_csv(category_overall_file, index=False)
            print(f"Category overall agreement saved to: {category_overall_file}")

    # Combine and save per-column statistics
    if all_per_column_stats:
        combined_per_column_df = pd.concat(all_per_column_stats, ignore_index=True)
        combined_per_column_df.to_csv(COLUMN_STATS_FILE, index=False)
        print(f"Per-column stats saved to: {COLUMN_STATS_FILE}")
        
        # Print per-column summary
        print("\n=== PER-COLUMN AGREEMENT SUMMARY BY DATA TYPE ===")
        for data_type in RAW_SCORES_FILES.keys():
            data_stats = combined_per_column_df[combined_per_column_df['Data_Type'] == data_type]
            if not data_stats.empty:
                print(f"\n{data_type.upper()}:")
                for _, row in data_stats.iterrows():
                    print(f"  {row['Label']}: {row['Agreement Rate (%)']}% agreement, {row['Positive Prevalence (%)']}% prevalence")

if __name__ == '__main__':
    main()
