import pandas as pd
import os

# File paths
INPUT_FILE = 'annotation/raw_scores.csv'
OUTPUT_DIR = 'output/annotation'
SOFT_LABELS_FILE = os.path.join(OUTPUT_DIR, 'soft_labels.csv')
OVERALL_STATS_FILE = os.path.join(OUTPUT_DIR, 'agreement_stats.csv')
COLUMN_STATS_FILE = os.path.join(OUTPUT_DIR, 'column_agreement_stats.csv')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_annotations(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    if ',' in first_line and '\t' not in first_line:
        sep = ','
    else:
        sep = '\t'
    
    df = pd.read_csv(file_path, sep=sep, header=1)
    df.columns = df.columns.str.strip()
    
    return df

def calculate_agreement(df, label_columns):
    total_labels = 0
    full_agreement = 0
    partial_agreement = 0
    no_agreement = 0

    soft_labels = []
    per_column_stats = []

    for _, row in df.iterrows():
        row_soft = {}
        for col in label_columns:
            val = row[col]
            row_soft[col] = val / 2.0  # Convert 0/1/2 → 0.0/0.5/1.0
        soft_labels.append(row_soft)

    for col in label_columns:
        col_values = df[col]
        full = (col_values == 2).sum()
        partial = (col_values == 1).sum()
        none = (col_values == 0).sum()
        total = full + partial + none

        pos_agree = full
        neg_agree = none
        disagree = partial
        percent_agree = (pos_agree + neg_agree) / total if total > 0 else 0
        prevalence = (col_values > 0).sum() / len(col_values)

        per_column_stats.append({
            'Label': col,
            'Positive Agreement (2)': pos_agree,
            'Partial Agreement (1)': disagree,
            'Negative Agreement (0)': neg_agree,
            'Total': total,
            'Agreement Rate (%)': round(percent_agree * 100, 2),
            'Positive Prevalence (%)': round(prevalence * 100, 2)
        })

        full_agreement += full
        partial_agreement += partial
        no_agreement += none
        total_labels += total

    overall_agreement = full_agreement / total_labels if total_labels > 0 else 0.0
    overall_stats = {
        'Total Labels': total_labels,
        'Full Agreement (2)': full_agreement,
        'Partial Agreement (1)': partial_agreement,
        'No Agreement (0)': no_agreement,
        'Gold Standard Agreement Rate (%)': round(overall_agreement * 100, 2)
    }

    soft_df = pd.DataFrame(soft_labels)
    per_column_df = pd.DataFrame(per_column_stats)
    return soft_df, overall_stats, per_column_df

def save_statistics(stats_dict, file_path):
    stats_df = pd.DataFrame([stats_dict])
    stats_df.to_csv(file_path, index=False)

def main():
    df = load_annotations(INPUT_FILE)

    label_columns = [
        'ask a genuine question',
        'ask a rhetorical question',
        'provide a fact or claim',
        'provide an observation',
        'express their opinion',
        'express others opinions',
        'money aid allocation', 'government critique',
        'societal critique', 'solutions/interventions', 'personal interaction',
        'media portrayal', 'not in my backyard', 'harmful generalization',
        'deserving/undeserving', 'racist'
    ]

    soft_df, overall_stats, per_column_df = calculate_agreement(df, label_columns)

    soft_df.to_csv(SOFT_LABELS_FILE, index=False)
    #save_statistics(overall_stats, OVERALL_STATS_FILE)
    per_column_df.to_csv(COLUMN_STATS_FILE, index=False)

    print(f"Soft labels saved to: {SOFT_LABELS_FILE}")
    #print(f"Overall agreement stats saved to: {OVERALL_STATS_FILE}")
    print(f"Per-column stats saved to: {COLUMN_STATS_FILE}")

if __name__ == '__main__':
    main()
