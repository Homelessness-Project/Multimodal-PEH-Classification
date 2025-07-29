import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import os
import scipy.stats as stats

# Create output directories if they don't exist
os.makedirs('output/charts', exist_ok=True)

# Mapping from soft label columns to classification columns
field_map = {
    'ask a genuine question': 'Comment_ask a genuine question',
    'ask a rhetorical question': 'Comment_ask a rhetorical question',
    'provide a fact or claim': 'Comment_provide a fact or claim',
    'provide an observation': 'Comment_provide an observation',
    'express their opinion': 'Comment_express their opinion',
    'express others opinions': 'Comment_express others opinions',
    'money aid allocation': 'Critique_money aid allocation',
    'government critique': 'Critique_government critique',
    'societal critique': 'Critique_societal critique',
    'solutions/interventions': 'Response_solutions/interventions',
    'personal interaction': 'Perception_personal interaction',
    'media portrayal': 'Perception_media portrayal',
    'not in my backyard': 'Perception_not in my backyard',
    'harmful generalization': 'Perception_harmful generalization',
    'deserving/undeserving': 'Perception_deserving/undeserving',
    'racist': 'racist'
}

# Update file paths for Reddit annotation and soft label data
RAW_SCORES_FILE = 'annotation/reddit_raw_scores.csv'
SOFT_LABELS_FILE = 'output/annotation/reddit_soft_labels.csv'

# Update file paths for all models classified outputs
LLAMA_CLASSIFIED_FILE = 'output/classified_comments_reddit_gold_subset_llama.csv'
QWEN_CLASSIFIED_FILE = 'output/classified_comments_reddit_gold_subset_qwen.csv'
GPT4_CLASSIFIED_FILE = 'output/reddit/gpt4/classified_comments_reddit_gold_subset_gpt4_none_flags.csv'
GROK_CLASSIFIED_FILE = 'output/reddit/grok/classified_comments_reddit_gold_subset_grok_none_flags.csv'
PHI4_CLASSIFIED_FILE = 'output/classified_comments_reddit_gold_subset_phi4.csv'

def load_classifications():
    """Load all model classification results for original and mitigated data."""
    try:
        print("Loading classification files...")
        llama_df = pd.read_csv(LLAMA_CLASSIFIED_FILE)
        qwen_df = pd.read_csv(QWEN_CLASSIFIED_FILE)
        
        # Load GPT-4 if available
        try:
            gpt4_df = pd.read_csv(GPT4_CLASSIFIED_FILE)
            print(f"Loaded GPT-4 data: {len(gpt4_df)} comments")
        except FileNotFoundError:
            print("GPT-4 file not found, skipping...")
            gpt4_df = None
        
        # Load Grok if available
        try:
            grok_df = pd.read_csv(GROK_CLASSIFIED_FILE)
            print(f"Loaded Grok data: {len(grok_df)} comments")
        except FileNotFoundError:
            print("Grok file not found, skipping...")
            grok_df = None
        
        # Load Phi-4 if available
        try:
            phi4_df = pd.read_csv(PHI4_CLASSIFIED_FILE)
            print(f"Loaded Phi-4 data: {len(phi4_df)} comments")
        except FileNotFoundError:
            print("Phi-4 file not found, skipping...")
            phi4_df = None
        
        soft_labels_df = pd.read_csv(SOFT_LABELS_FILE)
        
        print(f"Loaded {len(llama_df)} comments from Llama")
        print(f"Loaded {len(qwen_df)} comments from Qwen")
        print(f"Soft labels length: {len(soft_labels_df)}")
        
        print("\nSoft labels columns:")
        print(soft_labels_df.columns.tolist())
        print("\nLlama classification columns:")
        print(llama_df.columns.tolist())
        print("\nQwen classification columns:")
        print(qwen_df.columns.tolist())
        if gpt4_df is not None:
            print("\nGPT-4 classification columns:")
            print(gpt4_df.columns.tolist())
        if grok_df is not None:
            print("\nGrok classification columns:")
            print(grok_df.columns.tolist())
        if phi4_df is not None:
            print("\nPhi-4 classification columns:")
            print(phi4_df.columns.tolist())
        
        return llama_df, qwen_df, gpt4_df, grok_df, phi4_df, soft_labels_df
    except Exception as e:
        print(f"Error loading classification files: {e}")
        exit(1)

def calculate_kappa(llama_df, qwen_df, field):
    """Calculate Cohen's Kappa for a specific field."""
    # Get the values for both models
    llama_values = llama_df[field].fillna(0)  # Fill NaN with 0 for flag fields
    qwen_values = qwen_df[field].fillna(0)
    
    # Convert to int for flag fields (except racist)
    if field != "racist":
        llama_values = llama_values.astype(int)
        qwen_values = qwen_values.astype(int)
    else:
        # For racist, extract the value from the text
        def extract_racist_value(text):
            if pd.isna(text):
                return 0
            text = str(text).lower().strip()
            if text == "yes" or "racist: yes" in text:
                return 1  # Yes is positive (1)
            elif text == "no" or "racist: no" in text:
                return 0  # No is negative (0)
            return 0  # Default to negative (0) if not found
        
        llama_values = llama_values.apply(extract_racist_value)
        qwen_values = qwen_values.apply(extract_racist_value)
    
    # Calculate kappa
    kappa = cohen_kappa_score(llama_values, qwen_values)
    
    # Create confusion matrix with explicit labels
    cm = confusion_matrix(llama_values, qwen_values, labels=[0, 1])
    
    # Flip the matrix to put positive class (1) in top left
    cm = np.flip(cm, axis=(0, 1))
    
    # Calculate total positives for each model
    llama_positives = cm[0, 0] + cm[0, 1]  # Top row (1s)
    qwen_positives = cm[0, 0] + cm[1, 0]   # Left column (1s)
    
    return kappa, cm, llama_positives, qwen_positives

def calculate_soft_label_agreement(llama_df, qwen_df, soft_labels_df, field):
    """Calculate agreement between models and soft labels."""
    # Get the values for both models and soft labels, ensuring numeric dtype
    llama_values = pd.to_numeric(llama_df[field_map[field]], errors='coerce').fillna(0)
    qwen_values = pd.to_numeric(qwen_df[field_map[field]], errors='coerce').fillna(0)
    soft_values = pd.to_numeric(soft_labels_df[field], errors='coerce').fillna(0)

    # Binarize soft labels for confusion matrix
    soft_values_bin = (soft_values >= 0.5).astype(int)

    # Calculate agreement metrics
    llama_agreement = np.mean(np.abs(llama_values - soft_values) <= 0.5)  # Within 0.5
    qwen_agreement = np.mean(np.abs(qwen_values - soft_values) <= 0.5)    # Within 0.5

    # Create confusion matrices for soft labels (binarized)
    llama_cm = confusion_matrix(soft_values_bin, llama_values, labels=[0, 1])
    qwen_cm = confusion_matrix(soft_values_bin, qwen_values, labels=[0, 1])

    return {
        'llama_agreement': llama_agreement,
        'qwen_agreement': qwen_agreement,
        'llama_cm': llama_cm,
        'qwen_cm': qwen_cm,
        'soft_values': soft_values,
        'llama_values': llama_values,
        'qwen_values': qwen_values
    }

def calculate_multi_model_f1_scores(models_dict, soft_labels_df, fields):
    """Calculate F1 scores for multiple models against soft labels."""
    f1_results = {}
    
    for field in fields:
        field_results = {}
        soft_values = pd.to_numeric(soft_labels_df[field], errors='coerce').fillna(0)
        soft_bin = (soft_values == 1).astype(int)  # 1 if soft label is 1, else 0
        
        for model_name, model_df in models_dict.items():
            if model_df is not None:
                try:
                    # Get model predictions for this field
                    if field == 'racist':
                        # Handle racist field specially
                        if 'Racist_Flag' in model_df.columns:
                            model_pred = model_df['Racist_Flag'].fillna(0).astype(int)
                        else:
                            # Extract from racist text field
                            def extract_racist_value(text):
                                if pd.isna(text):
                                    return 0
                                text = str(text).lower().strip()
                                if text == "yes" or "racist: yes" in text:
                                    return 1
                                elif text == "no" or "racist: no" in text:
                                    return 0
                                return 0
                            model_pred = model_df['racist'].apply(extract_racist_value)
                    else:
                        # Handle other fields
                        field_col = field_map[field]
                        if field_col in model_df.columns:
                            model_pred = pd.to_numeric(model_df[field_col], errors='coerce').fillna(0).astype(int)
                        else:
                            model_pred = np.zeros(len(model_df))
                    
                    # Calculate F1 score
                    if len(model_pred) == len(soft_bin):
                        f1 = f1_score(soft_bin, model_pred, zero_division=0)
                        field_results[model_name] = f1
                    else:
                        field_results[model_name] = np.nan
                except Exception as e:
                    print(f"Error calculating F1 for {model_name} on {field}: {e}")
                    field_results[model_name] = np.nan
            else:
                field_results[model_name] = np.nan
        
        f1_results[field] = field_results
    
    return f1_results

def plot_confusion_matrix(cm, field, kappa, llama_positives, qwen_positives, ax, delta_info=None, llama_soft_agree=None, qwen_soft_agree=None):
    """Plot confusion matrix on the given axis, with optional soft label agreement annotations."""
    # Get labels based on field type
    if field == "racist":
        labels = ["Yes", "No"]  # Yes (1) is positive, No (0) is negative
    else:
        labels = ["1", "0"]  # 1 is positive, 0 is negative
    
    # Create heatmap with labels and consistent vmax
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=True, vmax=500)
    
    # Format field name for display
    if field == "racist":
        title = f"racist\nCohen's κ = {kappa:.2f}"
    elif field.startswith("Comment_"):
        category = "Direct Comment" if field == "Comment_direct" else "Reporting Comment"
        title = f"Comment: {category}\nCohen's κ = {kappa:.2f}"
    elif field.startswith("Critique_"):
        category = field.replace("Critique_", "").replace("_", " ").title()
        title = f"Critique: {category}\nCohen's κ = {kappa:.2f}"
    elif field.startswith("Response_"):
        category = field.replace("Response_", "").replace("_", " ").title()
        title = f"Response: {category}\nCohen's κ = {kappa:.2f}"
    elif field.startswith("Perception_"):
        category = field.replace("Perception_", "").replace("_", " ").title()
        title = f"Perception: {category}\nCohen's κ = {kappa:.2f}"
    else:
        # Default case - use the field name directly
        title = f"{field}\nCohen's κ = {kappa:.2f}"
    
    # Add delta information if provided
    if delta_info:
        title += f"\nΔκ = {delta_info['kappa_delta']:+.2f}"
        title += f"\nΔLlama +: {delta_info['llama_delta']:+d}"
        title += f"\nΔQwen +: {delta_info['qwen_delta']:+d}"
    else:
        title += f"\nLlama +: {llama_positives}"
        title += f"\nQwen +: {qwen_positives}"
    
    # Add soft label agreement annotation if provided
    if llama_soft_agree is not None and qwen_soft_agree is not None:
        title += f"\nLlama-Soft: {llama_soft_agree:.2f}  Qwen-Soft: {qwen_soft_agree:.2f}"
    
    # Add title with kappa score and positive counts
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Qwen Predictions', fontweight='bold')
    ax.set_ylabel('Llama Predictions', fontweight='bold')

def plot_delta_heatmap(original_results, mitigated_results, pdf_path):
    """Create a heatmap showing the changes in kappa scores."""
    fields = list(original_results.keys())
    kappa_deltas = [mitigated_results[f]['kappa'] - original_results[f]['kappa'] for f in fields]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create heatmap
    sns.heatmap(np.array(kappa_deltas).reshape(-1, 1), 
                annot=True, 
                fmt='+.2f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Δκ'},
                yticklabels=fields)
    
    plt.title('Changes in Cohen\'s Kappa Scores After Mitigation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig()
        plt.close()

def plot_soft_label_comparison(soft_results, pdf_path):
    """Create visualization comparing model predictions with soft labels."""
    fields = list(soft_results.keys())
    
    # Compute agreement breakdowns for each model and field
    llama_pos, llama_neg, llama_fp, llama_fn = [], [], [], []
    qwen_pos, qwen_neg, qwen_fp, qwen_fn = [], [], [], []
    soft_05 = []  # Single list for soft label 0.5 cases
    for f in fields:
        soft = np.array(soft_results[f]['soft_values'])
        llama = np.array(soft_results[f]['llama_values'])
        qwen = np.array(soft_results[f]['qwen_values'])
        
        # Positive agreement: model==1 & soft==1
        llama_pos.append(np.mean((llama == 1) & (soft == 1)))
        qwen_pos.append(np.mean((qwen == 1) & (soft == 1)))
        
        # Negative agreement: model==0 & soft==0
        llama_neg.append(np.mean((llama == 0) & (soft == 0)))
        qwen_neg.append(np.mean((qwen == 0) & (soft == 0)))
        
        # Soft label is 0.5 (same for both models)
        soft_05.append(np.mean(soft == 0.5))
        
        # False positives: model==1 & soft==0
        llama_fp.append(np.mean((soft == 0) & (llama == 1)))
        qwen_fp.append(np.mean((soft == 0) & (qwen == 1)))
        
        # False negatives: model==0 & soft==1
        llama_fn.append(np.mean((soft == 1) & (llama == 0)))
        qwen_fn.append(np.mean((soft == 1) & (qwen == 0)))
    
    # Prepare DataFrame for grouped, stacked bar chart
    agreement_df = pd.DataFrame({
        'Field': fields,
        'Llama Positive Agreement': llama_pos,
        'Llama Negative Agreement': llama_neg,
        'Llama False Positive': llama_fp,
        'Llama False Negative': llama_fn,
        'Soft Label 0.5': soft_05,  # Single column for soft label 0.5
        'Qwen Positive Agreement': qwen_pos,
        'Qwen Negative Agreement': qwen_neg,
        'Qwen False Positive': qwen_fp,
        'Qwen False Negative': qwen_fn
    })
    
    # Save agreement data to CSV
    agreement_df.to_csv('output/charts/llm_soft_label_agreement.csv', index=False)
    
    # Plot grouped, stacked bars
    fig, ax1 = plt.subplots(figsize=(9, 9))  # Changed to square with full width
    width = 0.35
    x = np.arange(len(fields))
    
    # Llama bars (reordered stacking)
    ax1.bar(x - width/2, agreement_df['Soft Label 0.5'], width, color='yellow', label='Soft Label 0.5', bottom=0)
    ax1.bar(x - width/2, agreement_df['Llama Positive Agreement'], width, color='lightblue', label='Llama Positive Agreement', 
            bottom=agreement_df['Soft Label 0.5'])
    ax1.bar(x - width/2, agreement_df['Llama Negative Agreement'], width, color='blue', label='Llama Negative Agreement', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Llama Positive Agreement'])
    ax1.bar(x - width/2, agreement_df['Llama False Positive'], width, color='orange', label='Llama False Positive', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Llama Positive Agreement'] + agreement_df['Llama Negative Agreement'])
    ax1.bar(x - width/2, agreement_df['Llama False Negative'], width, color='darkred', label='Llama False Negative', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Llama Positive Agreement'] + agreement_df['Llama Negative Agreement'] + agreement_df['Llama False Positive'])
    
    # Qwen bars (reordered stacking)
    ax1.bar(x + width/2, agreement_df['Soft Label 0.5'], width, color='yellow', label='Soft Label 0.5', bottom=0)
    ax1.bar(x + width/2, agreement_df['Qwen Positive Agreement'], width, color='lightgreen', label='Qwen Positive Agreement', 
            bottom=agreement_df['Soft Label 0.5'])
    ax1.bar(x + width/2, agreement_df['Qwen Negative Agreement'], width, color='green', label='Qwen Negative Agreement', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Qwen Positive Agreement'])
    ax1.bar(x + width/2, agreement_df['Qwen False Positive'], width, color='gold', label='Qwen False Positive', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Qwen Positive Agreement'] + agreement_df['Qwen Negative Agreement'])
    ax1.bar(x + width/2, agreement_df['Qwen False Negative'], width, color='red', label='Qwen False Negative', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Qwen Positive Agreement'] + agreement_df['Qwen Negative Agreement'] + agreement_df['Qwen False Positive'])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(fields, rotation=45, ha='right', fontsize=12)
    ax1.set_title('LLM Agreement with Soft Labels', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Proportion', fontsize=14)
    ax1.set_ylim(0, 1)
    
    # Custom legend (removing duplicate Soft Label 0.5 entry)
    handles = [
        plt.Rectangle((0,0),1,1,color='yellow'),
        plt.Rectangle((0,0),1,1,color='lightblue'),
        plt.Rectangle((0,0),1,1,color='blue'),
        plt.Rectangle((0,0),1,1,color='orange'),
        plt.Rectangle((0,0),1,1,color='darkred'),
        plt.Rectangle((0,0),1,1,color='lightgreen'),
        plt.Rectangle((0,0),1,1,color='green'),
        plt.Rectangle((0,0),1,1,color='gold'),
        plt.Rectangle((0,0),1,1,color='red')
    ]
    labels = [
        'Soft Label 0.5',
        'Llama Positive Agreement', 'Llama Negative Agreement', 'Llama False Positive', 'Llama False Negative',
        'Qwen Positive Agreement', 'Qwen Negative Agreement', 'Qwen False Positive', 'Qwen False Negative'
    ]
    ax1.legend(handles, labels, bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3, fontsize=12)
    plt.tight_layout()
    
    # Save agreement chart to PDF
    with PdfPages('output/charts/agreement.pdf') as pdf:
        pdf.savefig(fig)
        plt.close()
    
    # Plot distribution of soft labels with updated labels
    soft_dist = pd.DataFrame({
        'Field': fields,
        'Negative Agreement (0)': [np.mean(soft_results[f]['soft_values'] == 0) for f in fields],
        'Soft Label 0.5': [np.mean(soft_results[f]['soft_values'] == 0.5) for f in fields],
        'Positive Agreement (1)': [np.mean(soft_results[f]['soft_values'] == 1) for f in fields]
    })
    fig, ax2 = plt.subplots(figsize=(9, 7))
    soft_dist.plot(x='Field', y=['Negative Agreement (0)', 'Soft Label 0.5', 'Positive Agreement (1)'], 
                  kind='bar', stacked=True, ax=ax2, rot=45, color=['green', 'yellow', 'red'])
    ax2.set_title('Agreement Between Annotators', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Proportion', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45, ha='right')  # Added ha='right' for better alignment
    plt.tight_layout()
    
    # Save soft label distribution chart to PDF
    with PdfPages('output/charts/distribution.pdf') as pdf:
        pdf.savefig(fig)
        plt.close()

def gold_standard_by_city_size():
    # Load annotation data
    df = pd.read_csv(RAW_SCORES_FILE, header=1)
    
    # Define city groups
    small_cities = [
        'southbend', 'rockford', 'kzoo', 'scranton', 'fayetteville'
    ]
    large_cities = [
        'sanfrancisco', 'portland', 'baltimore', 'buffalo', 'elpaso'
    ]
    
    # Label columns (full agreement = 2)
    label_columns = [
        'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
        'provide an observation', 'express their opinion', 'express others opinions',
        'money aid allocation', 'government critique', 'societal critique',
        'solutions/interventions', 'personal interaction', 'media portrayal',
        'not in my backyard', 'harmful generalization', 'deserving/undeserving', 'racist'
    ]
    
    # For each city, compute prevalence (proportion of comments with full agreement for each label)
    df['city_size'] = df['City'].apply(lambda x: 'Small' if x in small_cities else ('Large' if x in large_cities else 'Other'))
    df = df[df['city_size'].isin(['Small', 'Large'])]
    
    # Prepare results
    rows = []
    for label in label_columns:
        small_vals = df[df['city_size']=='Small'][label] == 2
        large_vals = df[df['city_size']=='Large'][label] == 2
        small_prevs = small_vals.groupby(df[df['city_size']=='Small']['City']).mean()
        large_prevs = large_vals.groupby(df[df['city_size']=='Large']['City']).mean()
        small_mean = small_prevs.mean()
        small_std = small_prevs.std(ddof=0)
        large_mean = large_prevs.mean()
        large_std = large_prevs.std(ddof=0)
        # t-test for difference in means
        if len(small_prevs) > 1 and len(large_prevs) > 1:
            _, p_value = stats.ttest_ind(small_prevs, large_prevs, equal_var=False)
        else:
            p_value = float('nan')
        rows.append({
            'Category': label,
            'Large_Mean': large_mean,
            'Large_Std': large_std,
            'Small_Mean': small_mean,
            'Small_Std': small_std,
            'p_value': p_value
        })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv('output/charts/gold_standard_by_city_size_stats.csv', index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(label_columns))
    width = 0.35
    ax.bar(x - width/2, stats_df['Small_Mean'], width, yerr=stats_df['Small_Std'], label='Small Cities', capsize=5, color='skyblue')
    ax.bar(x + width/2, stats_df['Large_Mean'], width, yerr=stats_df['Large_Std'], label='Large Cities', capsize=5, color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(label_columns, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel('Prevalence (Full Agreement)', fontweight='bold')
    ax.set_xlabel('Category', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Gold Standard Annotation Prevalence by City Size (Full Agreement Only)\nMin=0, Max=1. Statistical test: difference in prevalence between clusters', fontweight='bold')
    legend = ax.legend(fontsize=12, title_fontproperties={'weight':'bold'}, prop={'weight':'bold'})
    plt.setp(legend.get_texts(), fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/charts/gold_standard_by_city_size.pdf')
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate F1 scores and agreement for multiple LLM models')
    args = parser.parse_args()
    
    # Load the classification results
    llama_df, qwen_df, gpt4_df, grok_df, phi4_df, soft_labels_df = load_classifications()
    
    # Create models dictionary
    models_dict = {
        'Llama': llama_df,
        'Qwen': qwen_df,
        'GPT-4': gpt4_df,
        'Grok': grok_df,
        'Phi-4': phi4_df
    }
    
    # Remove None models
    models_dict = {k: v for k, v in models_dict.items() if v is not None}
    
    print(f"\nLoaded {len(models_dict)} models: {list(models_dict.keys())}")
    
    # Fields to analyze - these match the column names in soft_labels.csv
    fields = list(field_map.keys())
    
    # Calculate F1 scores for all models
    print("\nCalculating F1 scores for all models...")
    f1_results = calculate_multi_model_f1_scores(models_dict, soft_labels_df, fields)
    
    # Create F1 scores DataFrame
    f1_data = []
    for field in fields:
        row = {'Field': field}
        for model_name in models_dict.keys():
            row[f'{model_name}_F1'] = f1_results[field].get(model_name, np.nan)
        f1_data.append(row)
    
    f1_df = pd.DataFrame(f1_data)
    f1_df.to_csv('output/charts/llm_f1_scores_multi_models.csv', index=False)
    print('Multi-model F1 scores saved to output/charts/llm_f1_scores_multi_models.csv')
    
    # Print F1 scores summary
    print("\n=== F1 Scores Summary ===")
    print(f"{'Field':<25} {'Llama':<8} {'Qwen':<8} {'GPT-4':<8} {'Grok':<8} {'Phi-4':<8}")
    print("-" * 70)
    for field in fields:
        row = f1_df[f1_df['Field'] == field].iloc[0]
        print(f"{field:<25} {row.get('Llama_F1', 'N/A'):<8.3f} {row.get('Qwen_F1', 'N/A'):<8.3f} {row.get('GPT-4_F1', 'N/A'):<8.3f} {row.get('Grok_F1', 'N/A'):<8.3f} {row.get('Phi-4_F1', 'N/A'):<8.3f}")
    
    # Calculate mean F1 scores per model
    print("\n=== Mean F1 Scores by Model ===")
    model_means = {}
    for model_name in models_dict.keys():
        model_f1s = [f1_results[field].get(model_name, np.nan) for field in fields]
        model_means[model_name] = np.nanmean(model_f1s)
        print(f"{model_name}: {model_means[model_name]:.3f}")
    
    # Save model comparison
    comparison_df = pd.DataFrame({
        'Model': list(model_means.keys()),
        'Mean_F1': list(model_means.values())
    })
    comparison_df.to_csv('output/charts/model_comparison_f1.csv', index=False)
    print('Model comparison saved to output/charts/model_comparison_f1.csv')
    
    # Create PDF for original data
    with PdfPages('output/charts/confusion_matrices.pdf') as pdf:
        # Calculate grid dimensions
        n_fields = len(original_results)
        n_cols = 3
        n_rows = (n_fields + n_cols - 1) // n_cols
        
        # Create figure with subplots - add extra height for title and row spacing
        fig = plt.figure(figsize=(15, 5 * n_rows + 1))
        
        # Add main title
        plt.suptitle('LLM Classification of Original Data', fontsize=16, fontweight='bold', y=0.98)
        
        # Create a gridspec with extra space at top and between rows
        gs = plt.GridSpec(n_rows, n_cols, top=0.9, hspace=0.6)  # Increased hspace from 0.4 to 0.6
        
        # Plot each confusion matrix
        for idx, (field, stats) in enumerate(original_results.items(), 1):
            row = (idx - 1) // n_cols
            col = (idx - 1) % n_cols
            ax = plt.subplot(gs[row, col])
            
            plot_confusion_matrix(
                stats['confusion_matrix'], 
                field, 
                stats['kappa'],
                stats['llama_positives'],
                stats['qwen_positives'],
                ax,
                llama_soft_agree=soft_results[field]['llama_agreement'],
                qwen_soft_agree=soft_results[field]['qwen_agreement']
            )
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    # Create soft label comparison visualization
    plot_soft_label_comparison(soft_results, 'output/charts/soft_label_comparison.pdf')
    
    # Calculate agreement statistics for mitigated data
    print("\nCalculating agreement statistics for mitigated data...")
    mitigated_results = {}
    for field in tqdm(fields, desc="Fields"):
        kappa, cm, llama_positives, qwen_positives = calculate_kappa(
            llama_mit_df, qwen_mit_df, field_map[field]
        )
        mitigated_results[field] = {
            'kappa': kappa,
            'confusion_matrix': cm,
            'llama_positives': llama_positives,
            'qwen_positives': qwen_positives
        }
    
    # Create PDF for mitigated data
    with PdfPages('output/charts/mitigated_confusion_matrices.pdf') as pdf:
        # Calculate grid dimensions
        n_fields = len(mitigated_results)
        n_cols = 3
        n_rows = (n_fields + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5 * n_rows + 1))
        
        # Add main title
        plt.suptitle('LLM Classification of Mitigated Data', fontsize=16, fontweight='bold', y=0.98)
        
        # Create a gridspec with extra space at top and between rows
        gs = plt.GridSpec(n_rows, n_cols, top=0.9, hspace=0.6)  # Increased hspace from 0.4 to 0.6
        
        # Plot each confusion matrix
        for idx, (field, stats) in enumerate(mitigated_results.items(), 1):
            row = (idx - 1) // n_cols
            col = (idx - 1) % n_cols
            ax = plt.subplot(gs[row, col])
            
            # Calculate deltas for display
            delta_info = {
                'kappa_delta': stats['kappa'] - original_results[field]['kappa'],
                'llama_delta': stats['llama_positives'] - original_results[field]['llama_positives'],
                'qwen_delta': stats['qwen_positives'] - original_results[field]['qwen_positives']
            }
            
            plot_confusion_matrix(
                stats['confusion_matrix'], 
                field, 
                stats['kappa'],
                stats['llama_positives'],
                stats['qwen_positives'],
                ax,
                delta_info
            )
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    # Create delta heatmap
    plot_delta_heatmap(original_results, mitigated_results, 'output/charts/kappa_deltas.pdf')
    
    # Print results
    print("\nInter-coder Reliability Results:")
    print("=" * 50)
    for field in fields:
        print(f"\n{field}:")
        print(f"Original Kappa: {original_results[field]['kappa']:.3f}")
        print(f"Agreement with Soft Labels - Llama: {soft_results[field]['llama_agreement']:.3f}, Qwen: {soft_results[field]['qwen_agreement']:.3f}")
        print(f"Mitigated Kappa: {mitigated_results[field]['kappa']:.3f}")
        print(f"Δκ: {mitigated_results[field]['kappa'] - original_results[field]['kappa']:+.3f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Field': fields,
        'Original_Kappa': [original_results[f]['kappa'] for f in fields],
        'Original_Llama_Positives': [original_results[f]['llama_positives'] for f in fields],
        'Original_Qwen_Positives': [original_results[f]['qwen_positives'] for f in fields],
        'Llama_Soft_Agreement': [soft_results[f]['llama_agreement'] for f in fields],
        'Qwen_Soft_Agreement': [soft_results[f]['qwen_agreement'] for f in fields],
        'Soft_Label_0.5': [np.mean(soft_results[f]['soft_values'] == 0.5) for f in fields],
        'Soft_Label_1': [np.mean(soft_results[f]['soft_values'] == 1) for f in fields],
        'Soft_Label_0': [np.mean(soft_results[f]['soft_values'] == 0) for f in fields],
        'Llama_Positive_Agreement': [np.mean((soft_results[f]['llama_values'] == 1) & (soft_results[f]['soft_values'] == 1)) for f in fields],
        'Llama_Negative_Agreement': [np.mean((soft_results[f]['llama_values'] == 0) & (soft_results[f]['soft_values'] == 0)) for f in fields],
        'Llama_False_Positive': [np.mean((soft_results[f]['soft_values'] == 0) & (soft_results[f]['llama_values'] == 1)) for f in fields],
        'Llama_False_Negative': [np.mean((soft_results[f]['soft_values'] == 1) & (soft_results[f]['llama_values'] == 0)) for f in fields],
        'Qwen_Positive_Agreement': [np.mean((soft_results[f]['qwen_values'] == 1) & (soft_results[f]['soft_values'] == 1)) for f in fields],
        'Qwen_Negative_Agreement': [np.mean((soft_results[f]['qwen_values'] == 0) & (soft_results[f]['soft_values'] == 0)) for f in fields],
        'Qwen_False_Positive': [np.mean((soft_results[f]['soft_values'] == 0) & (soft_results[f]['qwen_values'] == 1)) for f in fields],
        'Qwen_False_Negative': [np.mean((soft_results[f]['soft_values'] == 1) & (soft_results[f]['qwen_values'] == 0)) for f in fields],
        'Mitigated_Kappa': [mitigated_results[f]['kappa'] for f in fields],
        'Mitigated_Llama_Positives': [mitigated_results[f]['llama_positives'] for f in fields],
        'Mitigated_Qwen_Positives': [mitigated_results[f]['qwen_positives'] for f in fields],
        'Kappa_Delta': [mitigated_results[f]['kappa'] - original_results[f]['kappa'] for f in fields],
        'Llama_Positives_Delta': [mitigated_results[f]['llama_positives'] - original_results[f]['llama_positives'] for f in fields],
        'Qwen_Positives_Delta': [mitigated_results[f]['qwen_positives'] - original_results[f]['qwen_positives'] for f in fields]
    })
    results_df.to_csv("output/charts/intercoder_reliability_results.csv", index=False)
    print("\nResults saved to output/charts/intercoder_reliability_results.csv")

    # Create new CSV with counts
    counts_df = pd.DataFrame({
        'Field': fields,
        'Soft_Label_0.5_Count': [np.sum(soft_results[f]['soft_values'] == 0.5) for f in fields],
        'Soft_Label_1_Count': [np.sum(soft_results[f]['soft_values'] == 1) for f in fields],
        'Soft_Label_0_Count': [np.sum(soft_results[f]['soft_values'] == 0) for f in fields],
        'Llama_Positive_Count': [np.sum(soft_results[f]['llama_values'] == 1) for f in fields],
        'Llama_Negative_Count': [np.sum(soft_results[f]['llama_values'] == 0) for f in fields],
        'Qwen_Positive_Count': [np.sum(soft_results[f]['qwen_values'] == 1) for f in fields],
        'Qwen_Negative_Count': [np.sum(soft_results[f]['qwen_values'] == 0) for f in fields],
        'Mitigated_Llama_Positive_Count': [np.sum(llama_mit_df[field_map[f]] == 1) for f in fields],
        'Mitigated_Llama_Negative_Count': [np.sum(llama_mit_df[field_map[f]] == 0) for f in fields],
        'Mitigated_Qwen_Positive_Count': [np.sum(qwen_mit_df[field_map[f]] == 1) for f in fields],
        'Mitigated_Qwen_Negative_Count': [np.sum(qwen_mit_df[field_map[f]] == 0) for f in fields]
    })
    counts_df.to_csv("output/charts/category_counts.csv", index=False)
    print("Category counts saved to output/charts/category_counts.csv")

    # Calculate F1 scores for each field (treating soft label 0.5 as 0)
    f1_rows = []
    for field in fields:
        # Prepare ground truth and predictions
        soft_bin = (soft_results[field]['soft_values'] == 1).astype(int)  # 1 if soft label is 1, else 0
        llama_pred = (soft_results[field]['llama_values'] == 1).astype(int)
        qwen_pred = (soft_results[field]['qwen_values'] == 1).astype(int)
        # F1 scores
        llama_f1 = f1_score(soft_bin, llama_pred)
        qwen_f1 = f1_score(soft_bin, qwen_pred)
        f1_rows.append({
            'Field': field,
            'Llama_F1': llama_f1,
            'Qwen_F1': qwen_f1
        })
    f1_df = pd.DataFrame(f1_rows)
    f1_df.to_csv('output/charts/llm_f1_scores.csv', index=False)
    print('F1 scores saved to output/charts/llm_f1_scores.csv')

    # Calculate classification metrics for each field (treating soft label 0.5 as 0)
    metrics_rows_rounded = []
    metrics_rows_clear = []
    for field in fields:
        # Prepare ground truth and predictions for 'rounded' (0.5 -> 0)
        soft_bin_rounded = (soft_results[field]['soft_values'] == 1).astype(int)  # 1 if soft label is 1, else 0
        llama_pred_rounded = (soft_results[field]['llama_values'] == 1).astype(int)
        qwen_pred_rounded = (soft_results[field]['qwen_values'] == 1).astype(int)
        # Metrics for Llama (rounded)
        llama_f1_r = f1_score(soft_bin_rounded, llama_pred_rounded)
        llama_acc_r = accuracy_score(soft_bin_rounded, llama_pred_rounded)
        llama_prec_r = precision_score(soft_bin_rounded, llama_pred_rounded, zero_division=0)
        llama_rec_r = recall_score(soft_bin_rounded, llama_pred_rounded, zero_division=0)
        llama_kappa_r = cohen_kappa_score(soft_bin_rounded, llama_pred_rounded)
        # Metrics for Qwen (rounded)
        qwen_f1_r = f1_score(soft_bin_rounded, qwen_pred_rounded)
        qwen_acc_r = accuracy_score(soft_bin_rounded, qwen_pred_rounded)
        qwen_prec_r = precision_score(soft_bin_rounded, qwen_pred_rounded, zero_division=0)
        qwen_rec_r = recall_score(soft_bin_rounded, qwen_pred_rounded, zero_division=0)
        qwen_kappa_r = cohen_kappa_score(soft_bin_rounded, qwen_pred_rounded)
        metrics_rows_rounded.append({
            'Field': field,
            'Llama_F1': llama_f1_r,
            'Llama_Accuracy': llama_acc_r,
            'Llama_Precision': llama_prec_r,
            'Llama_Recall': llama_rec_r,
            'Llama_Kappa': llama_kappa_r,
            'Qwen_F1': qwen_f1_r,
            'Qwen_Accuracy': qwen_acc_r,
            'Qwen_Precision': qwen_prec_r,
            'Qwen_Recall': qwen_rec_r,
            'Qwen_Kappa': qwen_kappa_r
        })
        # Prepare ground truth and predictions for 'clear only' (exclude 0.5)
        mask_clear = (soft_results[field]['soft_values'] != 0.5)
        if np.sum(mask_clear) > 0:
            soft_bin_clear = (soft_results[field]['soft_values'][mask_clear] == 1).astype(int)
            llama_pred_clear = (soft_results[field]['llama_values'][mask_clear] == 1).astype(int)
            qwen_pred_clear = (soft_results[field]['qwen_values'][mask_clear] == 1).astype(int)
            # Metrics for Llama (clear)
            llama_f1_c = f1_score(soft_bin_clear, llama_pred_clear)
            llama_acc_c = accuracy_score(soft_bin_clear, llama_pred_clear)
            llama_prec_c = precision_score(soft_bin_clear, llama_pred_clear, zero_division=0)
            llama_rec_c = recall_score(soft_bin_clear, llama_pred_clear, zero_division=0)
            llama_kappa_c = cohen_kappa_score(soft_bin_clear, llama_pred_clear)
            # Metrics for Qwen (clear)
            qwen_f1_c = f1_score(soft_bin_clear, qwen_pred_clear)
            qwen_acc_c = accuracy_score(soft_bin_clear, qwen_pred_clear)
            qwen_prec_c = precision_score(soft_bin_clear, qwen_pred_clear, zero_division=0)
            qwen_rec_c = recall_score(soft_bin_clear, qwen_pred_clear, zero_division=0)
            qwen_kappa_c = cohen_kappa_score(soft_bin_clear, qwen_pred_clear)
        else:
            llama_f1_c = llama_acc_c = llama_prec_c = llama_rec_c = llama_kappa_c = np.nan
            qwen_f1_c = qwen_acc_c = qwen_prec_c = qwen_rec_c = qwen_kappa_c = np.nan
        metrics_rows_clear.append({
            'Field': field,
            'Llama_F1': llama_f1_c,
            'Llama_Accuracy': llama_acc_c,
            'Llama_Precision': llama_prec_c,
            'Llama_Recall': llama_rec_c,
            'Llama_Kappa': llama_kappa_c,
            'Qwen_F1': qwen_f1_c,
            'Qwen_Accuracy': qwen_acc_c,
            'Qwen_Precision': qwen_prec_c,
            'Qwen_Recall': qwen_rec_c,
            'Qwen_Kappa': qwen_kappa_c
        })
    metrics_df_rounded = pd.DataFrame(metrics_rows_rounded)
    # Round all values to 2 decimals before saving
    metrics_df_rounded = metrics_df_rounded.round(2)
    metrics_df_rounded.to_csv('output/charts/llm_classification_metrics_rounded.csv', index=False)
    print('Classification metrics (0.5 rounded to 0) saved to output/charts/llm_classification_metrics_rounded.csv')
    metrics_df_clear = pd.DataFrame(metrics_rows_clear)
    # Add human annotator agreement rate (proportion of clear cases)
    agreement_rates = []
    for field in fields:
        soft_vals = soft_results[field]['soft_values']
        n_clear = np.sum((soft_vals == 0) | (soft_vals == 1))
        agreement_rate = n_clear / len(soft_vals) if len(soft_vals) > 0 else np.nan
        agreement_rates.append(agreement_rate)
    metrics_df_clear['Human_Annotator_Agreement_Rate'] = agreement_rates
    metrics_df_clear = metrics_df_clear.round(2)
    metrics_df_clear.to_csv('output/charts/llm_classification_metrics_clear.csv', index=False)
    print('Classification metrics (0.5 excluded) saved to output/charts/llm_classification_metrics_clear.csv')

    # Add gold standard by city size analysis (for testing)
    gold_standard_by_city_size()

    # Correlation analysis between gold standard variables (all rows, ignore city)
    print('Computing correlation matrix between gold standard annotation variables (all data)...')
    gold_df = pd.read_csv(RAW_SCORES_FILE, header=1)
    label_columns = [
        'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
        'provide an observation', 'express their opinion', 'express others opinions',
        'money aid allocation', 'government critique', 'societal critique',
        'solutions/interventions', 'personal interaction', 'media portrayal',
        'not in my backyard', 'harmful generalization', 'deserving/undeserving', 'racist'
    ]
    corr_matrix = gold_df[label_columns].corr()
    corr_matrix.to_csv('output/charts/gold_standard_correlation_matrix.csv')
    print('Gold standard correlation matrix saved to output/charts/gold_standard_correlation_matrix.csv')
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix of Gold Standard Annotation Variables', fontsize=16, fontweight='bold')
    ax.set_xlabel('Variable', fontweight='bold')
    ax.set_ylabel('Variable', fontweight='bold')
    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('output/charts/gold_standard_correlation_matrix.pdf')
    plt.close()
    print('Gold standard correlation matrix heatmap saved to output/charts/gold_standard_correlation_matrix.pdf')

    # Correlation analysis between soft label variables (treat 0.5 as 0, ignore city)
    print('Computing correlation matrix between soft label variables (0.5 as 0, all data)...')
    soft_label_columns = [
        'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
        'provide an observation', 'express their opinion', 'express others opinions',
        'money aid allocation', 'government critique', 'societal critique',
        'solutions/interventions', 'personal interaction', 'media portrayal',
        'not in my backyard', 'harmful generalization', 'deserving/undeserving', 'racist'
    ]
    # Treat 0.5 as 0
    soft_bin_df = soft_labels_df[soft_label_columns].replace(0.5, 0)
    corr_matrix_soft = soft_bin_df.corr()
    corr_matrix_soft.to_csv('output/charts/soft_label_correlation_matrix.csv')
    print('Soft label correlation matrix saved to output/charts/soft_label_correlation_matrix.csv')
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_matrix_soft, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix of Soft Label Variables (0.5 as 0)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Variable', fontweight='bold')
    ax.set_ylabel('Variable', fontweight='bold')
    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('output/charts/soft_label_correlation_matrix.pdf')
    plt.close()
    print('Soft label correlation matrix heatmap saved to output/charts/soft_label_correlation_matrix.pdf')

if __name__ == "__main__":
    main() 