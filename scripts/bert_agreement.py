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

# Mapping from soft label columns to BERT classification columns
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
    'racist': 'Racist_Flag'
}

# File paths
BERT_CLASSIFIED_FILE = 'output/classified_comments_reddit_gold_subset_bert.csv'
SOFT_LABELS_FILE = 'output/annotation/reddit_soft_labels.csv'


def load_classifications():
    """Load BERT classification results and soft labels."""
    try:
        print("Loading classification files...")
        bert_df = pd.read_csv(BERT_CLASSIFIED_FILE)
        soft_labels_df = pd.read_csv(SOFT_LABELS_FILE)
        print(f"Loaded {len(bert_df)} comments from BERT")
        print(f"Soft labels length: {len(soft_labels_df)}")
        print("\nSoft labels columns:")
        print(soft_labels_df.columns.tolist())
        print("\nBERT classification columns:")
        print(bert_df.columns.tolist())
        return bert_df, soft_labels_df
    except Exception as e:
        print(f"Error loading classification files: {e}")
        exit(1)


def calculate_kappa(bert_df, soft_labels_df, field):
    """Calculate Cohen's Kappa for a specific field."""
    # Get the values for BERT predictions
    bert_values = bert_df[field].fillna(0).astype(int)
    
    # Get the values for soft labels (convert to binary)
    soft_values = soft_labels_df[field].fillna(0)
    soft_binary = (soft_values > 0.5).astype(int)
    
    # Calculate kappa
    kappa = cohen_kappa_score(soft_binary, bert_values)
    
    # Create confusion matrix
    cm = confusion_matrix(soft_binary, bert_values, labels=[0, 1])
    
    # Calculate metrics
    f1 = f1_score(soft_binary, bert_values, zero_division=0)
    accuracy = accuracy_score(soft_binary, bert_values)
    precision = precision_score(soft_binary, bert_values, zero_division=0)
    recall = recall_score(soft_binary, bert_values, zero_division=0)
    
    return kappa, cm, f1, accuracy, precision, recall


def calculate_soft_label_agreement(bert_df, soft_labels_df, field):
    """Calculate agreement with soft labels."""
    bert_values = bert_df[field_map[field]].fillna(0).astype(int)
    soft_values = soft_labels_df[field].fillna(0)
    
    # Calculate correlation
    correlation = np.corrcoef(bert_values, soft_values)[0, 1]
    
    # Calculate agreement for clear cases (0 or 1 in soft labels)
    clear_mask = (soft_values == 0) | (soft_values == 1)
    if clear_mask.sum() > 0:
        bert_clear = bert_values[clear_mask]
        soft_clear = soft_values[clear_mask].astype(int)
        clear_agreement = accuracy_score(soft_clear, bert_clear)
    else:
        clear_agreement = np.nan
    
    return {
        'correlation': correlation,
        'clear_agreement': clear_agreement,
        'soft_values': soft_values,
        'bert_values': bert_values
    }


def create_confusion_matrices(bert_df, soft_labels_df, fields):
    """Create confusion matrices for all fields."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, field in enumerate(fields):
        if i >= len(axes):
            break
            
        kappa, cm, f1, accuracy, precision, recall = calculate_kappa(bert_df, soft_labels_df, field)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[i])
        axes[i].set_title(f'{field}\nKappa: {kappa:.3f}, F1: {f1:.3f}')
        axes[i].set_xlabel('BERT Prediction')
        axes[i].set_ylabel('Gold Standard')
    
    # Hide unused subplots
    for i in range(len(fields), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('output/charts/bert_confusion_matrices.pdf', bbox_inches='tight')
    plt.close()


def create_correlation_matrix(bert_df, soft_labels_df, fields):
    """Create correlation matrix between BERT predictions and soft labels."""
    correlations = []
    
    for field in fields:
        bert_values = bert_df[field_map[field]].fillna(0).astype(int)
        soft_values = soft_labels_df[field].fillna(0)
        correlation = np.corrcoef(bert_values, soft_values)[0, 1]
        correlations.append(correlation)
    
    # Create correlation matrix
    corr_df = pd.DataFrame({
        'Field': fields,
        'Correlation': correlations
    })
    
    corr_df.to_csv('output/charts/bert_correlation_matrix.csv', index=False)
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(fields)), correlations)
    plt.xticks(range(len(fields)), fields, rotation=45, ha='right')
    plt.ylabel('Correlation with Gold Standard')
    plt.title('BERT vs Gold Standard Correlation')
    plt.tight_layout()
    plt.savefig('output/charts/bert_correlation_matrix.pdf', bbox_inches='tight')
    plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate agreement between BERT and gold standard')
    args = parser.parse_args()
    
    # Load the classification results
    bert_df, soft_labels_df = load_classifications()
    
    # Ensure we're comparing the same comments
    assert len(bert_df) == len(soft_labels_df), "Number of comments must match"
    
    # Fields to analyze
    fields = list(field_map.keys())
    
    # Calculate agreement statistics
    print("\nCalculating agreement statistics...")
    results = {}
    for field in tqdm(fields, desc="Fields"):
        kappa, cm, f1, accuracy, precision, recall = calculate_kappa(
            bert_df, soft_labels_df, field
        )
        results[field] = {
            'kappa': kappa,
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }
    
    # Calculate agreement with soft labels
    print("\nCalculating agreement with soft labels...")
    soft_results = {}
    for field in tqdm(fields, desc="Fields"):
        soft_results[field] = calculate_soft_label_agreement(
            bert_df, soft_labels_df, field
        )
    
    # Print results
    print("\n=== BERT vs Gold Standard Agreement ===")
    print(f"{'Field':<25} {'Kappa':<8} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8}")
    print("-" * 70)
    
    for field in fields:
        r = results[field]
        print(f"{field:<25} {r['kappa']:<8.3f} {r['f1']:<8.3f} {r['accuracy']:<8.3f} {r['precision']:<8.3f} {r['recall']:<8.3f}")
    
    # Calculate mean metrics
    mean_kappa = np.mean([results[f]['kappa'] for f in fields])
    mean_f1 = np.mean([results[f]['f1'] for f in fields])
    mean_accuracy = np.mean([results[f]['accuracy'] for f in fields])
    mean_precision = np.mean([results[f]['precision'] for f in fields])
    mean_recall = np.mean([results[f]['recall'] for f in fields])
    
    print("-" * 70)
    print(f"{'MEAN':<25} {mean_kappa:<8.3f} {mean_f1:<8.3f} {mean_accuracy:<8.3f} {mean_precision:<8.3f} {mean_recall:<8.3f}")
    
    # Save detailed results
    results_data = []
    for field in fields:
        r = results[field]
        s = soft_results[field]
        results_data.append({
            'Field': field,
            'Kappa': r['kappa'],
            'F1': r['f1'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'Correlation': s['correlation'],
            'Clear_Agreement': s['clear_agreement']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('output/charts/bert_agreement_results.csv', index=False)
    print(f"\nDetailed results saved to output/charts/bert_agreement_results.csv")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_confusion_matrices(bert_df, soft_labels_df, fields)
    create_correlation_matrix(bert_df, soft_labels_df, fields)
    print("Visualizations saved to output/charts/")


if __name__ == "__main__":
    main() 